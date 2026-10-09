"""Geometry association with optional appearance-verified bounded recovery.

Detection indices are frame-local; OSNet similarity is not biometric identity.
"""
import uuid

import numpy as np


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(area/max(1e-12, np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area))


def torso_depth(observation, box):
    """Median surface depth from an RGB torso grid, mapped through calibration."""
    if observation is None or observation.rgbd is None:
        return None
    import cv2
    from .rgbd_pose import camera_matrix, distortion_coefficients
    c = observation.rgbd.calibration
    x1, y1, x2, y2 = box
    xs = np.linspace(x1+.35*(x2-x1), x1+.65*(x2-x1), 5)
    ys = np.linspace(y1+.25*(y2-y1), y1+.55*(y2-y1), 5)
    pixels = np.array([(x,y) for y in ys for x in xs], float)
    rays = cv2.undistortPoints(pixels[:,None,:], camera_matrix(c['color_intrinsic']),
                               distortion_coefficients(c['color_distortion'])).reshape(-1,2)
    mapped, _ = cv2.projectPoints(np.c_[rays,np.ones(len(rays))], np.zeros(3), np.zeros(3),
        camera_matrix(c['aligned_depth_intrinsic']), distortion_coefficients(c['aligned_depth_distortion']))
    points = np.rint(mapped.reshape(-1,2)).astype(int)
    depth = observation.rgbd.aligned_depth_mm
    h,w = depth.shape
    values = [float(depth[y,x])/1000 for x,y in points if 0<=x<w and 0<=y<h
              and 250<=depth[y,x]<=6000]
    if len(values)<13 or np.percentile(values,90)-np.percentile(values,10)>.3:
        return None
    return float(np.median(values))


class PersonTarget:
    def __init__(self, encoder=None):
        self.encoder = encoder
        self.reset()

    def reset(self, anchor=None, *, mode='auto', selection_id=None):
        if mode not in ('auto','manual','paused'):
            raise ValueError('Unknown person selection mode')
        self.mode = mode
        self.selection_id = selection_id or uuid.uuid4().hex
        self.anchor = dict(anchor) if anchor is not None else None
        self.last_stamp = int(anchor['captured_mono_ns']) if anchor else None
        self.latched = False
        self.gallery = []
        self.gallery_frozen = False
        self.pending = None
        self.pending_count = 0
        self.pending_since = None
        self.pending_stamp = None
        self.last_feature = None
        self.gallery_cursor = 3

    def _enroll(self, feature, chosen, candidates, stable_geometry):
        """Keep bootstrap references and diverse temporally continuous poses.

        Never adapt during overlap, reacquisition, or missing depth. A changed
        body crop is not evidence of a new identity during continuous tracking.
        """
        clean=(not self.gallery_frozen and not self._ambiguous(chosen,candidates)
               and chosen['depth_m'] is not None)
        if clean and len(self.gallery)<3:
            self.gallery.append(feature.copy())
        elif (clean and stable_geometry and self.last_feature is not None
              and float(feature@self.last_feature)>=.8
              and float(np.max(np.asarray(self.gallery)@feature))<.97):
            if len(self.gallery)<20:self.gallery.append(feature.copy())
            else:
                self.gallery[self.gallery_cursor]=feature.copy()
                self.gallery_cursor=3+(self.gallery_cursor-3+1)%17
        self.last_feature=feature.copy()

    def _ambiguous(self, chosen, candidates):
        for other in candidates:
            if other is chosen: continue
            a,b=chosen['depth_m'],other['depth_m']
            if overlap(chosen['normalized_box'],other['normalized_box'])>.25 and (
                    a is None or b is None or abs(a-b)<.5): return True
        return False

    def _recover(self, candidates, features, stamp, info):
        """No identity reset and no gallery adaptation while suspended."""
        def wait(reason):
            self.pending=None;self.pending_count=0;self.pending_since=None;self.pending_stamp=None
            info.update(target_state=reason,reject_reason=reason)
            return None,info
        if len(self.gallery)<3 or features is None:
            return wait('目标丢失或身份有歧义；无外观参考，请重新选择人物')
        if stamp<=self.last_stamp:
            return wait('目标帧时间无效；继续暂停纠偏')
        if stamp-self.last_stamp>30_000_000_000:
            return wait('目标丢失超过30秒；请人工确认人物')
        ranked=sorted([(float(np.max(np.asarray(self.gallery)@features[c['index']])),c)
                       for c in candidates],key=lambda x:x[0],reverse=True)
        if not ranked:return wait('等待原标定者重新出现')
        score,chosen=ranked[0]
        margin=score-ranked[1][0] if len(ranked)>1 else 1.
        info.update(reid_similarity=score,reid_margin=margin)
        old=np.asarray(self.anchor['normalized_box']);box=np.asarray(chosen['normalized_box'])
        ratio=(box[2:]-box[:2])/(old[2:]-old[:2])
        a,b=self.anchor.get('depth_m'),chosen['depth_m']
        geometry=(np.all((ratio>=.35)&(ratio<=3)) and a is not None and b is not None
                  and abs(a-b)<=.6 and np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2)<=.4)
        if score<.85 or margin<.08 or not geometry or self._ambiguous(chosen,candidates):
            return wait('外观/深度或身份间隔不足；继续暂停纠偏')
        if (self.pending is None or overlap(self.pending,box)<.4
                or stamp<=self.pending_stamp or stamp-self.pending_stamp>500_000_000):
            self.pending_count=0;self.pending_since=stamp
        self.pending=box.copy();self.pending_count+=1;self.pending_stamp=stamp
        if self.pending_count<5 or stamp-self.pending_since<400_000_000:
            info.update(target_state=f'原目标恢复验证 {min(self.pending_count,5)}/5（至少0.4秒）',
                        reject_reason='连续多帧确认身份中；纠偏暂停')
            return None,info
        self.latched=False;self.pending=None;self.pending_count=0
        self.anchor=chosen.copy();self.last_stamp=stamp
        self.last_feature=features[chosen['index']].copy()
        info.update(selected_index=chosen['index'],target_state='原目标身份已验证，恢复跟踪',reid_recovered=True)
        return chosen['index'],info

    def choose(self, boxes, observation):
        boxes = np.asarray(boxes, float)
        if boxes.size==0:boxes=np.empty((0,4))
        if boxes.ndim!=2 or boxes.shape[1]<4 or not np.isfinite(boxes[:,:4]).all():
            raise ValueError('Invalid person bounding boxes')
        h,w = observation.frame_rgb.shape[:2]
        stamp = observation.captured_mono_ns
        candidates=[]
        for index, box in enumerate(boxes[:,:4]):
            box = np.clip(box, [0,0,0,0], [w,h,w,h])
            if np.any(box[2:]<=box[:2]):continue
            candidates.append(dict(index=index, box=box.tolist(),
                normalized_box=(box/np.array([w,h,w,h])).tolist(),
                depth_m=torso_depth(observation,box), captured_mono_ns=stamp))
        info = dict(people_detected=len(boxes), person_candidates=candidates,
                    target_id=self.selection_id, target_mode=self.mode, selected_index=None)
        features=None
        if self.encoder is not None and candidates:
            vectors=self.encoder(observation.frame_rgb,[c['box'] for c in candidates])
            if vectors.ndim!=2 or vectors.shape[0]!=len(candidates) or not np.isfinite(vectors).all():
                raise ValueError('无效人物外观特征')
            norms=np.linalg.norm(vectors,axis=1,keepdims=True)
            if np.any(norms<1e-8):raise ValueError('空人物外观特征')
            features={c['index']:v for c,v in zip(candidates,vectors/norms)}
        info['reid_enabled']=self.encoder is not None
        if self.encoder is not None:info['reid_config']=getattr(self.encoder,'metadata',{})
        def reject(reason, latch=False):
            self.latched |= latch
            if latch:self.gallery_frozen=True
            if latch and self.encoder is not None and len(self.gallery)>=3:
                reason=reason.replace('请重新选择人物','暂停，等待原目标身份恢复验证')
            info.update(target_state=reason, reject_reason=reason)
            return None, info
        if self.mode=='paused':return reject('人物选择已取消；视觉纠偏暂停')
        if self.latched:return self._recover(candidates,features,stamp,info)
        stable_geometry=False
        if self.anchor is None:
            if self.mode!='auto' or len(candidates)!=1:
                return reject('未检测到人' if not candidates else '多人画面：请点击人物框或选择目标')
            chosen=candidates[0]
        else:
            if stamp<=self.last_stamp or stamp-self.last_stamp>1_000_000_000:
                return reject('目标帧间隔过大；请重新选择人物',True)
            old=np.asarray(self.anchor['normalized_box'])
            center=(old[:2]+old[2:])/2
            scale=max(.05,float(np.linalg.norm(old[2:]-old[:2])))
            fits=[]
            for item in candidates:
                box=np.asarray(item['normalized_box'])
                shift=np.linalg.norm((box[:2]+box[2:])/2-center)/scale
                ratio=(box[2:]-box[:2])/(old[2:]-old[:2])
                a,b=self.anchor.get('depth_m'),item['depth_m']
                # Arm elevation widens the full-person box; its width is not
                # a rigid body measurement. Preserve centre/depth continuity.
                if (overlap(old,box)>=.25 and shift<=.35 and np.all((ratio>=.35)&(ratio<=3))
                        and (a is None or b is None or abs(a-b)<=.35)):
                    fits.append(item)
            if len(fits)!=1:
                return reject('目标丢失；请重新选择人物' if not fits else '多人交叉，身份有歧义；请重新选择人物',True)
            chosen=fits[0]
            if features is not None and self.gallery:
                info['reid_tracking_similarity']=float(np.max(np.asarray(self.gallery)@features[chosen['index']]))
            stable_geometry=(self.anchor.get('depth_m') is not None and chosen['depth_m'] is not None)
            # Even if only one passes temporal gates, a nearby occluder could
            # have replaced the target. Depth-separated people are less ambiguous.
            for other in candidates:
                if other is chosen:continue
                a,b=chosen['depth_m'],other['depth_m']
                if overlap(chosen['normalized_box'],other['normalized_box'])>.25 and (
                        a is None or b is None or abs(a-b)<.5):
                    return reject('人物框重叠，身份有歧义；请重新选择人物',True)
        self.anchor=chosen.copy()
        self.last_stamp=stamp
        if features is not None:self._enroll(features[chosen['index']],chosen,candidates,stable_geometry)
        state='目标已锁定（连续几何跟踪，OSNet 恢复待命）' if self.encoder is not None else '目标已锁定（短时几何跟踪）'
        if self.encoder is not None and len(self.gallery)<3:state+=f' · 外观参考采集 {len(self.gallery)}/3'
        info.update(selected_index=chosen['index'], target_state=state)
        return chosen['index'],info
