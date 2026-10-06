"""Latest-frame CPU hand inference; no camera/UDP ownership, no pose writes."""
from pathlib import Path
from threading import Condition
import hashlib
import time
import numpy as np
from PySide6 import QtCore
from .palm_geometry import lift_palm
from .rgbd_pose import camera_matrix,distortion_coefficients

DEFAULT_HAND_MODEL=Path(__file__).resolve().parents[2]/'no-improtent/hand-models/hand_landmarker.task'


def draw_palm_diagnostics(rgb,row,origin=(0,0)):
    """Diagnostic overlay only; never changes the observation or acceptance."""
    import cv2
    image=rgb.copy();offset=np.asarray(origin)
    def pixel(point):return tuple(np.rint(np.asarray(point)-offset).astype(int))
    roi=row.get('roi_xyxy')
    if roi:cv2.rectangle(image,pixel(roi[:2]),pixel(roi[2:]),(255,200,0),1)
    for points in row.get('detected_hands_xy',[]):
        for start,end in ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)):
            cv2.line(image,pixel(points[start]),pixel(points[end]),(255,255,0),1)
        for point in points:cv2.circle(image,pixel(point),2,(255,255,0),-1)
        cv2.circle(image,pixel(points[0]),6,(255,0,255),2)
    wrist=row.get('body_wrist_xy')
    if wrist is not None:
        cv2.circle(image,pixel(wrist),7,(0,255,255),2)
        for points in row.get('detected_hands_xy',[]):cv2.line(image,pixel(wrist),pixel(points[0]),(255,0,255),1)
    if row.get('accepted') and 'axes_crop_xy' in row and roi:
        axes=np.asarray(row['axes_crop_xy'])+np.asarray(roi[:2])
        for i,color in enumerate(((255,0,0),(0,255,0),(0,100,255))):cv2.line(image,pixel(axes[0]),pixel(axes[i+1]),color,2)
    label=f"2D hands:{row.get('detected_hand_count','?')} 3D:{'OK' if row.get('accepted') else 'REJECT'}"
    if 'wrist_distance_px' in row:label+=f" wrist:{row['wrist_distance_px']:.0f}/{row['wrist_gate_px']:.0f}px"
    cv2.putText(image,label,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return image


class PalmDetector:
    def __init__(self,path=DEFAULT_HAND_MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        path=Path(path)
        if not path.is_file():raise ValueError(f'缺少本地手部权重：{path}；不会自动下载')
        self.mp=mp;self.metadata={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mediapipe_version':mp.__version__}
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(path)),
            num_hands=2,min_hand_detection_confidence=.7,min_hand_presence_confidence=.7)
        self.detector=vision.HandLandmarker.create_from_options(options)

    def infer(self,observation,body):
        row={'captured_mono_ns':observation.captured_mono_ns,'accepted':False,'model':self.metadata,
             'detection_2d':False,'association_valid':False,'stage':'body_roi'}
        try:self._infer(observation,body,row)
        except ValueError as exc:row['reason']=str(exc)
        roi=row.get('roi_xyxy');origin=roi[:2] if roi else (0,0)
        rgb=observation.frame_rgb
        if roi:rgb=rgb[roi[1]:roi[3],roi[0]:roi[2]]
        if rgb.size:row['frame_rgb']=draw_palm_diagnostics(rgb,row,origin)
        return row

    def _infer(self,observation,body,row):
        import cv2
        rgb=observation.frame_rgb;h,w=rgb.shape[:2]
        if abs(body['captured_mono_ns']-observation.captured_mono_ns)>150_000_000:raise ValueError('身体关键点与掌部图像相差超过150ms')
        xy=np.asarray(body['keypoints_xy'],float);scores=np.asarray(body['confidence'],float)
        if not np.isfinite(scores[[7,9]]).all() or min(scores[7],scores[9])<.6 or not np.isfinite(xy[[7,9]]).all():raise ValueError('左肘腕二维观测无效，无法定位左手')
        wrist=xy[9];elbow=xy[7];centre=wrist+.3*(wrist-elbow)
        half=int(np.clip(np.linalg.norm(wrist-elbow)*.55,64,200))
        x0,y0=np.maximum(np.rint(centre-half).astype(int),0);x1,y1=np.minimum(np.rint(centre+half).astype(int),[w,h])
        crop=np.ascontiguousarray(rgb[y0:y1,x0:x1])
        row.update(roi_xyxy=[int(x0),int(y0),int(x1),int(y1)],body_wrist_xy=wrist.tolist(),body_elbow_xy=elbow.tolist())
        if min(crop.shape[:2])<64:raise ValueError('左手裁剪区域越界/过小')
        row['stage']='detection_2d'
        result=self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=crop))
        hands=[np.array([[p.x*crop.shape[1]+x0,p.y*crop.shape[0]+y0] for p in hand]) for hand in result.hand_landmarks]
        row.update(detected_hand_count=len(hands),detection_2d=bool(hands),detected_hands_xy=[p.tolist() for p in hands])
        if len(hands)!=1:raise ValueError(f'左手区域未检出唯一手掌（检出{len(hands)}个）')
        row['stage']='association';points=hands[0]
        dist=np.linalg.norm(points[0]-wrist)
        row.update(keypoints_xy=points.tolist(),wrist_distance_px=float(dist),wrist_gate_px=float(max(30.,half*.45)))
        if dist>max(30.,half*.45):raise ValueError('手部腕点与人体左腕不一致')
        if np.isfinite(xy[10]).all() and scores[10]>.6 and np.linalg.norm(points[0]-xy[10])<dist+20:raise ValueError('左右手关联不明确')
        row.update(association_valid=True,stage='depth_geometry')
        frame,quality,metric=lift_palm(observation,points)
        c=observation.rgbd.calibration
        endpoints=np.vstack([metric[0],(metric[0][:,None]+.04*frame).T])
        pixels=cv2.projectPoints(endpoints,np.zeros(3),np.zeros(3),camera_matrix(c['color_intrinsic']),distortion_coefficients(c['color_distortion']))[0].reshape(-1,2)-[x0,y0]
        row['axes_crop_xy']=pixels.tolist()
        category=result.handedness[0][0]
        row.update(accepted=True,stage='accepted',reason='掌面有效',R_camera_hand=frame.tolist(),quality=quality,
            keypoints_xy=points.tolist(),palm_points_camera_m=metric.tolist(),handedness_label=category.category_name,handedness_score=category.score,
            handedness_note='分数表示左右手分类，不是逐关键点置信度；左手身份来自人体腕点关联')

    def close(self):self.detector.close()


class PalmWorker(QtCore.QThread):
    error=QtCore.Signal(str)
    def __init__(self,parent=None,*,model=DEFAULT_HAND_MODEL):
        super().__init__(parent);self.model=model;self.condition=Condition();self.running=True;self.job=None;self.result=None

    def submit(self,observation,body):
        with self.condition:
            if self.running:self.job=(observation,body);self.condition.notify()

    def take_latest(self):
        with self.condition:result,self.result=self.result,None;return result

    def stop(self):
        with self.condition:self.running=False;self.job=None;self.condition.notify()

    def run(self):
        detector=None
        try:
            detector=PalmDetector(self.model)
            while True:
                with self.condition:
                    while self.running and self.job is None:self.condition.wait(.5)
                    if not self.running:break
                    observation,body=self.job;self.job=None
                try:row=detector.infer(observation,body)
                except ValueError as exc:row={'captured_mono_ns':observation.captured_mono_ns,'accepted':False,'reason':str(exc),'model':detector.metadata}
                row['completed_mono_ns']=time.monotonic_ns()
                with self.condition:self.result=row
        except Exception as exc:self.error.emit('手部视觉观测失败：'+str(exc))
        finally:
            if detector is not None:detector.close()
