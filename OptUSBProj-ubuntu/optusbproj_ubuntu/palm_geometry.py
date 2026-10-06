"""Left palm RGB-D geometry and held-out C0 fit; surface estimates, not GT."""
import numpy as np
from scipy.spatial.transform import Rotation
from .rgbd_pose import camera_matrix,distortion_coefficients
from .natural_hand import proper

PALM_IDS=(0,5,9,13,17)


def unit(v):
    v=np.asarray(v,float);length=np.linalg.norm(v)
    if not np.isfinite(v).all() or length<1e-8:raise ValueError('掌面方向退化')
    return v/length


def angle(a,b):return float(np.degrees(np.arccos(np.clip(unit(a)@unit(b),-1,1))))


def distance(a,b):return float(np.degrees(Rotation.from_matrix(proper(a).T@proper(b)).magnitude()))


def frame_from_points(points,cloud):
    """Semantic signed axes + robust PCA plane; left dorsum +X, proximal +Z."""
    p=np.asarray(points,float);q=np.asarray(cloud,float)
    if p.shape!=(5,3) or not np.isfinite(p).all() or q.ndim!=2 or q.shape[1]!=3 or len(q)<40 or not np.isfinite(q).all():raise ValueError('掌部深度有效点不足')
    longitudinal=p[0]-np.mean(p[1:],axis=0);radial=p[1]-p[4]
    if not .025<=np.linalg.norm(longitudinal)<=.16 or not .025<=np.linalg.norm(radial)<=.16:raise ValueError('掌部尺寸异常')
    if np.linalg.norm(np.cross(unit(longitudinal),unit(radial)))<.25:raise ValueError('掌部关键点接近共线')
    semantic=unit(np.cross(longitudinal,radial))
    # Iteratively remove points outside a robust, small local plane support.
    original=len(q)
    for _ in range(3):
        mu=q.mean(axis=0);cov=(q-mu).T@(q-mu)/len(q)
        values,vectors=np.linalg.eigh(cov);n=vectors[:,0]
        residual=np.abs((q-mu)@n);mad=np.median(residual)
        mask=residual<=max(.004,3*1.4826*mad)
        if mask.all():break
        q=q[mask]
        if len(q)<40:raise ValueError('掌面平面内点不足')
    mu=q.mean(axis=0);values,vectors=np.linalg.eigh((q-mu).T@(q-mu)/len(q));n=vectors[:,0]
    rms=float(np.sqrt(np.mean(((q-mu)@n)**2)))
    if len(q)/original<.7 or values[1]<1e-5 or values[0]/values[1]>.2 or rms>.006:raise ValueError('掌面平面不稳定/深度混合')
    if n@semantic<0:n=-n
    disagreement=angle(n,semantic)
    if disagreement>25:raise ValueError('点云法向量与掌部语义方向不一致')
    # Require a palmar view in v1 to avoid fitting the IMU's dorsal casing.
    facing=float(n@unit(-mu))
    if facing>-.3:raise ValueError('请让左掌心朝相机，避免侧面或手背设备')
    z=unit(longitudinal-n*(n@longitudinal));y=unit(np.cross(z,n));x=unit(np.cross(y,z))
    frame=proper(np.column_stack((x,y,z)))
    semantic_z=unit(longitudinal);semantic_y=unit(np.cross(semantic_z,semantic))
    semantic_frame=proper(np.column_stack((semantic,semantic_y,semantic_z)))
    return frame,dict(points=len(q),inlier_fraction=len(q)/original,plane_rms_mm=rms*1000,
        eigenvalues=values.tolist(),semantic_disagreement_deg=disagreement,view_cosine=facing,
        R_semantic_camera_hand=semantic_frame.tolist())


def lift_palm(observation,xy):
    """Distorted RGB rays -> aligned depth pixels -> COLOR optical metric frame."""
    import cv2
    xy=np.asarray(xy,float)
    if xy.shape!=(21,2) or not np.isfinite(xy).all() or observation.rgbd is None:raise ValueError('缺少手部像素/真实深度')
    c=observation.rgbd.calibration;rgb=observation.frame_rgb;depth=observation.rgbd.aligned_depth_mm
    kc=camera_matrix(c['color_intrinsic']);kd=camera_matrix(c['aligned_depth_intrinsic'])
    dc=distortion_coefficients(c['color_distortion']);dd=distortion_coefficients(c['aligned_depth_distortion'])
    if (c['color_intrinsic']['width'],c['color_intrinsic']['height'])!=(rgb.shape[1],rgb.shape[0]) or (c['aligned_depth_intrinsic']['width'],c['aligned_depth_intrinsic']['height'])!=(depth.shape[1],depth.shape[0]):raise ValueError('掌部内参与图像尺寸不一致')
    selected=xy[list(PALM_IDS)]
    if (selected<0).any() or (selected[:,0]>=rgb.shape[1]).any() or (selected[:,1]>=rgb.shape[0]).any():raise ValueError('掌部关键点越界')
    if np.linalg.norm(selected[1]-selected[4])<16:raise ValueError('掌部像素过小，请靠近相机')
    rays=cv2.undistortPoints(selected.reshape(-1,1,2),kc,dc).reshape(-1,2)
    projected=cv2.projectPoints(np.c_[rays,np.ones(5)],np.zeros(3),np.zeros(3),kd,dd)[0].reshape(-1,2)
    points=[]
    for ray,(u,v) in zip(rays,projected):
        x,y=np.rint([u,v]).astype(int)
        if x<3 or y<3 or x+3>=depth.shape[1] or y+3>=depth.shape[0]:raise ValueError('掌部深度越界')
        patch=depth[y-2:y+3,x-2:x+3];valid=patch[(patch>=250)&(patch<=4000)]
        if len(valid)<18:raise ValueError('掌部深度空洞')
        if np.percentile(valid,90)-np.percentile(valid,10)>30:raise ValueError('掌部关键点处深度混合')
        points.append(np.r_[ray,1]*np.median(valid)/1000)
    mask=np.zeros(depth.shape,np.uint8);hull=cv2.convexHull(np.rint(projected).astype(np.int32))
    cv2.fillConvexPoly(mask,hull,1);mask=cv2.erode(mask,np.ones((3,3),np.uint8))
    ys,xs=np.nonzero(mask);ys,xs=ys[::2],xs[::2]
    if len(xs)>3000:indices=np.linspace(0,len(xs)-1,3000).astype(int);ys,xs=ys[indices],xs[indices]
    z=depth[ys,xs];expected=np.median(np.asarray(points)[:,2])*1000
    good=(z>=250)&(z<=4000)&(abs(z-expected)<=50)
    if len(good)<40 or good.mean()<.7:raise ValueError('掌部区域深度支持不足')
    pixel=np.c_[xs[good],ys[good]].astype(float)
    cloud_rays=cv2.undistortPoints(pixel.reshape(-1,1,2),kd,dd).reshape(-1,2)
    cloud=np.c_[cloud_rays,np.ones(len(pixel))]*(z[good,None]/1000)
    frame,metrics=frame_from_points(points,cloud)
    return frame,metrics,np.asarray(points)


def estimate_c0(training,validation,manual):
    """Rows contain pure-IMU D and independently observed model palm C."""
    if len(training)<10 or len(validation)<10:raise ValueError('拟合和独立复测均至少需要10帧同期可信掌面')
    estimates=[proper(r['D']).T@proper(r['C']) for r in training]
    s=sum(estimates);u,_,vt=np.linalg.svd(s);c0=proper(u@np.diag([1,1,np.linalg.det(u@vt)])@vt)
    residual=np.array([distance(a,c0) for a in estimates]);keep=residual<=max(8.,float(np.median(residual)*2.5))
    if keep.sum()<10 or keep.mean()<.7:raise ValueError('初始掌面反推结果不一致，请重录')
    s=sum(a for a,k in zip(estimates,keep) if k);u,_,vt=np.linalg.svd(s);c0=proper(u@np.diag([1,1,np.linalg.det(u@vt)])@vt)
    train_errors=[distance(a,c0) for a,k in zip(estimates,keep) if k]
    checks=[distance(proper(r['D'])@c0,r['C']) for r in validation]
    median=float(np.median(checks));p90=float(np.percentile(checks,90))
    manual=proper(manual)
    # Closest Z-only rotation in chordal distance; report the lost tilt.
    numerator=c0[1,0]-c0[0,1];denominator=c0[0,0]+c0[1,1]
    observable=np.hypot(numerator,denominator)>1e-6
    alpha=float(np.degrees(np.arctan2(numerator,denominator)))
    z_only=Rotation.from_euler('z',alpha,degrees=True).as_matrix()
    tilt=angle(c0[:,2],[0,0,1])
    reasons=[]
    if np.median(train_errors)>=12:reasons.append('训练方向拟合误差过大')
    if median>=12 or p90>=25:reasons.append('独立掌面复测误差过大')
    if tilt>=20:reasons.append('自然长轴偏离模型Z超过20°，检查身体坐标对齐/自然姿态/映射')
    return dict(accepted=bool(np.median(train_errors)<12 and median<12 and p90<25 and tilt<20),neutral_long_axis_tilt_deg=tilt,
        C0_visual=c0.tolist(),C0_manual=manual.tolist(),training_samples=len(training),training_inliers=int(keep.sum()),
        training_median_deg=float(np.median(train_errors)),validation_samples=len(validation),validation_median_deg=median,validation_p90_deg=p90,
        manual_visual_so3_deg=distance(manual,c0),axis_difference_deg=[angle(manual[:,i],c0[:,i]) for i in range(3)],
        closest_z_offset_deg=alpha if observable else None,z_only_yaw_observable=bool(observable),
        z_only_residual_deg=distance(z_only,c0),reason='；'.join(reasons),
        note='表面几何与IMU的一致性，不是GT；完整SO(3)参考，不是长期航向纠偏')
