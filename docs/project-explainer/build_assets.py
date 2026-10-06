"""Reproducible, offline illustrations. Never opens sockets or hardware.

Run with OptUSBProj-yolo Python from any directory. Input recordings are read-only.
Only this page's assets and provenance JSON are generated; no calibration is applied
to the running GUI. Robot pictures use the real URDF, NOT RViz screenshots.
"""
from pathlib import Path
import sys
import json
import hashlib
import time
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'OptUSBProj-ubuntu')]
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PARTS
from optusbproj_ubuntu.x1_live import LiveRetargeter
from optusbproj_ubuntu.x1_retarget import _origin_transform
from optusbproj_ubuntu.pose_2d import Pose2DConfig, load_backend

OUT = Path(__file__).parent/'assets'
SESSION = ROOT/'OptUSBProj-ubuntu/recordings/20261004-135948-f649c2'
ANALYSIS = ROOT/'OptUSBProj-ubuntu/recordings/analysis_combined/20261004-135948-f649c2-00392a'
URDF = ROOT/'urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf'
MANIFEST = {'note': 'Real observations, computed reconstructions and schematics are distinct. No ground truth claims.', 'assets': {}}


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def register(name, kind, **data):
    p = OUT/name
    MANIFEST['assets'][name] = dict(kind=kind, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), **data)


def savefig(name, **data):
    plt.savefig(OUT/name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    register(name, **data)


def overlay(image, xy, conf):
    out = image.copy()
    for a, b in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12)]:
        if min(conf[a],conf[b]) >= .5 and np.isfinite(xy[[a,b]]).all():
            cv2.line(out, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0,210,255), 3, cv2.LINE_AA)
    for i, name in [(5,'LS'),(6,'RS'),(7,'LE'),(8,'RE'),(9,'LW'),(10,'RW'),(11,'LH'),(12,'RH')]:
        if conf[i]>=.5 and np.isfinite(xy[i]).all():
            pt=tuple(xy[i].astype(int))
            cv2.circle(out,pt,6,(100,255,60),-1,cv2.LINE_AA)
            cv2.putText(out,name,(pt[0]+8,pt[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    return out


def render_meshes(name, meshes, matrices, colors, label):
    plot=pv.Plotter(off_screen=True, window_size=(960,900))
    plot.set_background('#edf2f7')
    for mesh,matrix,color in zip(meshes,matrices,colors):
        actor=plot.add_mesh(mesh,color=color,smooth_shading=True)
        actor.user_matrix=matrix
    plot.add_text(label,position='upper_left',font_size=13,color='#172b4d')
    plot.show_axes()
    plot.camera_position=[(2.8,-3.2,1.3),(0,0,.15),(0,0,1)]
    plot.reset_camera()
    plot.camera.zoom(1.05)
    plot.screenshot(OUT/name)
    plot.close()


def robot_geometry(q):
    root=ET.parse(URDF).getroot()
    transforms={'world':np.eye(4)}
    pending=list(root.findall('joint'))
    while pending:
        remaining=[]
        for j in pending:
            parent=j.find('parent').get('link'); child=j.find('child').get('link')
            if parent not in transforms:
                remaining.append(j); continue
            motion=np.eye(4)
            axis=j.find('axis')
            if axis is not None and j.get('type') in ('revolute','continuous'):
                a=np.fromstring(axis.get('xyz'),sep=' '); a/=np.linalg.norm(a)
                motion[:3,:3]=Rotation.from_rotvec(a*q.get(j.get('name'),0)).as_matrix()
            transforms[child]=transforms[parent]@_origin_transform(j)@motion
        if len(remaining)==len(pending): raise ValueError('Disconnected URDF')
        pending=remaining
    meshes=[]; matrices=[]
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            path=ROOT/'urdf-model/pkg_robot_model'/mesh.get('filename').split('package://pkg_robot_model/')[-1]
            obj=pv.read(path)
            obj.points*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            meshes.append(obj);matrices.append(transforms[link.get('name')]@_origin_transform(visual))
    return meshes,matrices


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cameras=rows(SESSION/'camera.jsonl'); vision=rows(SESSION/'vision.jsonl')
    runtime=rows(SESSION/'runtime_pose.jsonl')
    manifest=json.loads((SESSION/'manifest.json').read_text())
    t0=manifest['started_mono_ns']
    report=json.loads((ANALYSIS/'wearing.json').read_text())
    parts=report['parts']; assignments={p:d['device_id'] for p,d in parts.items()}
    samples={}
    for p,d in parts.items():
        samples[p]=[r for r in runtime if r['device_id']==d['device_id']]
    snapshots={}
    for label,t in [('neutral',6),('forward',15),('lateral',31)]:
        camera=min(cameras,key=lambda r:abs((r['captured_mono_ns']-t0)/1e9-t))
        v=min(vision,key=lambda r:abs(r['captured_mono_ns']-camera['captured_mono_ns']))
        image=cv2.imread(str(SESSION/camera['frame']))
        cv2.imwrite(str(OUT/f'{label}-rgb.jpg'),image)
        register(f'{label}-rgb.jpg','recorded_rgb',source=str(SESSION/camera['frame']),time_s=(camera['captured_mono_ns']-t0)/1e9)
        annotated=overlay(image,np.asarray(v['keypoints_xy'],float),np.asarray(v['confidence']))
        cv2.imwrite(str(OUT/f'{label}-keypoints.jpg'),annotated)
        register(f'{label}-keypoints.jpg','recorded_keypoints_overlay',source=str(SESSION/'vision.jsonl'),timestamp_ns=v['captured_mono_ns'],camera_skew_ms=abs(v['captured_mono_ns']-camera['captured_mono_ns'])/1e6)
        snapshots[label]=(camera,v)
    # Independently rerun the user's preferred l weights on a real lateral frame.
    backend=load_backend(Pose2DConfig('yolo26',ROOT/'no-improtent/yolo-models/yolo26l-pose.pt',960))
    image=cv2.imread(str(SESSION/snapshots['lateral'][0]['frame']))
    xy,conf,info=backend.infer(image)
    cv2.imwrite(str(OUT/'lateral-yolo-l.jpg'),overlay(image,xy,conf))
    register('lateral-yolo-l.jpg','new_offline_inference',model=backend.metadata,diagnostics=info,source=str(SESSION/snapshots['lateral'][0]['frame']),arm_confidence=conf[5:11].tolist())
    del backend
    camera,v=snapshots['lateral']
    depth=np.load(SESSION/camera['rgbd']['aligned_depth_mm'])
    masked=np.ma.masked_where((depth<250)|(depth>4000),depth)
    fig,ax=plt.subplots(figsize=(11,6)); im=ax.imshow(masked,cmap='turbo_r',vmin=700,vmax=2600)
    points=np.asarray(v['keypoints_xy'],float)
    ax.scatter(points[5:13,0],points[5:13,1],s=35,c='white',edgecolors='black')
    ax.set_title('Registered real depth / selected 2D joint pixels');ax.axis('off');fig.colorbar(im,ax=ax,label='Depth (mm)',shrink=.8)
    savefig('registered-depth.png',kind='recorded_depth_visualization',source=str(SESSION/camera['rgbd']['aligned_depth_mm']))
    xyz=np.asarray(v['points_camera_m'],float); valid=np.asarray(v['valid'],bool)
    fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(projection='3d')
    for a,b in [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]:
        if valid[a] and valid[b]:ax.plot(*xyz[[a,b]].T,linewidth=3,color='#007c91')
    for i in range(8):
        if valid[i]:ax.scatter(*xyz[i],s=45,color='#e76f51');ax.text(*xyz[i],v['joint_names'][i],fontsize=7)
    ax.set(xlabel='Camera X / m',ylabel='Camera Y / m',zlabel='Camera Z / m',title='RGB-D surface-joint reconstruction (not bone centres)')
    ax.invert_yaxis();ax.view_init(elev=20,azim=-65)
    savefig('surface-skeleton.png',kind='computed_3d_observation',source=str(SESSION/'vision.jsonl'),timestamp_ns=v['captured_mono_ns'])
    # Replay poses in software only. Use paired raw-gauge neutral, not upright fit gauge.
    engine=PoseEngine(assignments,algorithm_modes={p:'gyro_tilt' for p in parts})
    for p,d in parts.items():
        engine.apply_calibration(p,d['device_id'],np.asarray(d['basis_matrix']),np.asarray(d['neutral_matrix']),{k:np.asarray(n) for k,n in d['neutral_matrices_by_mode'].items()})
        engine.current[d['device_id']]=engine.neutral[d['device_id']].copy()
        engine.received_at[d['device_id']]=time.monotonic()
    solver=LiveRetargeter(engine,URDF,max_age_s=1000,scale_mode='uniform')
    human_meshes=[pv.read(ROOT/'OptUSBProj-ubuntu/assets/models'/p.mesh) for p in PARTS]
    for mesh in human_meshes:mesh.points*=.001
    lookup={p:np.array([r['host_recv_mono_ns'] for r in rr],dtype=np.int64) for p,rr in samples.items()}
    def set_pose(t):
        for p,d in parts.items():
            index=np.searchsorted(lookup[p],t0+int(t*1e9));index=min(index,len(samples[p])-1)
            row=samples[p][index]
            engine.current[d['device_id']]=np.asarray(row['orientation_by_mode']['gyro_tilt'])
            engine.received_at[d['device_id']]=float(t)
    recorded_joint_q={}
    for t in np.arange(6,31.051,.05):
        set_pose(float(t));pos,status=solver.step(now=float(t))
        if any(abs(t-x)<.02 for x in (6,15,31)):
            recorded_joint_q[int(round(t))]=dict(zip(('1','2','3','4','5','6','51','52','53','54','55','56'),pos))
    for label,t in [('neutral',6),('forward',15),('lateral',31)]:
        set_pose(t)
        transforms=engine.transforms(include_natural=False)
        render_meshes(f'{label}-stl.png',human_meshes,transforms,[p.color for p in PARTS],f'STL / offline gyro+gravity / t={t}s')
        register(f'{label}-stl.png','computed_stl_reconstruction',source=str(SESSION/'runtime_pose.jsonl'),wearing=str(ANALYSIS/'wearing.json'),time_s=t,mesh_count=len(PARTS))
        q=recorded_joint_q[t];meshes,matrices=robot_geometry(q)
        render_meshes(f'{label}-x1.png',meshes,matrices,['#b6c6db']*len(meshes),f'X1 URDF / offline mimic IK / t={t}s')
        register(f'{label}-x1.png','computed_urdf_reconstruction',urdf=str(URDF),joint_angles_rad=q,time_s=t,mesh_count=len(meshes),note='Same URDF/FK as RViz; not an RViz screenshot. No hand sensor, wrist orientation not controlled.')
    # Actual recorded offline direction residuals. This is a short record, not 5-min GT.
    cp=ROOT/'OptUSBProj-ubuntu/recordings/analysis_rgbd/20261004-103131-25da07-guided-b88d89/comparison.jsonl'
    comparison=rows(cp)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for ax,part in zip(axes.flat,parts):
        ts=[];pure=[];fused=[]
        for row in comparison:
            data=row.get('parts',{}).get(part)
            if not data or data.get('vision_direction') is None:continue
            d=np.asarray(data['vision_direction'],float)
            if not np.isfinite(d).all():continue
            ts.append(row['time_s'])
            for key,values in [('direction_pure',pure),('direction_fused',fused)]:
                a=np.asarray(data[key],float)
                values.append(np.degrees(np.arccos(np.clip(np.dot(a,d)/(np.linalg.norm(a)*np.linalg.norm(d)),-1,1))))
        ax.plot(ts,pure,color='#d76a33',linewidth=1,label='IMU only')
        ax.plot(ts,fused,color='#067a85',linewidth=1,label='RGB-D corrected')
        ax.set_title(part.replace('_',' '));ax.grid(alpha=.2);ax.set_ylim(0,60)
    axes[0,0].legend();axes[1,0].set_xlabel('Record time / s');axes[1,1].set_xlabel('Record time / s')
    axes[0,0].set_ylabel('Direction residual / deg');axes[1,0].set_ylabel('Direction residual / deg')
    fig.suptitle('Same-data short-record replay: disagreement with noisy RGB-D, NOT GT error')
    savefig('direction-residuals.png',kind='recorded_comparison_plot',source=str(cp),duration_s=comparison[-1]['time_s'],note='Includes fit/validation sections; not an independent 5-minute drift benchmark.')
    # Deliberately illustrative geometry: gravity does not determine heading.
    fig=plt.figure(figsize=(10,5));ax=fig.add_subplot(projection='3d')
    ax.quiver(0,0,0,0,0,1.2,color='#2171b5',linewidth=3)
    for angle,color in [(0,'#d76a33'),(50,'#067a85')]:
        a=np.deg2rad(angle);v=np.array([.85*np.cos(a),.85*np.sin(a),-.5])
        ax.quiver(0,0,0,*v,color=color,linewidth=4)
    a=np.linspace(0,2*np.pi,100);ax.plot(.85*np.cos(a),.85*np.sin(a),np.full(100,-.5),'--',color='#9aa6b3')
    ax.text(0,0,1.3,'gravity/up axis');ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-.7,1.5),title='Illustration: equal tilt, different heading')
    ax.set_axis_off();ax.view_init(elev=20,azim=-65)
    savefig('yaw-observability.png',kind='mathematical_schematic',note='Synthetic vectors; not measured drift or device performance.')
    MANIFEST['reconstruction_note']='Raw gyro_tilt matrices sampled at requested times; paired neutral+B from the same capture. X1 uses current LiveRetargeter, warm-start at 20 Hz, hands off; generated offline, not historical joint outputs.'
    archived=ROOT/'OptUSBProj-ubuntu/recordings/analysis_combined/20261004-144730-de9b33-0ab2c0/report.json'
    d=json.loads(archived.read_text())
    MANIFEST['accepted_report']=dict(source=str(archived),sha256=hashlib.sha256(archived.read_bytes()).hexdigest(),raw_images_available=False,parts={p:dict(wearing_status=d['wearing']['parts'][p]['status'],camera_accepted=c['accepted'],camera_fit_median_deg=c['fit_median_deg'],camera_fit_p90_deg=c['fit_p90_deg']) for p,c in d['camera']['parts'].items()})
    for name,kind,source in [('rviz-historical.png','historical_screencast_frame','/home/hzb/视频/Screencasts/Screencast From 2026-09-28 15-47-19.webm'),('architecture.html','reused_validated_archify','/home/hzb/Work/WT901/.archify/architecture-wt901-overview-20261005-202518/wt901-overview.html')]:
        if (OUT/name).is_file():register(name,kind,source=source)
    (OUT/'provenance.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+'\n')
    print('Generated',len(MANIFEST['assets']),'assets',OUT,flush=True)


if __name__=='__main__':main()
