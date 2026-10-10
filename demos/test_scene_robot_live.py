"""Opt-in integration checks. Small arm motion restored; never saves scene."""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'OptUSBProj-ubuntu')]


def check_live():
    from optusbproj_ubuntu.coppeliasim_bridge import SceneBridge
    from optusbproj_ubuntu.scene_robot_model import SceneArm,body_axes,matrix
    b=SceneBridge()
    original=b.exchange()
    try:
        models={s:SceneArm(a,*np.array(a['limits']).T) for s,a in b.data['arms'].items()}
        C=body_axes(models)
        assert np.linalg.det(C)>.999999
        # Arm-only commands: all thirty finger joints remain unchanged.
        hand_code="""local r={}
for _,h in ipairs(sim.getObjectsInTree(sim.handle_scene,sim.object_joint_type,0)) do
 local p=sim.getObjectAlias(h,1)
 if string.find(p,'jointLH') or string.find(p,'jointRH') then r[p]=sim.getJointPosition(h) end
end
return r"""
        fingers=b.run(hand_code)
        errors=[]
        for side in ('left','right'):
            m=models[side];q=original[side]
            target=q+np.deg2rad(np.array([1,-1,1,-1,1,1]))
            s,e,w=m.points(target)
            directions=[(e-s)/m.lengths[0],(w-e)/m.lengths[1]]
            fitted,err,_=m.fit(directions,q,np.deg2rad(2),m.tool_frame(target)[:3,:3])
            actual=b.exchange({side:fitted})
            np.testing.assert_allclose(actual[side],fitted,atol=1e-8)
            path=b.data['arms'][side]['objects'][5]['path']
            got=b.run("return sim.getObjectMatrix(sim.getObject(%r),sim.getObject('/BaseLinkHull'))"%path)
            # Joint six base position agrees with the virtual wrist point.
            diff=np.linalg.norm(matrix(got)[:3,3]-m.tool_frame(fitted)[:3,3]);errors.append(diff)
            assert diff<1e-7
        assert b.run(hand_code)==fingers
        print('PASS live: left/right IK -> batched set/readback; wrist FK max error m',max(errors))
        print('PASS: thirty finger angles unchanged')
    finally:
        b.exchange(original)
        for s,q in b.exchange().items():np.testing.assert_allclose(q,original[s],atol=1e-8)
        b.close()
        print('PASS: all twelve original arm angles restored; scene not saved')


def check_gui():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from optusbproj_ubuntu.pose import PoseEngine
    from optusbproj_ubuntu.scene_robot_gui import SceneRobotDialog
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer=QtWidgets.QWidget();viewer.engine=PoseEngine()
    dialog=SceneRobotDialog(viewer);dialog.show();dialog.connect_scene()
    assert dialog.task.currentData()=='positions'
    deadline=time.monotonic()+10
    while not dialog.connected and time.monotonic()<deadline:
        app.processEvents();time.sleep(.02)
    assert dialog.connected,dialog.status.text()
    assert not dialog.mailbox.get()[0]
    dialog.begin()
    assert not dialog.mailbox.get()[0]  # Missing IMUs must not enable output.
    dialog.grab().save('/tmp/scene-robot-preview.png')
    dialog.close();assert dialog.worker is None
    print('PASS GUI: connect without output, readiness blocks missing devices, worker exits')


def check_follow(task='positions'):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from optusbproj_ubuntu.pose import PoseEngine
    from optusbproj_ubuntu.scene_robot_gui import SceneRobotDialog
    from optusbproj_ubuntu.coppeliasim_bridge import SceneBridge
    from scipy.spatial.transform import Rotation
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    observer=SceneBridge();original=observer.exchange()
    viewer=QtWidgets.QWidget();viewer.engine=PoseEngine({'right_upper_arm':'probe-a','right_forearm':'probe-b'})
    for d in ('probe-a','probe-b'):
        viewer.engine.current[d]=Rotation.from_rotvec([.2,.15,0]).as_matrix();viewer.engine.neutral[d]=np.eye(3)
        viewer.engine.received_at[d]=time.monotonic()
    dialog=SceneRobotDialog(viewer);dialog.left.setChecked(False);dialog.manual.setChecked(True)
    dialog.task.setCurrentIndex(0 if task=='positions' else 1)
    dialog.logging.setChecked(True)
    try:
        dialog.connect_scene();deadline=time.monotonic()+10
        while not dialog.connected and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert dialog.connected,dialog.status.text()
        for d in viewer.engine.received_at:viewer.engine.received_at[d]=time.monotonic()
        dialog.begin();assert dialog.mailbox.get()[0]
        deadline=time.monotonic()+1.2
        while time.monotonic()<deadline:
            for d in viewer.engine.received_at:viewer.engine.received_at[d]=time.monotonic()
            app.processEvents();time.sleep(.01)
        actual=observer.exchange()
        assert np.max(np.abs(actual['right']-original['right']))>np.deg2rad(1)
        np.testing.assert_allclose(actual['left'],original['left'],atol=1e-8)
        # Short gap must hold without revoking follow authorization.
        deadline=time.monotonic()+.5
        while time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert dialog.mailbox.get()[0] and dialog.mailbox.get()[1] is None
        assert dialog.timer.isActive() and dialog.input_gap
        held=observer.exchange()
        deadline=time.monotonic()+.1
        while time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        np.testing.assert_allclose(observer.exchange()['right'],held['right'],atol=1e-8)
        for d in viewer.engine.received_at:viewer.engine.received_at[d]=time.monotonic()
        dialog.sample()
        assert dialog.mailbox.get()[0] and dialog.mailbox.get()[1] is not None
        assert not dialog.input_gap
        # A sustained gap still requires explicit restart.
        deadline=time.monotonic()+1.2
        while time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert not dialog.mailbox.get()[0]
        assert not dialog.timer.isActive()
        print('PASS',task,'movement; short gap held; fresh input automatically resumed; long gap stopped')
    finally:
        dialog.close()
        assert dialog.worker is None,'Worker still active; refusing concurrent restoration'
        observer.exchange(original);observer.close()
        print('PASS: synthetic follow test restored both arms')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--follow',action='store_true')
    args=p.parse_args()
    if not(args.live or args.gui or args.follow):p.error('Explicitly select --live, --gui or --follow')
    if args.live:check_live()
    if args.gui:check_gui()
    if args.follow:
        check_follow('positions');check_follow('vectors')
