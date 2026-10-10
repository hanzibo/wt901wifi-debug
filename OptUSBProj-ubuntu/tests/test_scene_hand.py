import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6 import QtWidgets
from optusbproj_ubuntu.scene_hand import HandMailbox,hand_pose,hand_step
from optusbproj_ubuntu.scene_robot_gui import SceneRobotDialog
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.gripper_gesture_gui import BilateralGestureWindow,SharedGestureSource
from optusbproj_ubuntu.coppeliasim_bridge import SceneBridge
from optusbproj_ubuntu.scene_robot_gui import SceneWorker,Mailbox


class HandTests(unittest.TestCase):
    def test_pose_validation(self):
        bounds=[[0,np.pi/2]]*15
        np.testing.assert_array_equal(hand_pose([0]*15,bounds),np.zeros(15))
        for q in ([0]*14,[np.nan]*15,[-.01]*15,[2]*15):
            with self.assertRaises(ValueError):hand_pose(q,bounds)

    def test_straight_path_speed_and_gap_cap(self):
        target=np.linspace(0,np.pi/2,15)
        q=hand_step(np.zeros(15),target,120,.05)
        np.testing.assert_allclose(q,target*(6/90))
        self.assertAlmostEqual(np.rad2deg(hand_step(np.zeros(15),target,120,2)[-1]),12)
        np.testing.assert_array_equal(hand_step(target,target,120,.05),target)

    def test_mailbox_freshness_copy_stop_and_sides(self):
        box=HandMailbox();q=np.ones(15)
        box.set('left',q,10,120,'gesture');q[:]=2
        out,_,rev=box.targets({'left':np.zeros(15),'right':np.zeros(15)},10.05,.05)
        self.assertEqual(set(out),{'left'})
        self.assertEqual(box.snapshot()[0]['left']['target'][0],1)
        self.assertFalse(box.targets({'left':np.zeros(15)},10.31,.05)[0])
        box.stop('left');self.assertGreater(box.snapshot()[1],rev);self.assertFalse(box.snapshot()[0])

    def test_hand_gui_authorization_and_stop(self):
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host=QtWidgets.QWidget();host.engine=PoseEngine()
        dialog=SceneRobotDialog(host);dialog.connected=True
        hands={s:{'limits':[[0,np.pi/2]]*15,'objects':[{'path':s+str(i)} for i in range(15)]} for s in ('left','right')}
        dialog.config={'scene':{'hands':hands}}
        dialog.show_hand_config();form=dialog.hand_dialog
        dialog.hand_angles={s:np.zeros(15) for s in hands};dialog.hand_readback_at=time.monotonic()
        form.capture('left','open')
        with self.assertRaises(ValueError):form.endpoint('left','closed')
        for p in form.widgets['left'][0]:p[1].setValue(45)
        form.confirm('left','open');form.confirm('left','closed')
        dialog.receive_hand('left',{'enabled':True,'valid':True,'state':True,'stamp':time.monotonic()})
        self.assertEqual(dialog.hand_mailbox.snapshot()[0]['left']['source'],'gesture')
        self.assertFalse(dialog.mailbox.get()[0])  # No arms or arm IMUs required.
        dialog.receive_hand('left',{'enabled':True,'valid':False,'state':True,'stamp':time.monotonic()})
        self.assertFalse(dialog.hand_mailbox.snapshot()[0])
        form.begin_preview('left','closed');self.assertIn('left',form.preview)
        form.widgets['left'][0][0][1].setValue(40)
        self.assertFalse(form.preview);self.assertFalse(dialog.hand_mailbox.snapshot()[0])
        with self.assertRaises(ValueError):form.endpoint('left','closed')
        form.close();dialog.close();host.close()

    def test_bilateral_gesture_stop_does_not_change_rviz_convention(self):
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host=QtWidgets.QWidget();host.engine=PoseEngine()
        host._show_x1_preview=lambda:None;host._show_scene_robot=lambda:None
        window=BilateralGestureWindow(SharedGestureSource(),'/tmp/unused-gesture-tests',host)
        messages=[];window.hand_output.connect(lambda side,data:messages.append((side,data)))
        for s,p in window.panels.items():
            self.assertEqual(p.target_joint.currentData(),'7' if s=='left' else '57')
            p.send_scene.setChecked(True)
        window.stop_outputs()
        self.assertTrue(messages)
        self.assertTrue(all(not p.send_scene.isChecked() for p in window.panels.values()))
        window.shutdown();host.close()

    def test_bridge_validates_whole_hand_batch_before_any_call(self):
        from unittest.mock import Mock
        b=SceneBridge.__new__(SceneBridge)
        b.data={'arms':{},'hands':{'left':{'objects':[{'h':i} for i in range(15)],'limits':[[0,1]]*15}}}
        b.run=Mock()
        for q in (np.zeros(14),np.full(15,np.nan),np.full(15,2)):
            with self.assertRaises(ValueError):b.exchange(hand_targets={'left':q})
        b.run.assert_not_called()

    def test_worker_publishes_hands_without_arms_and_stops(self):
        import copy
        from scipy.spatial.transform import Rotation
        homes=[]
        for i,axis in enumerate(('y','x','z','y','x','z')):
            T=np.eye(4)
            if axis=='y':T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
            if axis=='x':T[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
            T[:3,3]=[0,.13,-i*.08];homes.append(T[:3].reshape(-1).tolist())
        data={'uid':1,'baseMatrix':[0]*12,'arms':{},'hands':{}}
        for side in ('left','right'):
            arm={'q':[0]*6,'joints':copy.deepcopy(homes),'palm':homes[-1],
                 'limits':[[-3,3]]*6,'objects':[]}
            if side=='right':
                for h in arm['joints']:h[7]=-.13
            data['arms'][side]=arm
            data['hands'][side]={'q':[0]*15,'limits':[[0,1.57]]*15,'objects':[]}
        instances=[];done=[False];writes=[]
        class FakeBridge:
            def __init__(self,port):
                self.data=copy.deepcopy(data);self.hand_angles={s:np.zeros(15) for s in data['hands']};instances.append(self)
            def exchange(self,targets=None,hand_targets=None):
                assert not targets  # The arm mailbox is never enabled.
                if hand_targets:
                    writes.append({s:q.copy() for s,q in hand_targets.items()})
                    self.hand_angles.update(hand_targets)
                return {s:np.zeros(6) for s in data['arms']}
            def close(self):pass
        box=HandMailbox();worker=SceneWorker(23000,Mailbox(),120,None,hand_mailbox=box)
        worker.isInterruptionRequested=lambda:done[0]
        reports=[]
        def report(row):
            reports.append(row)
            if row.get('connected'):box.set('left',np.ones(15)*.5,time.monotonic(),120,'gesture')
            if row.get('hand_targets'):
                box.stop();done[0]=True
        worker.report.connect(report)
        worker.failed.connect(lambda text:(done.__setitem__(0,True),self.fail(text)))
        with patch('optusbproj_ubuntu.coppeliasim_bridge.SceneBridge',FakeBridge):worker.run()
        self.assertEqual(len(writes),1);self.assertEqual(set(writes[0]),{'left'})
        self.assertTrue(any(row.get('hand_angles') for row in reports))

    def test_single_side_save_load_requires_confirmation(self):
        import tempfile,json
        from pathlib import Path
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host=QtWidgets.QWidget();host.engine=PoseEngine();dialog=SceneRobotDialog(host)
        dialog.connected=True
        dialog.config={'scene':{'hands':{s:{'limits':[[0,np.pi/2]]*15,
            'objects':[{'path':s+str(i)} for i in range(15)]} for s in ('left','right')}}}
        dialog.show_hand_config();form=dialog.hand_dialog
        for p in form.widgets['left'][0]:p[1].setValue(30)
        form.confirm('left','open');form.confirm('left','closed')
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'poses.json')
            with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(path,'')):form.save()
            self.assertEqual(set(json.loads(Path(path).read_text())['sides']),{'left'})
            with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(path,'')):form.load()
            with self.assertRaises(ValueError):form.endpoint('left','closed')
            self.assertFalse(dialog.hand_mailbox.snapshot()[0])
            form.confirm('left','open');form.confirm('left','closed')
            np.testing.assert_allclose(form.endpoint('left','closed'),np.deg2rad(np.full(15,30)))
        form.close();dialog.close();host.close()


if __name__=='__main__':unittest.main()
