from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from PySide6 import QtWidgets
from optusbproj_ubuntu.hand_demo import HandBackend,HandDemoWorker,HandDemoWindow,local_model


def config(roi=False):
    return dict(centre_roi=roi,point_threshold=.3,backend='rtm',device='cpu',det_threshold=.3)


class HandDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_local_only_and_ambiguous_models(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):local_model(root/'missing.onnx')
            with self.assertRaises(ValueError):local_model(root)
            p=root/'model.onnx';p.write_bytes(b'placeholder');self.assertEqual(local_model(root),p)
            (root/'other.onnx').write_bytes(b'placeholder')
            with self.assertRaises(ValueError):local_model(root)
            z=root/'model.zip';z.write_bytes(b'placeholder')
            with self.assertRaises(ValueError):local_model(z)

    def test_no_hand_boxes_never_runs_pose(self):
        b=HandBackend.__new__(HandBackend);b.config=config();b.mp=None;b.det=Mock(return_value=np.empty((0,4)));b.pose=Mock()
        row=b.infer(np.zeros((200,300,3),np.uint8),42)
        self.assertEqual(row['visible_count'],0);self.assertEqual(row['detected_count'],0);b.pose.assert_not_called()

    def test_rtm_roi_coords_and_rgb_bgr(self):
        b=HandBackend.__new__(HandBackend);b.config=config(True);b.mp=None
        b.det=Mock(return_value=np.array([[10,10,90,80]]))
        b.pose=Mock(return_value=(np.tile([[[40.,30.]]],(1,21,1)),np.ones((1,21))))
        image=np.zeros((200,400,3),np.uint8);image[:,:,0]=200
        row=b.infer(image,42)
        np.testing.assert_array_equal(b.det.call_args.args[0][0,0],[0,0,200])
        self.assertEqual(b.det.call_args.args[0].shape,(100,200,3))
        self.assertEqual(row['keypoints_xy'][0][0],[140,80]);self.assertEqual(row['boxes_xyxy'][0],[110,60,190,130])
        self.assertEqual(row['visible_count'],1)

    def test_mediapipe_fullframe_no_body_dependency(self):
        b=HandBackend.__new__(HandBackend);b.config=config();b.mp=SimpleNamespace(Image=lambda **kw:kw,ImageFormat=SimpleNamespace(SRGB=1))
        b.model=Mock();b.model.detect.return_value=SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=.3,y=.4) for _ in range(21)]],handedness=[[SimpleNamespace(category_name='Left')]])
        row=b.infer(np.zeros((200,300,3),np.uint8),1)
        self.assertEqual(row['visible_count'],1);self.assertEqual(row['keypoints_xy'][0][0],[90,80])
        self.assertEqual(b.model.detect.call_args.args[0]['data'].shape,(200,300,3))

    def test_worker_latest_mailbox_and_stop(self):
        worker=HandDemoWorker(config());worker.submit(np.zeros((2,2,3),np.uint8),1);worker.submit(np.zeros((2,2,3),np.uint8),2)
        self.assertEqual(worker.job[1],2);self.assertEqual(worker.skipped,1)
        worker.stop();self.assertIsNone(worker.job);worker.submit(None,3);self.assertIsNone(worker.job)

    def test_gui_image_does_not_start_camera_and_close(self):
        w=HandDemoWindow();w.timer.stop();w.static_rgb=np.zeros((100,200,3),np.uint8)
        worker=Mock();worker.take_latest.return_value=None;worker.isRunning.return_value=False
        with patch('optusbproj_ubuntu.hand_demo.HandDemoWorker',return_value=worker),patch('optusbproj_ubuntu.hand_demo.OrbbecWorker') as camera:
            w.start();self.assertTrue(w.pending);self.assertFalse(w.backend.isEnabled())
            w.on_ready('CPU');camera.assert_not_called();worker.submit.assert_called_once()
            self.assertIsNone(w.camera);w.stop();self.assertFalse(w.pending);worker.stop.assert_called()
            w.worker=None;w.reap();self.assertTrue(w.backend.isEnabled());w.close()


if __name__=='__main__':unittest.main()
