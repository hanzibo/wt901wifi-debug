"""Offline tests: geometry, known official printouts and duplicate handling inputs."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from apriltag_pose.core import Detector, Pose, camera_model, estimate_pose, square_points
from apriltag_pose.diagnostics import PoseWindow, quality_reasons


class Tests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[850., 0, 320], [0, 850, 240], [0, 0, 1.]])
        self.d = np.zeros(8)

    def test_pose_and_scale(self):
        r = np.array([2.8, .2, -.1])
        t = np.array([.03, -.02, .5])
        corners = cv2.projectPoints(square_points(.025), r, t, self.k, self.d)[0]
        pose = estimate_pose(corners, .025, self.k, self.d)
        np.testing.assert_allclose(pose.tvec.flatten(), t, atol=1e-6)
        np.testing.assert_allclose(cv2.Rodrigues(pose.rvec)[0], cv2.Rodrigues(r)[0], atol=1e-6)
        doubled = estimate_pose(corners, .05, self.k, self.d)
        np.testing.assert_allclose(doubled.tvec, 2*pose.tvec, atol=1e-6)

    def test_calibration_validation(self):
        calibration = {'color_intrinsic': dict(width=640, height=480, fx=850, fy=850, cx=320, cy=240),
                       'color_distortion': dict(model='OB_DISTORTION_BROWN_CONRADY',
                                                **{n: 0 for n in ('k1','k2','k3','k4','k5','k6','p1','p2')})}
        k, d = camera_model(calibration, (480, 640, 3))
        np.testing.assert_array_equal(k, self.k)
        np.testing.assert_array_equal(d, self.d)
        with self.assertRaises(ValueError):
            camera_model(calibration, (720, 1280, 3))
        for unsupported in ('FISHEYE', 'INVERSE_BROWN_CONRADY', 'MODIFIED_BROWN_CONRADY'):
            calibration['color_distortion']['model'] = unsupported
            with self.assertRaises(ValueError):
                camera_model(calibration, (480, 640, 3))

    def test_nonzero_distortion_order(self):
        coefficients = dict(k1=.01, k2=.02, p1=.03, p2=.04, k3=.05, k4=.06, k5=.07, k6=.08)
        cal = {'color_intrinsic': dict(width=640, height=480, fx=850, fy=850, cx=320, cy=240),
               'color_distortion': dict(model='BROWN_CONRADY', **coefficients)}
        _, d = camera_model(cal, (480, 640))
        np.testing.assert_allclose(d, list(coefficients.values()))

    def test_offcenter_frontal_square(self):
        corners = np.array([[100., 150], [300., 150], [300., 350], [100., 350]])
        k = np.array([[1000., 0, 640], [0, 1000, 360], [0, 0, 1.]])
        pose = estimate_pose(corners, .05, k, self.d)
        self.assertLess(pose.error_px, .01)

    def test_empty(self):
        self.assertEqual(Detector().detect(np.full((480, 640, 3), 255, np.uint8)), [])

    def test_static_variation_and_gap(self):
        window = PoseWindow()
        pose = Pose(np.array([.1, .2, .3]), np.array([.01, .02, .5]), .2, False)
        for n in range(16):
            result = window.update(0, n*.1, 50., pose)
        self.assertTrue(result.ready)
        self.assertLess(result.position_mm, 1e-8)
        self.assertLess(result.rotation_deg, 1e-5)
        self.assertEqual(window.update(0, 3., 50., pose).count, 1)
        self.assertEqual(window.update(0, 3.1, 25., pose).count, 1)
        self.assertEqual(window.update(0, 3.0, 25., pose).count, 1)
        window.retain(set())
        self.assertFalse(window.samples)

    def test_variation_known_amplitude(self):
        window = PoseWindow()
        for n in range(20):
            sign = (-1)**n
            pose = Pose(np.array([0., 0., sign*np.deg2rad(3)]),
                        np.array([sign*.004, 0., .5]), .2, False)
            result = window.update(0, n*.1, 50., pose)
        self.assertAlmostEqual(result.position_mm, 4.)
        self.assertAlmostEqual(result.rotation_deg, 3.)
        self.assertEqual(len(quality_reasons(50., pose, result)), 2)
        self.assertFalse(quality_reasons(50., pose, result, max_position=5., max_rotation=4.))

    def test_geometry_quality_and_window_bound(self):
        pose = Pose(np.zeros(3), np.array([0., 0., .5]), 2., True)
        self.assertEqual(len(quality_reasons(20., pose, None)), 3)
        window = PoseWindow()
        for n in range(1000):
            result = window.update(1, n*.01, 25., pose)
        self.assertLessEqual(result.count, 120)
        self.assertLessEqual(result.span_s, 2.)

    def test_generated_dictionary_ids(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((350, 950, 3), 255, np.uint8)
        for tag_id in range(3):
            marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 240)
            image[50:290, 35+tag_id*300:275+tag_id*300] = marker[:, :, None]
        self.assertEqual(sorted(i for i, _ in Detector().detect(image)), [0, 1, 2])

    def test_official_source_ids(self):
        root = Path('/home/hzb/Work/apriltag-imgs/tag36h11')
        if not root.is_dir():
            self.skipTest('Optional locally cloned official images not present')
        for tag_id in range(3):
            source = cv2.imread(str(root / f'tag36_11_{tag_id:05d}.png'))
            image = cv2.resize(source, (400, 400), interpolation=cv2.INTER_NEAREST)
            self.assertEqual([i for i, _ in Detector().detect(image)], [tag_id])

    def test_gui_diagnostics_without_camera(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OptUSBProj-ubuntu'))
        from PySide6 import QtWidgets
        from apriltag_pose.gui import Window
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = Window()
        window.timer.stop()
        try:
            corners = cv2.projectPoints(square_points(.05), np.array([2.8, .2, -.1]),
                                        np.array([.03, -.02, .5]), self.k, self.d)[0].reshape(4, 2)
            window.detector = SimpleNamespace(detect=lambda rgb: [(0, corners)])
            cal = {'color_intrinsic': dict(width=640, height=480, fx=850, fy=850, cx=320, cy=240),
                   'color_distortion': dict(model='NONE')}
            obs = SimpleNamespace(frame_rgb=np.full((480, 640, 3), 255, np.uint8),
                                  rgbd=SimpleNamespace(calibration=cal), captured_mono_ns=0)
            window.measure.setChecked(True)
            for n in range(16):
                obs.captured_mono_ns = int(n*1e8)
                window.show_observation(obs)
            self.assertEqual(window.table.rowCount(), 1)
            self.assertNotEqual(window.table.item(0, 5).text(), '-')
            self.assertLess(float(window.table.item(0, 6).text()), .001)
            self.assertIn('16 / 1.5', window.table.item(0, 8).text())
            window.detector.detect = lambda rgb: [(0, corners), (0, corners)]
            window.show_observation(obs)
            self.assertIn('重复 ID', window.table.item(0, 9).text())
            self.assertFalse(window.pose_window.samples)
            window.detector.detect = lambda rgb: []
            window.show_observation(obs)
            self.assertEqual(window.table.rowCount(), 0)
            app.processEvents()
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
