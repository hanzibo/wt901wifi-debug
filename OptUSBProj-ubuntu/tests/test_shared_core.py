"""Frozen numerical checks and dependency boundaries for shared math."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import get_type_hints

import numpy as np

from optusbproj_ubuntu.core import rigid, retarget
from optusbproj_ubuntu.core.contracts import JointAnglesRad, PointMeters
from optusbproj_ubuntu.x1_kinematics import PoEArmKinematics
from optusbproj_ubuntu.x1_tasks import ArmTask
from optusbproj_ubuntu.x1_ik import IKResult, solve

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((Path(__file__).parent / "fixtures/core_math_baseline.json").read_text())
URDF = PROJECT.parent / "urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf"


class SharedCoreTests(unittest.TestCase):
    def test_frozen_rigid_values(self):
        for case in BASELINE['rigid']:
            xi, T = np.array(case['xi']), np.array(case['T'])
            with self.subTest(theta=np.linalg.norm(xi[:3])):
                values = {'T': rigid.exp_se3(xi), 'log': rigid.log_se3(T),
                          'inverse': rigid.inverse(T), 'adjoint': rigid.adjoint(T),
                          'left': rigid.so3_left(xi[:3]),
                          'right_inverse': rigid.so3_right_inverse(xi[:3]),
                          'se3_left_inverse': rigid.se3_left_inverse(xi)}
                for key, value in values.items():
                    np.testing.assert_allclose(value, case[key], rtol=1e-13, atol=1e-13)

    def test_exp_log_roundtrip_and_inverse(self):
        for case in BASELINE['rigid']:
            T = np.array(case['T'])
            np.testing.assert_allclose(rigid.exp_se3(rigid.log_se3(T)), T, atol=1e-12)
            np.testing.assert_allclose(rigid.inverse(T) @ T, np.eye(4), atol=1e-12)

    def test_adjoint_conjugates_body_motion(self):
        T = np.array(BASELINE['rigid'][-2]['T'])
        xi = np.array([.03, -.02, .01, .01, .02, -.03])
        np.testing.assert_allclose(T @ rigid.exp_se3(xi) @ rigid.inverse(T),
                                   rigid.exp_se3(rigid.adjoint(T) @ xi), atol=1e-12)

    def test_frozen_scaling_and_unit_conversion(self):
        for case in BASELINE['scaling']:
            h0, h, r = (tuple(np.array(p) for p in case[key])
                        for key in ('human_neutral', 'human_current', 'robot_neutral'))
            with self.subTest(mode=case['mode'], human_units=h0[0][0]):
                np.testing.assert_allclose(retarget.retarget_scales(h0, r, case['mode']),
                                           case['scales'], rtol=1e-13, atol=1e-13)
                e, w, bend = retarget.retarget_bone_points(h0, h, r, scale_mode=case['mode'])
                np.testing.assert_allclose(e, case['elbow'], atol=1e-13)
                np.testing.assert_allclose(w, case['wrist'], atol=1e-13)
                self.assertAlmostEqual(bend, case['bend_deg'], places=12)

    def test_legacy_imports_are_same_functions(self):
        from optusbproj_ubuntu import x1_kinematics, x1_retarget
        for name in ('skew', 'inverse', 'adjoint', 'so3_left', 'so3_right_inverse',
                     'exp_se3', 'log_se3', 'se3_left_inverse'):
            self.assertIs(getattr(x1_kinematics, name), getattr(rigid, name))
        for name in ('_angle_degrees', 'retarget_scales', 'retarget_bone_points'):
            self.assertIs(getattr(x1_retarget, name), getattr(retarget, name))

    def test_core_has_no_robot_io_or_gui_imports(self):
        code = '''import sys
from optusbproj_ubuntu.core import rigid, retarget, contracts
for name in sys.modules:
    assert not name.startswith(('PySide', 'PyQt', 'pybullet', 'coppeliasim',
                               'wt901.', 'wt901_viz.')), name
    assert not name.startswith(('optusbproj_ubuntu.x1_',
                               'optusbproj_ubuntu.scene_robot_')), name
'''
        subprocess.run([sys.executable, '-c', code], check=True, cwd=PROJECT)

    def test_scene_geometry_does_not_import_x1_model(self):
        code = '''import sys
from optusbproj_ubuntu.scene_robot_model import SceneArm
assert 'optusbproj_ubuntu.x1_retarget' not in sys.modules
assert 'optusbproj_ubuntu.x1_kinematics' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], check=True, cwd=PROJECT)

    def test_contract_metadata_does_not_convert_arrays(self):
        task_hints = get_type_hints(ArmTask, include_extras=True)
        self.assertEqual(task_hints['elbow'], PointMeters)
        self.assertEqual(task_hints['previous'], JointAnglesRad)
        self.assertEqual(get_type_hints(IKResult, include_extras=True)['q'], JointAnglesRad)
        p, q = np.ones(3), np.zeros(6)
        task = ArmTask(None, p, p, q)
        self.assertIs(task.elbow, p)
        self.assertIs(task.previous, q)

    def test_frozen_x1_fk_ik_both_sides_and_objectives(self):
        for case in BASELINE['x1']:
            with self.subTest(side=case['side'], mode=case['mode']):
                model = PoEArmKinematics(URDF, case['side'])
                targetq = np.array(case['target_q'])
                np.testing.assert_allclose(model.fk_space(targetq), case['fk'], atol=1e-12)
                _, elbow, wrist = model.points(targetq)
                task = ArmTask(model, elbow, wrist, np.zeros(6), 25,
                               model.wrist_relative_rotation(targetq),
                               model.fk_space(targetq) if case['mode'] == 'pose' else None)
                result = solve(task, 'least_squares', .15)
                np.testing.assert_allclose(result.q, case['q'], atol=1e-7, rtol=0)
                self.assertAlmostEqual(result.position_error_m, case['position_error_m'], places=9)
                self.assertAlmostEqual(result.cost, case['cost'], places=9)
                if case['orientation_error_deg'] is not None:
                    self.assertAlmostEqual(result.orientation_error_deg,
                                           case['orientation_error_deg'], places=6)


if __name__ == '__main__':
    unittest.main()
