"""Geometry and fail-closed policy tests; no ROS graph or hardware involved."""
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'OptUSBProj-ubuntu'), str(ROOT/'urdf-model/script_python')]
from x1_collision_model import CollisionModel, partition_surface, convex_object
from x1_collision_gate import CollisionGate
from optusbproj_ubuntu.x1_retarget import ArmKinematics
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = CollisionModel()
        cls.zero = {n: 0. for n,j in cls.model.joints.items() if j[2] != 'fixed'}

    def test_full_fk_matches_existing_arm_chain(self):
        q = self.zero.copy()
        for n in ('1','2','3','4','5','6'):
            q[n] = .1
        frames = self.model.frames(q)
        base_inv = np.linalg.inv(frames['base_link_0'])
        for side in ('left', 'right'):
            arm = ArmKinematics(ROOT/'urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf', side)
            expected = arm.frames(np.array([q[n] for n in arm.joint_names]))
            for n in arm.joint_names:
                link = self.model.joints[n][1]
                np.testing.assert_allclose(base_inv@frames[link], expected[n], atol=1e-12)

    def test_zero_clearance_and_moving_toward_torso(self):
        self.assertEqual(len(self.model.shapes), 22)
        self.assertEqual(len(self.model.pairs), 200)
        self.assertTrue(self.model.check_motion(self.zero,self.zero,budget=1)['safe'])
        q = self.zero.copy(); q['2'] = float(np.deg2rad(20.))
        result = self.model.check_motion(self.zero,q,budget=1)
        self.assertFalse(result['safe'])
        self.assertIn('base_link_0', result['pair'])
        self.assertEqual(result['category'], 'collision')

    def test_compound_shoulders_release_false_positive_and_certify_path(self):
        q = self.zero.copy();q['2'] = q['52'] = float(np.deg2rad(-20))
        result = self.model.check_motion(self.zero, q, budget=1)
        self.assertTrue(result['safe'], result)
        self.assertGreater(result['samples'], 1)
        pieces = {s['link']: len(s['pieces']) for s in self.model.shapes}
        for link in ('base_link_0','L2','R52'):
            self.assertGreater(pieces[link], 1)

    def test_partition_preserves_surface_area_and_closed_volume_envelope(self):
        import trimesh
        # Both parts of a cut solid are enclosed, including cut-face interiors.
        mesh = trimesh.creation.box(extents=[1., .7, .3])
        pieces = partition_surface(mesh.copy(), depth=6)
        self.assertGreaterEqual(sum(p.area for p in pieces), mesh.area)
        hulls = [trimesh.convex.convex_hull(p.vertices) for p in pieces]
        self.assertAlmostEqual(sum(h.volume for h in hulls), mesh.volume, places=6)
        # Interior cells have no original exterior triangles, but must remain.
        self.assertEqual(len(pieces),64)
        self.assertTrue(any(np.all(np.abs(p.vertices.mean(axis=0)) < [.25,.175,.075])
                            for p in pieces))
        for p, hull in zip(pieces, hulls):
            from scipy.spatial import ConvexHull
            planes = ConvexHull(hull.vertices).equations
            self.assertLessEqual(float(np.max(p.vertices@planes[:,:3].T+planes[:,3])), 1e-8)

    def test_planar_fragment_not_dropped(self):
        obj, points = convex_object([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
        self.assertGreater(float(np.ptp(points[:,2])), 0.)
        self.assertIsNotNone(obj)

    def test_uncertain_cut_keeps_parent_envelope(self):
        import trimesh
        from unittest.mock import patch
        from scipy.spatial import QhullError
        mesh=trimesh.creation.box()
        with patch('x1_collision_model.ConvexHull',side_effect=QhullError('uncertain')):
            pieces=partition_surface(mesh.copy(),depth=6)
        self.assertEqual(len(pieces),1)
        np.testing.assert_allclose(pieces[0].bounds,mesh.bounds)
        self.assertAlmostEqual(pieces[0].volume,mesh.volume)

    def test_timeout_has_distinct_category(self):
        result = self.model.check_motion(self.zero,self.zero,budget=-1)
        self.assertFalse(result['safe'])
        self.assertEqual(result['category'],'timeout')

    def test_clearance_is_not_reported_as_intersection(self):
        model=object.__new__(CollisionModel)
        model.margins=np.array([.002]);model.version='test'
        model.shapes=[dict(link='a'),dict(link='b')];model.pairs=[(0,1,{'x':1.})]
        model.validate=lambda q:None
        model.distances=lambda q,deadline,indices=None:[.001]
        result=model.check_motion({'x':0.},{'x':0.})
        self.assertFalse(result['safe']);self.assertEqual(result['category'],'clearance')

    def test_limits_unknown_geometry_budget_fail_closed(self):
        q = self.zero.copy(); q['1'] = 3.
        self.assertFalse(self.model.check_motion(self.zero,q)['safe'])
        q = self.zero.copy(); q.pop('7')
        self.assertFalse(self.model.check_motion(self.zero,q)['safe'])
        self.assertFalse(self.model.check_motion(self.zero,self.zero,budget=-1)['safe'])

    def test_safe_endpoints_are_not_sufficient(self):
        # Analytic distance oracle with a collision only at the path midpoint.
        model = object.__new__(CollisionModel)
        model.margin = .002; model.margins=np.array([.002]); model.version='test'
        model.shapes=[dict(link='a'),dict(link='b')]
        model.pairs=[(0,1,{'x':1.})]
        model.validate=lambda q: None
        model.distances=lambda q, deadline, indices=None: [abs(q['x']-.5)-.1]
        self.assertGreater(model.distances({'x':0.},0)[0],0)
        self.assertGreater(model.distances({'x':1.},0)[0],0)
        self.assertFalse(model.check_motion({'x':0.},{'x':1.})['safe'])

    def test_certificate_subdivides_and_rejects_unresolved(self):
        model=object.__new__(CollisionModel)
        model.margin=.002;model.margins=np.array([.002]);model.version='test'
        model.shapes=[dict(link='a'),dict(link='b')];model.pairs=[(0,1,{'x':1.})]
        model.validate=lambda q:None
        model.distances=lambda q,deadline,indices=None:[.05]
        result=model.check_motion({'x':0.},{'x':1.},budget=1)
        self.assertTrue(result['safe']);self.assertGreater(result['samples'],1)
        self.assertFalse(model.check_motion({'x':0.},{'x':1.},max_depth=0)['safe'])


class GateTests(unittest.TestCase):
    def test_isolated_worker_acceptance_and_rejection(self):
        q={n:0. for n in ('101','102','1','2','3','4','5','6','7','51','52','53','54','55','56','57')}
        g=CollisionGate(sys.executable)
        try:
            g.configure(True,q)
            deadline=time.monotonic()+15
            while g.state=='初始化' and time.monotonic()<deadline:
                g.poll();time.sleep(.01)
            self.assertEqual(g.state,'正常',g.reason)
            accepted=g.accepted.copy()
            goal=q.copy();goal['2']=.2
            self.assertTrue(g.submit(goal, 123))
            self.assertFalse(g.submit(q,124))
            while g.pending and time.monotonic()<deadline:
                g.poll();time.sleep(.005)
            self.assertIn(g.state,('已拦截','截短'))
            self.assertLess(g.accepted['2'],goal['2'])
            if g.state == '已拦截':
                self.assertEqual(g.accepted,accepted)
            else:
                self.assertGreater(g.accepted['2'],accepted['2'])
                self.assertTrue(g.last_result['safe'])
                self.assertLess(g.last_result['alpha'],1.)
            self.assertTrue(g.status()['enabled'])
            g.configure(False,q)
            self.assertFalse(g.enabled);self.assertEqual(g.state,'关闭')
        finally:g.close()

    def test_missing_worker_never_silently_disables(self):
        g=CollisionGate('/nonexistent/x1-python')
        g.configure(True,{'x':0.})
        self.assertEqual(g.state,'故障');self.assertTrue(g.enabled)
        self.assertIsNone(g.status()['accepted'])
        g.close()

    def test_stale_or_wrong_response_cannot_commit(self):
        g=CollisionGate();g.enabled=True;g.ready=True;g.state='正常';g.model_version='v'
        g.accepted={'x':0.};g.pending=dict(id=2,goal={'x':1.},metadata=7,initial=False,at=time.monotonic())
        g.inbox.put(dict(type='result',id=1,safe=True,model_version='v'))
        self.assertIsNone(g.poll());self.assertEqual(g.accepted,{'x':0.})
        g.pending['at']-=1
        g.inbox.put(dict(type='result',id=2,safe=True,model_version='v'))
        self.assertIsNone(g.poll());self.assertEqual(g.state,'故障')
        self.assertEqual(g.accepted,{'x':0.})

    def test_shortened_path_commits_only_verified_fraction(self):
        g=CollisionGate();g.enabled=True;g.ready=True;g.state='正常';g.model_version='v'
        g.accepted={'x':0.};g.pending=dict(id=1,goal={'x':1.},metadata=7,initial=False,at=time.monotonic())
        g.inbox.put(dict(type='result',id=1,safe=True,alpha=.125,model_version='v'))
        commit=g.poll()
        self.assertEqual(commit,({'x':.125},7));self.assertEqual(g.state,'截短')


class GUITests(unittest.TestCase):
    def test_status_does_not_display_fcl_sentinel_as_penetration_depth(self):
        from PySide6 import QtWidgets
        from optusbproj_ubuntu.x1_live_gui import X1PreviewDialog
        from test_x1_live import prepared_engine
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host=QtWidgets.QMainWindow();host.engine=prepared_engine()
        host.calibration_dialog=None;host.settings_dialog=None
        host.x1_preview_button=QtWidgets.QPushButton()
        dialog=X1PreviewDialog(host)
        try:
            dialog._safety=dict(state='已拦截',reason='计算超时，安全性未确认',result=dict(
                category='timeout',distance_m=-1.,requested_result=dict(category='collision')))
            dialog._show_collision_status()
            text=dialog.collision_status.text()
            self.assertIn('超时未确认',text)
            self.assertIn('不是穿透深度',text)
            self.assertIn('原目标：几何相交',text)
            self.assertNotIn('-1000',text)
        finally:
            dialog._shutdown();host.close()
        self.assertIsNotNone(app)

    def test_optional_checkbox_requires_confirmed_protection(self):
        from PySide6 import QtWidgets
        from optusbproj_ubuntu.x1_live_gui import X1PreviewDialog
        from test_x1_live import prepared_engine
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host=QtWidgets.QMainWindow();host.engine=prepared_engine()
        host.calibration_dialog=None;host.settings_dialog=None
        host.x1_preview_button=QtWidgets.QPushButton()
        dialog=X1PreviewDialog(host)
        try:
            self.assertFalse(dialog.collision_protection.isChecked())
            dialog.collision_protection.setChecked(True)
            dialog._begin_collision_policy()
            self.assertFalse(dialog.collision_protection.isEnabled())
            self.assertFalse(dialog._collision_ready())
            dialog._policy_ack=True
            dialog._safety=dict(enabled=True,state='初始化',output_speed_limit=True)
            self.assertFalse(dialog._collision_ready())
            dialog._safety['state']='正常'
            self.assertTrue(dialog._collision_ready())
            dialog._safety['state']='故障'
            with self.assertRaises(ValueError):dialog._collision_ready()
            self.assertTrue(dialog.collision_protection.isChecked())
        finally:
            dialog._shutdown();host.close()
        self.assertIsNotNone(app)

    def test_ik_candidate_does_not_become_next_start(self):
        from scipy.spatial.transform import Rotation
        from test_x1_live import prepared_engine
        from optusbproj_ubuntu.x1_live import LiveRetargeter
        engine=prepared_engine()
        device=engine.assignments['left_upper_arm']
        engine.current[device]=Rotation.from_euler('x',20,degrees=True).as_matrix()
        retargeter=LiveRetargeter(engine,ROOT/'urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf')
        q,_=retargeter.step(commit=False)
        self.assertGreater(np.linalg.norm(q),1e-6)
        for side in ('left','right'):np.testing.assert_array_equal(retargeter.previous[side],np.zeros(6))
        retargeter.accept_published(q)
        np.testing.assert_array_equal(retargeter.previous['left'],q[:6])


if __name__=='__main__':unittest.main()
