"""Spawn process numerical equivalence, bounded protocol and failure handling."""
from pathlib import Path
from dataclasses import replace
import sys
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'OptUSBProj-ubuntu')]
from jetson_teleop.process_ik import ProcessIK, describe, _forbid_devices
from jetson_teleop.fast_geometry import prepare
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_bilateral import BilateralIK, BilateralLimits, right_defaults
from optusbproj_ubuntu.real_robot_shared import SharedArmIK
from optusbproj_ubuntu.real_robot_protocol import encode
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PART_BY_KEY


def pose(side):
    ts = PoseEngine().transforms(include_natural=False)
    points = [ts[PART_BY_KEY[side+'_'+p]][:3,3] for p in ('upper_arm','forearm','hand')]
    return dict(points=points,directions=[(b-a)/np.linalg.norm(b-a) for a,b in zip(points,points[1:])],
                hand=ts[PART_BY_KEY[side+'_hand']][:3,:3],
                rotations={p:ts[PART_BY_KEY[side+'_'+p]][:3,:3] for p in ('upper_arm','forearm')})


class ProcessTests(unittest.TestCase):
    def test_both_modes_preserve_seed_geometry_results_and_terminate(self):
        frame=encode([90.]*15,[0]*10)
        for solver in (SharedArmIK(frame,ArmLimits()),
                       BilateralIK(frame,BilateralLimits(replace(right_defaults(),enabled=(True,)*6),
                                                         ArmLimits(enabled=(True,)*6)))):
            prepare(solver)
            arms=solver.solvers if hasattr(solver,'solvers') else {'left':solver}
            for arm in arms.values():arm.previous=arm.previous+.01
            worker=ProcessIK.from_solver(solver)
            try:
                for _ in range(2):
                    payload={s:pose(s) for s in arms}
                    if not hasattr(solver,'solvers'):payload=payload['left']
                    actual=worker.solve(payload);expected=solver.solve(payload)
                    np.testing.assert_allclose(actual['target'],expected['target'],atol=1e-7,rtol=0)
                    self.assertAlmostEqual(actual['position_error_mm'],expected['position_error_mm'],places=7)
                self.assertEqual(worker.sequence,2)
                self.assertEqual(worker.last_solve_timing['backend'],'spawn_process')
                self.assertGreater(worker.last_solve_timing['thread_cpu_ms'],0)
            finally:worker.close()
            self.assertFalse(worker.process.is_alive())
            worker.close()

    def test_child_failure_is_not_silently_retried(self):
        worker=ProcessIK.from_solver(SharedArmIK(encode([90.]*15,[0]*10),ArmLimits()))
        try:
            with self.assertRaises(RuntimeError):worker.solve({})
            self.assertTrue(worker.faulted)
            with self.assertRaises(RuntimeError):worker.solve(pose('left'))
        finally:worker.close()

    def test_child_death_and_concurrent_submission_fail_promptly(self):
        worker=ProcessIK.from_solver(SharedArmIK(encode([90.]*15,[0]*10),ArmLimits()))
        try:
            with worker.lock:
                with self.assertRaisesRegex(RuntimeError,'并发排队'):worker.solve(pose('left'))
            worker.process.terminate();worker.process.join(1)
            start=time.monotonic()
            with self.assertRaises((RuntimeError,BrokenPipeError)):worker.solve(pose('left'))
            self.assertLess(time.monotonic()-start,1.)
        finally:worker.close()

    def test_child_device_audit_and_unsupported_solver(self):
        with self.assertRaises(RuntimeError):_forbid_devices('socket.__new__',())
        with self.assertRaises(RuntimeError):_forbid_devices('open',('/dev/ttyUSB0',))
        with self.assertRaises(TypeError):describe(object())


if __name__=='__main__':unittest.main()
