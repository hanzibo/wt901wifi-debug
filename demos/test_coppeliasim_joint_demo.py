import math
import unittest
from unittest.mock import Mock
from coppeliasim_joint_demo import Backend, LatestTargets


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend.__new__(Backend)
        self.backend.sim = Mock(simulation_stopped=0)
        self.backend.sim.getSimulationState.return_value = 0
        self.backend.sim.getObject.return_value = 1
        self.backend.rows = [dict(path='/joint',handle=1,dependent=-1,lower=-1.,upper=1.)]

    def test_invalid_and_dependent_rejected(self):
        for value in [math.nan,math.inf,2.]:
            with self.assertRaises(ValueError):self.backend.set_position('/joint',value)
        self.backend.rows[0]['dependent'] = 4
        with self.assertRaises(ValueError):self.backend.set_position('/joint',.2)
        self.backend.sim.setJointPosition.assert_not_called()

    def test_running_or_replaced_scene_rejected(self):
        self.backend.sim.executeLuaCode.return_value=(True,dict(ok=False,error='仿真正在运行'))
        with self.assertRaises(ValueError):self.backend.set_position('/joint',.2)
        self.backend.sim.executeLuaCode.return_value=(True,dict(ok=False,error='场景对象已改变'))
        with self.assertRaises(ValueError):self.backend.set_position('/joint',.2)
        self.backend.sim.setJointPosition.assert_not_called()

    def test_write_readback(self):
        self.backend.sim.executeLuaCode.return_value=(True,dict(ok=True,values={'/joint':.2}))
        self.assertEqual(self.backend.set_position('/joint',.2),.2)
        self.backend.sim.executeLuaCode.assert_called_once()
        code=self.backend.sim.executeLuaCode.call_args.args[0]
        self.assertLess(code.index('sim.getSimulationState'),code.index('sim.setJointPosition'))
        self.assertLess(code.index('sim.getJointInterval'),code.index('sim.setJointPosition'))

    def test_batch_validates_all_before_rpc(self):
        with self.assertRaises(ValueError):self.backend.exchange({'/joint':2.},read=True)
        self.backend.sim.executeLuaCode.assert_not_called()
        self.backend.sim.executeLuaCode.return_value=(False,'script error')
        with self.assertRaises(RuntimeError):self.backend.read()

    def test_latest_targets_bounded_and_disable_clears(self):
        box=LatestTargets();box.put('/joint',.1);self.assertEqual(box.take(),{})
        box.enable(True)
        for i in range(10000):box.put('/joint',i)
        self.assertEqual(box.take(),{'/joint':9999})
        box.put('/joint',3);box.enable(False);self.assertEqual(box.take(),{})


if __name__ == '__main__':unittest.main()
