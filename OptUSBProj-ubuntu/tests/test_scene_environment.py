import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from optusbproj_ubuntu.scene_environment import launch_environment,port_in_use,EnvironmentAction


class EnvironmentTests(unittest.TestCase):
    def test_launch_validates_paths_and_isolates_qt_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'coppeliaSim.sh').touch();scene=root/'scene.ttt';scene.touch()
            with patch.dict(os.environ,{'QT_PLUGIN_PATH':'bad','QT_QPA_PLATFORM':'offscreen',
                                        'LIBGL_ALWAYS_SOFTWARE':'1'}),patch('subprocess.Popen') as popen:
                launch_environment(root,scene,23001)
                args,kwargs=popen.call_args
                self.assertEqual(args[0][-1],'-GzmqRemoteApi.rpcPort=23001')
                self.assertEqual(args[0][2],str(scene.resolve()))
                self.assertNotIn('QT_PLUGIN_PATH',kwargs['env'])
                self.assertEqual(kwargs['env']['QT_QPA_PLATFORM'],'xcb')
                self.assertEqual(kwargs['env']['__NV_PRIME_RENDER_OFFLOAD'],'1')
                self.assertEqual(kwargs['env']['__GLX_VENDOR_LIBRARY_NAME'],'nvidia')
                self.assertNotIn('LIBGL_ALWAYS_SOFTWARE',kwargs['env'])
                self.assertTrue(kwargs['start_new_session'])
            with self.assertRaises(ValueError):launch_environment(root,root/'missing.ttt',23000)
            with self.assertRaises(ValueError):launch_environment(root/'missing',scene,23000)

    def test_occupied_or_ambiguous_port_is_not_free(self):
        with patch('socket.create_connection',side_effect=ConnectionRefusedError):self.assertFalse(port_in_use(23000))
        with patch('socket.create_connection',side_effect=TimeoutError):self.assertTrue(port_in_use(23000))
        with patch('socket.create_connection') as connect:self.assertTrue(port_in_use(23000))

    def test_quit_uses_own_client_and_no_joint_or_save_calls(self):
        client=Mock()
        with patch('coppeliasim_zmqremoteapi_client.RemoteAPIClient',return_value=client) as factory:
            EnvironmentAction(23001,closing=True).run()
        factory.assert_called_once_with('127.0.0.1',23001)
        self.assertEqual([c.args for c in client.call.call_args_list],
                         [('sim.getSimulationState',[]),('sim.quitSimulator',[])])
        client.socket.close.assert_called_once_with(0)


if __name__=='__main__':unittest.main()
