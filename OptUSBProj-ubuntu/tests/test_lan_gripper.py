import importlib.util
from pathlib import Path
import unittest
import json
import threading
import socket
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
import sys
sys.path.insert(0, str(Path(__file__).parents[2]/'urdf-model'/'script_python'))
from x1_imu_live import GripperPreview, parse_gripper

spec = importlib.util.spec_from_file_location('lan_gripper', Path(__file__).parents[1]/'lan_gripper_demo.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.state = module.ControlState(lambda: self.now)

    def cmd(self, seq=0, **kwargs):
        self.state.command(dict(version=1, client='phone123', seq=seq,
                                enabled=True, closure=1., **kwargs))

    def test_rate_and_timeout_freeze(self):
        self.cmd()
        self.now = .08
        self.assertAlmostEqual(self.state.snapshot()['closure'], .08)
        self.now = 1.
        result = self.state.snapshot()
        self.assertFalse(result['enabled'])
        self.assertAlmostEqual(result['closure'], .08)

    def test_bad_values(self):
        for value in (float('nan'), -1., 1.1, True):
            with self.assertRaises(ValueError):
                self.state.command(dict(version=1, client='phone123', seq=0,
                                        enabled=True, closure=value))

    def test_single_owner_and_sequence(self):
        self.cmd()
        with self.assertRaises(ValueError):
            self.cmd()
        with self.assertRaises(ValueError):
            self.state.command(dict(version=1, client='another123', seq=1,
                                    enabled=True, closure=.5))
        self.state.stop()
        self.assertFalse(self.state.snapshot()['enabled'])


class HTTPTests(unittest.TestCase):
    def test_http_contract(self):
        server = module.Server(('127.0.0.1', 0), token='testtoken')
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        client = build_opener(ProxyHandler({}))
        try:
            self.assertIn(b'<html', client.open(base+'/?token=testtoken').read())
            with self.assertRaises(HTTPError) as caught:
                client.open(base+'/state?token=wrong')
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            data = json.dumps(dict(version=1, client='testphone', seq=1,
                                   enabled=True, closure=.7)).encode()
            req = Request(base+'/command?token=testtoken', data=data,
                          headers={'Origin': base})
            self.assertTrue(json.load(client.open(req))['enabled'])
            req = Request(base+'/stop?token=testtoken', data=b'{}',
                          headers={'Origin': 'http://untrusted.invalid'})
            with self.assertRaises(HTTPError) as caught:
                client.open(req)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


class GripperBridgeTests(unittest.TestCase):
    def test_fresh_joint_has_single_input_owner(self):
        bridge=GripperPreview()
        phone=('127.0.0.1',9000);imu=('127.0.0.1',9001)
        self.assertTrue(bridge.accept(self.frame(),phone,1.))
        self.assertFalse(bridge.accept(self.frame(seq=2,stamp=1.1),imu,1.1))
        self.assertTrue(bridge.accept(self.frame(joint='7',seq=2,stamp=1.1),imu,1.1))
        self.assertTrue(bridge.accept(self.frame(seq=3,stamp=1.4),imu,1.4))

    def frame(self, joint='57', seq=1, position=1., stamp=1.):
        return json.dumps(dict(version=1, type='gripper', joint=joint, seq=seq,
                               position=position, sent_mono_ns=round(stamp*1e9))).encode()

    def test_validation(self):
        self.assertEqual(parse_gripper(self.frame(), now_ns=1_000_000_000), (1, '57', 1.))
        for frame in (self.frame(joint='5'), self.frame(seq=0),
                      self.frame(position=-.1), self.frame(position=float('nan')),
                      self.frame(stamp=0.)):
            with self.assertRaises(ValueError):
                parse_gripper(frame, now_ns=1_000_000_000)

    def test_independent_hold_limit_and_replay(self):
        bridge = GripperPreview()
        self.assertTrue(bridge.accept(self.frame(), ('127.0.0.1', 9000), 1.))
        self.assertFalse(bridge.accept(self.frame(), ('127.0.0.1', 9000), 1.))
        bridge.step(1.)
        self.assertEqual(bridge.step(1.1), ['57'])
        self.assertAlmostEqual(bridge.positions['57'], .1)
        self.assertEqual(bridge.positions['7'], 0.)
        self.assertEqual(bridge.step(1.5), [])
        self.assertAlmostEqual(bridge.positions['57'], .1)
        bridge.accept(self.frame(joint='7', seq=2, stamp=1.5), ('127.0.0.1', 9000), 1.5)
        self.assertEqual(bridge.step(1.6), ['7'])
        self.assertAlmostEqual(bridge.positions['7'], .1)

    def test_forwarder_opt_in_and_mapping(self):
        server = module.Server(('127.0.0.1', 0))
        stopped = threading.Event()
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.15)
        worker = threading.Thread(target=module.forward_preview,
                                  args=(server, stopped, receiver.getsockname()[1]))
        worker.start()
        try:
            server.state.command(dict(version=1, client='testphone', seq=1,
                                      enabled=True, closure=0.))
            with self.assertRaises(socket.timeout):
                receiver.recvfrom(2048)
            server.state.command(dict(version=1, client='testphone', seq=2,
                                      enabled=True, closure=0., preview=True, joint='7'))
            payload = json.loads(receiver.recvfrom(2048)[0])
            self.assertEqual((payload['joint'], payload['position']), ('7', 1.))
            server.state.command(dict(version=1, client='testphone', seq=3,
                                      enabled=True, closure=0., preview=True,
                                      joint='7', reverse=True))
            payload = json.loads(receiver.recvfrom(2048)[0])
            self.assertEqual(payload['position'], 0.)
        finally:
            stopped.set()
            worker.join(1)
            receiver.close()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
