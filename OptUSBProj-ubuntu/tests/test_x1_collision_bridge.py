"""Actual UDP/worker integration with a fake ROS publisher (no ROS graph)."""
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'urdf-model/script_python'))
import x1_imu_live as bridge


class BridgeIntegrationTests(unittest.TestCase):
    def test_merged_grippers_and_arms_go_through_one_gate(self):
        stop=threading.Event();published=[];errors=[]
        class Message:
            def __init__(self):self.header=SimpleNamespace(stamp=None)
        class Node:
            def __init__(self,*a):pass
            def create_publisher(self,*a):return SimpleNamespace(
                publish=lambda message:published.append(dict(zip(message.name,message.position))),
                get_subscription_count=lambda:1)
            def create_timer(self,period,callback):self.callback=callback
            def get_clock(self):return SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:0))
            def destroy_node(self):pass
        def spin(node,**kwargs):
            node.callback();time.sleep(.005)
        modules={'rclpy':SimpleNamespace(init=lambda:None,ok=lambda:not stop.is_set(),
                    spin_once=spin,shutdown=lambda:None),
                 'rclpy.node':SimpleNamespace(Node=Node),
                 'sensor_msgs':SimpleNamespace(),
                 'sensor_msgs.msg':SimpleNamespace(JointState=Message)}
        def port():
            s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',0))
            value=s.getsockname()[1];s.close();return value
        arm_port,grip_port=port(),port()
        arms=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        control=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        control.settimeout(.1)
        def send(payload, sock=control, dest=grip_port):
            sock.sendto(json.dumps(dict(version=1,sent_mono_ns=time.monotonic_ns(),**payload)).encode(),
                        ('127.0.0.1',dest))
        def wait_for(predicate,timeout=12):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                send(dict(type='gripper_status_query'))
                try:data=json.loads(control.recvfrom(8192)[0])
                except socket.timeout:continue
                if predicate(data):return data
                if errors:raise errors[0]
            self.fail('Timed out waiting for bridge status')
        def run():
            try:bridge.main()
            except Exception as exc:errors.append(exc);stop.set()
        with patch.dict(sys.modules,modules),patch.object(sys,'argv',[
                'x1_imu_live.py','--port',str(arm_port),'--gripper-port',str(grip_port),
                '--collision-python',sys.executable,'--collision-start-enabled']):
            thread=threading.Thread(target=run)
            thread.start()
            try:
                startup=wait_for(lambda d:d.get('type')=='gripper_status')
                self.assertTrue(startup['safety']['enabled'])
                send(dict(type='collision_config',request_id='protect',enabled=True))
                ready=wait_for(lambda d:d.get('safety',{}).get('state')=='正常')
                self.assertTrue(ready['safety']['enabled'])
                self.assertTrue(published)
                # Gripper-only inputs also pass through geometry validation.
                send(dict(type='gripper',seq=1,joint='7',position=.05))
                wait_for(lambda d:d.get('type')=='gripper_status' and d['joints']['7']['position']>0)
                q=[0.]*12;q[1]=.2
                send(dict(type='frame',seq=1,joint_names=bridge.ARMS,positions=q),arms,arm_port)
                wait_for(lambda d:d.get('safety',{}).get('state')=='已拦截')
                # Improved geometry may admit a certified prefix, but never
                # the colliding full goal. A freeze at zero is not required.
                self.assertTrue(all(0. <= p['2'] < .2 for p in published))
                # A policy change while inputs are fresh must not disable protection.
                send(dict(type='frame',seq=2,joint_names=bridge.ARMS,positions=q),arms,arm_port)
                send(dict(type='collision_config',request_id='off-busy',enabled=False))
                blocked=wait_for(lambda d:d.get('request_id')=='off-busy')
                self.assertIsNotNone(blocked['error']);self.assertTrue(blocked['safety']['enabled'])
                time.sleep(.4)
                send(dict(type='collision_config',request_id='off-idle',enabled=False))
                disabled=wait_for(lambda d:d.get('request_id')=='off-idle')
                self.assertIsNone(disabled['error']);self.assertFalse(disabled['safety']['enabled'])
                send(dict(type='frame',seq=3,joint_names=bridge.ARMS,positions=q),arms,arm_port)
                wait_for(lambda d:any(p['2']==.2 for p in published))
            finally:
                stop.set();thread.join(5);arms.close();control.close()
            self.assertFalse(thread.is_alive());self.assertFalse(errors,errors)


if __name__=='__main__':unittest.main()
