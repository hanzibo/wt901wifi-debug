"""Actual UDP/worker integration with a fake ROS publisher (no ROS graph)."""
import json
import math
from pathlib import Path
import socket
import sys
import threading
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'urdf-model/script_python'))
import x1_imu_live as bridge


class BridgeIntegrationTests(unittest.TestCase):
    def test_final_speed_limit_preserves_one_straight_path(self):
        start=dict.fromkeys(bridge.JOINT_NAMES,0.)
        goal=start.copy();goal.update({'3':.8,'53':-.4,'7':.1})
        for dt in (.005, .033, .1, 10., 0.):
            limited,alpha=bridge.limited_step(start,goal,dt)
            self.assertTrue(0 <= alpha <= 1)
            for name in goal:
                self.assertAlmostEqual(limited[name],alpha*goal[name])
                speed=math.radians(120) if name in bridge.ARMS else 1.
                self.assertLessEqual(abs(limited[name]),speed*max(0,dt)+1e-12)
            self.assertLessEqual(abs(limited['3']),math.radians(9)+1e-12)

    def test_merged_grippers_and_arms_go_through_one_gate(self):
        stop=threading.Event();published=[];errors=[]
        publish_times=[]
        class Message:
            def __init__(self):self.header=SimpleNamespace(stamp=None)
        class Node:
            def __init__(self,*a):pass
            def create_publisher(self,*a):
                def publish(message):
                    published.append(dict(zip(message.name,message.position)))
                    publish_times.append(time.monotonic())
                return SimpleNamespace(publish=publish, get_subscription_count=lambda:1)
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
        with tempfile.TemporaryDirectory() as log_root, patch.dict(sys.modules,modules),patch.object(sys,'argv',[
                'x1_imu_live.py','--port',str(arm_port),'--gripper-port',str(grip_port),
                '--collision-python',sys.executable,'--collision-start-enabled',
                '--angle-log-root', log_root]):
            thread=threading.Thread(target=run)
            thread.start()
            try:
                startup=wait_for(lambda d:d.get('type')=='gripper_status')
                self.assertTrue(startup['safety']['enabled'])
                self.assertFalse(startup['angle_log']['enabled'])
                send(dict(type='angle_log_config',request_id='log-on',enabled=True))
                logging=wait_for(lambda d:d.get('request_id')=='log-on')
                self.assertTrue(logging['angle_log']['enabled'])
                log_path=Path(logging['angle_log']['path'])
                # Retried controls must not create another recording.
                send(dict(type='angle_log_config',request_id='log-on',enabled=True))
                repeated=wait_for(lambda d:d.get('request_id')=='log-on')
                self.assertEqual(repeated['angle_log']['path'],str(log_path))
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
                for a,b,ta,tb in zip(published,published[1:],publish_times,publish_times[1:]):
                    for name in bridge.ARMS:
                        self.assertLessEqual(abs(b[name]-a[name]),math.radians(120)*(tb-ta)+.002)
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
                send(dict(type='angle_log_config',request_id='log-off',enabled=False))
                stopped=wait_for(lambda d:d.get('request_id')=='log-off')
                self.assertFalse(stopped['angle_log']['enabled'])
                rows=[json.loads(line) for line in log_path.read_text().splitlines()]
                events={r['event'] for r in rows}
                self.assertTrue({'start','receive','publish','check_submit','check_result','stop'} <= events)
                received=next(r for r in rows if r['event']=='receive' and r['seq']==1)
                self.assertEqual(received['positions'],q)
                joint_index=rows[0]['joint_names'].index('2')
                self.assertTrue(any(r['event']=='publish' and r['positions'][joint_index]==.2 for r in rows))
                self.assertTrue(all(a['mono_ns']<=b['mono_ns'] for a,b in zip(rows,rows[1:])))
            finally:
                stop.set();thread.join(5);arms.close();control.close()
            self.assertFalse(thread.is_alive());self.assertFalse(errors,errors)


if __name__=='__main__':unittest.main()
