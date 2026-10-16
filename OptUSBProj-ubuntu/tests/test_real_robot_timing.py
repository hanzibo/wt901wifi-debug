"""Observations and discarded IK logging; no real serial devices."""
import time
import unittest
import socket
from types import SimpleNamespace
from optusbproj_ubuntu.real_robot_timing import observe
from optusbproj_ubuntu.real_robot_arm_worker import ArmIKWorker
from optusbproj_ubuntu.real_robot_timing import acquisition_poll, acquisition_feed
from wt901.gui_udp import GuiUdpSource


class Events:
    def __init__(self): self.rows=[]
    def record(self,event,**data): self.rows.append((event,data))


class TimingTests(unittest.TestCase):
    def test_existing_udp_socket_kernel_timestamp_and_multiple_observers(self):
        source=GuiUdpSource(port=0,host='127.0.0.1')
        first=[];second=[]
        source.enable_diagnostics(True)
        source.add_diagnostic_sink(first.append);source.add_diagnostic_sink(second.append)
        sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            # Linux enables global RX timestamping asynchronously; warm up before
            # asserting queue residence, rather than racing the first packet.
            time.sleep(.05)
            raw=bytearray(54);raw[:12]=b'WT5500011381';raw[52:]=b'\r\n'
            sender.sendto(raw,source.sock.getsockname())
            time.sleep(.03)  # Demonstrate socket buffering, not a second receiver.
            packets=source.poll_timed()
            self.assertEqual(len(packets),1)
            self.assertEqual(packets[0][1].device_id,'WT5500011381')
            datagram=next(r for r in first if r['event']=='udp_datagram')
            if source._kernel_stamp_option is not None:
                self.assertIsNotNone(datagram['kernel_received_wall_ns'])
                self.assertGreater(datagram['socket_queue_ms'],15.)
            self.assertTrue(any(r['event']=='imu_parsed' for r in second))
            self.assertTrue(any(r['event']=='udp_datagram' for r in source.take_diagnostics()))
            source.remove_diagnostic_sink(first.append)
            count=len(first);source.diagnostic_event('test_remaining_observer')
            self.assertEqual(len(first),count)
            self.assertEqual(second[-1]['event'],'test_remaining_observer')
            source.remove_diagnostic_sink(second.append)
            self.assertIsNone(source._observer_state)
            self.assertIsNone(source._kernel_stamp_option)
            # Unsubscribing real-robot tracing must not disable existing diagnostics.
            source.diagnostic_event('legacy_still_enabled')
            self.assertEqual(source.take_diagnostics()[-1]['event'],'legacy_still_enabled')
        finally:
            sender.close();source.close()

    def test_main_poll_and_model_feed_observations(self):
        events=Events()
        source=SimpleNamespace(diagnostic_event=events.record)
        owner=SimpleNamespace(source=source,engine=SimpleNamespace(received_at={'imu':12.}))
        packet=SimpleNamespace(device_id='imu')
        now=time.monotonic()
        acquisition_poll(owner,now-.02)
        acquisition_poll(owner,now)
        acquisition_poll(owner,now,[(round(now*1e9),packet)])
        acquisition_feed(owner,packet,round(now*1e9),now)
        self.assertAlmostEqual(events.rows[1][1]['interval_ms'],20.)
        self.assertEqual(events.rows[-1][1]['engine_received_at'],12.)
        self.assertEqual(events.rows[-1][1]['device'],'imu')

    def test_live_device_ages_continue_when_pose_is_stale(self):
        engine=SimpleNamespace(assignments={'left_upper_arm':'a','left_forearm':'b','left_hand':'c'},
                               received_at={'a':9.99,'b':9.98,'c':9.7})
        dialog=SimpleNamespace(viewer=SimpleNamespace(engine=engine),_selected_sides=lambda:('left',))
        first=observe(dialog,10.)
        self.assertAlmostEqual(first['devices']['left_hand']['age_ms'],300.)
        engine.received_at['a']=10.09
        second=observe(dialog,10.1)
        self.assertAlmostEqual(second['gui_poll_gap_ms'],100.)
        self.assertFalse(second['devices']['left_hand']['changed'])
        self.assertTrue(second['devices']['left_upper_arm']['changed'])
        self.assertAlmostEqual(second['devices']['left_hand']['age_ms'],400.)
        del engine.received_at['c']
        self.assertIsNone(observe(dialog,10.2)['devices']['left_hand']['age_ms'])

    def test_ik_expiration_is_logged_before_and_after_solve(self):
        class Slow:
            def solve(self,side):
                time.sleep(.06)
                return {'target':[90.]*6}
        events=Events();worker=ArmIKWorker(Slow(),recorder=events)
        worker.start()
        try:
            worker.submit({},time.monotonic()-.1,{'devices':{'left_hand':{'age_ms':300.}}})
            end=time.monotonic()+1.
            while not any(e=='arm_ik_discard' for e,_ in events.rows) and time.monotonic()<end: time.sleep(.005)
            worker.submit({},time.monotonic()+.03,{'gui_poll_gap_ms':20.})
            end=time.monotonic()+1.
            while len([e for e,_ in events.rows if e=='arm_ik_discard'])<2 and time.monotonic()<end: time.sleep(.005)
        finally:
            worker.shutdown();self.assertTrue(worker.wait(2000))
        drops=[d for e,d in events.rows if e=='arm_ik_discard']
        self.assertEqual([d['stage'] for d in drops],['before_solve','after_solve'])
        self.assertGreater(drops[1]['ik_ms'],30.)
        self.assertEqual(drops[0]['input_timing']['devices']['left_hand']['age_ms'],300.)
