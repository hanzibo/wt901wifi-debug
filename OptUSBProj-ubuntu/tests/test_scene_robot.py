import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from optusbproj_ubuntu.scene_robot_model import SceneArm,position_targets,segment_targets,palm_target
from optusbproj_ubuntu.scene_robot_gui import Mailbox, pose_sample,SceneWorker,SceneRobotDialog,dispatch_reason,StaleImuInput,temporary_imu_gap,joint_step,cycle_wait_ms
from optusbproj_ubuntu.x1_retarget import retarget_bone_points
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.coppeliasim_bridge import SceneBridge
from unittest.mock import Mock


def fixture():
    homes=[]
    for i,axis in enumerate(('y','x','z','y','x','z')):
        T=np.eye(4)
        if axis=='y':T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        if axis=='x':T[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        T[:3,3]=[0,.13,-i*.08]
        homes.append(T[:3].reshape(-1).tolist())
    return {'q':[0]*6,'joints':homes,'palm':homes[-1]}


class ModelTests(unittest.TestCase):
    def test_staged_palm_target_cannot_change_arm_stage(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        q0=np.zeros(6);_,e,w=m.points(q0)
        goals={s:m.segment_frame(q0,s)[:3,:3] for s in ('upper_arm','forearm')}
        targets=[m.tool_frame(q0)[:3,:3],Rotation.from_euler('xyz',[.7,.4,.8]).as_matrix()]
        results=[m.fit(None,q0,.1,R,positions=(e,w),segment_rotations=goals,solver_mode='staged')[0] for R in targets]
        np.testing.assert_array_equal(results[0][:4],results[1][:4])
        self.assertGreater(np.linalg.norm(results[0][4:]-results[1][4:]),.01)
        for q in results:self.assertLessEqual(np.max(np.abs(q-q0)),.100001)

    def test_staged_wrist_motion_keeps_segment_rotations(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.));q0=np.zeros(6)
        _,e,w=m.points(q0);truth=q0.copy();truth[5]=.7
        R=m.tool_frame(truth)[:3,:3];goals={s:m.segment_frame(q0,s)[:3,:3] for s in ('upper_arm','forearm')}
        q=q0.copy()
        for _ in range(20):q,err,ori=m.fit(None,q,.1,R,positions=(e,w),segment_rotations=goals,solver_mode='staged')
        self.assertLess(ori,1.)
        self.assertLess(max(m.segment_errors(q,goals).values()),1.)

    def test_output_limits_and_scheduler_match_x1(self):
        self.assertAlmostEqual(joint_step(120,.05),np.deg2rad(6))
        self.assertAlmostEqual(joint_step(120,.125),np.deg2rad(9))
        self.assertEqual(cycle_wait_ms(0,.125),0)
        self.assertEqual(cycle_wait_ms(0,.02),30)

    def test_cached_prefixes_match_original_frames(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.));q=np.array([.1,.2,.3,-.5,.6,.7])
        prefixes=m.prefixes(q)
        for i in range(6):
            np.testing.assert_allclose(prefixes[i]@m.homes[i],m.frame(q,m.homes[i],i),atol=1e-12)
        np.testing.assert_allclose(prefixes[6]@m.tool,m.tool_frame(q),atol=1e-12)

    def test_palm_bias_neutral_and_local_axis(self):
        C=Rotation.from_euler('xyz',[.2,.3,.4]).as_matrix()
        H0=Rotation.from_euler('y',.7).as_matrix()
        R0=Rotation.from_euler('z',.6).as_matrix()
        np.testing.assert_allclose(palm_target(C,H0,H0,R0),R0,atol=1e-12)
        for axis in ('x','y','z'):
            K=Rotation.from_euler(axis,90,degrees=True).as_matrix()
            actual=palm_target(C,H0,H0,R0,{'enabled':True,'axis':axis,'degrees':90})
            np.testing.assert_allclose(actual,R0@K,atol=1e-12)
            np.testing.assert_allclose(actual.T@actual,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(actual),1)

    def test_palm_disabled_bias_matches_previous_target(self):
        C=Rotation.from_euler('x',.5).as_matrix();H=Rotation.from_euler('y',.4).as_matrix()
        R0=Rotation.from_euler('z',.2).as_matrix()
        result=palm_target(C,H,np.eye(3),R0,{'enabled':False,'axis':'x','degrees':90})
        np.testing.assert_array_equal(result,C@H@C.T@R0)
        plus=palm_target(C,np.eye(3),np.eye(3),R0,{'enabled':True,'axis':'x','degrees':90})
        minus=palm_target(C,np.eye(3),np.eye(3),R0,{'enabled':True,'axis':'x','degrees':-90})
        np.testing.assert_allclose((R0.T@plus)@(R0.T@minus),np.eye(3),atol=1e-12)

    def test_dialog_short_gap_resume_and_long_gap_halt(self):
        from types import SimpleNamespace
        e=PoseEngine({'right_upper_arm':'a','right_forearm':'b'})
        for d in ('a','b'):
            e.current[d]=np.eye(3);e.neutral[d]=np.eye(3);e.received_at[d]=time.monotonic()
        data,signature=pose_sample(e,['right'],[],True)
        box=Mailbox();box.resume(data)
        checked=Mock();checked.isChecked.return_value=True
        choice=Mock();choice.currentData.return_value='positions'
        weight=Mock();weight.value.return_value=.15
        d=SimpleNamespace(viewer=SimpleNamespace(engine=e),selected=lambda:(['right'],[]),
            manual=checked,task=choice,scale=choice,segment_orientation=checked,segment_weight=weight,
            signature=signature,mailbox=box,input_gap=False,last_sample_at=data['at'],status=Mock(),halt=Mock())
        e.received_at['b']=time.monotonic()-.5
        SceneRobotDialog.sample(d)
        self.assertTrue(d.input_gap);self.assertTrue(box.get()[0]);self.assertIsNone(box.get()[1])
        d.halt.assert_not_called()
        for dev in ('a','b'):e.received_at[dev]=time.monotonic()
        SceneRobotDialog.sample(d)
        self.assertFalse(d.input_gap);self.assertIsNotNone(box.get()[1]);d.halt.assert_not_called()
        e.received_at['b']=time.monotonic()-1.1
        SceneRobotDialog.sample(d);d.halt.assert_called_once()

    def test_calibration_change_after_gap_does_not_resume(self):
        from types import SimpleNamespace
        e=PoseEngine({'right_upper_arm':'a','right_forearm':'b'})
        for dev in ('a','b'):
            e.current[dev]=np.eye(3);e.neutral[dev]=np.eye(3);e.received_at[dev]=time.monotonic()
        data,signature=pose_sample(e,['right'],[],True)
        box=Mailbox();box.resume(data);box.set(True,None)
        manual=Mock();manual.isChecked.return_value=True
        d=SimpleNamespace(viewer=SimpleNamespace(engine=e),selected=lambda:(['right'],[]),manual=manual,
                          signature=signature,mailbox=box,input_gap=True,status=Mock(),halt=Mock())
        e.neutral['b']=Rotation.from_euler('x',.2).as_matrix()
        SceneRobotDialog.sample(d)
        d.halt.assert_called_once();self.assertIsNone(box.get()[1])

    def test_only_short_positive_imu_age_allows_waiting(self):
        for age,expected in ((.3,True),(.8,True),(1.,False),(1.1,False),(-.2,False)):
            self.assertEqual(temporary_imu_gap(StaleImuInput('right_forearm','b',age)),expected)
        self.assertFalse(temporary_imu_gap(ValueError('标定改变')))

    def test_oldest_selected_imu_determines_long_gap(self):
        e=PoseEngine({'right_upper_arm':'a','right_forearm':'b'})
        for d in ('a','b'):e.current[d]=np.eye(3);e.neutral[d]=np.eye(3)
        now=time.monotonic();e.received_at.update(a=now-.3,b=now-1.1)
        with self.assertRaises(StaleImuInput) as caught:pose_sample(e,['right'],[],True)
        self.assertGreater(caught.exception.age,1.)
        self.assertFalse(temporary_imu_gap(caught.exception))

    def test_dispatch_diagnostic_reasons(self):
        old={'at':1.};new={'at':1.29}
        self.assertEqual(dispatch_reason(False,1,1,new,old,1.3),'sending_stopped')
        self.assertEqual(dispatch_reason(True,1,2,new,old,1.3),'generation_changed')
        self.assertEqual(dispatch_reason(True,1,1,None,old,1.3),'missing_snapshot')
        self.assertEqual(dispatch_reason(True,1,1,new,old,1.3),'solve_snapshot_expired')
        self.assertIsNone(dispatch_reason(True,1,1,new,old,1.2))

    def test_stop_reason_is_preserved(self):
        box=Mailbox();box.set(False,reason='IMU 数据不新鲜：right_forearm / b')
        box.set(False)
        self.assertIn('right_forearm',box.stop_reason)

    def test_segment_frames_include_third_and_fourth_joint(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        q=np.zeros(6);raised=q.copy();raised[2]=.7
        np.testing.assert_allclose(m.points(q),m.points(raised),atol=1e-9)
        self.assertGreater(np.linalg.norm(m.segment_frame(q,'upper_arm')[:3,:3]-
                                         m.segment_frame(raised,'upper_arm')[:3,:3]),.5)
        bent=q.copy();bent[3]=.6
        np.testing.assert_allclose(m.segment_frame(q,'upper_arm'),m.segment_frame(bent,'upper_arm'))
        self.assertGreater(np.linalg.norm(m.segment_frame(q,'forearm')[:3,:3]-
                                         m.segment_frame(bent,'forearm')[:3,:3]),.5)

    def test_rotation_constraint_resolves_straight_arm_twist(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        truth=np.array([0.,0.,.7,0.,0.,0.]);_,e,w=m.points(truth)
        goals={s:m.segment_frame(truth,s)[:3,:3] for s in ('upper_arm','forearm')}
        pure=np.zeros(6);corrected=pure.copy()
        for _ in range(25):
            pure=m.fit(None,pure,.1,positions=(e,w))[0]
            corrected=m.fit(None,corrected,.1,positions=(e,w),segment_rotations=goals,segment_weight=.15)[0]
        self.assertLess(max(m.segment_errors(corrected,goals).values()),2.)
        self.assertGreater(max(m.segment_errors(pure,goals).values()),20.)

    def test_zero_weight_exactly_preserves_old_task(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.));q=np.zeros(6)
        _,e,w=m.points(q)
        goals={s:m.segment_frame(m.neutral,s)[:3,:3] for s in ('upper_arm','forearm')}
        old=m.fit(None,q,.1,positions=(e,w))[0]
        zero=m.fit(None,q,.1,positions=(e,w),segment_rotations=goals,segment_weight=0)[0]
        np.testing.assert_array_equal(old,zero)

    def test_segment_neutral_targets_and_coordinate_conjugation(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        C=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
        H0=Rotation.from_euler('x',.3).as_matrix()
        rotations={s:H0 for s in ('upper_arm','forearm')}
        targets=segment_targets(m,C,rotations,rotations)
        for s,T in targets.items():np.testing.assert_allclose(T,m.segment_frame(m.neutral,s)[:3,:3],atol=1e-12)

    def test_position_mapping_matches_x1(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        neutral=tuple(np.array(p,dtype=float) for p in ([0,0,0],[0,0,-2],[0,0,-4]))
        current=tuple(np.array(p,dtype=float) for p in ([1,0,0],[1,1,-1],[2,1,-2]))
        for mode in ('uniform','segments'):
            expected=retarget_bone_points(neutral,current,m.points(m.neutral),scale_mode=mode)
            actual=position_targets(m,np.eye(3),neutral,current,mode)
            np.testing.assert_allclose(actual[0],expected[0]);np.testing.assert_allclose(actual[1],expected[1])
        e,w=position_targets(m,np.eye(3),neutral,neutral)
        np.testing.assert_allclose(e,m.points(m.neutral)[1]);np.testing.assert_allclose(w,m.points(m.neutral)[2])

    def test_position_ik_bounded_and_improves(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        target=np.array([.2,-.2,.1,-.3,.1,0.]);_,e,w=m.points(target)
        q=np.zeros(6)
        first=m.fit(None,q,.03,positions=(e,w))[1]
        for _ in range(30):
            new,err,_=m.fit(None,q,.03,positions=(e,w))
            self.assertLessEqual(np.max(np.abs(new-q)),.030001);q=new
        self.assertLess(err,first)

    def test_pause_requires_explicit_resume(self):
        box=Mailbox();box.resume({'at':1});box.pause()
        box.set(True,{'at':2})
        self.assertFalse(box.get()[0]);self.assertIsNone(box.get()[1])
        box.resume({'at':3});self.assertTrue(box.get()[0])

    def test_identity_rejects_new_scene(self):
        b=SceneBridge.__new__(SceneBridge)
        b.data={'uid':2,'baseMatrix':[0]*12,'arms':{}}
        with self.assertRaisesRegex(ValueError,'场景更换'):b.validate_identity({'uid':1,'baseMatrix':[0]*12,'arms':{}})

    def test_worker_timeout_recovers_without_replaying_write(self):
        import copy
        import zmq
        from unittest.mock import patch
        data={'uid':1,'baseMatrix':[0]*12,'arms':{}}
        for side,y in (('left',.13),('right',-.13)):
            a=fixture()
            for raw in a['joints']:raw[7]=y
            a['palm'][7]=y
            a['limits']=[[-3,3]]*6
            a['objects']=[{'path':side+str(i),'h':i,'uid':i} for i in range(6)]
            data['arms'][side]=a
        instances=[];stop=[False];reports=[];actual={s:np.zeros(6) for s in data['arms']}
        class FakeBridge:
            validate_identity=SceneBridge.validate_identity
            def __init__(self,port):
                self.data=copy.deepcopy(data);self.calls=[];instances.append(self)
                for side,a in self.data['arms'].items():
                    m=SceneArm(data['arms'][side],np.full(6,-3.),np.full(6,3.))
                    a['q']=actual[side].tolist()
                    a['joints']=[m.frame(actual[side],home,i)[:3].reshape(-1).tolist() for i,home in enumerate(m.homes)]
                    a['palm']=m.tool_frame(actual[side])[:3].reshape(-1).tolist()
            def exchange(self,targets=None):
                self.calls.append(targets)
                if len(instances)==1 and len(self.calls)==2:
                    # The write DID apply, but its response was lost.
                    for s,q in targets.items():actual[s]=q.copy()
                    raise zmq.Again()
                return {s:q.copy() for s,q in actual.items()}
            def close(self):pass
        worker=SceneWorker(23000,Mailbox(),60,150)
        worker.isInterruptionRequested=lambda:stop[0]
        def received(value):
            reports.append(value)
            if value.get('connected'):
                worker.mailbox.resume({'at':time.monotonic(),'task':'vectors','sides':{
                    'right':{'directions':[np.array([0.,0.,-1.])]*2,'hand':None}}})
            if value.get('recovered'):stop[0]=True
        worker.report.connect(received)
        worker.failed.connect(lambda text: (stop.__setitem__(0,True),self.fail(text)))
        with patch('optusbproj_ubuntu.coppeliasim_bridge.SceneBridge',FakeBridge):worker.run()
        self.assertEqual(len(instances),2)
        self.assertTrue(any(r.get('recovering') for r in reports))
        self.assertTrue(any(r.get('recovered') for r in reports))
        self.assertTrue(worker.mailbox.blocked)
        self.assertIsNotNone(instances[0].calls[1])
        self.assertTrue(all(t is None for t in instances[1].calls))
        recovered=next(r for r in reports if r.get('recovered'))
        np.testing.assert_allclose(recovered['angles']['right'],actual['right'])

    def test_invalid_batch_never_calls_server(self):
        b=SceneBridge.__new__(SceneBridge)
        b.data={'arms':{'left':{'objects':[{'h':i} for i in range(6)],'limits':[[-1,1]]*6}}}
        b.run=Mock()
        for q in (np.full(6,np.nan),np.full(6,2),np.zeros(5)):
            with self.assertRaises(ValueError):b.exchange({'left':q})
        b.run.assert_not_called()

    def test_jacobian(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        q=np.array([.1,-.2,.3,-.4,.2,.1]);eps=1e-6
        for index in (1,3,5):
            numeric=np.column_stack([(m.frame(q+np.eye(6)[i]*eps,m.homes[index],index)[:3,3]
                -m.frame(q-np.eye(6)[i]*eps,m.homes[index],index)[:3,3])/(2*eps) for i in range(6)])
            np.testing.assert_allclose(m.point_jacobian(q,index),numeric,atol=1e-8)

    def test_bounded_tracking_improves(self):
        m=SceneArm(fixture(),np.full(6,-3.),np.full(6,3.))
        target=np.array([.1,-.1,.2,-.5,.2,0.])
        s,e,w=m.points(target)
        directions=[(e-s)/m.lengths[0],(w-e)/m.lengths[1]]
        q=np.zeros(6);start=m.fit(directions,q,.03)[1]
        for _ in range(30):
            new,err,_=m.fit(directions,q,.03)
            self.assertLessEqual(np.max(np.abs(new-q)),.030001);q=new
        self.assertLess(err,start)

    def test_mailbox_discards_stopped_samples(self):
        m=Mailbox();m.set(True,{'at':1});generation=m.get()[2]
        m.set(False);self.assertIsNone(m.get()[1]);m.set(True,{'at':2})
        self.assertNotEqual(generation,m.get()[2])

    def test_readiness_calibration_and_stale(self):
        e=PoseEngine({'right_upper_arm':'a','right_forearm':'b'})
        for device in ('a','b'):
            e.current[device]=np.eye(3);e.neutral[device]=np.eye(3);e.received_at[device]=time.monotonic()
        with self.assertRaisesRegex(ValueError,'穿戴标定'):pose_sample(e,['right'],[],False)
        data,sig=pose_sample(e,['right'],[],True)
        self.assertIsNone(data['sides']['right']['hand'])
        e.neutral['a']=Rotation.from_euler('x',.1).as_matrix()
        self.assertNotEqual(sig,pose_sample(e,['right'],[],True)[1])
        e.received_at['b']-=1
        with self.assertRaisesRegex(ValueError,'不新鲜'):pose_sample(e,['right'],[],True)


if __name__=='__main__':unittest.main()
