"""Lossless analysis reuse: exact outputs, one read, and no stale/poisoned cache."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from optusbproj_ubuntu.analysis_context import AnalysisContext
from optusbproj_ubuntu.combined_calibration import load_capture, build_joint_report, validate_camera
from optusbproj_ubuntu.rgbd_camera_analysis import analyze_camera, window
from optusbproj_ubuntu.rgbd_fusion import load_live_series
from test_combined_calibration import capture


def options(session, output, context):
    manifest,guide,parts,windows=load_capture(session,context=context)
    origin=manifest['started_mono_ns']
    seconds=lambda ns:(ns-origin)/1e9
    from optusbproj_ubuntu.camera_protocol import camera_phases
    train_phases,_=camera_phases(guide)
    return SimpleNamespace(session=session,output=output,parts=parts,live_state=True,
        static_train_only=True,max_offset_deg=guide.get('heading_max_offset_deg',30.),
        neutral=window(':'.join(str(seconds(ns)) for ns in windows['neutral'])),
        train=window(f'{seconds(windows["neutral"][0])}:{max(seconds(windows[p][1]) for p in train_phases)}'))


class AnalysisContextTests(unittest.TestCase):
    def test_shared_and_independent_runs_have_exact_reports_and_exports(self):
        for corrupt,visual in ((False,False),(True,False),(False,True)):
            with self.subTest(corrupt=corrupt,visual=visual),TemporaryDirectory() as tmp:
                root=Path(tmp);session=root/'capture';session.mkdir();capture(session,corrupt=corrupt,camera_oblique=True)
                separate=AnalysisContext(session)
                camera=analyze_camera(options(session,root/'separate',separate),context=separate)
                expected=build_joint_report(session,camera,'visual' if visual else 'imu',prior_weight=0)
                shared=AnalysisContext(session)
                camera=analyze_camera(options(session,root/'shared',shared),context=shared)
                actual=build_joint_report(session,camera,'visual' if visual else 'imu',prior_weight=0,context=shared)
                self.assertEqual(actual,expected)
                for path in (root/'separate').glob('*.jsonl'):
                    self.assertEqual(path.read_bytes(),(root/'shared'/path.name).read_bytes())

    def test_each_source_read_once_and_inputs_unchanged(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session,camera_oblique=True)
            names=('manifest.json','camera_guide.json','imu.csv','runtime_pose.jsonl','vision.jsonl')
            before={n:hashlib.sha256((session/n).read_bytes()).digest() for n in names}
            counts={n:0 for n in names};original=Path.open
            def counted(path,*args,**kwargs):
                if path.parent==session and path.name in counts:counts[path.name]+=1
                return original(path,*args,**kwargs)
            ctx=AnalysisContext(session)
            with patch.object(Path,'open',counted):
                args=options(session,root/'output',ctx)
                camera=analyze_camera(args,context=ctx)
                raw_before=deepcopy(ctx.runtime)
                with patch('optusbproj_ubuntu.combined_calibration._validate_camera',wraps=__import__('optusbproj_ubuntu.combined_calibration',fromlist=['_validate_camera'])._validate_camera) as validation:
                    build_joint_report(session,camera,context=ctx)
                    validation.assert_not_called()
                self.assertEqual(raw_before,ctx.runtime)
            self.assertEqual(counts,{n:1 for n in names})
            self.assertEqual(before,{n:hashlib.sha256((session/n).read_bytes()).digest() for n in names})

    def test_returned_diagnostics_and_interpolation_cannot_poison_cache(self):
        with TemporaryDirectory() as tmp:
            session=Path(tmp);capture(session)
            ctx=AnalysisContext(session);_,guide,parts,windows=load_capture(session,context=ctx)
            device=guide['devices'][parts[0]]
            series,up,rows,bootstrap=load_live_series(session,device,windows['neutral'],context=ctx)
            stamp=rows[0]['host_recv_mono_ns'];expected=series.at(stamp).copy()
            matrix=series.at(stamp);matrix[:]=0;up[:]=0
            rows[0]['orientation_by_mode']['gyro_tilt'][0][0]=777;rows[0]['new']='diagnostic'
            bootstrap['new']='diagnostic'
            again,up2,rows2,info2=load_live_series(session,device,windows['neutral'],context=ctx)
            np.testing.assert_array_equal(again.at(stamp),expected)
            self.assertNotEqual(rows2[0]['orientation_by_mode']['gyro_tilt'][0][0],777)
            self.assertNotIn('new',rows2[0]);self.assertNotIn('new',info2)
            self.assertGreater(np.linalg.norm(up2),0)

    def test_source_change_and_wrong_session_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session)
            ctx=AnalysisContext(session);_ = ctx.runtime
            with self.assertRaisesRegex(ValueError,'目录不一致'):ctx.check_session(root)
            with (session/'runtime_pose.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'发生变化'):ctx.check_session(session)

    def test_report_edit_does_not_bypass_independent_validation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session,camera_oblique=True)
            ctx=AnalysisContext(session);camera=analyze_camera(options(session,root/'output',ctx),context=ctx)
            camera['parts']['right_upper_arm']['Q_camera_from_imu_world']=[[1,0,0],[0,1,0],[0,0,2]]
            checked=validate_camera(session,camera,context=ctx)
            self.assertFalse(checked['parts']['right_upper_arm']['accepted'])

    def test_failed_validation_retains_evidence_on_second_pass(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir()
            capture(session,corrupt=True,camera_oblique=True)
            ctx=AnalysisContext(session)
            camera=analyze_camera(options(session,root/'output',ctx),context=ctx)
            first=validate_camera(session,camera,context=ctx)
            failed=first['parts']['left_upper_arm']
            self.assertFalse(failed['accepted'])
            self.assertIn('lateral_repeat',failed['independent_validation'])
            self.assertIn('neutral',failed['camera_diagnostics']['windows'])
            self.assertIn('return_vs_initial',failed['camera_diagnostics'])
            expected=deepcopy(first)
            self.assertEqual(validate_camera(session,camera,context=ctx),expected)
            # Separate context must also keep rejected-part metrics intact.
            self.assertEqual(validate_camera(session,deepcopy(camera)),expected)

    def test_diagnostics_do_not_change_fit_or_acceptance(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session,camera_oblique=True)
            ctx=AnalysisContext(session)
            camera=analyze_camera(options(session,root/'output',ctx),context=ctx)
            before=deepcopy(camera)
            checked=validate_camera(session,camera,context=ctx)
            for part,data in checked['parts'].items():
                self.assertTrue(data['accepted'])
                for name in ('Q_camera_from_imu_world','bone_axis_sensor','fit_median_deg','fit_p90_deg'):
                    self.assertEqual(data[name],before['parts'][part][name])
                diagnostics=data['camera_diagnostics']
                self.assertAlmostEqual(diagnostics['return_vs_initial']['imu_orientation_change_deg'],0,places=7)
                self.assertAlmostEqual(diagnostics['return_vs_initial']['visual_direction_change_deg'],0,places=7)
            with patch('optusbproj_ubuntu.combined_calibration.camera_diagnostics',side_effect=ValueError('diagnostic only')):
                changed=validate_camera(session,deepcopy(before))
            self.assertTrue(all(d['accepted'] for d in changed['parts'].values()))
            self.assertEqual(changed['parts']['right_forearm']['camera_diagnostics']['error'],'diagnostic only')
