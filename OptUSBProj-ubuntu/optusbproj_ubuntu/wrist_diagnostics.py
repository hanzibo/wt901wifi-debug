"""Read-only finite differences of input targets and simulator angle readbacks."""
import numpy as np
from scipy.spatial.transform import Rotation


class WristDiagnostics:
    def __init__(self):
        self.inputs={};self.outputs={}

    def reset(self):
        self.inputs.clear();self.outputs.clear()

    def measure(self,side,input_at,readback_at,target,actual_rotation,q,previous,step,lower,upper,
                relative_target=None):
        old_input=self.inputs.get(side);old_output=self.outputs.get(side)
        target_speed=None;relative_speed=None;input_dt=None;output_dt=None;velocity=None
        if target is not None:
            if old_input is not None:
                dt=input_at-old_input[0]
                if 1e-6<dt<=.5:
                    input_dt=dt
                    target_speed=float(np.rad2deg(Rotation.from_matrix(target@old_input[1].T).magnitude())/dt)
                    if relative_target is not None and old_input[2] is not None:
                        relative_speed=float(np.rad2deg(Rotation.from_matrix(relative_target@old_input[2].T).magnitude())/dt)
            if old_input is None or input_at>old_input[0]:
                self.inputs[side]=(input_at,target.copy(),None if relative_target is None else relative_target.copy())
        else:self.inputs.pop(side,None)
        if old_output is not None:
            dt=readback_at-old_output[0]
            if 1e-6<dt<=.5:
                output_dt=dt;velocity=np.rad2deg((q-old_output[1])/dt)
        self.outputs[side]=(readback_at,q.copy())
        error=None if target is None else np.rad2deg(Rotation.from_matrix(target.T@actual_rotation).as_rotvec())
        steps=np.broadcast_to(np.asarray(step,dtype=float),(6,))
        return {
            'input_mono_s':input_at,'readback_mono_s':readback_at,
            'input_dt_s':input_dt,'readback_dt_s':output_dt,
            'target_rotation_base':None if target is None else target.tolist(),
            'actual_rotation_base':actual_rotation.tolist(),
            'target_relative_forearm_rotation':None if relative_target is None else relative_target.tolist(),
            'target_speed_deg_s':target_speed,'target_relative_forearm_speed_deg_s':relative_speed,
            'joint_velocity_deg_s':None if velocity is None else velocity.tolist(),
            'joint5_velocity_deg_s':None if velocity is None else float(velocity[4]),
            'joint6_velocity_deg_s':None if velocity is None else float(velocity[5]),
            'orientation_error_deg':None if error is None else float(np.linalg.norm(error)),
            'orientation_error_rotvec_deg':None if error is None else error.tolist(),
            'step_limit_deg':float(np.rad2deg(steps[0])),
            'joint_step_limits_deg':np.rad2deg(steps).tolist(),
            'wrist_step_limits_deg':np.rad2deg(steps[4:]).tolist(),
            'wrist_step_limit_active':(np.abs(np.abs(q[4:]-previous[4:])-steps[4:])<np.deg2rad(.02)).tolist(),
            'wrist_joint_limit_active':(np.minimum(q[4:]-lower[4:],upper[4:]-q[4:])<np.deg2rad(.05)).tolist(),
        }
