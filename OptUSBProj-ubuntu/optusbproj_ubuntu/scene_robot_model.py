"""Scene-derived PoE model and bounded IK. Kinematic preview, not a safety controller."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .core.rigid import exp_se3, adjoint
from .core.retarget import retarget_scales


def position_targets(model, C, human_neutral, human_current, scale_mode='uniform'):
    """Original X1 neutral bone-displacement targets, with explicit frame conversion."""
    robot_neutral=model.points(model.neutral)
    su,sf=retarget_scales(human_neutral,robot_neutral,scale_mode)
    hs0,he0,hw0=human_neutral
    hs,he,hw=human_current
    du=C@((he-hs)-(he0-hs0))
    df=C@((hw-he)-(hw0-he0))
    return robot_neutral[1]+su*du, robot_neutral[2]+su*du+sf*df


def matrix(raw):
    result = np.eye(4)
    result[:3] = np.asarray(raw).reshape(3, 4)
    return result


def palm_target(C,human,human_neutral,robot_neutral,offset=None):
    """Fixed bias is right-multiplied in the neutral robot palm frame."""
    target=C@human@human_neutral.T@C.T@robot_neutral
    if offset and offset.get('enabled',False):
        axis=offset['axis'];degrees=float(offset['degrees'])
        if axis not in ('x','y','z') or not np.isfinite(degrees):
            raise ValueError('手掌目标偏置轴/角度无效')
        target=target@Rotation.from_euler(axis,degrees,degrees=True).as_matrix()
    return target


def segment_targets(model, C, rotations, human_neutral=None):
    """Map spatial motion, not CAD axes themselves, onto neutral segment frames."""
    neutral=human_neutral or {key:np.eye(3) for key in ('upper_arm','forearm')}
    return {key:C@rotations[key]@neutral[key].T@C.T@
            model.segment_frame(model.neutral,key)[:3,:3] for key in neutral}


class SceneArm:
    def __init__(self, data, lower, upper):
        self.reference = np.asarray(data['q'], dtype=float)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.neutral = np.array([0., 0., 0., -np.pi/2, 0., 0.])
        self.homes = [matrix(raw) for raw in data['joints']]
        self.screws = np.column_stack([
            np.r_[T[:3, 2], -np.cross(T[:3, 2], T[:3, 3])] for T in self.homes])
        self.tool = matrix(data['palm'])
        # A virtual frame rigidly attached to the palm, centered at joint six.
        self.tool[:3, 3] = self.homes[5][:3, 3]
        self.lengths = (np.linalg.norm(self.homes[3][:3,3]-self.homes[1][:3,3]),
                        np.linalg.norm(self.homes[5][:3,3]-self.homes[3][:3,3]))
        if min(self.lengths) < 1e-5 or np.any(self.lower >= self.upper):
            raise ValueError('机器人几何或限位无效')
        if np.any(self.neutral < self.lower) or np.any(self.neutral > self.upper):
            raise ValueError('预览限位不包含机器人中立位')

    def frame(self, q, home, count=6):
        P = np.eye(4)
        for i in range(count):
            P = P @ exp_se3(self.screws[:,i]*(q[i]-self.reference[i]))
        return P @ home

    def prefixes(self,q):
        result=[np.eye(4)]
        for i in range(6):result.append(result[-1]@exp_se3(self.screws[:,i]*(q[i]-self.reference[i])))
        return result

    def points(self, q):
        return tuple(self.frame(q,self.homes[i],i)[:3,3] for i in (1,3,5))

    def tool_frame(self, q):
        return self.frame(q,self.tool)

    def segment_frame(self, q, segment):
        # Joint4 BASE is rigid on Link3 (includes q3, excludes q4).
        # Joint5 BASE is rigid on Link4 (includes q4, excludes q5).
        # These are virtual link frames, NOT the base of the joint we constrain.
        index={'upper_arm':3,'forearm':4}[segment]
        return self.frame(q,self.homes[index],index)

    def segment_errors(self,q,targets):
        return {key:float(np.rad2deg(Rotation.from_matrix(
            target.T@self.segment_frame(q,key)[:3,:3]).magnitude()))
            for key,target in (targets or {}).items()}

    def point_jacobian(self, q, index):
        p = self.frame(q,self.homes[index],index)[:3,3]
        J, P = np.zeros((3,6)), np.eye(4)
        for i in range(index):
            xi = adjoint(P) @ self.screws[:,i]
            J[:,i] = np.cross(xi[:3],p)+xi[3:]
            P = P @ exp_se3(self.screws[:,i]*(q[i]-self.reference[i]))
        return J

    def fit(self, directions, previous, step, rotation=None, *, positions=None,
            segment_rotations=None,segment_weight=.15,solver_mode='joint'):
        if solver_mode not in ('joint','staged'):raise ValueError('未知关节任务分配模式')
        if not np.isfinite(segment_weight) or segment_weight<0:
            raise ValueError('臂段朝向权重须为非负有限数')
        lo=np.maximum(self.lower,previous-step)
        hi=np.minimum(self.upper,previous+step)
        if np.any(lo >= hi):
            raise ValueError('当前角度超出预览限位，请调整后重新连接')
        def residual(q,include_palm=True):
            compiled = getattr(self, '_compiled_geometry', None)
            if compiled is not None:
                # Optional board backend: only geometry/linear residuals are
                # compiled. Keep SciPy's principal rotation logarithm unchanged.
                base, prefixes = compiled(q, previous, directions, positions)
                r = [base]
            else:
                prefixes = self.prefixes(q)
                s,e,w=tuple((prefixes[i]@self.homes[i])[:3,3] for i in (1,3,5))
                if positions is None:
                    r=[5*(e-s-self.lengths[0]*directions[0]),
                       5*(w-e-self.lengths[1]*directions[1])]
                else:
                    r=[5*(w-positions[1]),2.5*(e-positions[0])]
                r.extend([.06*(q-previous), .008*(q-self.neutral)])
            # Reuse one PoE chain instead of recalculating it for each point/frame.
            if segment_weight>0:
                for key,target in (segment_rotations or {}).items():
                    r.append(segment_weight*Rotation.from_matrix(
                        target.T@(prefixes[{'upper_arm':3,'forearm':4}[key]]@
                                  self.homes[{'upper_arm':3,'forearm':4}[key]])[:3,:3]).as_rotvec())
            if rotation is not None and include_palm:
                r.append(.3*Rotation.from_matrix(rotation.T@(prefixes[6]@self.tool)[:3,:3]).as_rotvec())
            elif rotation is None:
                r.append(.04*(q[4:]-self.neutral[4:]))
            return np.concatenate(r)
        initial=np.clip(previous,lo,hi)
        def optimize(seed,indices,include_palm):
            def local(values):
                candidate=seed.copy();candidate[indices]=values
                return residual(candidate,include_palm)
            result=least_squares(local,seed[indices],bounds=(lo[indices],hi[indices]),max_nfev=25,
                                 ftol=1e-5,xtol=1e-5,gtol=1e-5)
            if not np.isfinite(result.x).all() or np.linalg.norm(local(result.x))>np.linalg.norm(local(seed[indices]))+1e-8:
                raise ValueError('IK 无有限改进解')
            q=seed.copy();q[indices]=result.x
            return q
        if solver_mode=='staged':
            # Palm residual must never influence shoulder/elbow allocation.
            q=optimize(initial,slice(0,4),False)
            q=optimize(q,slice(4,6),True)
        else:
            q=optimize(initial,slice(0,6),True)
        s,e,w=self.points(q)
        err=(max(np.linalg.norm(e-s-self.lengths[0]*directions[0]),
                 np.linalg.norm(w-e-self.lengths[1]*directions[1])) if positions is None else
             max(np.linalg.norm(e-positions[0]),np.linalg.norm(w-positions[1])))
        orient=None if rotation is None else np.rad2deg(Rotation.from_matrix(
            rotation.T@self.tool_frame(q)[:3,:3]).magnitude())
        return q, float(err), None if orient is None else float(orient)


def body_axes(models):
    """GUI +X forward,+Y left,+Z up -> CAD base frame, at configured neutral."""
    left=models['left'].points(models['left'].neutral)
    right=models['right'].points(models['right'].neutral)
    y=left[0]-right[0]; y/=np.linalg.norm(y)
    z=(left[0]-left[1])+(right[0]-right[1])
    z-=y*(y@z); z/=np.linalg.norm(z)
    x=np.cross(y,z)
    C=np.column_stack((x,y,z))
    if not np.isfinite(C).all() or not np.allclose(C.T@C,np.eye(3),atol=1e-6):
        raise ValueError('机器人中立位无法确定人体坐标轴')
    return C
