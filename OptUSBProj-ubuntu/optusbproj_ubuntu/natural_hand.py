"""Display-only natural palm reference; no visual measurement or robot target."""
import copy
import numpy as np


def proper(value):
    r=np.asarray(value,float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1.,atol=1e-6):
        raise ValueError('自然掌面矩阵必须属于SO(3)')
    return r


def validate_reference(data):
    result=copy.deepcopy(data)
    if result.get('format')!='natural-palm-reference-v1' or result.get('part')!='left_hand':
        raise ValueError('首版仅支持左手自然掌面参考')
    if not result.get('device_id') or not result.get('epoch') or result.get('mode') not in ('vendor_rpy','gyro_tilt'):
        raise ValueError('缺少有效设备/积分/算法绑定')
    for name in ('anchor','basis','mapping_basis','neutral_pose'):
        result[name]=proper(result[name]).tolist()
    if 'standard_neutral' in result:result['standard_neutral']=proper(result['standard_neutral']).tolist()
    if not isinstance(result.get('enabled'),bool):raise ValueError('自然显示开关无效')
    return result


def rotation(data,current):
    b=proper(data['basis']);r0=proper(data['anchor']);c0=proper(data['neutral_pose'])
    return b @ (r0.T @ proper(current)) @ b.T @ c0
