"""Versioned camera-only pose windows; mounting keeps its original actions."""
OBLIQUE_PROTOCOL = 'visible-oblique-v1'


def camera_phases(guide):
    protocol = guide.get('camera_pose_protocol')
    if protocol is None:
        return ('neutral', 'forward', 'lateral'), ('forward_repeat', 'lateral_repeat', 'neutral_return')
    if protocol != OBLIQUE_PROTOCOL:
        raise ValueError('未知相机动作协议，不能回退到旧窗口')
    return ('neutral', 'oblique', 'lateral'), ('oblique_repeat', 'lateral_repeat', 'neutral_return')


def add_oblique_steps(steps, preparation, hold, scope):
    arm = {'right': '右臂', 'left': '左臂', 'bilateral': '双臂'}[scope]
    direction = '分别向身体左右外侧展开' if scope == 'bilateral' else ('向身体左外侧展开' if scope == 'left' else '向身体右外侧展开')
    instruction = f'{arm}从正前方{direction}约30–45°，抬高约45–70°，肘腕伸直；保持身体朝向，让肩肘腕在画面中分开。角度不必精确。'
    result = []
    for step in steps:
        result.append(step)
        if step[0] == 'down':
            result.append(('oblique', arm+'斜前举（相机拟合）', instruction, preparation, hold))
        elif step[0] == 'forward_repeat':
            result.append(('oblique_repeat', '再次斜前举（相机独立复测）', instruction+' 尽量重复刚才的姿态。', preparation, hold))
    return result
