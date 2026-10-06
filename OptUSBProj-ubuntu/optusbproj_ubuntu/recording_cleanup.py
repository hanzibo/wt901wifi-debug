"""Delete exactly one completed, identity-checked calibration capture, never its reports."""
import json
from pathlib import Path
import re
import shutil

RECORDINGS_ROOT = Path(__file__).resolve().parents[1]/'recordings'
SESSION_NAME = re.compile(r'\d{8}-\d{6}-[0-9a-f]{6}')


def validate_capture(path, started_mono_ns, *, root=RECORDINGS_ROOT):
    target = Path(path).absolute();base = Path(root).resolve()
    if target.is_symlink() or target.resolve() != target or target.parent != base:
        raise ValueError('只能删除 recordings 下的一个原始录制目录，禁止链接、上级目录或分析目录')
    if not SESSION_NAME.fullmatch(target.name) or not target.is_dir():raise ValueError('原始录制目录名称无效或不存在')
    manifest = json.loads((target/'manifest.json').read_text())
    guide = json.loads((target/'camera_guide.json').read_text())
    if manifest.get('format') not in ('wt901-fusion-session-v1','wt901-fusion-session-v2'):
        raise ValueError('不是联合录制目录')
    if manifest.get('started_mono_ns') != int(started_mono_ns):raise ValueError('录制身份变化，拒绝删除')
    if not manifest.get('ended_mono_ns') or manifest.get('error'):raise ValueError('录制未正常结束，拒绝删除')
    if guide.get('status') != 'complete' or guide.get('workflow') != 'combined_wearing_camera_calibration':
        raise ValueError('不是本次已完成的一体标定录制')
    if guide.get('recording_started_mono_ns') != int(started_mono_ns):raise ValueError('引导记录身份不一致')
    return target


def delete_capture(path, started_mono_ns, *, root=RECORDINGS_ROOT):
    target = validate_capture(path, started_mono_ns, root=root)
    # CPython uses fd-based rmtree where supported; inner symlinks are unlinked,
    # not followed. Refuse unsafe platforms rather than expanding the target.
    if not shutil.rmtree.avoids_symlink_attacks:raise ValueError('平台不支持安全目录删除，已拒绝执行')
    shutil.rmtree(target)
    return target
