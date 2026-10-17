"""Local parameter-only hand Checkpoints. Loading never authorizes output."""
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from .real_robot_hand import HandConfig

ROOT = Path(__file__).resolve().parents[1]/'config'/'real_robot_hand_checkpoints'


def _pairs(rows):
    data={}
    for k,v in rows:
        if k in data: raise ValueError('Checkpoint 包含重复字段')
        data[k]=v
    return data


def read_profile(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>65536: raise ValueError('预设文件不合法或过大')
    data=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=_pairs)
    if (not isinstance(data,dict) or set(data)!={'version','name','sides'} or type(data['version']) is not int
            or data['version']!=1 or not isinstance(data['name'],str) or not 1<=len(data['name'])<=100
            or not isinstance(data['sides'],dict) or set(data['sides'])!={'left','right'}):
        raise ValueError('不是有效的真机五指预设')
    checked={}
    fields=set(HandConfig.__dataclass_fields__)-{'enabled'}
    for side,row in data['sides'].items():
        if not isinstance(row,dict) or set(row)!=fields: raise ValueError('手指参数字段不匹配')
        checked[side]=HandConfig(**row)  # Permissions always default off.
    return data['name'],checked


def save_profile(name, sides, root=ROOT):
    if not isinstance(name,str) or not 1<=len(name.strip())<=100 or set(sides)!={'left','right'}:
        raise ValueError('须提供名称及左右手参数')
    rows={}
    for side,c in sides.items():
        c=HandConfig(**asdict(c)); row=asdict(c); row.pop('enabled'); rows[side]=row
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(uuid4())+'.json')
    payload=json.dumps(dict(version=1,name=name.strip(),sides=rows),ensure_ascii=False,indent=2,allow_nan=False)
    fd,tmp=tempfile.mkstemp(dir=root,prefix='.hand-',suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(payload+'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return path
