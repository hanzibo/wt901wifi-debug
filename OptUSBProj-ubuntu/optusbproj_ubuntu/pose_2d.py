"""Interchangeable 2D-only models in original RGB pixels; no depth prediction."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_YOLO = Path(__file__).resolve().parents[2]/'no-improtent/yolo-models/yolo26m-pose.pt'


@lru_cache(maxsize=16)
def _hash_file(path, size, mtime):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():return {'path':str(path),'sha256':None}
    stat=path.stat()
    return {'path':str(path),'sha256':_hash_file(str(path),stat.st_size,stat.st_mtime_ns)}


@dataclass(frozen=True)
class Pose2DConfig:
    backend: str = 'rtmw'
    model: Path = DEFAULT_YOLO
    imgsz: int = 640

    def __post_init__(self):
        if self.backend not in ('rtmw','yolo26'):raise ValueError('未知二维后端')
        if self.imgsz not in (640,960):raise ValueError('YOLO输入尺寸必须为640或960')

    def metadata(self):
        cache=Path.home()/'.cache/rtmlib/hub/checkpoints'
        weights=([self.model] if self.backend=='yolo26' else [
            cache/'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx',
            cache/'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx'])
        config={'backend':self.backend,'weights':[file_identity(p) for p in weights],
                'imgsz':self.imgsz if self.backend=='yolo26' else None,
                'person_conf':.25 if self.backend=='yolo26' else None,
                'native_keypoints':17 if self.backend=='yolo26' else 133,
                'output_layout':'coco133-compatible; absent points NaN/zero',
                'identity_policy':'single_person_default; optional_user_target_geometry_v1'}
        config['signature']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        return config


def empty_keypoints():
    return np.full((133,2),np.nan),np.zeros(133)


def infer_rtmw(model,bgr,*,target=None,observation=None):
    boxes=model.det_model(bgr);xy,scores=empty_keypoints()
    info={'people_detected':len(boxes),'inference_backend':'RTMW3D 2D outputs / CUDA'}
    selected = 0 if len(boxes)==1 else None
    if target is not None:
        selected,selection=target.choose(boxes,observation)
        info.update(selection)
    if selected is not None:
        _xyz,score_batch,_simcc,points=model.pose_model(bgr,bboxes=np.asarray(boxes)[selected:selected+1])
        xy,scores=np.asarray(points[0]),np.asarray(score_batch[0])
    elif 'reject_reason' not in info:info['reject_reason']='未检测到人' if len(boxes)==0 else '多人画面：暂不自动选择身份'
    return xy,scores,info


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


class RTMWBackend:
    def __init__(self,config):
        import onnxruntime as ort
        from rtmlib import Wholebody3d
        self.metadata=config.metadata()
        paths=[Path(w['path']) for w in self.metadata['weights']]
        if not all(p.is_file() for p in paths):raise RuntimeError('缺少本地RTMW/YOLOX模型；请先运行rtmw3d_probe.py，本窗口不自动下载')
        if hasattr(ort,'preload_dlls'):ort.preload_dlls(directory='')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('未发现ONNX CUDA执行后端；不回退CPU')
        self.model=Wholebody3d(det=str(paths[0]),pose=str(paths[1]),backend='onnxruntime',device='cuda')
        for part in (self.model.det_model,self.model.pose_model):
            if 'CUDAExecutionProvider' not in part.session.get_providers():raise RuntimeError('RTMW未使用CUDA')

    def infer(self,bgr,*,target=None,observation=None):
        return infer_rtmw(self.model,bgr,target=target,observation=observation)


class YOLOBackend:
    def __init__(self,config):
        try:
            import torch
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            import sys
            raise RuntimeError(f'当前解释器 {sys.executable} 缺少YOLO依赖（{exc.name}）。请关闭窗口，使用桌面“WT901-RGBD-YOLO对比”或OptUSBProj-yolo环境启动；无需在旧环境重复安装。') from exc
        if not Path(config.model).is_file():raise RuntimeError(f'缺少本地YOLO权重：{config.model}；不会自动下载')
        if not torch.cuda.is_available():raise RuntimeError('PyTorch CUDA不可用；不回退CPU')
        self.config=config;self.metadata=config.metadata()
        self.model=YOLO(str(config.model),task='pose')
        if self.model.task!='pose':raise RuntimeError('请使用-pose姿态权重，不是检测权重')
        # Warm-up is excluded from per-frame timings.
        self.model.predict(np.zeros((480,640,3),np.uint8),device=0,imgsz=config.imgsz,conf=.25,verbose=False)

    def infer(self,bgr,*,target=None,observation=None):
        result=self.model.predict(bgr,device=0,imgsz=self.config.imgsz,conf=.25,verbose=False)[0]
        count=len(result.boxes) if result.boxes is not None else 0
        xy,scores=empty_keypoints()
        info={'people_detected':count,'inference_backend':'YOLO26 Pose / CUDA'}
        selected = 0 if count==1 else None
        if target is not None:
            boxes=_numpy(result.boxes.xyxy) if count else np.empty((0,4))
            selected,selection=target.choose(boxes,observation)
            info.update(selection)
        if selected is None:
            if 'reject_reason' not in info:info['reject_reason']='未检测到人' if count==0 else '多人画面：暂不自动选择身份'
        elif result.keypoints is None or result.keypoints.conf is None:
            info['reject_reason']='YOLO没有关键点或置信度'
        else:
            points=_numpy(result.keypoints.xy);confidence=_numpy(result.keypoints.conf)
            if points.shape!=(count,17,2) or confidence.shape!=(count,17):raise ValueError('YOLO输出应为17个COCO关键点')
            xy[:17]=points[selected];scores[:17]=confidence[selected]
        return xy,scores,info


def load_backend(config):
    return YOLOBackend(config) if config.backend=='yolo26' else RTMWBackend(config)
