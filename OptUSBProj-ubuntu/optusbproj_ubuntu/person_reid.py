"""Local OSNet inference; no downloads or third-party package initialization."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'no-improtent'
SOURCE = ROOT / 'deep-person-reid/torchreid/models/osnet.py'
WEIGHTS = ROOT / 'reid-models/osnet_x1_0_msmt17.pth'


class OSNetEncoder:
    def __init__(self):
        import torch
        if not SOURCE.is_file() or not WEIGHTS.is_file():
            raise RuntimeError('缺少本地 OSNet 源码或 MSMT17 权重')
        if not torch.cuda.is_available():
            raise RuntimeError('OSNet CUDA 不可用')
        spec = importlib.util.spec_from_file_location('wt901_osnet', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.osnet_x1_0(num_classes=1, pretrained=False)
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k,v in state.items()
                 if not k.removeprefix('module.').startswith('classifier.')}
        missing, extra = self.model.load_state_dict(state, strict=False)
        if set(missing) != {'classifier.weight','classifier.bias'} or extra:
            raise RuntimeError(f'OSNet 权重不匹配: {missing}, {extra}')
        self.model = self.model.eval().cuda()
        self.metadata = dict(model='OSNet x1.0 MSMT17',
            weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def __call__(self, rgb, boxes):
        import cv2
        import torch
        if not len(boxes): return np.empty((0,512))
        crops=[]
        h,w=rgb.shape[:2]
        for box in boxes:
            x1,y1,x2,y2=np.asarray(box[:4],int)
            crop=rgb[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if not crop.size: raise ValueError('OSNet 空人物框')
            crops.append(cv2.resize(crop,(128,256)))
        tensor=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).cuda().float()/255
        mean=tensor.new_tensor([.485,.456,.406])[None,:,None,None]
        std=tensor.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            features=self.model((tensor-mean)/std)
            features=torch.nn.functional.normalize(features,dim=1)
        return features.cpu().numpy()
