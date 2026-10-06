"""Standalone 2-D hand backend comparison. No IMU, calibration or robot writes."""
from pathlib import Path
from threading import Condition
import time
import numpy as np
from PySide6 import QtCore,QtGui,QtWidgets
from .orbbec_worker import OrbbecWorker,DEFAULT_PYTHON
from .palm_worker import DEFAULT_HAND_MODEL

MODEL_ROOT=Path(__file__).resolve().parents[2]/'no-improtent'/'hand-models'
EDGES=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))


def local_model(value):
    path=Path(value)
    if path.is_dir():
        files=sorted(path.rglob('*.onnx'))
        if len(files)!=1:raise ValueError(f'{path} 中需要唯一一个ONNX文件，当前{len(files)}个；请直接选择文件')
        path=files[0]
    if not path.is_file():raise ValueError(f'本地模型不存在：{path}；请先下载并解压，不自动联网')
    if path.suffix.lower() not in ('.onnx','.task'):raise ValueError('请选择解压后的.onnx或MediaPipe.task，不能直接加载zip')
    return path


class HandBackend:
    def __init__(self,config):
        self.config=config;self.mp=None;self.providers='CPU'
        if config['backend']=='mediapipe':
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self.mp=mp
            self.model=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(local_model(config['mp_model']))),
                num_hands=4,min_hand_detection_confidence=config['det_threshold'],
                min_hand_presence_confidence=config['det_threshold']))
        else:
            from rtmlib import RTMDet,RTMPose
            # Explicit file validation prevents rtmlib auto-download fallback.
            det=str(local_model(config['det_model']));pose=str(local_model(config['pose_model']))
            self.det=RTMDet(det,model_input_size=(320,320),backend='onnxruntime',device=config['device'],score_thr=config['det_threshold'])
            self.pose=RTMPose(pose,model_input_size=(256,256),backend='onnxruntime',device=config['device'],to_openpose=False)
            providers=[self.det.session.get_providers(),self.pose.session.get_providers()]
            self.providers=str(providers)
            if config['device']=='cuda' and any('CUDAExecutionProvider' not in p for p in providers):
                raise RuntimeError('请求CUDA但ONNX会话未使用CUDA，请选CPU或检查CUDA/cuDNN依赖')

    def infer(self,rgb,stamp):
        import cv2
        started=time.perf_counter();h,w=rgb.shape[:2]
        if self.config['centre_roi']:
            x0,y0,x1,y1=w//4,h//4,3*w//4,3*h//4
        else:x0,y0,x1,y1=0,0,w,h
        crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]);offset=np.array([x0,y0])
        hands=[];boxes=[];labels=[];scores=[]
        if self.mp is not None:
            result=self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=crop))
            for i,points in enumerate(result.hand_landmarks):
                points=np.array([[p.x*crop.shape[1],p.y*crop.shape[0]] for p in points])+offset
                hands.append(points);scores.append(np.ones(21)) # No per-landmark confidence in this API.
                boxes.append(np.r_[points.min(axis=0),points.max(axis=0)])
                labels.append(result.handedness[i][0].category_name+' (MP)')
        else:
            bgr=cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
            detected=np.asarray(self.det(bgr),float).reshape(-1,4)
            # RTMPose(empty boxes) falls back to full-image pose: avoid hallucinated hands.
            if len(detected):
                points,conf=self.pose(bgr,bboxes=detected)
                for box,pts,score in zip(detected,points,conf):
                    hands.append(np.asarray(pts)+offset);scores.append(np.asarray(score))
                    boxes.append(box+np.r_[offset,offset]);labels.append('RTM hand')
        image=rgb.copy();threshold=self.config['point_threshold'];visible=0
        def pixel(p):return tuple(np.rint(p).astype(int))
        if self.config['centre_roi']:cv2.rectangle(image,(x0,y0),(x1,y1),(255,180,0),2)
        for box,points,score,label in zip(boxes,hands,scores,labels):
            valid=np.isfinite(points).all(axis=1)&np.isfinite(score)&(score>=threshold)
            if valid.sum()<5:continue
            visible+=1;cv2.rectangle(image,pixel(box[:2]),pixel(box[2:]),(0,255,255),2)
            cv2.putText(image,label,pixel(box[:2]+[0,-5]),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            for a,b in EDGES:
                if valid[a] and valid[b]:cv2.line(image,pixel(points[a]),pixel(points[b]),(255,255,0),2)
            for i,p in enumerate(points):
                if valid[i]:cv2.circle(image,pixel(p),4,(255,0,255) if i==0 else (255,255,0),-1)
        return dict(frame_rgb=image,captured_mono_ns=stamp,inference_ms=(time.perf_counter()-started)*1000,
                    detected_count=len(hands),visible_count=visible,valid_points=[int((np.isfinite(s)&(s>=threshold)).sum()) for s in scores],
                    keypoints_xy=[p.tolist() for p in hands],boxes_xyxy=[np.asarray(b).tolist() for b in boxes])

    def close(self):
        if self.mp is not None:self.model.close()


class HandDemoWorker(QtCore.QThread):
    ready=QtCore.Signal(str);error=QtCore.Signal(str)
    def __init__(self,config,parent=None):
        super().__init__(parent);self.config=config;self.condition=Condition();self.running=True;self.job=None;self.result=None;self.skipped=0
    def submit(self,rgb,stamp):
        with self.condition:
            if self.running:
                if self.job is not None:self.skipped+=1
                self.job=(rgb,stamp);self.condition.notify()
    def take_latest(self):
        with self.condition:result,self.result=self.result,None;return result
    def stop(self):
        with self.condition:self.running=False;self.job=None;self.condition.notify()
    def run(self):
        backend=None
        try:
            backend=HandBackend(self.config);self.ready.emit(backend.providers)
            while True:
                with self.condition:
                    while self.running and self.job is None:self.condition.wait(.5)
                    if not self.running:break
                    rgb,stamp=self.job;self.job=None
                result=backend.infer(rgb,stamp)
                with self.condition:self.result=result
        except Exception as exc:self.error.emit('手部后端失败：'+str(exc))
        finally:
            if backend is not None:backend.close()


class HandDemoWindow(QtWidgets.QWidget):
    def __init__(self,orbbec_python=DEFAULT_PYTHON):
        super().__init__();self.orbbec_python=Path(orbbec_python);self.worker=None;self.camera=None;self.static_rgb=None;self.pending=False
        self.processed=self.detected=0;self.last_result=None;self.providers=''
        self.setWindowTitle('Gemini · 手部21点后端对比（独立Demo）');self.resize(1050,850)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('仅2D检测，不做深度/标定/IMU/X1控制。先关闭其他占用Gemini的程序；切换后端前先停止。MediaPipe左右标签仅供观察，分数不能与RTM直接比较。')
        note.setWordWrap(True);layout.addWidget(note)
        self.backend=QtWidgets.QComboBox();self.backend.addItem('MediaPipe HandLandmarker（全图）','mediapipe');self.backend.addItem('RTMDet-nano＋RTMPose-m Hand（全图）','rtm')
        self.device=QtWidgets.QComboBox();self.device.addItems(['cpu','cuda'])
        self.det_threshold=QtWidgets.QDoubleSpinBox();self.det_threshold.setRange(.05,.95);self.det_threshold.setSingleStep(.05);self.det_threshold.setValue(.5)
        self.point_threshold=QtWidgets.QDoubleSpinBox();self.point_threshold.setRange(.05,.95);self.point_threshold.setSingleStep(.05);self.point_threshold.setValue(.3)
        self.backend.currentIndexChanged.connect(lambda:self.det_threshold.setValue(.5 if self.backend.currentData()=='mediapipe' else .3))
        self.roi=QtWidgets.QCheckBox('只检测中央50%区域（把手放入黄框；对照小目标）')
        form=QtWidgets.QFormLayout();layout.addLayout(form)
        for name,widget in (('后端',self.backend),('RTM执行设备（MediaPipe始终CPU）',self.device),('检测阈值（两模型含义不同）',self.det_threshold),('RTM关键点显示阈值（不用于MediaPipe）',self.point_threshold)):form.addRow(name,widget)
        layout.addWidget(self.roi)
        self.paths={}
        for key,title,default in (('mp_model','MediaPipe权重',DEFAULT_HAND_MODEL),('det_model','RTMDet ONNX',MODEL_ROOT/'rtm-det'),('pose_model','RTMPose-Hand ONNX',MODEL_ROOT/'rtm-pose')):
            entry=QtWidgets.QLineEdit(str(default));self.paths[key]=entry
            row=QtWidgets.QHBoxLayout();row.addWidget(entry);button=QtWidgets.QPushButton('选择…');row.addWidget(button)
            button.clicked.connect(lambda _,k=key:self.choose_model(k));form.addRow(title,row)
        row=QtWidgets.QHBoxLayout();self.start_button=QtWidgets.QPushButton('启动Gemini测试');self.stop_button=QtWidgets.QPushButton('停止');self.image_button=QtWidgets.QPushButton('选择图片离线测试…');self.camera_button=QtWidgets.QPushButton('切回Gemini')
        for button in (self.start_button,self.stop_button,self.image_button,self.camera_button):row.addWidget(button)
        layout.addLayout(row);self.start_button.clicked.connect(self.start);self.stop_button.clicked.connect(self.stop);self.image_button.clicked.connect(self.choose_image);self.camera_button.clicked.connect(self.select_camera)
        self.preview=QtWidgets.QLabel('等待启动：原始彩色图 → 手框 → 21点，无人体裁剪依赖');self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);self.preview.setMinimumSize(480,270);self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Ignored);layout.addWidget(self.preview,1)
        self.status=QtWidgets.QLabel('未启动；模型只从本地加载，不自动下载。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(30);self.timer.timeout.connect(self.poll);self.timer.start();self.controls(False)

    def controls(self,running):
        for widget in (self.backend,self.device,self.det_threshold,self.point_threshold,self.roi,self.image_button,self.camera_button,*self.paths.values()):widget.setEnabled(not running)
        self.start_button.setEnabled(not running);self.stop_button.setEnabled(running)
    def choose_model(self,key):
        if self.worker is not None:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择本地模型',str(MODEL_ROOT),'Models (*.onnx *.task)')
        if path:self.paths[key].setText(path)
    def choose_image(self):
        import cv2
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择测试图片','','Images (*.jpg *.jpeg *.png)')
        if not path:return
        bgr=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:self.status.setText('图片读取失败');return
        self.static_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);self.start_button.setText('启动图片测试');self.show_rgb(self.static_rgb)
    def select_camera(self):
        self.static_rgb=None;self.start_button.setText('启动Gemini测试');self.status.setText('已选择Gemini；启动前关闭其他相机程序')
    def start(self):
        if self.worker is not None or self.camera is not None:return
        config=dict(backend=self.backend.currentData(),device=self.device.currentText(),det_threshold=self.det_threshold.value(),point_threshold=self.point_threshold.value(),centre_roi=self.roi.isChecked(),**{k:v.text() for k,v in self.paths.items()})
        self.pending=True;self.controls(True);self.processed=self.detected=0;self.last_result=None;self.providers=''
        self.status.setText('正在加载后端；首次CUDA会话初始化可能较慢…')
        self.worker=HandDemoWorker(config,self);self.worker.ready.connect(self.on_ready);self.worker.error.connect(self.fail);self.worker.finished.connect(self.reap);self.worker.start()
    def on_ready(self,providers):
        if not self.pending:return
        self.providers=providers
        self.status.setText('已加载；实际执行后端：'+providers)
        if self.static_rgb is not None:self.worker.submit(self.static_rgb,time.monotonic_ns())
        else:
            self.camera=OrbbecWorker(self.orbbec_python,fps=15,warmup=2,parent=self)
            self.camera.error.connect(self.fail);self.camera.finished.connect(self.camera_finished);self.camera.start()
    def fail(self,message):self.stop();self.status.setText(message)
    def stop(self):
        self.pending=False
        if self.camera is not None or self.worker is not None:
            self.status.setText('正在停止相机与推理后端，请等待控件恢复可用…');self.stop_button.setEnabled(False)
        if self.camera is not None:self.camera.stop()
        if self.worker is not None:self.worker.stop()
        if self.camera is None and self.worker is None:self.controls(False)
    def reap(self):
        if self.worker is not None and not self.worker.isRunning():self.worker.deleteLater();self.worker=None
        if self.worker is None and self.camera is None:self.controls(False)
    def camera_finished(self):
        if self.camera is not None:self.camera.deleteLater();self.camera=None
        if self.pending:self.fail('Gemini采集已结束，请停止后重新启动')
        self.reap()
    def show_rgb(self,rgb):
        h,w=rgb.shape[:2];image=QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.preview.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))
    def poll(self):
        if not self.pending:return
        if self.camera is not None:
            observation=self.camera.take_latest()
            if observation is not None:
                if self.last_result is None:self.show_rgb(observation.frame_rgb)
                if self.worker is not None:self.worker.submit(observation.frame_rgb,observation.captured_mono_ns)
        if self.worker is not None:
            row=self.worker.take_latest()
            if row is not None:
                self.last_result=row;self.processed+=1;self.detected+=int(row['visible_count']>0);self.show_rgb(row['frame_rgb'])
                age=(time.monotonic_ns()-row['captured_mono_ns'])/1e6
                self.status.setText(f"检测手框 {row['detected_count']}；显示手掌 {row['visible_count']}；有效点 {row['valid_points']}；推理 {row['inference_ms']:.1f} ms；帧龄 {age:.0f} ms\n已处理 {self.processed} 帧，至少一手可显示 {self.detected/self.processed:.0%}（不是精度/GT）；跳过 {self.worker.skipped} 个等待帧；黄色21点，品红腕点。\n实际执行：{self.providers}")
    def closeEvent(self,event):
        self.stop()
        threads=[t for t in (self.worker,self.camera) if t is not None]
        if all(t.wait(1500) for t in threads):event.accept()
        else:self.status.setText('后端/相机正在退出，请稍后再关闭');event.ignore()


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--orbbec-python',type=Path,default=DEFAULT_PYTHON)
    args=parser.parse_args();app=QtWidgets.QApplication([]);window=HandDemoWindow(args.orbbec_python);window.show();return app.exec()
