#!/usr/bin/env python3
"""Compare raw visual surface points, pure six-axis and offline fused right arm."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from optusbproj_ubuntu.rgbd_pose import RGBDPoseEstimate
from optusbproj_ubuntu.rgbd_pose_gui import MetricSkeletonWidget
from optusbproj_ubuntu.camera_preview import CameraPreview


class SharedViewCanvas(MetricSkeletonWidget):
    view_changed=QtCore.Signal(float,float,float)

    def mouseMoveEvent(self,event):
        super().mouseMoveEvent(event)
        self.view_changed.emit(float(self.yaw),float(self.tilt),self.zoom)

    def wheelEvent(self,event):
        super().wheelEvent(event)
        self.view_changed.emit(float(self.yaw),float(self.tilt),self.zoom)


class ComparisonWindow(QtWidgets.QWidget):
    def __init__(self, analysis):
        super().__init__()
        self.report=json.loads((analysis/'report.json').read_text())
        self.rows=[json.loads(l) for l in (analysis/'comparison.jsonl').open()]
        if not self.rows:
            raise ValueError('没有回放帧')
        self.source=Path(self.report['source_session'])
        self.setWindowTitle('右臂离线对比 · RGB-D / 纯IMU / 航向融合（非实时控制）')
        self.resize(1200,850)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('左：相机表面点；中：纯陀螺＋加计；右：同一IMU＋视觉低频航向纠偏。三图使用相同视角/比例。IMU图肩点来自相机，骨长固定为标定中值；不是STL模型或机器人轨迹。前段标定，后段才融合。')
        note.setWordWrap(True);layout.addWidget(note)
        self.image=CameraPreview();self.image.setFixedHeight(260)
        self.image.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(self.image)
        panels=QtWidgets.QHBoxLayout();self.canvases=[]
        shoulder_points=[r['vision']['points_camera_m'][1] for r in self.rows if r['vision']['valid'][1]]
        center=np.nanmedian(np.array(shoulder_points,float),axis=0)+np.array([0,.15,0])
        for title in ('RGB-D原始表面观测（并非真值）','纯IMU（固定方向标定）','IMU＋视觉低频航向纠偏'):
            column=QtWidgets.QVBoxLayout()
            heading=QtWidgets.QLabel(title);heading.setFixedHeight(24);column.addWidget(heading)
            canvas=SharedViewCanvas();canvas.view_center=center;canvas.view_span=.75
            canvas.title_text='右臂 · 拖动旋转 / 滚轮缩放'
            canvas.view_changed.connect(self.sync_view)
            column.addWidget(canvas,1);panels.addLayout(column);self.canvases.append(canvas)
        layout.addLayout(panels,1)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        controls=QtWidgets.QHBoxLayout()
        self.play=QtWidgets.QPushButton('播放');self.play.clicked.connect(self.toggle_play);controls.addWidget(self.play)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.slider.setRange(0,len(self.rows)-1)
        self.slider.valueChanged.connect(self.show_frame);controls.addWidget(self.slider,1);layout.addLayout(controls)
        self.timer=QtCore.QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.advance)
        self.show_frame(0)

    def sync_view(self,yaw,tilt,zoom):
        for canvas in self.canvases:
            canvas.yaw=yaw;canvas.tilt=tilt;canvas.zoom=zoom;canvas.update()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop();self.play.setText('播放')
        else:
            if self.slider.value()==self.slider.maximum():self.slider.setValue(0)
            self.play.setText('暂停');self.timer.start(1)

    def advance(self):
        index=self.slider.value()
        if index>=self.slider.maximum():
            self.play.setText('播放');return
        self.slider.setValue(index+1)
        following=min(index+2,self.slider.maximum())
        dt=self.rows[following]['time_s']-self.rows[index+1]['time_s']
        self.timer.start(max(1,round(dt*1000)))

    def show_frame(self,index):
        row=self.rows[index]
        self.image.display_image(QtGui.QImage(str(self.source/row['frame'])))
        visual=RGBDPoseEstimate.from_record(row['vision'])
        only_right=np.zeros(8,bool);only_right[[1,3,5]]=visual.valid[[1,3,5]]
        visual=replace(visual,valid=only_right)
        self.canvases[0].estimate=visual
        shoulder=visual.points_camera_m[1]
        for mode,canvas in zip(('pure','fused'),self.canvases[1:]):
            xyz=np.full((8,3),np.nan);valid=np.zeros(8,bool)
            if visual.valid[1]:
                xyz[1]=shoulder;valid[1]=True
                for part,a,b in [('right_upper_arm',1,3),('right_forearm',3,5)]:
                    info=self.report['parts'].get(part,{})
                    direction=row['parts'].get(part,{}).get('direction_'+mode)
                    if valid[a] and direction is not None:
                        xyz[b]=xyz[a]+np.asarray(direction)*info['bone_length_m'];valid[b]=True
            canvas.estimate=replace(visual,points_camera_m=xyz,valid=valid)
        for canvas in self.canvases:canvas.update()
        states=[f"{part}: {value.get('offset_deg',0):+.2f}° · {value['reason']}" for part,value in row['parts'].items()]
        self.status.setText(f"t={row['time_s']:.2f}s · 帧 {index+1}/{len(self.rows)} · "+' | '.join(states))

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('analysis',type=Path)
    args=parser.parse_args();app=QtWidgets.QApplication([])
    window=ComparisonWindow(args.analysis);window.show();app.exec()


if __name__=='__main__':main()
