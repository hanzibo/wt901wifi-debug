"""Right-arm configuration widgets; no hardware operations."""
from PySide6 import QtWidgets
from .real_robot_bilateral import right_defaults
from .real_robot_protocol import ARM_LIMITS


def build_right_panel(dialog):
    panel = QtWidgets.QGroupBox('右臂六轴：默认关闭；先逐轴验证，再进行双臂测试')
    grid = QtWidgets.QGridLayout(panel)
    for i,text in enumerate(('输出','最小 °','最大 °','速度 °/s','加速度 °/s²','场景→协议方向')):
        grid.addWidget(QtWidgets.QLabel(text),0,i)
    rows, defaults = [], right_defaults()
    for i in range(6):
        enabled = QtWidgets.QCheckBox(f'R{i+1}')
        enabled.toggled.connect(dialog._invalidate_capture)
        lo, hi = ARM_LIMITS[i]
        sign = QtWidgets.QComboBox()
        sign.addItems(['+1（待逐轴验证）','−1（反向）'])
        sign.currentIndexChanged.connect(dialog._invalidate_capture)
        row = (enabled, dialog._spin(lo,hi,defaults.lower[i],''),
               dialog._spin(lo,hi,defaults.upper[i],''),
               dialog._spin(.1,250,100.,''),dialog._spin(.1,250,150.,''),sign)
        rows.append(row)
        for col,widget in enumerate(row):
            grid.addWidget(widget,i+1,col)
    palm = QtWidgets.QCheckBox('右手掌朝向（需要右手背 IMU；只作用于 R5/R6）')
    palm.setChecked(True)
    offset = QtWidgets.QCheckBox('右手掌固定转向（局部坐标，默认关闭）')
    axis = QtWidgets.QComboBox()
    axis.addItems(['x','y','z']); axis.setCurrentText('z')
    angle = dialog._spin(-180,180,90.,' °')
    copy = QtWidgets.QPushButton('复制仿真窗口右手转向配置')
    copy.clicked.connect(dialog._copy_right_offset)
    palm.toggled.connect(dialog._invalidate_capture)
    offset.toggled.connect(dialog._invalidate_capture)
    axis.currentIndexChanged.connect(dialog._invalidate_capture)
    grid.addWidget(palm,7,0,1,6)
    for col,widget in enumerate((offset,axis,angle,copy)):
        grid.addWidget(widget,8,col,1,2 if col == 3 else 1)
    text = QtWidgets.QLabel('右臂范围来自原 PC 软件，尚非现场验证限位。未勾选轴冻结在 90°；\n'
                           '右臂全部关闭时不要求右侧传感器。任一参与侧断流，两臂同步暂停。')
    grid.addWidget(text,9,0,1,6)
    return panel, rows, (palm,offset,axis,angle,copy)
