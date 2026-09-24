"""姿态角 -> 旋转矩阵, 以及坐标系约定的验证工具。

WT901WIFI 输出三个欧拉角:
    Roll  = 绕 X 轴
    Pitch = 绕 Y 轴
    Yaw   = 绕 Z 轴   (9 轴算法下为绝对航向角, 以东北天为参考)

把三个角转成旋转矩阵 R 后:
    * 机体坐标系的三个基向量在世界系中的方向 = R 的三列
    * 用它们即可在 3D 场景里画出随传感器转动的三条轴线

关于旋转顺序 (order):
    不同顺序算出的 R 不同。这里提供两种常见约定, 并用加速度计
    做交叉验证 —— 静止时加速度计测到的重力方向, 必须和 R 推算出的
    重力方向一致。`best_order()` 会自动挑出与实测加速度最吻合的那种。
"""

from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix(roll: float, pitch: float, yaw: float,
                    order: str = "ZYX") -> np.ndarray:
    """把欧拉角 (单位: 度) 转成 3x3 旋转矩阵。

    order:
        "ZYX" -> R = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (航空航天常用)
        "XYZ" -> R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    """
    rx, ry, rz = _rot_x(roll * DEG), _rot_y(pitch * DEG), _rot_z(yaw * DEG)
    if order == "ZYX":
        return rz @ ry @ rx
    if order == "XYZ":
        return rx @ ry @ rz
    raise ValueError(f"未知旋转顺序: {order}")


def body_axes(roll: float, pitch: float, yaw: float,
              order: str = "ZYX") -> np.ndarray:
    """返回 3x3 矩阵, 第 i 列 = 机体第 i 轴在世界系中的单位向量。"""
    return rotation_matrix(roll, pitch, yaw, order)


def up_body(roll: float, pitch: float, yaw: float,
            order: str = "ZYX") -> np.ndarray:
    """由姿态角推算"世界系竖直向上"方向在机体坐标系下的单位向量。

    静止时加速度计测的就是这个方向 (加速度计读的是支持力/比力, 向上为正),
    所以它应当与归一化后的加速度读数一致。
    """
    R = rotation_matrix(roll, pitch, yaw, order)
    return R.T @ np.array([0.0, 0.0, 1.0])


def accel_direction(acc_x: float, acc_y: float, acc_z: float) -> np.ndarray:
    """加速度计测到的"上"方向 (单位向量)。"""
    v = np.array([acc_x, acc_y, acc_z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def fit_error(packet, order: str = "ZYX") -> float:
    """用一帧数据算: 姿态角推算的"上"方向 与 实测加速度方向 的夹角 (度)。"""
    a = accel_direction(packet.acc_x, packet.acc_y, packet.acc_z)
    g = up_body(packet.roll, packet.pitch, packet.yaw, order)
    cos = float(np.clip(np.dot(a, g), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def best_order(packets, candidates=("ZYX", "XYZ")) -> tuple[str, dict[str, float]]:
    """在若干旋转顺序里, 挑出与加速度计最吻合的那个。

    返回 (最佳顺序, {顺序: 平均夹角误差})。误差越小说明约定越正确。
    """
    scores: dict[str, float] = {}
    for order in candidates:
        errs = [fit_error(p, order) for p in packets]
        scores[order] = float(np.mean(errs)) if errs else float("inf")
    best = min(scores, key=scores.get)
    return best, scores
