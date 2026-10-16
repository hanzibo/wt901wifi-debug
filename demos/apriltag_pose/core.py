"""OpenCV tag36h11 detection and metric, camera-relative square pose."""
from dataclasses import dataclass
import cv2
import numpy as np

DEFAULT_SIZES_MM = {0: 50., 1: 25., 2: 16.}


def camera_model(calibration, image_shape):
    """Reject mismatched geometry rather than silently guessing intrinsics."""
    i = calibration['color_intrinsic']
    if (i['width'], i['height']) != (image_shape[1], image_shape[0]):
        raise ValueError('彩色内参与图像尺寸不符，拒绝估计姿态')
    k = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1.]], float)
    if not np.isfinite(k).all() or min(i['fx'], i['fy']) <= 0:
        raise ValueError('无效相机内参')
    d = calibration['color_distortion']
    model = str(d['model']).upper()
    if 'NONE' in model:
        distortion = np.zeros(8)
    elif 'BROWN_CONRADY' in model and not any(x in model for x in ('INVERSE', 'MODIFIED')):
        distortion = np.array([d[n] for n in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')], float)
    else:
        raise ValueError(f'尚未验证的畸变模型：{model}；不使用猜测参数')
    if not np.isfinite(distortion).all():
        raise ValueError('无效畸变参数')
    return k, distortion


def square_points(size_m):
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError('标签黑框尺寸必须为正数')
    h = size_m/2
    # OpenCV canonical corners: top-left, top-right, bottom-right, bottom-left.
    # Tag +x right, +y up; +z out of the printed front face.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray
    error_px: float
    ambiguous: bool


def estimate_pose(corners, size_m, k, d):
    points = square_points(size_m)
    corners = np.asarray(corners, float).reshape(4, 2)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        points, corners, k, d, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    candidates = []
    for r, t in zip(rotations, translations):
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            continue
        # IPPE can lose precision for near-frontal squares. Refine each seed
        # against the actual distorted pixel observations, retaining both seeds.
        r, t = cv2.solvePnPRefineLM(points, corners, k, d, r.copy(), t.copy())
        rotation = cv2.Rodrigues(r)[0]
        if np.any((points @ rotation.T + t.reshape(1, 3))[:, 2] <= 0):
            continue
        projection = cv2.projectPoints(points, r, t, k, d)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projection-corners)**2, axis=1))))
        candidates.append((error, r, t))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    ambiguous = len(candidates) > 1 and candidates[1][0] <= max(.15, best[0]*1.2)
    return Pose(best[1], best[2], best[0], ambiguous)


class Detector:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, rgb):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        return [] if ids is None else [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)]
