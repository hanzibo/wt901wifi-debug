"""Standalone RTMW3D webcam probe; does not modify the existing camera GUI."""

import argparse
import time

import cv2
import onnxruntime as ort
from rtmlib import Wholebody3d, draw_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMW3D webcam and GPU smoke test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda":
        # Loads CUDA/cuDNN libraries installed with onnxruntime-gpu[cuda,cudnn].
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            parser.error("ONNX Runtime 未提供 CUDAExecutionProvider；请检查 GPU 版安装")

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        camera.release()
        parser.error(f"无法打开摄像头 {args.camera_index}；请确认没有其他程序占用")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        print("正在加载检测器和 RTMW3D；首次运行会下载模型，请稍候…", flush=True)
        model = Wholebody3d(backend="onnxruntime", device=args.device)
        for name, session in (("检测器", model.det_model.session),
                              ("姿态模型", model.pose_model.session)):
            providers = session.get_providers()
            print(f"{name}执行后端: {providers}", flush=True)
            if args.device == "cuda" and "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"{name}未启用 CUDA，停止以避免误把 CPU 推理当作 GPU")

        last_report = time.monotonic()
        frame_count = 0
        print("按 Q 或 Esc 退出。画面是 2D 投影；终端显示 3D 输出形状。", flush=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("摄像头读取失败")

            keypoints_3d, scores, _, keypoints_2d = model(frame)
            frame_count += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                fps = frame_count / (now - last_report)
                sample = (keypoints_3d[0, 5].round(3).tolist()
                          if len(keypoints_3d) else "无人")
                print(f"{fps:.1f} 帧/s | 3D {keypoints_3d.shape} | "
                      f"2D {keypoints_2d.shape} | 分数 {scores.shape} | "
                      f"3D 点样例 {sample}",
                      flush=True)
                frame_count = 0
                last_report = now

            if len(keypoints_2d):
                frame = draw_skeleton(frame, keypoints_2d, scores,
                                      openpose_skeleton=False, kpt_thr=0.5)
            cv2.imshow("RTMW3D webcam probe - Q / Esc to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
