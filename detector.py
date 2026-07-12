#!/usr/bin/env python3
"""
detector.py — per-frame YOLOv8 inference timer.

CONTRACT with benchmark.py:
  Prints ONE line per frame, in MILLISECONDS:
      inference time: <ms>
  benchmark.py scrapes stdout for lines starting with "inference time:".

BACKEND is chosen by --framework, which just picks the model file.
Ultralytics auto-detects the runtime from the file EXTENSION, so one code
path covers every board:

    framework   ext        runtime            boards
    ---------   --------   ----------------   -----------------------------
    onnx        .onnx      ONNX Runtime (CPU) Pi Zero 2W  (+ CPU baseline on any board)
    rknn        .rknn      RKNN-Lite          Radxa Zero 3W (RK3566), Khadas Edge 2 (RK3588S)
    tensorrt    .engine    TensorRT           Jetson AGX Orin

By default the model file is derived as  "{imgsz}_yolov8n.{ext}".
Override with --model if your filenames differ.

Timing uses Ultralytics' own reported inference time
(results[0].speed['inference'], already in ms), which excludes pre/post-proc
so you measure the accelerator, not image loading.

--- If you'd rather NOT depend on Ultralytics ---
Swap the `run_ultralytics()` call for a raw backend:
  * ONNX:     onnxruntime.InferenceSession(...).run(...)
  * RKNN:     rknnlite.api.RKNNLite -> load_rknn / init_runtime / inference
  * TensorRT: tensorrt + pycuda, or trtexec
Keep printing "inference time: <ms>" per frame and benchmark.py won't care.
"""

import argparse
import os
import sys
import numpy as np

EXT = {
    "onnx": "onnx",
    "rknn": "rknn",
    "tensorrt": "engine",
    "engine": "engine",  # alias
}


def resolve_model(framework, imgsz, model_arg):
    if model_arg:
        return model_arg
    ext = EXT.get(framework)
    if ext is None:
        sys.exit(f"[detector] unknown framework '{framework}'. "
                 f"Choose from {list(EXT)}.")
    return f"{imgsz}_yolov8n.{ext}"


def make_input(imgsz, image_path):
    """Real image if given, else a fixed synthetic frame (pure-inference bench)."""
    if image_path:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            sys.exit(f"[detector] could not read image: {image_path}")
        return img
    # Deterministic synthetic frame so runs are comparable.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)


def run_ultralytics(model_path, img, imgsz, frames, warmup, core_mask):
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        sys.exit(f"[detector] model not found: {model_path} "
                 f"(convert it first with convert_model.py)")

    model = YOLO(model_path, task="detect")

    # RK3588 (Khadas) can spread across 3 NPU cores; pass through if supported.
    predict_kwargs = dict(imgsz=imgsz, verbose=False)

    # Warm up (first calls include lazy init / graph build — don't measure them).
    for _ in range(max(0, warmup)):
        model(img, **predict_kwargs)

    for _ in range(frames):
        r = model(img, **predict_kwargs)
        ms = r[0].speed.get("inference")
        if ms is None:
            continue
        print(f"inference time: {ms:.3f} ms", flush=True)


def main():
    p = argparse.ArgumentParser(description="YOLOv8 per-frame inference timer")
    p.add_argument("--framework", required=True,
                   choices=list(EXT), help="Selects model file + runtime")
    p.add_argument("--model", default=None,
                   help="Explicit model path (overrides derived name)")
    p.add_argument("--imgsz", type=int, default=640, help="Square input size")
    p.add_argument("--frames", type=int, default=200,
                   help="Number of frames to time")
    p.add_argument("--warmup", type=int, default=10,
                   help="Untimed warmup frames")
    p.add_argument("--image", default=None,
                   help="Optional real image; omit for synthetic frame")
    p.add_argument("--core-mask", default=None,
                   help="RK3588 NPU core mask hint (unused with Ultralytics; "
                        "kept for a raw-RKNN backend)")
    args = p.parse_args()

    model_path = resolve_model(args.framework, args.imgsz, args.model)
    img = make_input(args.imgsz, args.image)
    run_ultralytics(model_path, img, args.imgsz,
                    args.frames, args.warmup, args.core_mask)


if __name__ == "__main__":
    main()
