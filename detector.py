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

Use --validate-results with real images to print every detection and save
annotated frames under --results-dir (default: ./result). Validation output is
processed inside the measured loop so loop FPS represents the annotated pipeline;
the per-frame "inference time" value remains accelerator inference-only.

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
import time
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


def load_inputs(imgsz, image_path, images_dir):
    """Preload (display name, image) pairs outside the measured loop."""
    if images_dir:
        if not os.path.isdir(images_dir):
            sys.exit(f"[detector] image directory not found: {images_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))
                 if os.path.splitext(name)[1].lower() in extensions]
        if not paths:
            sys.exit(f"[detector] no supported images found in: {images_dir}")
        return [(os.path.basename(path), make_input(imgsz, path)) for path in paths]
    name = os.path.basename(image_path) if image_path else "synthetic.png"
    return [(name, make_input(imgsz, image_path))]


def print_and_save_results(records, results_dir):
    """Print boxes and save annotated frames as part of the measured pipeline."""
    import cv2

    os.makedirs(results_dir, exist_ok=True)
    for frame_index, source_name, result in records:
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        print(f"detections: frame={frame_index + 1} source={source_name} count={count}")

        if boxes is not None:
            for detection_index, box in enumerate(boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                class_name = result.names.get(class_id, str(class_id)) \
                    if isinstance(result.names, dict) else result.names[class_id]
                print(
                    f"detection: frame={frame_index + 1} index={detection_index + 1} "
                    f"class_id={class_id} class={class_name!r} "
                    f"confidence={confidence:.6f} "
                    f"box_xyxy=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                )

        stem, extension = os.path.splitext(source_name)
        extension = extension if extension.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"
        output_name = f"frame_{frame_index + 1:05d}_{stem}{extension}"
        output_path = os.path.join(results_dir, output_name)
        if not cv2.imwrite(output_path, result.plot()):
            print(f"[detector] warning: failed to save {output_path}", file=sys.stderr)
        else:
            print(f"result saved: {output_path}")


def run_ultralytics(model_path, images, imgsz, frames, warmup, core_mask,
                    validate_results, results_dir, conf, iou):
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        sys.exit(f"[detector] model not found: {model_path} "
                 f"(convert it first with convert_model.py)")

    model = YOLO(model_path, task="detect")

    # RK3588 (Khadas) can spread across 3 NPU cores; pass through if supported.
    predict_kwargs = dict(imgsz=imgsz, verbose=False, conf=conf, iou=iou)

    # Warm up (first calls include lazy init / graph build — don't measure them).
    for _ in range(max(0, warmup)):
        model(images[0][1], **predict_kwargs)

    loop_start = time.perf_counter()
    for index in range(frames):
        source_name, img = images[index % len(images)]
        r = model(img, **predict_kwargs)
        ms = r[0].speed.get("inference")
        if ms is None:
            continue
        print(f"inference time: {ms:.3f} ms", flush=True)
        if validate_results:
            print_and_save_results([(index, source_name, r[0])], results_dir)
    loop_elapsed = time.perf_counter() - loop_start
    print(f"measured loop elapsed: {loop_elapsed:.9f} s", flush=True)


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
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold")
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--image", default=None,
                        help="Optional real image; omit for synthetic input")
    inputs.add_argument("--images", default=None,
                        help="Optional image directory; inputs are preloaded and cycled")
    p.add_argument("--core-mask", default=None,
                   help="RK3588 NPU core mask hint (unused with Ultralytics; "
                        "kept for a raw-RKNN backend)")
    p.add_argument("--validate-results", action="store_true",
                   help="Print and save annotated detections inside the measured loop")
    p.add_argument("--results-dir", default="result",
                   help="Annotated output directory used by --validate-results")
    args = p.parse_args()

    model_path = resolve_model(args.framework, args.imgsz, args.model)
    images = load_inputs(args.imgsz, args.image, args.images)
    run_ultralytics(model_path, images, args.imgsz,
                    args.frames, args.warmup, args.core_mask,
                    args.validate_results, args.results_dir,
                    args.conf, args.iou)


if __name__ == "__main__":
    main()
