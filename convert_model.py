#!/usr/bin/env python3
"""
convert_model.py — export a YOLOv8 .pt into every board's runtime format.

Replaces convert2onnx.py. A plain ONNX export is only enough for the CPU path
(Pi Zero 2W). NPU targets need a second, board-specific conversion — which is
why the old "idt this works" comment was right.

Usage:
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target onnx
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target rknn --rknn-name rk3566   # Radxa Zero 3W
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target rknn --rknn-name rk3588   # Khadas Edge 2 (RK3588S uses rk3588)
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target tensorrt                  # MUST run ON the Jetson

    target        output          board / runtime
    -----------   -------------   --------------------------------
    onnx          .onnx           Pi Zero 2W (ONNX Runtime, CPU)
    rknn          .rknn           Radxa (rk3566) / Khadas (rk3588), RKNN-Lite
    tensorrt      .engine         Jetson AGX Orin (TensorRT)

IMPORTANT:
  * RKNN export needs rknn-toolkit2 installed and the correct --rknn-name target.
  * TensorRT .engine is hardware/driver specific — build it ON the Jetson AGX Orin
    itself (JetPack + TensorRT), not on your laptop.
"""

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8 for a target board")
    p.add_argument("--pt", required=True, help="Source .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--target", required=True,
                   choices=["onnx", "rknn", "tensorrt"])
    p.add_argument("--rknn-name", default="rk3588",
                   help="RKNN target SoC: rk3566 (Radxa Zero 3W) or rk3588 (Khadas Edge 2)")
    p.add_argument("--half", action="store_true",
                   help="FP16 (recommended for TensorRT on Jetson)")
    args = p.parse_args()

    model = YOLO(args.pt)

    if args.target == "onnx":
        out = model.export(format="onnx", imgsz=args.imgsz,
                           dynamic=False, simplify=True)
    elif args.target == "rknn":
        # Ultralytics drives rknn-toolkit2 under the hood.
        out = model.export(format="rknn", imgsz=args.imgsz, name=args.rknn_name)
    else:  # tensorrt
        out = model.export(format="engine", imgsz=args.imgsz, half=args.half)

    print(f"[convert] wrote: {out}")
    print("[convert] rename to '{imgsz}_yolov8n.<ext>' so detector.py finds it "
          "automatically, or pass --model to benchmark.py.")


if __name__ == "__main__":
    main()
