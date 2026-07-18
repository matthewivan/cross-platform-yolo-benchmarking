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
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target tensorrt --precision fp16
    python3 convert_model.py --pt 640_yolov8n_v9.pt --imgsz 640 --target tensorrt --precision int8 --data data.yaml

    target        output          board / runtime
    -----------   -------------   --------------------------------
    onnx          .onnx           Pi Zero 2W (ONNX Runtime, CPU)
    rknn          .rknn           Radxa (rk3566) / Khadas (rk3588), RKNN-Lite
    tensorrt      .engine         Jetson AGX Orin (TensorRT)

IMPORTANT:
  * RKNN export needs rknn-toolkit2 installed and the correct --rknn-name target.
  * TensorRT .engine is hardware/driver specific — build it ON the Jetson AGX Orin
    itself (JetPack + TensorRT), not on your laptop.
  * TensorRT INT8 calibration must use representative data from your deployment.
"""

import argparse
import os
import shutil
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8 for a target board")
    p.add_argument("--pt", required=True, help="Source .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--target", required=True,
                   choices=["onnx", "rknn", "tensorrt"])
    p.add_argument("--rknn-name", default="rk3588",
                   help="RKNN target SoC: rk3566 (Radxa Zero 3W) or rk3588 (Khadas Edge 2)")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32",
                   help="TensorRT precision (INT8 requires representative --data)")
    p.add_argument("--half", action="store_true",
                   help="Deprecated alias for --precision fp16")
    p.add_argument("--data", default=None,
                   help="Dataset YAML used for representative INT8 calibration")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of the calibration dataset used for INT8")
    p.add_argument("--batch", type=int, default=1,
                   help="TensorRT engine/calibration batch size")
    p.add_argument("--workspace", type=float, default=None,
                   help="TensorRT workspace limit in GiB (default: automatic)")
    p.add_argument("--output", default=None,
                   help="Optional final output path, useful for precision-specific names")
    args = p.parse_args()

    if args.half:
        if args.precision not in ("fp32", "fp16"):
            p.error("--half cannot be combined with --precision int8")
        args.precision = "fp16"
    if args.target == "tensorrt" and args.precision == "int8" and not args.data:
        p.error("TensorRT INT8 requires --data with your representative dataset YAML")
    if not 0 < args.fraction <= 1:
        p.error("--fraction must be greater than 0 and at most 1")

    model = YOLO(args.pt)

    if args.target == "onnx":
        out = model.export(format="onnx", imgsz=args.imgsz,
                           dynamic=False, simplify=True)
    elif args.target == "rknn":
        # Ultralytics drives rknn-toolkit2 under the hood.
        out = model.export(format="rknn", imgsz=args.imgsz, name=args.rknn_name)
    else:  # tensorrt
        export_args = dict(format="engine", imgsz=args.imgsz, batch=args.batch,
                           workspace=args.workspace)
        if args.precision == "fp16":
            export_args["quantize"] = 16
        elif args.precision == "int8":
            export_args.update(quantize=8, data=args.data, fraction=args.fraction)
        out = model.export(**export_args)

    if args.output:
        destination = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(str(out), destination)
        out = destination

    print(f"[convert] wrote: {out}")
    if args.target == "tensorrt":
        print(f"[convert] TensorRT precision: {args.precision}")
    print("[convert] rename to '{imgsz}_yolov8n.<ext>' so detector.py finds it "
          "automatically, or pass --model to benchmark.py.")


if __name__ == "__main__":
    main()
