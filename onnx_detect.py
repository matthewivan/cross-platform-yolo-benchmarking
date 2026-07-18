#!/usr/bin/env python3
"""
onnx_detect.py — Raspberry Pi Zero 2W ONNX runner that MIRRORS your RKNN script.

This is your RKNN test script with three changes so the comparison is fair:
  1. backend: RKNN_model_container -> onnxruntime CPU session
  2. input:   RKNN took uint8 HWC; the float ONNX needs /255 float32 NCHW
  3. output order is reconciled by channel/shape (onnxruntime may order the 6
     branches differently than RKNN) before post_process runs.

Everything else — letter_box, DFL decode, box_process, filter, NMS, thresholds,
class list, and the timing scope (inference + post_process together) — is kept
IDENTICAL to your RKNN script, so latency differences reflect hardware, not code.

IMPORTANT: point --model at the SAME ONNX that was fed to rknn-toolkit2 (the
rknn_model_zoo / airockchip-style export with 6 outputs). A plain
`ultralytics export format=onnx` produces the merged [1, 4+nc, 8400] layout and
will NOT work with this post_process.

Prints per image (SECONDS, matching your script):
    inference time: <seconds>
So run benchmark.py with --latency-unit s.

Deps (aarch64 wheels exist for all): onnxruntime numpy opencv-python-headless
Also needs your py_utils/coco_utils.py (COCO_test_helper) on the path — it's
pure numpy/cv2, no RKNN dependency.

Usage:
  python3 onnx_detect.py --model model_256.onnx --imgs ./images --imgsz 256
  python3 benchmark.py \
      --external-cmd "python3 onnx_detect.py --model model_256.onnx --imgs ./images --imgsz 256" \
      --latency-unit s --frameworks onnx --output rpi_benchmark_256.csv
"""

import os
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from py_utils.coco_utils import COCO_test_helper

# ---- thresholds (overridable via CLI; kept as globals like your script) ----
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (w, h) — set from --imgsz

CLASSES = ('Black Buoy', 'Blue Buoy', 'Green Buoy', 'Maroon Buoy', 'Or', 'Orange Buoy', 'Red Buoy', 'Wader', 'White Buoy', 'Yellow Buoy', 'Zebra Buoy')


# ===========================================================================
# Post-processing — copied VERBATIM from your RKNN script (pure numpy).
# ===========================================================================
def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float64))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores


def class_name(cl):
    return CLASSES[cl] if 0 <= cl < len(CLASSES) else f"class_{cl}"


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (class_name(cl), top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(class_name(cl), score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def img_check(path):
    for _type in ['.jpg', '.jpeg', '.png', '.bmp']:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


# ===========================================================================
# ONNX-specific bits
# ===========================================================================
def organize_outputs(outputs):
    """
    Reorder branches into what post_process expects, regardless of how
    onnxruntime ordered them and regardless of 2-per-branch (box,cls) or
    3-per-branch (box,cls,score) exports.

    Group tensors by spatial size (H,W), then within each group sort by channel
    count DESCENDING -> [box(64), cls(nc), score(1)]. post_process indexes
    box at pair*i and cls at pair*i+1, so this ordering is correct for both
    6-output and 9-output models.
    """
    groups = {}
    for o in outputs:
        groups.setdefault(tuple(o.shape[2:]), []).append(o)
    ordered = []
    for key in sorted(groups, key=lambda k: k[0], reverse=True):  # big grid -> small
        ordered += sorted(groups[key], key=lambda t: t.shape[1], reverse=True)
    return ordered


def main():
    global OBJ_THRESH, NMS_THRESH, IMG_SIZE, CLASSES

    p = argparse.ArgumentParser(description="Pi Zero 2W ONNX runner (mirrors RKNN script)")
    p.add_argument("--model", required=True, help="6-output ONNX (the one converted to RKNN)")
    p.add_argument("--imgs", default="./images", help="Image directory")
    p.add_argument("--imgsz", type=int, default=256, help="Square input size")
    p.add_argument("--loops", type=int, default=1, help="Repeat image set N times")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup runs")
    p.add_argument("--conf", type=float, default=OBJ_THRESH)
    p.add_argument("--iou", type=float, default=NMS_THRESH)
    p.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0=auto)")
    p.add_argument("--save", action="store_true", help="Save annotated results (off by default)")
    args = p.parse_args()

    OBJ_THRESH, NMS_THRESH = args.conf, args.iou
    IMG_SIZE = (args.imgsz, args.imgsz)

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, sess_options=so,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    file_list = sorted(os.listdir(args.imgs))
    img_list = [f for f in file_list if img_check(f)]
    co_helper = COCO_test_helper(enable_letter_box=True)

    # Warmup (not timed)
    if img_list:
        w_src = cv2.imread(os.path.join(args.imgs, img_list[0]))
        w_img = co_helper.letter_box(im=w_src.copy(),
                                     new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        w_img = cv2.cvtColor(w_img, cv2.COLOR_BGR2RGB)
        w_blob = np.ascontiguousarray(
            np.transpose(w_img.astype(np.float32) / 255.0, (2, 0, 1))[None])
        for _ in range(max(0, args.warmup)):
            session.run(None, {input_name: w_blob})

    measured_count = 0
    loop_start = time.perf_counter()
    for _ in range(args.loops):
        for img_name in img_list:
            img_path = os.path.join(args.imgs, img_name)
            img_src = cv2.imread(img_path)
            if img_src is None:
                continue

            img = co_helper.letter_box(im=img_src.copy(),
                                       new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # ONNX float NCHW input (RKNN used uint8 HWC — this is the key diff)
            blob = np.ascontiguousarray(
                np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None])

            # --- timed region: inference + post_process (same scope as RKNN) ---
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: blob})
            outputs = organize_outputs(outputs)
            boxes, classes, scores = post_process(outputs)
            inference_time = time.perf_counter() - start_time
            measured_count += 1

            print(f"inference time: {inference_time}")   # SECONDS (use --latency-unit s)
            print('IMG: {}'.format(img_name))

            if boxes is not None:
                real = co_helper.get_real_box(boxes)
                for box, score, cl in zip(real, scores, classes):
                    print(f"{class_name(cl)}: {score}")
                if args.save:
                    img_p = img_src.copy()
                    draw(img_p, real, scores, classes)
                    os.makedirs('./result', exist_ok=True)
                    cv2.imwrite(os.path.join('./result', img_name), img_p)
    loop_elapsed = time.perf_counter() - loop_start
    if measured_count:
        print(f"measured loop elapsed: {loop_elapsed:.9f} s")


if __name__ == '__main__':
    main()
