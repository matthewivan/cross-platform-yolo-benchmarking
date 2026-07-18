import os
import cv2
import time

from py_utils.rknn_executor import RKNN_model_container
from py_utils.coco_utils import COCO_test_helper
import numpy as np

# confidence threshold
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ('Black Buoy', 'Blue Buoy', 'Green Buoy', 'Maroon Buoy', 'Or', 'Orange Buoy', 'Red Buoy', 'Wader', 'White Buoy', 'Yellow Buoy', 'Zebra Buoy') 


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
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
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
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
    keep = np.array(keep)
    return keep


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def dfl(position):
    # Distribution Focal Loss (DFL)
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

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
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


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


if __name__ == '__main__':
    # init var
    model_path = './models/yolov8n_second_buoy.rknn'
    target = 'rk3566'
    img_folder = './images'

    # init model
    platform = 'rknn'
    model = RKNN_model_container(model_path, target, None)

    file_list = sorted(os.listdir(img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # run test; model construction is deliberately outside this measured loop.
    measured_count = 0
    loop_start = time.perf_counter()
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i + 1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        pad_color = (0, 0, 0)
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.perf_counter()
        outputs = model.run([img])
        boxes, classes, scores = post_process(outputs)
        inference_time = time.perf_counter() - start_time
        measured_count += 1

        print(f"inference time: {inference_time}")
        print('IMG: {}'.format(img_name))
        img_p = img_src.copy()

        if boxes is not None:
            draw(img_p, co_helper.get_real_box(boxes), scores, classes)
            for box, score, cl in zip(boxes, scores, classes):
                print(f"{CLASSES[cl]}: {score}")

        if not os.path.exists('./result'):
            os.mkdir('./result')

        result_path = os.path.join('./result', img_name)
        cv2.imwrite(result_path, img_p)
        print('Detection result save to {}\n\n'.format(result_path))

        # cv2.imshow("full post process result", img_p)
        # cv2.waitKeyEx(0)
    loop_elapsed = time.perf_counter() - loop_start
    if measured_count:
        print(f"measured loop elapsed: {loop_elapsed:.9f} s")
