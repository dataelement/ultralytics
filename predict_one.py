import argparse
import math
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ImageFile
from tqdm import tqdm

from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_boxes(
    image: np.ndarray,
    boxes,
    arrowedLine=False,
    scores=None,
    drop_score=0.5,
) -> np.ndarray:
    """
    Args:
        image: np.ndarray
        boxes: ndarray[n, 4, 2]
    """
    if scores is None:
        scores = [1] * len(boxes)
    for box, score in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, COLORS[2], 3)
        if arrowedLine:
            image = cv2.arrowedLine(
                image,
                (int(box[0][0][0]), int(box[0][0][1])),
                (int(box[1][0][0]), int(box[1][0][1])),
                color=(0, 255, 0),
                thickness=3,
                line_type=cv2.LINE_4,
                shift=0,
                tipLength=0.1,
            )
    return image


def start_point_boxes(boxes, boxes_cos, boxes_sin):
    """
    确定boxes中每个框的起始点（左上点）并重新排序顶点

    参数:
    boxes: 包含多个矩形框坐标的数组，每个框由8个值组成(4个点的x,y坐标)
    boxes_cos: 每个框的预测方向的余弦值
    boxes_sin: 每个框的预测方向的正弦值

    返回:
    boxes: 重新排序后的矩形框坐标数组
    """
    for box_index in range(len(boxes)):
        # 步骤1: 归一化预测的方向向量
        cos_value = boxes_cos[box_index]
        sin_value = boxes_sin[box_index]
        cos_value_norm = 2 * cos_value - 1
        sin_value_norm = 2 * sin_value - 1
        # 标准化向量长度为1
        cos_value = cos_value_norm / math.sqrt(math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))
        sin_value = sin_value_norm / math.sqrt(math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))

        # 步骤2: 计算预测方向的角度(0-360度)
        cos_angle = math.acos(cos_value) * 180 / np.pi
        sin_angle = math.asin(sin_value) * 180 / np.pi
        # 根据余弦和正弦值确定实际角度所在的象限
        if cos_angle <= 90 and sin_angle <= 0:
            angle = 360 + sin_angle
        elif cos_angle <= 90 and sin_angle > 0:
            angle = sin_angle
        elif cos_angle > 90 and sin_angle > 0:
            angle = cos_angle
        elif cos_angle > 90 and sin_angle <= 0:
            angle = 360 - cos_angle

        # 步骤3: 计算实际框的方向角度
        box = boxes[box_index]
        box = box[:8].reshape((4, 2))  # 重塑为4个点的坐标
        box_angle_vector = box[1] - box[0]  # 计算第一条边的方向向量
        # 计算实际框的方向余弦和正弦值
        box_cos_value = box_angle_vector[0] / np.linalg.norm(box_angle_vector)
        box_sin_value = box_angle_vector[1] / np.linalg.norm(box_angle_vector)

        # 计算实际框的角度
        box_cos_angle = math.acos(box_cos_value) * 180 / np.pi
        box_sin_angle = math.asin(box_sin_value) * 180 / np.pi
        # 确定实际角度所在的象限
        if box_cos_angle <= 90 and box_sin_angle <= 0:
            box_angle = 360 + box_sin_angle
        elif box_cos_angle <= 90 and box_sin_angle > 0:
            box_angle = box_sin_angle
        elif box_cos_angle > 90 and box_sin_angle > 0:
            box_angle = box_cos_angle
        elif box_cos_angle > 90 and box_sin_angle <= 0:
            box_angle = 360 - box_cos_angle

        # 计算框的四个可能的角度（每隔90度）
        box_angle = np.array([box_angle, (box_angle + 90) % 360, (box_angle + 180) % 360, (box_angle + 270) % 360])

        # 步骤4: 找出最接近预测角度的点作为起始点
        delta_angle = np.append(np.abs(box_angle - angle), 360 - np.abs(box_angle - angle))
        start_point_index = np.argmin(delta_angle) % 4

        # 步骤5: 根据起始点重新排序所有点
        box = box[
            [start_point_index, (start_point_index + 1) % 4, (start_point_index + 2) % 4, (start_point_index + 3) % 4]
        ]
        boxes[box_index] = box.reshape((-1))  # 转回一维数组
    return boxes


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1]), bounding_box[-1]


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO prediction with command line interface')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model weights')
    parser.add_argument('--data-dir', type=Path, required=True, help='Base directory containing the dataset')
    parser.add_argument(
        '--save-dir', type=Path, help='Directory to save predictions (default: model_path/../predictions_yolo11x)'
    )
    parser.add_argument('--batch-size', type=int, default=24, help='Batch size for prediction (default: 24)')
    parser.add_argument('--device', type=str, default='0', help='Device to run prediction on (default: 0)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum number of detections (default: 1000)')
    parser.add_argument('--task', type=str, default='detect', help='Task to run (default: detect)')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size (default: 1280)')
    return parser.parse_args()


if __name__ == "__main__":
    model_path = 'table_det_yolo11s_test/yolo11s-obb_table_det_epoch50_imgsz1024_bs2564/weights/best.pt'
    image_path = './roate.jpg'
    image_size = 1024
    task = 'obb'

    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        save=True,
        device='0',
        max_det=1000,
        conf=0.4,
        task=task,
        show_conf=True,
        show_labels=True,
        imgsz=image_size,
        verbose=True,
    )

    #
    image = cv2.imread(image_path)
    xyxyxyxy = results[0].obb.xyxyxyxy.cpu().numpy()
    r = results[0].obb.xywhr[0][-1].cpu().numpy()
    # r = np.pi * 11 / 6
    pred_degree = np.rad2deg(r)
    print('r: ', r)
    print('degree: ', np.rad2deg(r))

    _bbox, _, degree = get_mini_boxes(xyxyxyxy)
    bbox = start_point_boxes([np.array(_bbox)], [np.cos(r)], [np.sin(r)])
    # cos, sin = np.cos(r), np.sin(r)
    # # box = start_point_boxes([bbox], [cos], [sin])
    img = draw_boxes(image, bbox, arrowedLine=True)
    cv2.imwrite('output.jpg', img)
    print(f'save {image_path} to output.jpg')

    # image = cv2.imread(image_path)
