import argparse
import json
import math
import os
import re
import shutil
import urllib
import wave
from collections import Counter, defaultdict
from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


def is_numeric(string):
    pattern = re.compile(r'^[0-9]+$')
    return bool(pattern.match(string))


def get_image_size_and_channels(image_path):
    i = Image.open(image_path)
    w, h = i.size
    c = len(i.getbands())
    return w, h, c


def get_audio_duration(audio_path):
    with wave.open(audio_path, mode='r') as f:
        return f.getnframes() / float(f.getframerate())


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_polygon_area(x, y):
    """https://en.wikipedia.org/wiki/Shoelace_formula"""

    assert len(x) == len(y)

    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    assert len(x) == len(y)

    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def get_annotator(item, default=None, int_id=False):
    """Get annotator id or email from annotation"""
    annotator = item['completed_by']
    if isinstance(annotator, dict):
        annotator = annotator.get('email', default)
        return annotator

    if isinstance(annotator, int) and int_id:
        return annotator

    return str(annotator)


def get_json_root_type(filename):
    char = 'x'
    with open(filename, "r", encoding='utf-8') as f:
        # Read the file character by character
        while char != '':
            char = f.read(1)

            # Skip any whitespace
            if char.isspace():
                continue

            # If the first non-whitespace character is '{', it's a dict
            if char == '{':
                return "dict"

            # If the first non-whitespace character is '[', it's an array
            if char == '[':
                return "list"

            # If neither, the JSON file is invalid
            return "invalid"

    # If the file is empty, return "empty"
    return "empty"


def prettify_result(v):
    """
    :param v: list of regions or results
    :return: label name as is if there is only 1 item in result `v`, else list of label names
    """
    out = []
    tag_type = None
    for i in v:
        j = deepcopy(i)
        tag_type = j.pop('type')
        if tag_type == 'Choices' and len(j['choices']) == 1:
            out.append(j['choices'][0])
        elif tag_type == 'TextArea' and len(j['text']) == 1:
            out.append(j['text'][0])
        else:
            out.append(j)
    return out[0] if tag_type in ('Choices', 'TextArea') and len(out) == 1 else out


def convert_annotation_to_yolo(label):
    """
    Convert LS annotation to Yolo format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - width (float): Width of the object.
            - height (float): Height of the object.
            - x (float): X-coordinate of the top-left corner of the object.
            - y (float): Y-coordinate of the top-left corner of the object.

    Returns:
        tuple or None: If the conversion is successful, returns a tuple (x, y, w, h) representing
        the coordinates and dimensions of the object in Yolo format, where (x, y) are the center
        coordinates of the object, and (w, h) are the width and height of the object respectively.
    """

    if not ("x" in label and "y" in label and 'width' in label and 'height' in label):
        return None

    w = label['width']
    h = label['height']

    x = (label['x'] + w / 2) / 100
    y = (label['y'] + h / 2) / 100
    w = w / 100
    h = h / 100

    return x, y, w, h


def convert_annotation_to_yolo_obb(label):
    """
    Convert LS annotation to Yolo OBB format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - original_width (int): Original width of the image.
            - original_height (int): Original height of the image.
            - x (float): X-coordinate of the top-left corner of the object in percentage of the original width.
            - y (float): Y-coordinate of the top-left corner of the object in percentage of the original height.
            - width (float): Width of the object in percentage of the original width.
            - height (float): Height of the object in percentage of the original height.
            - rotation (float, optional): Rotation angle of the object in degrees (default is 0).

    Returns:
        list of tuple or None: List of tuples containing the coordinates of the object in Yolo OBB format.
            Each tuple represents a corner of the bounding box in the order:
            (top-left, top-right, bottom-right, bottom-left).
    """

    if not (
        "original_width" in label
        and "original_height" in label
        and 'x' in label
        and 'y' in label
        and 'width' in label
        and 'height' in label
        and 'rotation' in label
    ):
        return None

    org_width, org_height = label['original_width'], label['original_height']
    x = label['x'] / 100 * org_width
    y = label['y'] / 100 * org_height
    w = label['width'] / 100 * org_width
    h = label['height'] / 100 * org_height

    rotation = math.radians(label.get("rotation", 0))
    cos, sin = math.cos(rotation), math.sin(rotation)

    coords = [
        (x, y),
        (x + w * cos, y + w * sin),
        (x + w * cos - h * sin, y + w * sin + h * cos),
        (x - h * sin, y + h * cos),
    ]

    # Normalize coordinates
    return [(coord[0] / org_width, coord[1] / org_height) for coord in coords]


def save_data(data, output_dir, split):
    img_dir = Path(output_dir, 'images', split)
    ensure_dir(img_dir)
    label_dir = Path(output_dir, 'labels', split)
    ensure_dir(label_dir)

    # 将图片保存到output_dir
    for image_path, annotations in data.items():
        shutil.copy(image_path, img_dir / Path(image_path).name)
        with open(label_dir / Path(image_path).with_suffix('.txt').name, 'w') as f:
            for annotation in annotations:
                f.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")


def create_yaml(output_dir, names2id):
    """
        Create a yaml file for YOLOv8.
        # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ../datasets/coco8 # dataset root dir
    train: images/train # train images (relative to 'path') 4 images
    val: images/val # val images (relative to 'path') 4 images
    test: # test images (optional)

    # Classes (80 COCO classes)
    names:
        0: person
        1: bicycle
        2: car
        # ...
        77: teddy bear
        78: hair drier
        79: toothbrush
    """
    with open(Path(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train # train images (relative to 'path') 4 images\n")
        f.write(f"val: images/val # val images (relative to 'path') 4 images\n")
        f.write(f"test: # test images (optional)\n")
        f.write(f"names:\n")
        for name, _id in names2id.items():
            f.write(f"    {_id}: {name}\n")


def main():
    default_train_json_path = '/workspace/datasets/layout/unsv2_layout_yolo_data/data/layout_train.json'
    default_val_json_path = '/workspace/datasets/layout/unsv2_layout_yolo_data/data/layout_val.json'
    default_image_dir = '/workspace/datasets/layout/unsv2_layout'
    default_output_dir = '/workspace/datasets/layout/unsv2_layout_yolo_data'
    # json路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_path', type=str, default=default_train_json_path)
    parser.add_argument('--val_json_path', type=str, default=default_val_json_path)
    parser.add_argument('--image_dir', type=str, default=default_image_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # 定义标签映射字典
    label_mapping = {
        'table_caption': 'caption',
        'figure': 'caption',
        'page_number': 'abandon',
        'page_footer': 'abandon',
        'page_header': 'abandon',
    }

    names2id = {'title': 0, 'caption': 1, 'table': 2, 'figure': 3, 'abandon': 4, 'plain_text': 5}

    for split in ['train', 'val']:
        json_path = args.train_json_path if split == 'train' else args.val_json_path
        # 读取json文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        convert_data = defaultdict(list)
        # 遍历data，将每个item转换为yolo格式
        for item in tqdm(data, desc=f'Converting {split} data'):
            image_path = Path(args.image_dir, Path(item['image']).parents[1].name, 'images', Path(item['image']).name)
            for annos in item['label']:
                x, y, w, h = convert_annotation_to_yolo(annos)
                labels = annos['rectanglelabels']

                for label in labels:
                    label = label_mapping.get(label, label)
                    if is_numeric(label) or label not in names2id:
                        continue
                    convert_data[image_path].append([names2id[label], x, y, w, h])

        save_data(convert_data, args.output_dir, split)
        print(f'{split} data saved to {args.output_dir}')

    create_yaml(args.output_dir, names2id)
    print(f'YAML file created at {args.output_dir}/data.yaml')


if __name__ == '__main__':
    main()
