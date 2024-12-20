import argparse
import json
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from nanoid import generate
from pydantic import BaseModel, Field, HttpUrl, field_validator


def generate_id():
    """生成10位随机ID"""
    return generate(size=10)


def euclidean_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2)


def xywh2xyxyxyxy(bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y, x + w, y + h, x, y + h]).reshape(4, 2)


def bbox2ls(bbox, original_width, original_height):
    bbox = np.array(bbox)
    tl, tr, br, bl = bbox

    x, y = tl

    x_ = x / original_width * 100
    y_ = y / original_height * 100

    # width and height
    w = euclidean_distance(tl, tr) / original_width * 100
    h = euclidean_distance(tl, bl) / original_height * 100

    # get top line vector
    dy = tr[1] - tl[1]
    dx = tr[0] - tl[0]
    # get randians
    angle = np.arctan2(dy, dx)
    # convert to degrees
    r = angle * 180 / np.pi

    # fix value range
    if r < 0:
        r += 360
    # if (r >= 360): r -= 360 # don't really need this since -pi <= arctan2(x, y) <= pi

    x_ = x_.clip(0, 100)
    y_ = y_.clip(0, 100)
    w = w.clip(0, 100)
    h = h.clip(0, 100)
    r = r.clip(0, 360)
    return x_, y_, w, h, r


class DataModel(BaseModel):
    image: str
    Index: int
    Tag: str


class ValueModel(BaseModel):
    """矩形标注值模型"""

    x: float = Field(..., ge=0, le=100, description="矩形左上角X坐标(百分比)")
    y: float = Field(..., ge=0, le=100, description="矩形左上角Y坐标(百分比)")
    width: float = Field(..., gt=0, le=100, description="矩形宽度(百分比)")
    height: float = Field(..., gt=0, le=100, description="矩形高度(百分比)")
    rotation: float = Field(..., ge=0, lt=360, description="旋转角度")
    rectanglelabels: List[str] = Field(..., min_items=1, description="矩形标签列表")

    class Config:
        json_schema_extra = {
            "example": {
                "x": 37.28506787330317,
                "y": 8.573256557901471,
                "width": 24.07239819004525,
                "height": 3.326935380678183,
                "rotation": 0,
                "rectanglelabels": ["Airplane"],
            }
        }


class RelationModel(BaseModel):
    """关系模型"""

    from_id: str = Field(..., description="起始ID")
    to_id: str = Field(..., description="结束ID")
    type: str = "relation"
    direction: str = "right"


class ResultModel(BaseModel):
    """标注结果模型"""

    original_width: int = Field(..., gt=0, description="原始图像宽度")
    original_height: int = Field(..., gt=0, description="原始图像高度")
    image_rotation: float = Field(default=0, ge=0, lt=360, description="图像旋转角度")
    value: ValueModel = Field(..., description="标注值")
    id: str = Field(..., description="标注ID")
    type: str = Field(default="rectanglelabels", description="标注类型")
    origin: str = Field(default="manual", description="标注来源")
    from_name: str = Field(default="label", description="标签名称")
    to_name: str = Field(default="image", description="目标名称")

    class Config:
        json_schema_extra = {
            "example": {
                "original_width": 1920,
                "original_height": 1080,
                "image_rotation": 0,
                "value": {...},  # ValueModel example
                "id": "ZMl4GhsMXL",
                "type": "rectanglelabels",
                "origin": "manual",
                "from_name": "label",
                "to_name": "image",
            }
        }


class AnnotationModel(BaseModel):
    """标注模型"""

    result: List[ResultModel | RelationModel] = Field(..., min_items=1, description="标注结果列表")


class Template(BaseModel):
    """完整的标注模板"""

    annotations: List[AnnotationModel] = Field(..., description="标注列表")
    data: DataModel = Field(..., description="基础数据")


def create_annotation(
    image_url: str,
    bboxes: List[np.ndarray],
    original_width: int,
    original_height: int,
    labels: List[str],
    box_ids: List[str],
    image_idx: int,
    box_orders: List[int],
) -> Template:
    """
    创建标注模板

    Args:
        image_url: 图像URL
        bbox: 边界框坐标
        original_width: 原始图像宽度
        original_height: 原始图像高度
        label: 标签

    Returns:
        Template: 标注模板
    """
    results = []
    for bbox, label, box_id, box_order in zip(bboxes, labels, box_ids, box_orders):
        x, y, w, h, r = bbox2ls(bbox, original_width, original_height)
        value_model = ValueModel(x=x, y=y, width=w, height=h, rotation=r, rectanglelabels=[label, str(box_order)])
        result_model = ResultModel(
            original_width=original_width, original_height=original_height, value=value_model, id=box_id
        )
        results.append(result_model)

    return Template(
        annotations=[AnnotationModel(result=results)], data=DataModel(image=image_url, Index=image_idx, Tag="Images")
    )


@dataclass
class ConvertLayoutDataToLabelStudio:
    data_dir: str
    coco_json_file: str
    output_file: str
    url_prefix: str
    category_mapping: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        # 验证 images 文件夹是否存在
        if not Path(self.data_dir).exists():
            raise ValueError(f"images 文件夹不存在: {self.data_dir}")
        with open(self.coco_json_file, 'r') as fin:
            self.coco_data = json.load(fin)

        self.image_id2annos = self._get_image_id2annos()
        self.image_id2url = self._get_image_id2url()
        self.id2category = self._get_id2category()
        self.image_id2image_info = self._get_image_id2image_info()

    def _get_image_id2image_info(self):
        image_id2image_info = {}
        for image_data in self.coco_data['images']:
            image_id2image_info[image_data['id']] = image_data
        return image_id2image_info

    def _get_image_url(self, image_name: str):
        return "{}/{}".format(self.url_prefix, image_name)

    def _get_image_id2url(self):
        image_id2url = {}
        for image_data in self.coco_data['images']:
            image_id2url[image_data['id']] = self._get_image_url(Path(image_data['file_name']).name)
        return image_id2url

    def _get_image_id2annos(self):
        image_id2annos = defaultdict(list)
        for anno in self.coco_data['annotations']:
            image_id2annos[anno['image_id']].append(anno)
        return image_id2annos

    def _get_id2category(self):
        id2category = {}
        for category in self.coco_data['categories']:
            if category['name'] in self.category_mapping:
                id2category[category['id']] = self.category_mapping[category['name']]
            else:
                id2category[category['id']] = category['name']

        return id2category

    def _gen_label_html(self):
        prefix = """
<RectangleLabels name="label" toName="image" choice="multiple">
"""
        suffix = """
</RectangleLabels>
"""
        html_template = prefix
        for category in self.coco_data['categories'] + [*range(40)]:
            if isinstance(category, int):
                html_template += f'    <Label value="{category}"/>\n'
            else:
                html_template += f'    <Label value="{category["name"]}"/>\n'
        html_template += suffix
        return html_template

    def convert(self):
        content = []
        for image_id, annos in self.image_id2annos.items():
            image_url = self.image_id2url[image_id]
            image_info = self.image_id2image_info[image_id]
            bboxes = [xywh2xyxyxyxy(anno['bbox']) for anno in annos]
            labels = [self.id2category[anno['category_id']] for anno in annos]
            box_ids = [generate_id() for _ in annos]
            # relations = [anno['relation'] for anno in annos]
            box_orders = [i for i in range(len(annos))]
            template = create_annotation(
                image_url=image_url,
                bboxes=bboxes,
                labels=labels,
                box_ids=box_ids,
                image_idx=image_id,
                original_width=image_info['width'],
                original_height=image_info['height'],
                box_orders=box_orders,
            )
            content.append(template.model_dump())
        with open(self.output_file, 'w') as fout:
            json.dump(content, fout, indent=4)
        label_html = self._gen_label_html()
        with open(self.output_file.replace('.json', '.html'), 'w') as fout:
            fout.write(label_html)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='input data path', type=str, default=None)
    parser.add_argument('--coco_json_file', help='coco json file path', type=str, default=None)
    parser.add_argument('--output_file', help='output file path', type=str, default=None)
    parser.add_argument('--url_prefix', help='url prefix path', type=str, default='http://192.168.106.8/datasets')

    return parser.parse_args()


def main():
    args = get_args()
    convert = ConvertLayoutDataToLabelStudio(args.data_dir, args.coco_json_file, args.output_file, args.url_prefix)
    convert.convert()


if __name__ == '__main__':
    main()
