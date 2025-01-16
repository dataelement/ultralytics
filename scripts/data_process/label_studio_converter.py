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
import yaml
from nanoid import generate
from ordering import ReadingOrderClient
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl, field_validator
from tqdm import tqdm


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

    predictions: List[AnnotationModel] = Field(..., description="标注列表")
    data: DataModel = Field(..., description="基础数据")


def create_annotation(
    image_url: str,
    bboxes: List[List[float]],
    original_width: int,
    original_height: int,
    labels: List[str],
    box_ids: List[str],
    image_idx: int,
    image_type: str,
    box_orders: List[int],
    dataset_name: str,
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
        if dataset_name.lower() == 'doclaynet':
            if label == 'caption':
                if 'table' in labels:
                    label = 'table_caption'
                elif 'figure' in labels:
                    label = 'figure_caption'
        value_model = ValueModel(x=x, y=y, width=w, height=h, rotation=r, rectanglelabels=[str(box_order), label])
        result_model = ResultModel(
            original_width=original_width, original_height=original_height, value=value_model, id=box_id
        )
        results.append(result_model)

    # if box_orders:
    #     sorted_box_order_pairs = [
    #         (box_id, order) for box_id, order in sorted(list(zip(box_ids, box_orders)), key=lambda x: x[1])
    #     ]
    #     for i in range(len(sorted_box_order_pairs) - 1):
    #         if int(sorted_box_order_pairs[i][1]) == -1 or int(sorted_box_order_pairs[i + 1][1]) == -1:
    #             continue
    #         relation_model = RelationModel(from_id=sorted_box_order_pairs[i][0], to_id=sorted_box_order_pairs[i + 1][0])
    #         results.append(relation_model)

    return Template(
        predictions=[AnnotationModel(result=results)],
        data=DataModel(image=image_url, Index=image_idx, Tag=image_type),
    )


@dataclass
class ConvertLayoutDataToLabelStudio:
    data_dir: str
    coco_json_file: str
    output_file: str
    url_prefix: str
    category_mapping_file: Optional[str] = None

    def __post_init__(self):
        # 验证 images 文件夹是否存在
        if not Path(self.data_dir).exists():
            raise ValueError(f"文件夹不存在: {self.data_dir}")
        with open(self.coco_json_file, 'r') as fin:
            self.coco_data = json.load(fin)

        if self.category_mapping_file:
            with open(self.category_mapping_file, 'r') as fin:
                self.category_mapping = yaml.safe_load(fin)
        else:
            self.category_mapping = {}

        self.dataset_name = Path(self.data_dir).name
        self.image_id2annos = self._get_image_id2annos()
        self.image_id2url = self._get_image_id2url()
        self.id2category = self._get_id2category()
        self.image_id2image_info = self._get_image_id2image_info()

        self.reading_order_client = ReadingOrderClient()

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
        if self.category_mapping:
            for category in self.coco_data['categories']:
                if category['name'] in self.category_mapping:
                    id2category[category['id']] = self.category_mapping[category['name']]
                else:
                    print(f"{category['name']} not in category mapping")
        else:
            for category in self.coco_data['categories']:
                id2category[category['id']] = category['name']

        return id2category

    def _gen_label_html(self):
        prefix = """
<View>
<Image name="image" value="$image"/>
<RectangleLabels name="label" toName="image" choice="multiple">
"""
        suffix = """
</RectangleLabels>
</View>
"""
        html_template = prefix
        if self.category_mapping:
            for category in list(set(self.category_mapping.values())) + [*range(55)]:
                html_template += f'    <Label value="{category}"/>\n'
        else:
            for category in self.coco_data['categories'] + [*range(55)]:
                if isinstance(category, int):
                    html_template += f'    <Label value="{category}"/>\n'
                else:
                    html_template += f'    <Label value="{category["name"]}"/>\n'
        html_template += suffix
        return html_template

    def convert(self):
        content = []
        for image_id, annos in tqdm(self.image_id2annos.items()):
            image_url = self.image_id2url[image_id]
            image_info = self.image_id2image_info[image_id]
            image_type = image_info['doc_category']
            if image_type in ['newspaper', 'handwrite_note', 'photo_doc', 'other']:
                continue

            abandon_bboxes = []
            abandon_labels = []
            abandon_box_ids = []
            bboxes = []
            labels = []
            box_ids = []
            for anno in annos:
                if anno['category_id'] not in self.id2category:
                    continue
                label = self.id2category[anno['category_id']]
                if label.lower() in ['page_footer', 'page_header', 'page_number']:
                    abandon_bboxes.append(xywh2xyxyxyxy(anno['bbox']).tolist())
                    abandon_labels.append(self.id2category[anno['category_id']])
                    abandon_box_ids.append(generate_id())
                else:
                    bboxes.append(xywh2xyxyxyxy(anno['bbox']).tolist())
                    labels.append(self.id2category[anno['category_id']])
                    box_ids.append(generate_id())
            image_path = Path(self.data_dir, 'images') / Path(image_url).name

            try:
                box_orders = self.reading_order_client.predict(image_path, bboxes, labels)
            except Exception as e:
                print(e)
                print(image_path)
                continue

            bboxes += abandon_bboxes
            labels += abandon_labels
            box_ids += abandon_box_ids
            box_orders += [-1] * len(abandon_bboxes)

            template = create_annotation(
                image_url=image_url,
                bboxes=bboxes,
                labels=labels,
                box_ids=box_ids,
                image_idx=image_id,
                original_width=image_info['width'],
                original_height=image_info['height'],
                image_type=image_type,
                box_orders=box_orders,
                dataset_name=self.dataset_name,
            )
            content.append(template.model_dump())

        with open(self.output_file, 'w') as fout:
            json.dump(content, fout, indent=4)
        label_html = self._gen_label_html()
        with open(self.output_file.replace('.json', '.html'), 'w') as fout:
            fout.write(label_html)

        # print(self._statistic())


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='input data path', type=str, default=None)
    parser.add_argument('--coco_json_file', help='coco json file path', type=str, default=None)
    parser.add_argument('--output_file', help='output file path', type=str, default=None)
    parser.add_argument('--url_prefix', help='url prefix path', type=str, default='http://192.168.106.8/datasets')
    parser.add_argument('--category_mapping_file', help='category mapping file path', type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    convert = ConvertLayoutDataToLabelStudio(
        args.data_dir,
        args.coco_json_file,
        args.output_file,
        args.url_prefix,
        args.category_mapping_file,
    )
    convert.convert()


if __name__ == '__main__':
    main()
