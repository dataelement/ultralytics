from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from ultralytics.utils.plotting import Annotator, colors

data_base_dir = Path('/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet')
output_dir = Path('workspace/visualization/') / data_base_dir.stem
output_dir.mkdir(exist_ok=True, parents=True)
label_dir = data_base_dir / 'labels'
image_dir = data_base_dir / 'images'
dataset_name = 'doclaynet'


cfg_dir = Path(__file__).parent.parent / 'ultralytics/cfg/datasets'
with open(cfg_dir / f'{dataset_name}.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    id2label = cfg['names']


def yolo_to_xyxy(box, width, height):
    """Convert YOLO format (n_x1, n_y1, n_x2, n_y2, n_x3, n_y3, n_x4, n_y4) to (x1, y1, x3, y3)"""
    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, box)
    return [x1 * width, y1 * height, x3 * width, y3 * height]


# 处理每个标签文件
for label_path in tqdm(list(label_dir.glob('*.txt'))[:400]):
    # 获取对应的图片路径
    image_path = image_dir / f"{label_path.stem}.jpg"
    if not image_path.exists():
        image_path = image_dir / f"{label_path.stem}.png"

    if not image_path.exists():
        print(f"Cannot find image for {label_path}")
        continue

    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot read image: {image_path}")
        continue

    height, width = image.shape[:2]

    # 创建标注器
    annotator = Annotator(image)

    # 读取标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    original_image = image.copy()
    
    # 处理每个边界框
    for line in lines:
        cls_id, *box = line.strip().split()
        cls_id = int(cls_id)

        # 转换坐标格式
        xyxy = yolo_to_xyxy(box, width, height)

        # 绘制边界框和标签
        label = id2label[cls_id]
        annotator.box_label(xyxy, label, color=colors(cls_id, True))

    # 水平拼接原图和标注后的图片
    concatenated_image = np.hstack((original_image, image))
    
    # 保存拼接后的结果
    output_path = output_dir / f"{label_path.stem}_vis.jpg"
    cv2.imwrite(str(output_path), concatenated_image)

print("Visualization completed!")
