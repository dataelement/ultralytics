import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":

    model_path = '/workspace/youjiachen/workspace/ultralytics/layout_yolo11x/yolo11x_dataelem_layout_epoch500_imgsz1280_bs24/weights/best.pt'
    data_base_dir = Path('/workspace/datasets/layout/dataelem_layout/yolo_format_merge_all')
    save_dir = Path(model_path).parent.parent / 'predictions_yolo11x'
    save_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 24

    model = YOLO(model_path)
    val_txt = data_base_dir / 'val.txt'

    with open(val_txt, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]

    dataset = [str(data_base_dir / img_path) for img_path in img_paths]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader):
        results = model.predict(source=batch, save=False, conf=0.3, device='0')
        for result in results:
            img_stem = Path(result.path).stem
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            scores = result.boxes.conf

            predictions = []
            for box, cls_id, conf in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.tolist()
                pred = map(str, [x1, y1, x2, y1, x2, y2, x1, y2, cls_id.item(), conf.item()])
                predictions.append(pred)

            with open(save_dir / f'{img_stem}.txt', 'w') as f:
                f.write('\n'.join(','.join(pred) for pred in predictions))
