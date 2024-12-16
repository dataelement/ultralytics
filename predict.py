import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.common import mask_to_bboxes

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    args = parse_args()

    model_path = args.model_path
    data_base_dir = args.data_dir
    save_dir = args.save_dir or Path(model_path).parent.parent / 'predictions'
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    val_txt = data_base_dir / 'val.txt'

    with open(val_txt, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]

    dataset = [str(data_base_dir / img_path) for img_path in img_paths]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for batch in tqdm(dataloader):
        if args.task == 'seg':
            retina_masks = True
        else:
            retina_masks = False
        results = model.predict(
            source=batch,
            save=False,
            device=args.device,
            max_det=args.max_det,
            conf=args.conf,
            retina_masks=retina_masks,
            task=args.task,
            imgsz=args.imgsz,
            verbose=False,
        )
        for result in results:
            img_stem = Path(result.path).stem
            if args.task == 'obb':
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()
                scores = result.obb.conf.cpu().numpy()

            elif args.task == 'detect':
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

            elif args.task == 'seg':
                if not result.masks:
                    continue
                masks = result.masks.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                boxes, scores, classes = mask_to_bboxes(masks, scores, result.orig_shape, classes)

            predictions = []
            for box, cls_id, conf in zip(boxes, classes, scores):
                cls_id = torch.tensor(cls_id)
                conf = torch.tensor(conf)
                h, w = result.orig_shape
                flatten_box = box.flatten()
                if len(flatten_box) == 4:
                    flatten_box[[0, 2]] = flatten_box[[0, 2]].clip(0, w - 1)
                    flatten_box[[1, 3]] = flatten_box[[1, 3]].clip(0, h - 1)
                    x1, y1, x2, y2 = flatten_box
                    pred = map(str, [x1, y1, x2, y1, x2, y2, x1, y2, cls_id.item(), conf.item()])
                elif len(flatten_box) == 8:
                    flatten_box[[0, 2, 4, 6]] = flatten_box[[0, 2, 4, 6]].clip(0, w - 1)
                    flatten_box[[1, 3, 5, 7]] = flatten_box[[1, 3, 5, 7]].clip(0, h - 1)
                    pred = map(str, [*flatten_box, int(cls_id.item()), conf.item()])
                predictions.append(pred)

            with open(save_dir / f'{img_stem}.txt', 'w') as f:
                f.write('\n'.join(','.join(pred) for pred in predictions))
