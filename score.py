import argparse
import os
import pdb
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import Polygon as plg
from tqdm import tqdm

# from shapely.geometry import Polygon as plg


class eval_IOU(object):
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh

    def __call__(self, gt_boxes_list, boxes_list):
        detMatched_list = []
        numDetCare_list = []
        numGtCare_list = []
        for i in range(len(gt_boxes_list)):
            gt_boxes = gt_boxes_list[i]
            boxes = boxes_list[i]
            detMatched, numDetCare, numGtCare = self.eval(gt_boxes, boxes)
            detMatched_list.append(detMatched)
            numDetCare_list.append(numDetCare)
            numGtCare_list.append(numGtCare)
        matchedSum = np.sum(np.array(detMatched_list))
        numGlobalCareDet = np.sum(np.array(numDetCare_list))
        numGlobalCareGt = np.sum(np.array(numGtCare_list))
        methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )
        return methodPrecision, methodRecall, methodHmean

    def eval(self, gt_boxes, boxes):
        detMatched = 0
        numDetCare = 0
        numGtCare = 0
        if gt_boxes is None:
            return 0, 0, 0

        gtPols = []
        detPols = []
        detDontCarePolsNum = []
        iouMat = np.empty([1, 1])
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            gtPols.append(self.polygon_from_box(gt_box))

        if boxes is None:
            return 0, 0, len(gtPols)

        for box in boxes:
            detPol = self.polygon_from_box(box)
            detPols.append(detPol)

        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            pairs = []
            detMatchedNums = []
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = self.get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_thresh:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        numGtCare = len(gtPols)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        return detMatched, numDetCare, numGtCare

    def get_intersection(self, pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def get_union(self, pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - self.get_intersection(pD, pG)

    def get_intersection_over_union(self, pD, pG):
        try:
            return self.get_intersection(pD, pG) / self.get_union(pD, pG)
        except Exception:
            return 0

    def polygon_from_box(self, box):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(box[0][0])
        resBoxes[0, 4] = int(box[0][1])
        resBoxes[0, 1] = int(box[1][0])
        resBoxes[0, 5] = int(box[1][1])
        resBoxes[0, 2] = int(box[2][0])
        resBoxes[0, 6] = int(box[2][1])
        resBoxes[0, 3] = int(box[3][0])
        resBoxes[0, 7] = int(box[3][1])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def main(gt_dir, pred_dir, iou=0.7, multi_class=False):
    det_eval = eval_IOU(iou_thresh=iou)

    if not multi_class:
        names = load_files(gt_dir)
        boxes_list = []
        gt_boxes_list = []
        for name in names:
            boxes = []
            if os.path.exists(os.path.join(pred_dir, name)):
                for line in open(os.path.join(pred_dir, name)):
                    line = line.strip()
                    lines = line.split(',')
                    lines = list(map(float, lines))
                    box = np.array(lines).reshape([4, 2])
                    boxes.append(np.int0(np.round(box)))
                    # boxes.append(np.int0(box))

            boxes = np.array(boxes, dtype=np.int32)
            boxes_list.append(boxes)

            gt_boxes = []
            for line in open(os.path.join(gt_dir, name)):
                line = line.strip()
                lines = line.split(',')[:8]
                lines = list(map(float, lines))
                box = np.array(lines).reshape([4, 2])
                gt_boxes.append(np.int0(np.round(box)))
                # gt_boxes.append(np.int0(box))

            gt_boxes = np.array(gt_boxes, dtype=np.int32)
            gt_boxes_list.append(gt_boxes)

        precision, recall, hmean = det_eval(gt_boxes_list, boxes_list)
        return precision, recall, hmean

    elif multi_class:
        names = load_files(gt_dir)
        pred_boxes_dict = defaultdict(list)
        gt_boxes_dict = defaultdict(list)
        for name in tqdm(names):
            if any(not Path(file_dir, name).exists() for file_dir in [pred_dir, gt_dir]):
                continue

            pred_boxes = {i: [] for i in structure_class_names}
            if os.path.exists(os.path.join(pred_dir, name)):
                for line in open(os.path.join(pred_dir, name)):
                    line = line.strip().split(',')
                    lines = line[:8]
                    # label_idx = int(line[-1])
                    label_idx = int(float(line[-2]))
                    lines = list(map(float, lines))
                    box = np.array(lines).reshape([4, 2])
                    pred_boxes[idx2label[label_idx]].append(np.int0(box))

                for label, pred in pred_boxes.items():
                    pred_boxes_dict[label].append(pred)

            gt_boxes = {i: [] for i in structure_class_names}
            for line in open(os.path.join(gt_dir, name)):
                line = line.strip().split(',')
                if len(line) == 8:
                    line = line + ["text"]
                lines = line[:8]
                label_name = line[-1]
                if label_name not in structure_class_names:
                    continue
                lines = list(map(float, lines))
                box = np.array(lines).reshape([4, 2])
                # gt_boxes[label_name].append(np.int0(np.round(box)))
                gt_boxes[label_name].append(np.int0(box))

            for label, gt in gt_boxes.items():
                gt_boxes_dict[label].append(gt)

        result = dict()
        for label_class in structure_class_names:
            gt_boxes_list, boxes_list = (
                gt_boxes_dict[label_class],
                pred_boxes_dict[label_class],
            )
            precision, recall, hmean = det_eval(gt_boxes_list, boxes_list)
            result[label_class] = {
                'precision': precision,
                'recall': recall,
                'hmean': hmean,
            }

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default=None, type=str, required=True, help='gt txts.')
    parser.add_argument('--pred_dir', default=None, type=str, required=True, help='predict txts.')
    parser.add_argument('--multi_class', action='store_true', help='multi_class or not')
    parser.add_argument('--dataset_name', default='table_cell_det', type=str, help='table_cell_det or layout')
    parser.add_argument('--iou', default=0.7, type=float)
    args = parser.parse_args()

    if args.dataset_name == 'table_cell_det':
        structure_class_names = ['cell']
        label2idx = {label: ids for ids, label in enumerate(structure_class_names)}
    elif args.dataset_name == 'table_row_col':
        structure_class_names = [
            'table column',
            'table row',
            'table spanning cell',
        ]
        label2idx = {label: ids for ids, label in enumerate(structure_class_names)}
    elif args.dataset_name == 'table_det':
        structure_class_names = [
            'wired_table',
            'lineless_table',
        ]
        label2idx = {label: ids + 1 for ids, label in enumerate(structure_class_names)}
    elif args.dataset_name == 'dataelem_layout':
        structure_class_names = ['印章', '图片', '标题', '段落', '表格', '页眉', '页码', '页脚']
        label2idx = {label: ids + 1 for ids, label in enumerate(structure_class_names)}
    elif args.dataset_name == 'doclaynet':
        structure_class_names = [
            'Caption',
            'Footnote',
            'Formula',
            'List-item',
            'Page-footer',
            'Page-header',
            'Picture',
            'Section-header',
            'Table',
            'Text',
            'Title',
        ]
        label2idx = {label: ids for ids, label in enumerate(structure_class_names)}
    elif args.dataset_name == 'text_det':
        structure_class_names = ['text']
        label2idx = {label: ids for ids, label in enumerate(structure_class_names)}
    else:
        raise ValueError(f'task {args.task} not supported')

    idx2label = {v: k for k, v in label2idx.items()}

    if not args.multi_class:
        precision, recall, hmean = main(args.gt_dir, args.pred_dir, args.iou)
        print('precision:{}, recall:{}, hmean:{}'.format(precision, recall, hmean))

    elif args.multi_class:
        multi_class_metrics = main(args.gt_dir, args.pred_dir, args.iou, args.multi_class)
        metrics_df = pd.DataFrame.from_dict(multi_class_metrics, orient='index')
        print(metrics_df.to_string())
        print(
            f'avg_p: {metrics_df["precision"].mean():.4f}, avg_r: {metrics_df["recall"].mean():.4f}, avg_h: {metrics_df["hmean"].mean():.4f}'
        )
