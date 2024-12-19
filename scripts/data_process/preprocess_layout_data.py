import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from ultralytics.utils.plotting import plot_labels


class DataPath(Enum):
    """Enum class for dataset paths.

    Contains paths for different layout datasets including DocLayNet, M6Doc, CDLA and ElemLayout.
    Each dataset has train/val/test splits.
    """

    # DocLayNet dataset paths
    DocLayNet: Path = Path('/workspace/datasets/layout/DocLayNet')
    DocLayNet_train: Path = DocLayNet / 'COCO/train.json'
    DocLayNet_val: Path = DocLayNet / 'COCO/val.json'
    DocLayNet_test: Path = DocLayNet / 'COCO/test.json'
    DocLayNet_images: Path = DocLayNet / 'PNG'

    # M6Doc dataset paths
    M6Doc: Path = Path('/workspace/datasets/layout/M6Doc')
    M6Doc_train: Path = M6Doc / 'annotations/instances_train2017.json'
    M6Doc_val: Path = M6Doc / 'annotations/instances_val2017.json'
    M6Doc_test: Path = M6Doc / 'annotations/instances_test2017.json'
    M6Doc_train_images: Path = M6Doc / 'train2017'
    M6Doc_val_images: Path = M6Doc / 'val2017'
    M6Doc_test_images: Path = M6Doc / 'test2017'

    # CDLA dataset paths
    CDLA: Path = Path('/workspace/datasets/layout/CDLA_DATASET')
    CDLA_train: Path = CDLA / 'CDLA_DATASET_COCO_TRAIN/annotations.json'
    CDLA_val: Path = CDLA / 'CDLA_DATASET_COCO_VAL/annotations.json'
    CDLA_train_images: Path = CDLA / 'CDLA_DATASET_COCO_TRAIN/JPEGImages'
    CDLA_val_images: Path = CDLA / 'CDLA_DATASET_COCO_VAL/JPEGImages'

    # ElemLayout dataset paths
    ElemLayout: Path = Path('/workspace/datasets/layout/dataelem_layout')
    ElemLayout_caibao_train: Path = ElemLayout / 'coco_财报/train.json'
    ElemLayout_caibao_val: Path = ElemLayout / 'coco_财报/val.json'
    ElemLayout_caibao_images: Path = ElemLayout / 'coco_财报/images'

    ElemLayout_hetong_train: Path = ElemLayout / 'coco_合同/train.json'
    ElemLayout_hetong_val: Path = ElemLayout / 'coco_合同/val.json'
    ElemLayout_hetong_images: Path = ElemLayout / 'coco_合同/images'

    ElemLayout_lunwen_train: Path = ElemLayout / 'coco_论文/train.json'
    ElemLayout_lunwen_val: Path = ElemLayout / 'coco_论文/val.json'
    ElemLayout_lunwen_images: Path = ElemLayout / 'coco_论文/images'

    ElemLayout_yanbao_train: Path = ElemLayout / 'coco_研报/train.json'
    ElemLayout_yanbao_val: Path = ElemLayout / 'coco_研报/val.json'
    ElemLayout_yanbao_images: Path = ElemLayout / 'coco_研报/images'

    # D4LA dataset paths
    D4LA: Path = Path('/workspace/datasets/layout/DocLayout-YOLO/layout_data/D4LA')
    D4LA_train: Path = D4LA / 'train.json'
    D4LA_val: Path = D4LA / 'test.json'
    D4LA_images: Path = D4LA / 'images'


class CocoAnnotation(BaseModel):
    """COCO标注数据模型"""

    images: list[dict]
    annotations: list[dict]
    categories: list[dict]

    def update_image_index(self):
        id_map = {}
        for i, image in enumerate(self.images):
            cur_image_id = image['id']
            update_image_id = i
            id_map[cur_image_id] = update_image_id
        for anno in self.annotations:
            anno['image_id'] = id_map[anno['image_id']]
        for image in self.images:
            image['id'] = id_map[image['id']]
        return self


class LayoutDataset(BaseModel):
    train: CocoAnnotation
    val: CocoAnnotation
    test: Optional[CocoAnnotation] = None
    image_paths: list[Path]
    dataset_name: str
    file_name2image_path: dict[str, Path] = {}

    def model_post_init(self, *args, **kwargs):
        self.file_name2image_path = {image_path.name: image_path for image_path in self.image_paths}

    def get_images(self, split: str) -> list[Path]:
        """Get image paths for a given split."""
        images = []
        for image in tqdm(self.train.images if split == 'train' else self.val.images):
            file_name = Path(image['file_name']).name
            images.append(self.file_name2image_path[file_name])
        return images

    def export_dataset(self, output_dir: Path):
        # 将图片保存到 output_dir/images 目录下
        output_dir = output_dir / self.dataset_name
        image_dir = output_dir / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        for image in self.train.images + self.val.images:
            file_name = Path(image['file_name']).name
            shutil.copy(self.file_name2image_path[file_name], image_dir / file_name)

        # train cocoannotation BaseModel dump 成 json 保存到 output dir 下
        with open(output_dir / 'train.json', 'w') as f:
            json.dump(self.train.model_dump(), f, ensure_ascii=False, indent=4)

        # val cocoannotation dump 成 json 保存到 output dir 下
        with open(output_dir / 'val.json', 'w') as f:
            json.dump(self.val.model_dump(), f, ensure_ascii=False, indent=4)

        meta_data = {
            'dataset_name': self.dataset_name,
            'categories': self.train.categories,
            'doc_categories': dict(
                Counter([i['doc_category'] for i in self.train.images] + [i['doc_category'] for i in self.val.images])
            ),
            'nums': {
                'train': len(self.train.images),
                'val': len(self.val.images),
            },
        }
        with open(output_dir / 'meta.json', 'w') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
    
    def convert_to_label_studio_format(self):
        pass


@dataclass
class LayoutDatasets:
    def __post_init__(self):
        self.datasets = [
            self.get_doclaynet_dataset(),
            self.get_m6doc_dataset(),
            self.get_cdla_dataset(),
            self.get_elemlayout_dataset(),
            self.get_d4la_dataset(),
        ]

        # get all doc_category
        self.doc_categories_cnt = defaultdict(int)
        for dataset in self.datasets:
            for image in dataset.train.images:
                self.doc_categories_cnt[image['doc_category']] += 1
        logger.info(f'All doc_categories: {self.doc_categories_cnt}')

    def get_doclaynet_dataset(self) -> LayoutDataset:
        logger.info('Loading DocLayNet dataset...')
        files = {
            'train': DataPath.DocLayNet_train.value,
            'val': DataPath.DocLayNet_val.value,
            'test': DataPath.DocLayNet_test.value,
        }
        output_dir = Path('/workspace/datasets/layout/DocLayNet/_PNG')
        with open(files['train'], 'r') as f:
            train_data = CocoAnnotation.model_validate_json(f.read())

        with open(files['val'], 'r') as f:
            val_data = CocoAnnotation.model_validate_json(f.read())

        with open(files['test'], 'r') as f:
            test_data = CocoAnnotation.model_validate_json(f.read())

        return LayoutDataset(
            train=train_data,
            val=val_data,
            test=test_data,
            image_paths=list(DataPath.DocLayNet_images.value.glob('*')),
            dataset_name='DocLayNet',
        )

    def get_m6doc_dataset(self) -> LayoutDataset:
        logger.info('Loading M6Doc dataset...')
        files = {
            'train': DataPath.M6Doc_train.value,
            'val': DataPath.M6Doc_val.value,
            'test': DataPath.M6Doc_test.value,
        }
        with open(files['train'], 'r') as f:
            train_data = CocoAnnotation.model_validate_json(f.read())
            logger.debug(f'Train data: {len(train_data.images)}')
            for image in train_data.images:
                image['doc_category'] = 'unknown'
        with open(files['val'], 'r') as f:
            val_data = CocoAnnotation.model_validate_json(f.read())
            logger.debug(f'Val data: {len(val_data.images)}')
            for image in val_data.images:
                image['doc_category'] = 'unknown'
        with open(files['test'], 'r') as f:
            test_data = CocoAnnotation.model_validate_json(f.read())
            logger.debug(f'Test data: {len(test_data.images)}')
            for image in test_data.images:
                image['doc_category'] = 'unknown'
        all_images = (
            list(DataPath.M6Doc_train_images.value.glob('*'))
            + list(DataPath.M6Doc_val_images.value.glob('*'))
            + list(DataPath.M6Doc_test_images.value.glob('*'))
        )

        return LayoutDataset(
            train=train_data,
            val=val_data,
            test=test_data,
            image_paths=all_images,
            dataset_name='M6Doc',
        )

    def get_cdla_dataset(self) -> LayoutDataset:
        logger.info('Loading CDLA dataset...')
        all_images = list(DataPath.CDLA_train_images.value.glob('*.jpg')) + list(
            DataPath.CDLA_val_images.value.glob('*.jpg')
        )
        with open(DataPath.CDLA_train.value, 'r') as f:
            train_data = CocoAnnotation.model_validate_json(f.read())
            for image in train_data.images:
                image['doc_category'] = 'paper_zh'
        with open(DataPath.CDLA_val.value, 'r') as f:
            val_data = CocoAnnotation.model_validate_json(f.read())
            for image in val_data.images:
                image['doc_category'] = 'paper_zh'
        return LayoutDataset(
            train=train_data,
            val=val_data,
            image_paths=all_images,
            dataset_name='CDLA',
        )

    def get_elemlayout_dataset(self) -> LayoutDataset:
        logger.info('Loading ElemLayout dataset...')
        all_images = []
        train_data = defaultdict(list)
        val_data = defaultdict(list)
        train_id = 0
        val_id = 0
        for scene in ['caibao', 'hetong', 'lunwen', 'yanbao']:
            train_json = getattr(DataPath, f'ElemLayout_{scene}_train').value
            val_json = getattr(DataPath, f'ElemLayout_{scene}_val').value
            all_images.extend(list(getattr(DataPath, f'ElemLayout_{scene}_images').value.glob('*')))
            train_id_map = {}
            val_id_map = {}
            with open(train_json, 'r') as f:
                data = json.load(f)
            for image in data['images']:
                image['doc_category'] = scene

                pre_image_id = image['id']
                image['id'] = train_id

                train_id_map[pre_image_id] = train_id
                train_id += 1
            for ann in data['annotations']:
                ann['image_id'] = train_id_map[ann['image_id']]

            train_data['annotations'].extend(data['annotations'])
            train_data['images'].extend(data['images'])
            for category in data['categories']:
                if category not in train_data['categories']:
                    train_data['categories'].append(category)

            with open(val_json, 'r') as f:
                data = json.load(f)
            for image in data['images']:
                image['doc_category'] = scene

                pre_image_id = image['id']
                image['id'] = val_id
                val_id_map[pre_image_id] = val_id
                val_id += 1
            for ann in data['annotations']:
                ann['image_id'] = val_id_map[ann['image_id']]

            val_data['annotations'].extend(data['annotations'])
            val_data['images'].extend(data['images'])
            for category in data['categories']:
                if category not in val_data['categories']:
                    val_data['categories'].append(category)

        return LayoutDataset(
            train=train_data,
            val=val_data,
            image_paths=all_images,
            dataset_name='ElemLayout',
        )

    def get_d4la_dataset(self) -> LayoutDataset:
        logger.info('Loading D4LA dataset...')
        with open(DataPath.D4LA_train.value, 'r') as f:
            train_data = CocoAnnotation.model_validate_json(f.read())
            for image in train_data.images:
                image['doc_category'] = Path(image['file_name']).stem.split('_')[0]
        with open(DataPath.D4LA_val.value, 'r') as f:
            val_data = CocoAnnotation.model_validate_json(f.read())
            for image in val_data.images:
                image['doc_category'] = Path(image['file_name']).stem.split('_')[0]

        return LayoutDataset(
            train=train_data,
            val=val_data,
            image_paths=list(DataPath.D4LA_images.value.glob('*')),
            dataset_name='D4LA',
        )


@dataclass
class DataSampler:
    layout_dataset: LayoutDataset
    random_seed: int

    def __post_init__(self):
        random.seed(self.random_seed)
        self.id2label = self._get_dataset_id2label()
        self.label2id = self._get_dataset_label2id()
        self.dataset_scenes = self._get_dataset_scenes()

        self.train_img_id2annos, self.val_img_id2annos = self._get_dataset_img_id2annos()
        self.train_id2img, self.val_id2img = self._get_dataset_id2img_info()
        self.train_scene_to_images, self.val_scene_to_images = self._get_dataset_scene2images()
        self.train_scene_to_annos, self.val_scene_to_annos = self._get_dataset_scene2annos()

    def _get_dataset_scenes(self) -> Set[str]:
        return set(image['doc_category'] for image in self.layout_dataset.train.images) & set(
            image['doc_category'] for image in self.layout_dataset.val.images
        )

    def _get_dataset_id2img_info(self) -> Tuple[Dict[int, dict], Dict[int, dict]]:
        return {img['id']: img for img in self.layout_dataset.train.images}, {
            img['id']: img for img in self.layout_dataset.val.images
        }

    def _get_dataset_img_id2annos(self) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
        train_image_to_annos = defaultdict(list)
        val_image_to_annos = defaultdict(list)
        for anno in self.layout_dataset.train.annotations:
            train_image_to_annos[anno['image_id']].append(anno)
        for anno in self.layout_dataset.val.annotations:
            val_image_to_annos[anno['image_id']].append(anno)
        return train_image_to_annos, val_image_to_annos

    def _get_dataset_id2label(self) -> Dict[int, str]:
        return {cat['id']: cat['name'] for cat in self.layout_dataset.train.categories}

    def _get_dataset_label2id(self) -> Dict[str, int]:
        return {cat['name']: cat['id'] for cat in self.layout_dataset.train.categories}

    def _get_dataset_scene2images(self) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
        train_scene_to_images = defaultdict(list)
        val_scene_to_images = defaultdict(list)
        for image in self.layout_dataset.train.images:
            train_scene_to_images[image['doc_category']].append(image)
        for image in self.layout_dataset.val.images:
            val_scene_to_images[image['doc_category']].append(image)
        return train_scene_to_images, val_scene_to_images

    def _get_dataset_scene2annos(self) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
        train_scene_to_annos = defaultdict(list)
        val_scene_to_annos = defaultdict(list)
        for anno in self.layout_dataset.train.annotations:
            train_scene_to_annos[anno['image_id']].append(anno)
        for anno in self.layout_dataset.val.annotations:
            val_scene_to_annos[anno['image_id']].append(anno)
        return train_scene_to_annos, val_scene_to_annos

    def sample_by_scene(
        self, scene_configs: Dict[str, Dict[str, int]], target_labels: Optional[Set[str]] = None
    ) -> LayoutDataset:
        logger.info(f'Sampling {self.layout_dataset.dataset_name} dataset...')
        # logger.info(f'Original scene_configs: {scene_configs}')

        train_sample_images = []
        val_sample_images = []
        for scene in self.dataset_scenes:
            for split, count in scene_configs[scene].items():
                logger.debug(f'scene: {scene}, split: {split}, count: {count}')
                if split == 'train':
                    images_info = self.train_scene_to_images[scene]
                    annos = self.train_scene_to_annos[scene]
                    imgid2annos = self.train_img_id2annos
                    sample_image_result = train_sample_images
                else:
                    images_info = self.val_scene_to_images[scene]
                    annos = self.val_scene_to_annos[scene]
                    imgid2annos = self.val_img_id2annos
                    sample_image_result = val_sample_images

                used_img_ids = set()
                sample_num = min(count, len(images_info))
                for anno in annos:
                    if anno['category_id'] in target_labels and anno['image_id'] not in used_img_ids:
                        used_img_ids.add(anno['image_id'])

                _sample_nums = sample_num - len(used_img_ids)
                if _sample_nums > 0:
                    remaining_image_ids = [img['id'] for img in images_info if img['id'] not in used_img_ids]
                    sampled_image_ids = random.sample(remaining_image_ids, _sample_nums)
                    used_img_ids.update(sampled_image_ids)

                sample_image_result.extend(list(used_img_ids))
                logger.debug(f"scene: {scene}, split: {split}, sample_result: {len(used_img_ids)}")

        logger.debug(f'Train sample images: {len(train_sample_images)}')
        logger.debug(f'Val sample images: {len(val_sample_images)}')

        train_sample_data = defaultdict(list)
        val_sample_data = defaultdict(list)

        for split, sample_img_res in {'train': train_sample_images, 'val': val_sample_images}.items():
            img_id_map = {}
            imgid2annos = self.train_img_id2annos if split == 'train' else self.val_img_id2annos
            id2img = self.train_id2img if split == 'train' else self.val_id2img
            sample_data = train_sample_data if split == 'train' else val_sample_data
            for cur_img_id, pre_img_id in enumerate(sample_img_res):
                pre_img_info = id2img[pre_img_id]
                pre_img_annos = imgid2annos[pre_img_id]
                img_id_map[cur_img_id] = pre_img_id

                pre_img_info['id'] = cur_img_id
                for anno in pre_img_annos:
                    anno['image_id'] = cur_img_id

                sample_data['annotations'].extend(pre_img_annos)
                sample_data['images'].append(pre_img_info)

        return LayoutDataset(
            train=CocoAnnotation(
                images=train_sample_data['images'],
                annotations=train_sample_data['annotations'],
                categories=self.layout_dataset.train.categories,
            ),
            val=CocoAnnotation(
                images=val_sample_data['images'],
                annotations=val_sample_data['annotations'],
                categories=self.layout_dataset.val.categories,
            ),
            image_paths=self.layout_dataset.image_paths,
            dataset_name=self.layout_dataset.dataset_name,
        )


if __name__ == '__main__':
    output_dir = Path('/workspace/datasets/layout/unsv2_layout')
    layout_datasets = LayoutDatasets()
    scene_configs = {
        'financial_reports': {'train': 500, 'val': 100},
        'scientific_articles': {'train': 1000, 'val': 100},
        'government_tenders': {'train': 500, 'val': 100},
        'laws_and_regulations': {'train': 500, 'val': 100},
        'manuals': {'train': 500, 'val': 100},
        'patents': {'train': 500, 'val': 100},
        'unknown': {'train': 3000, 'val': 600},
        'paper_zh': {'train': 1000, 'val': 200},
        'caibao': {'train': 1000, 'val': 200},
        'hetong': {'train': 1000, 'val': 200},
        'lunwen': {'train': 1000, 'val': 200},
        'yanbao': {'train': 1000, 'val': 200},
        'letter': {'train': 300, 'val': 100},
        'email': {'train': 300, 'val': 60},
        'scientific': {'train': 300, 'val': 60},
        'budget': {'train': 300, 'val': 60},
        'form': {'train': 300, 'val': 60},
        'invoice': {'train': 300, 'val': 60},
        'specification': {'train': 300, 'val': 60},
        'memo': {'train': 300, 'val': 60},
        'news': {'train': 300, 'val': 60},
        'presentation': {'train': 300, 'val': 60},
        'resume': {'train': 300, 'val': 60},
    }
    target_labels = {'table', 'Table'}
    for layout_ds in tqdm(layout_datasets.datasets):
        sampler = DataSampler(layout_ds, random_seed=42)
        sample_dataset = sampler.sample_by_scene(scene_configs, target_labels)
        train_images = sample_dataset.get_images('train')
        val_images = sample_dataset.get_images('val')
        print(f'Train images: {len(train_images)}')
        print(f'Val images: {len(val_images)}')
        sample_dataset.export_dataset(output_dir)
