import io
import json
import base64
import pathlib
import logging
import collections

import cv2
import numpy
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as alb


def _load_image(image_data_b64):
    # note(will.brennan) - from https://github.com/wkentaro/labelme/blob/f20a9425698f1ac9b48b622e0140016e9b73601a/labelme/utils/image.py#L17
    image_data = base64.b64decode(image_data_b64)
    image_data = numpy.fromstring(image_data, dtype=numpy.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _create_masks(shapes, image_width: int, image_height: int):
    for shape in shapes:
        mask = numpy.zeros((image_height, image_width), dtype=numpy.uint8)
        points = numpy.array(shape['points']).reshape((-1, 1, 2))
        points = numpy.round(points).astype(numpy.int32)

        cv2.fillPoly(mask, [points], (1, ))
        mask = mask.astype(numpy.uint8)
        yield mask


def _create_bboxs(shapes):
    for shape in shapes:
        points = numpy.array(shape['points'])
        xmin, ymin = numpy.min(points, axis=0)
        xmax, ymax = numpy.max(points, axis=0)

        yield [xmin, ymin, xmax, ymax]


class ToTensor(alb.ImageOnlyTransform):
    def __init__(self):
        super().__init__(always_apply=True)

    def apply(self, image, **params):
        return transforms.ToTensor()(image)

    def get_params(self):
        return {}


class LabelMeDataset(data.Dataset):
    def __init__(self, directory: str, use_augmentation: bool):
        self.directory = pathlib.Path(directory)
        self.use_augmentation = use_augmentation
        assert self.directory.exists()
        assert self.directory.is_dir()

        self.labelme_paths = []
        self.categories = collections.defaultdict(list)

        for labelme_path in self.directory.rglob('*.json'):
            with open(labelme_path, 'r') as labelme_file:
                labelme_json = json.load(labelme_file)

                required_keys = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
                assert all(key in labelme_json for key in required_keys), (required_keys, labelme_json.keys())

                self.labelme_paths += [labelme_path]

                for shape in labelme_json['shapes']:
                    label = shape['label']
                    self.categories[label] += [labelme_path]

        for category, paths in self.categories.items():
            for path in paths:
                logging.debug(f'{category} - {path}')
        self.categories = sorted(list(self.categories.keys()))

        logging.info(f'loaded {len(self)} annotations from {self.directory}')
        logging.info(f'use augmentation: {self.use_augmentation}')
        logging.info(f'categories: {self.categories}')

        aug_transforms = [ToTensor()]
        if self.use_augmentation:
            aug_transforms = [
                alb.HueSaturationValue(always_apply=True),
                alb.RandomBrightnessContrast(always_apply=True),
                alb.HorizontalFlip(),
                alb.RandomGamma(always_apply=True),
            ] + aug_transforms
        bbox_params = alb.BboxParams(format='pascal_voc', min_area=0.0, min_visibility=0.0, label_fields=['labels'])
        self.transforms = alb.Compose(transforms=aug_transforms, bbox_params=bbox_params)

    def __len__(self):
        return len(self.labelme_paths)

    def __getitem__(self, idx: int):
        labelme_path = self.labelme_paths[idx]
        logging.debug('parsing labelme json')

        with open(labelme_path, 'r') as labelme_file:
            labelme_json = json.load(labelme_file)

        image_width = labelme_json['imageWidth']
        image_height = labelme_json['imageHeight']

        image = _load_image(labelme_json['imageData'])
        assert image.shape == (image_height, image_width, 3)

        labelme_shapes = labelme_json['shapes']
        labelme_shapes = [i for i in labelme_json['shapes'] if len(i['points']) > 2]
        assert all(i['shape_type'] == 'polygon' for i in labelme_shapes)

        masks = list(_create_masks(labelme_shapes, image_width, image_height))

        bboxes = list(_create_bboxs(labelme_shapes))

        labels = [self.categories.index(shape['label']) for shape in labelme_shapes]

        logging.debug('applying transforms to image and targets')

        target = self.transforms(image=image, bboxes=bboxes, labels=labels, masks=masks)

        image = target.pop('image')

        target['masks'] = torch.as_tensor(numpy.stack(target['masks']), dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['iscrowd'] = torch.zeros_like(target['labels'], dtype=torch.int64)
        target['image_id'] = torch.tensor([idx], dtype=torch.int64)

        bboxes = torch.as_tensor(target.pop('bboxes'), dtype=torch.float32)

        target['area'] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target['boxes'] = bboxes

        return image, target
