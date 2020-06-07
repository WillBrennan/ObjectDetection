import logging

import torch
from torch import nn
from torchvision.models import detection


class MaskRCNN(nn.Module):
    @staticmethod
    def load(state_dict):
        # todo(will.brennan) - improve this... might want to save a categories file with this instead
        category_prefix = '_categories.'
        categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
        categories = [k[len(category_prefix):] for k in categories]

        model = MaskRCNN(categories)
        model.load_state_dict(state_dict)
        return model

    def __init__(self, categories):
        super().__init__()
        logging.info(f'creating model with categories: {categories}')

        # todo(will.brennan) - find a nicer way of saving the categories in the state dict...
        self._categories = nn.ParameterDict({i: nn.Parameter(torch.Tensor(0)) for i in categories})
        num_categories = len(self._categories)

        self.model = detection.maskrcnn_resnet50_fpn(pretrained=True)

        logging.debug('changing num_categories for bbox predictor')

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_categories)

        logging.debug('changing num_categories for mask predictor')

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, 256, num_categories
        )

    @property
    def categories(self):
        return list(self._categories.keys())

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def filter_by_threshold(result, bbox_thresh: float, mask_thresh: float):
    scores_mask = result['scores'] > bbox_thresh
    result = {k: v[scores_mask] for k, v in result.items()}

    result['masks'] = result['masks'][:, 0] >= mask_thresh

    return result