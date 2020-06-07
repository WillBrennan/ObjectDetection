from typing import List
from typing import Dict
import itertools

import torch
import cv2
import numpy


def draw_results(image: torch.Tensor, target: Dict[str, torch.Tensor], categories: List[str]):
    image = (255 * image).to(torch.uint8).cpu().numpy()
    image = numpy.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    colours = (
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    )

    for label, mask in zip(target['labels'], target['masks']):
        label = label.item()
        mask = mask.cpu().bool().numpy()

        category = categories[label]
        colour = colours[label % len(colours)]

        image[mask] = 0.5 * image[mask] + 0.5 * numpy.array(colour)

    for label, bbox in zip(target['labels'], target['boxes']):
        label = label.item()

        category = categories[label]
        colour = colours[label % len(colours)]

        bbox = bbox.cpu().numpy()
        bbox = numpy.round(bbox).astype(int).tolist()
        bbox_tl = tuple(bbox[:2])
        bbox_br = tuple(bbox[2:])
        cv2.rectangle(image, bbox_tl, bbox_br, colour, 3)

        text_point = (bbox_tl[0], bbox_tl[1] - 10)
        cv2.putText(image, category, text_point, cv2.FONT_HERSHEY_PLAIN, 2, colour)

    return image
