import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision.transforms import functional as F

from object_detection import MaskRCNN
from object_detection import filter_by_threshold
from object_detection import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    assert args.display or args.save

    logging.info(f'loading model from {args.model}')
    model = MaskRCNN.load(torch.load(args.model))
    model.cuda().eval()

    image_dir = pathlib.Path(args.images)

    fn_filter = functools.partial(filter_by_threshold, bbox_thresh=args.threshold, mask_thresh=args.threshold)

    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        logging.info(f'finding objects in {image_file} with threshold of {args.threshold}')

        image = cv2.imread(str(image_file))
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            image = F.to_tensor(image)
            image = image.cuda().unsqueeze(0)

            results = model(image)
            results = [fn_filter(i) for i in results]

        image = draw_results(image[0], results[0], categories=model.categories)

        if args.save:
            output_name = f'results_{image_file.name}'
            logging.info(f'writing output to {output_name}')
            cv2.imwrite(str(output_name), image)

        if args.display:
            cv2.imshow('image', image)

            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()
