import argparse
import os
import cv2

import numpy as np

from os import path

sys.path.append("../Few-Shot-Patch-Based-Training/_tools/")
import config

def background_substract(image_paths):
    bg_substractor = cv2.createBackgroundSubstractorMOG2(detectShadows=True)
    img = cv2.imread(image_paths[0])

    masks = []
    for path in image_paths[1:]:
        fg_mask = bg_substractor.apply(img)

        masks.append(fg_mask)

        cv2.imshow("FG Mask", fg_mask)
        cv2.imshow("image", img)
        k = cv2.waitKey(30)

        if k == 27:
            break

        img = cv2.imread(path)
    
    return masks


def process_folder(input_path, output_path):
    images = os.listdir(input_path)
    images.sort()

    images = images[int(config.frameFirst) - 1: int(config.frameLast)]
    image_paths = [path.join(input_path, x) for x in images]
    mask_paths = list(map(lambda x: str(path.join(output_path, x)), images))

    masks_forward = background_substract(image_paths)
    masks_backward = background_substract(image_paths[::-1])

    if not path.isdir(output_path):
        os.mkdir(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""" Creates masks for body""")

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
    
    parser.add_argument("--output-path", "-o",
        help="The folder the masks are writen to",
        required=True
    )

    args = parser.parse_args()
    process_folder(args.input_path, args.output_path)