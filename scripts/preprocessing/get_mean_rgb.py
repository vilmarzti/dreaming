from segmentation.helper import get_image_paths

import cv2
import numpy as np

import argparse

def main(input_path):
    image_paths = get_image_paths(input_path)
    chunked_paths = np.array_split(image_paths, 10)
    mean_rgb = np.zeros(3)
    for paths in chunked_paths:
        imgs = np.array([cv2.imread(image_path) for image_path in paths])
        mean_rgb += np.mean(imgs, axis=(0, 1, 2))
    mean_rgb /= 10
    print(mean_rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""" Creates masks for body""")

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
    
    args = parser.parse_args()
    main(args.input_path)