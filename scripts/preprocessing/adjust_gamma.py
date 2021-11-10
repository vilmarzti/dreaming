import cv2
import argparse

import numpy as np

import os
from os import path

import sys
sys.path.append("../Few-Shot-Patch-Based-Training/_tools/")
import config

def gamma_lut(gamma):
    lut = np.empty((1, 256), np.uint8)
    lut[0] = [np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255) for i in range(256)]
    return lut

def adjust_gamma(gamma, input_path, output_path):
    image_names = os.listdir(input_path)
    image_names.sort()

    image_names = image_names[int(config.frameFirst) - 1: int(config.frameLast)]
    image_paths = [path.join(input_path, x) for x in image_names]
    output_paths = [path.join(output_path, x) for x in image_names]

    if not path.isdir(output_path):
        os.mkdir(output_path)

    lut = gamma_lut(gamma)

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        out_img = cv2.LUT(img, lut)
        cv2.imwrite(output_paths[i], out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""" Creates masks for body""")

    parser.add_argument("--gamma", "-g",
        help="Gamma for gamma correction",
        type=float,
        required=True
    )

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
    
    parser.add_argument("--output-path", "-o",
        help="The folder the masks are writen to",
        required=True
    )

    args = parser.parse_args()
    adjust_gamma(args.gamma, args.input_path, args.output_path)