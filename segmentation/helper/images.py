import os
import sys

from os import path

sys.path.append("./Few-Shot-Patch-Based-Training/_tools/")
import config

def get_image_paths(input_path):
    image_names = os.listdir(input_path)
    image_names.sort()

    image_names = image_names[int(config.frameFirst) - 1: int(config.frameLast)]
    image_paths = [path.join(input_path, x) for x in image_names]
    return image_paths

def get_output_paths(images_path, output_path):
    base_names = [path.basename(i) for i in images_path]
    output_paths = [path.join(output_path, b) for b in base_names]
    return output_paths