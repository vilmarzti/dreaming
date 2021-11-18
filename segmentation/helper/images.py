import os
from os import path

def get_image_paths(input_path, frame_first=1, frame_last=-1):
    image_names = os.listdir(input_path)
    image_names.sort()

    image_names = image_names[frame_first - 1: frame_last]
    image_paths = [path.join(input_path, x) for x in image_names]
    return image_paths

def get_output_paths(images_path, output_path):
    base_names = [path.basename(i) for i in images_path]
    output_paths = [path.join(output_path, b) for b in base_names]
    return output_paths