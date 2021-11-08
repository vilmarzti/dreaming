import os
import cv2
import os.path as path
import sys
import math
import shutil
import argparse


sys.path.append("./Few-Shot-Patch-Based-Training/_tools/")
import add_gauss
import count_black
import config

def compute_max_thresh(mask_paths):
    num_max = 0
    for p in mask_paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        perc_black =  1- (cv2.countNonZero(img) / img.size)
        perc_black = math.floor(perc_black * 100) + 3
        num_max = max(num_max, perc_black)
    return num_max

def compute_individual_zero(mask_paths):
    individual = []
    individual = [count_black.count_pixels(p) for p in mask_paths]
    return individual 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Computes gauss with given mask""")

    parser.add_argument("--mask-folder", "-m",
        help="The path to the folder with the masks",
        required=True)

    parser.add_argument("--threshold", "-t",
        help="Threshold in percent",
        type=int,
        default=3
    )

    args = parser.parse_args()

    def copy_file(image_name):
        shutil.copy(
            path.join(args.mask_folder, image_name),
            path.join(config.maskDir, image_name)
        ) 
    
    max_threshold = args.threshold
    mask_paths = os.listdir(args.mask_folder)
    mask_paths.sort()
    
    full_paths = [path.join(args.mask_folder, p) for p in mask_paths]

    start = int(config.frameFirst)
    end = int(config.frameLast)
    full_paths = full_paths[start - 1: end]

    individual_zero = compute_individual_zero(full_paths)

    os.chdir("./Few-Shot-Patch-Based-Training/_tools")
    add_gauss.loop(max_threshold, 10, copy_file, individual_zero)