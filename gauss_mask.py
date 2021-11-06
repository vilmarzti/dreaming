import glob
import os
import cv2
import os.path as path
import sys
import math
import shutil
import argparse

from numpy.core.numeric import full

sys.path.append("./Few-Shot-Patch-Based-Training/_tools/")
import tool_gauss
import add_gauss

def compute_max_thresh(mask_paths):
    num_max = 0
    for p in mask_paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        perc_black =  1- (cv2.countNonZero(img) / img.size)
        perc_black = math.floor(perc_black * 100) + 3
        num_max = max(num_max, perc_black)
    return num_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Computes gauss with given mask""")

    parser.add_argument("--mask-folder", "-m",
        help="The path to the folder with the masks",
        required=True)
    

    args = parser.parse_args()
    def copy_file(image_name):
        shutil.copy(
            path.join(args.mask_folder, image_name),
            path.join(tool_gauss.maskDir, image_name)
        ) 
    
    mask_paths = os.listdir(args.mask_folder)
    mask_paths.sort()
    
    full_paths = [path.join(args.mask_folder, p) for p in mask_paths]

    start = int(tool_gauss.frameFirst)
    end = int(tool_gauss.frameLast)
    full_paths = full_paths[start - 1: end + 1]

    #max_threshold = compute_max_thresh(full_paths)
    max_threshold = 35
    print(max_threshold)
    os.chdir("./Few-Shot-Patch-Based-Training/_tools")
    add_gauss.loop(max_threshold, 10, copy_file)