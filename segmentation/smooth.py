import cv2
import sys
import os
import argparse
import numpy as np
import multiprocessing as mp

from itertools import repeat
from os import path

sys.path.append("../Few-Shot-Patch-Based-Training/_tools/")
import config

length = 10 # range will be equal to 2* length + 1
impact = np.log(0.05) / length # adjust such that furthest away image has a weight of 0.1
mu = 0
sig = 5
start = int(config.frameFirst) - 1
end = int(config.frameLast)
gauss_vals = {}


def load_images(mask_folder):
    images = []
    for x in range(start + 1, end + 1):
        image_name = f"{x:04}.png"
        p = path.join(mask_folder, image_name)
        i = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        images.append(i)
    return np.array(images)

def gaussian(x):
    if x in gauss_vals:
        return gauss_vals[x]
    else:
        r = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
        gauss_vals[x] = r
        return r

def create_ranges():
    ranges = []
    for i in range(end):
        r_start = i - length if i - length >= start else start
        r_end  = i + length + 1 if i + length + 1 < end else end
        r = np.arange(r_start, r_end)
        ranges.append(r)
    return ranges

def smooth_image(images, rang, i):
    relative = abs(rang - i)
    #weight = np.exp(relative * impact)
    weight = [gaussian(x) for x in relative]
    total_weight = sum(weight)
    imgs = images[rang]

    # Interpolate accoding to the given values
    weighted_images = np.array([w * i for w, i in zip(weight, imgs)]) / total_weight
    smoothed_image = np.sum(weighted_images, axis=0)
    smoothed_image = np.where(smoothed_image < 127.5, 0, 255).astype("uint8")

    return smoothed_image

    # save image
    smoothed_path = path.join(smooth_folder, f"{i + 1:04}.png")
    cv2.imwrite(str(smoothed_path), smoothed_image)

def save_image(smooth_folder, index, image):
    smoothed_path = path.join(smooth_folder, f"{index + 1:04}.png")
    cv2.imwrite(str(smoothed_path), image)


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

    mask_folder = args.input_path
    smooth_folder = args.output_path

    ranges = create_ranges()
    images = load_images(mask_folder)
    indices = np.arange(len(ranges))

    if not path.isdir(smooth_folder):
        os.mkdir(smooth_folder)

    pool = mp.Pool(mp.cpu_count())

    # compute smoothed image
    smoothed_images = pool.starmap(smooth_image, zip(repeat(images), ranges, indices))

    # save images
    pool.starmap(save_image, zip(repeat(smooth_folder), indices, smoothed_images))
