import cv2
import sys
import os
import numpy as np
import multiprocessing as mp

from os import path

sys.path.append("./Few-Shot-Patch-Based-Training/_tools/")
import config

length = 5 # range will be equal to 2* length + 1
impact = np.log(0.05) / length # adjust such that furthest away image has a weight of 0.1
smooth_folder = "./smooth_mask"
mask_folder = "./masks"
start = int(config.frameFirst) - 1
end = int(config.frameLast)

def load_images():
    images = []
    for x in range(start + 1, end + 1):
        image_name = f"{x:04}.png"
        p = path.join(mask_folder, image_name)
        i = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        images.append(i)
    return np.array(images)

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
    weight = np.exp(relative * impact)
    total_weight = sum(weight)
    imgs = images[rang]

    # Interpolate accoding to the given values
    weighted_images = np.array([w * i for w, i in zip(weight, imgs)]) / total_weight
    smoothed_image = np.sum(weighted_images, axis=0)
    smoothed_image = np.where(smoothed_image < 127.5, 0, 255).astype("uint8")

    # save image
    smoothed_path = path.join(smooth_folder, f"{i + 1:04}.png")
    cv2.imwrite(str(smoothed_path), smoothed_image)



if __name__ == "__main__":
    ranges = create_ranges()
    images = load_images()
    indices = np.arange(len(ranges))

    if not path.isdir(smooth_folder):
        os.mkdir(smooth_folder)

    for r, i in zip(ranges, indices):
        smooth_image(images, r, i)

    pool = mp.Pool(mp.cpu_count())
    #pool.starmap(smooth_image, zip(images, ranges, indices))
    #smooth_image(images, ranges[0], indices[0])
