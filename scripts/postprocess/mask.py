import cv2
import os

import numpy as np

from os import path


def read_mask(mask_path):
    if ".png" in mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    else:
        mask = np.load(mask_path)
    return mask

def main():
    images = "/home/martin/Videos/ondrej_et_al/bf/test"
    mask1 = "data/masks_ensemble/"
    mask2 = "data/masks_backsub/"

    image_names = os.listdir(images)
    image_names.sort()

    mask1_names = os.listdir(mask1)
    mask1_names.sort()

    mask2_names = os.listdir(mask2)
    mask2_names.sort()

    length = len(image_names)

    mask1_names = mask1_names[:length]
    mask2_names = mask2_names[:length]

    for image_name, m1_name, m2_name in zip(image_names, mask1_names, mask2_names):
        img_original = cv2.imread(path.join(images, image_name), cv2.IMREAD_COLOR)

        m1 = read_mask(path.join(mask1, m1_name)) / 255
        m2 = read_mask(path.join(mask2, m2_name)) / 255

        combined_mask = m1  #* m2

        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 5)
        masked_image = np.array(img_original * combined_mask, dtype=np.uint8)
        combined_mask = np.array(combined_mask * 255, dtype=np.uint8)

        cv2.imshow("combined", img_original)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()