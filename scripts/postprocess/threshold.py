import cv2
import os

import numpy as np

def main():
    input_path = "./data/masks_ensemble/"
    output_path = "./data/masks_ensemble_threshold/"

    image_names = os.listdir(input_path)
    image_names.sort()

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for image_name in image_names:
        image = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE)
        image = np.where(image < 128, 0, 255)
        cv2.imwrite(os.path.join(output_path, image_name), image)

if __name__ == "__main__":
    main()