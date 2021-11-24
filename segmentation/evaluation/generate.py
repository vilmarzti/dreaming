import os
import cv2
import torch

import numpy as np

from os import path

from ..helper import transforms 

def generate_segmentations(dataset, output_path, model, save_png=False):
    # Check if cuda is available
    device="cuda:0" if torch.cuda.is_available else "cpu"

    # Put model to gpu if possible
    model = model.to(device)

    # create mask-dir
    if not path.isdir(output_path):
        os.mkdir(output_path)

    # Go through images and create the segmentation
    for i, image in enumerate(dataset):
        # Create Segmentation
        with torch.no_grad():
            image_tensor = torch.tensor(np.single(image), device=device)
            segmentation = model(image_tensor).detach().cpu().numpy()

        image = image[0]
        # Find out how much space is padded to left and right
        pad_left = (segmentation.shape[3] - 720) // 2
        pad_top = (segmentation.shape[2] - 1280) // 2

        segmentation = segmentation[0, 0, pad_top: pad_top + 1280, pad_left : pad_left + 720]

        # For debugging purposes
        #cv2.imshow("Segmentation", segmentation)
        #cv2.imshow("Original", original_image)
        #cv2.waitKey(20)

        #image_name = path.basename(dataset.paths[0][i])
        image_name = f"{i+1:04d}.png"

        # prepare for saving
        if save_png:
            segmentation = np.array(segmentation * 255, np.uint8)
            cv2.imwrite(path.join(output_path, image_name), segmentation)
        else:
            np.save(path.join(output_path, image_name.replace(".png", ".npy")), segmentation)