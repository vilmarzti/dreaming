import os
import cv2
import torch

import numpy as np

from os import path

from ..helper import preprocessing

def generate_segmentations(input_path, output_path, model, pad_to=None):
    # Check if cuda is available
    device="cuda:0" if torch.cuda.is_available else "cpu"

    # Put model to gpu if possible
    model = model.to(device)

    # preprocess functions for the images
    if pad_to is not None:
        preprocess = [
            lambda x: preprocessing.pad_reflect(x, (pad_to[0], pad_to[1]))
        ]
    else: 
        preprocess = []

    preprocess.append(preprocessing.add_encoding)
    preprocess.append(preprocessing.subtract_mean)

    # Compose the functions
    preprocess = preprocessing.compose(*preprocess)

    # create mask-dir
    if not path.isdir(output_path):
        os.mkdir(output_path)

    # Go through images and create the segmentation
    image_names = os.listdir(input_path)
    image_names.sort()
    for i_name in image_names:
        # Read image
        original_image = cv2.imread(path.join(input_path, i_name), cv2.IMREAD_COLOR)

        # preprocess
        image = np.transpose([original_image], [0, 3, 1, 2])
        image = preprocess(image)

        # Create Segmentation
        with torch.no_grad():
            image_tensor = torch.tensor(np.single(image), device=device)
            segmentation = model(image_tensor).detach().cpu().numpy()

        # Find out how much space is padded to left and right
        if pad_to is not None:
            pad_left = (pad_to[0] - original_image.shape[1]) // 2
            pad_top = (pad_to[1] - original_image.shape[0]) // 2
        else:
            pad_left = 0
            pad_top = 0

        segmentation = segmentation[0, 0, pad_top: pad_top + original_image.shape[0], pad_left : pad_left + original_image.shape[1]]
        cv2.imshow("Segmentation", segmentation)
        cv2.imshow("Original", original_image)
        cv2.waitKey(20)

        # prepare for saving
        segmentation = np.array(segmentation * 255, np.uint8)
        #cv2.imwrite(path.join(output_path, i_name), segmentation)