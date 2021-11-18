import os
import cv2
import math
import torch

import numpy as np

from os import path

from ..helper import preprocessing


def compute_segmentation(image, tiles_x, tiles_y, size, cnn, device):
    segmentation = np.zeros((1, image.shape[2], image.shape[3]))
    for y in range(tiles_y):
        for x in range(tiles_x):
            tile = image[:, :, y * size: (y + 1) * size, x * size: (x + 1) * size]
            tile = torch.tensor(np.single(tile), device=device)
            output = cnn(tile)

            segmentation[:,y * size: (y + 1) * size, x * size: (x + 1) * size] = output[0, 0].detach().cpu()
    return segmentation

def generate_segmentations(input_path, output_path, create_model, config, checkpoint_path, image_size, tile_size):
    # Tile image to into parts of tile_size x tile_size
    tiles_x = math.ceil(image_size[0] / tile_size)
    tiles_y = math.ceil(image_size[1] / tile_size)

    # Check if cuda is available
    device="cuda:0" if torch.cuda.is_available else "cpu"

    # Load model from checkpoint
    model = create_model(config)
    checkpoint = torch.load(path.join(checkpoint_path, "checkpoint"))
    model.load_state_dict(checkpoint[0])

    # Put model to gpu if possible
    cnn = model.to(device)

    # preprocess function for the images
    preprocess = [
        lambda x: preprocessing.pad_reflect(x, (tiles_x * tile_size, tiles_y * tile_size)) ,
        preprocessing.add_encoding,
        preprocessing.subtract_mean
    ]
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
        segmentation = None
        with torch.no_grad():
            segmentation = compute_segmentation(image, tiles_x, tiles_y, tile_size, cnn, device)

        # save segment image
        segmentation = np.array(segmentation[0, :image_size[1], :image_size[0]] * 255, np.uint8)
        cv2.imwrite(path.join(output_path, i_name), segmentation)