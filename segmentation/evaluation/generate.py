import os
import cv2
import math
import torch

import numpy as np

from os import path

from segmentation.models.cnn import CNNSegmentation
from segmentation.models.unet import UNet
from segmentation.training.train import crop_or_scale

from ..helper import preprocessing


def compute_segmentation(image, tile_size, model, device="cpu", output_tile=None):
    num_t_x = image.shape[3] // tile_size 
    num_t_y = image.shape[2] //  tile_size

    if output_tile:
        num_t_x = image.shape[3] // output_tile
        num_t_y = image.shape[2] // output_tile

    segmentation = np.zeros((1, image.shape[2], image.shape[3]))
    for y in range(num_t_y):
        for x in range(num_t_x):
            # Change stride of tiles depending on the output_tile size
            if not output_tile:
                tile = image[:, :, y * tile_size: (y + 1) * tile_size, x * tile_size: (x + 1) * tile_size]
            else:
                tile = image[:, :, y * output_tile: y * output_tile + tile_size, x * output_tile: x * output_tile + tile_size]

            tile = torch.tensor(np.single(tile), device=device)
            output = model(tile)
            
            # If output smaller than input (i.e we're using CNN without pad), scale output
            if type(model) is CNNSegmentation:
                output, _ = crop_or_scale(output, np.zeros((1, 1, tile_size, tile_size)), "scale")
                segmentation[:, y * tile_size: (y + 1) * tile_size, x * tile_size: (x + 1) * tile_size] = output[0, 0].detach().cpu()

            if type(model) is UNet and output_tile:
                segmentation[:, y * output_tile: (y + 1) * output_tile, x * output_tile: (x + 1) * output_tile] = output[0, 0].detach().cpu()
            
    return segmentation

def generate_segmentations(input_path, output_path, model, tile_size, pad_to, output_tile=None):
    # Tile image to into parts of tile_size x tile_size
    pad_x = pad_to[0]
    pad_y = pad_to[1]

    # Check if cuda is available
    device="cuda:0" if torch.cuda.is_available else "cpu"

    # Put model to gpu if possible
    model = model.to(device)

    # preprocess function for the images
    preprocess = [
        lambda x: preprocessing.pad_reflect(x, (pad_x, pad_y)) ,
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
            segmentation = compute_segmentation(
                image,
                tile_size, 
                model, 
                device,
                output_tile
            )

        # Cut segmentation appropriately
        pad_left = (pad_x - original_image.shape[1]) // 2
        pad_top = (pad_y - original_image.shape[0]) // 2
        segmentation = segmentation[0, pad_top: pad_top + original_image.shape[0], pad_left : pad_left + original_image.shape[1]]
        cv2.imshow("Segmentation", segmentation)
        cv2.imshow("Original", original_image)
        cv2.waitKey(20)

        # prepare for saving
        segmentation = np.array(segmentation * 255, np.uint8)
        #cv2.imwrite(path.join(output_path, i_name), segmentation)