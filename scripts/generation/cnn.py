import os
import cv2
import math
import torch

import numpy as np

from os import path
from ray.tune import Analysis
from torch.nn.functional import interpolate
from segmentation.helper import create_cnn
from segmentation.constants import BGR_MEAN
from ray.tune.analysis.experiment_analysis import Analysis

experiment_path = "/home/martin/Documents/code/python/dreaming/data/raytune/cnn"
input_path = "/home/martin/Videos/ondrej_et_al/bf/bf_gen/input_filtered"
output_path = "/home/martin/Documents/code/python/dreaming/data/masks"
transform ="scale"

if __name__ == "__main__":
    analysis = Analysis(experiment_path)

    # Get best trial. There's probably a quicker way to do this
    best_trial = None
    best_score = 0
    for k in analysis.trial_dataframes:
        score = analysis.trial_dataframes[k]["val_j_index"].iloc[-1]

        best_trial = k if score > best_score else best_trial
        best_score = score if score > best_score else best_score 
        
    print(f"Loaded trial {best_trial} with  score {best_score}")

    # Get config and ceckpoint_path from best_trial
    best_config = analysis.get_all_configs()[best_trial]
    best_cp_path = analysis.get_trial_checkpoints_paths(best_trial)[-1][0]

    # Load model from checkpoint
    cnn = create_cnn(best_config)
    checkpoint = torch.load(path.join(best_cp_path, "checkpoint"))
    cnn.load_state_dict(checkpoint[0])

    # Process input images
    image_paths = os.listdir(input_path)
    for image_path in image_paths:
        # Read Image and preprocess
        image = cv2.imread(path.join(input_path, image_path), cv2.IMREAD_COLOR)
        image = image - np.expand_dims(BGR_MEAN, axis=(0, 1, 2))
        image = np.transpose(image, (0, 3, 1, 2))[0]

        # Tile image to into parts of 252x252
        tiles_x = math.ceil(720 / 252)
        tiles_y = math.ceil(1280 / 252)

        # Calculate how much the image should be padded at the end of the x and y-axis
        num_pad_x = (tiles_x * 252) - 720
        num_pad_y = (tiles_y * 252) - 1280

        # Pad Image
        image = np.pad(image, pad_width=((0, 0), (0, num_pad_y), (0, num_pad_x)), mode="reflect")

        # Pass through the tiles and create a new output image
        output_image = np.zeros((1, 1280, 720))
        for y in range(tiles_y):
            for x in range(tiles_x):
                pos_x = x * 252
                pos_y = y * 252
                tile = torch.Tensor([image[:, pos_y: pos_y + 252, pos_x: pos_x + 252 ]])

                out_tile = cnn(tile)
                 
                # Rearrange to scale out_tile back
                out_tile = interpolate(out_tile, (252, 252), mode="bilinear", align_corners=False)
                out_tile = out_tile.detach().cpu().numpy()

                output_image[:, pos_y: pos_y + 252, pos_x: pos_x + 255]

        cv2.imshow("Segmentation", output_image)
        




        # Save resulting Masks
        pass
