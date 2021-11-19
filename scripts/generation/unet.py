import math
import torch

from os import path

from segmentation.helper.create_model import create_unet
from segmentation.evaluation.generate import generate_segmentations

from ray.tune.analysis.experiment_analysis import Analysis

def main(experiment_path, input_path, output_path):
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
    model = create_unet(best_config)
    checkpoint = torch.load(path.join(best_cp_path, "checkpoint"))
    model.load_state_dict(checkpoint[0])

    pad_x = 812
    pad_y = 1372

    generate_segmentations(
        input_path,
        output_path,
        model,
        (pad_x, pad_y),
    )

if __name__ == "__main__":
    main(
        "/home/martin/Documents/code/python/dreaming/data/raytune/unet_run1",
        "/home/martin/Videos/ondrej_et_al/bf/bf_gen/input_filtered",
        "/home/martin/Documents/code/python/dreaming/data/masks",
    )