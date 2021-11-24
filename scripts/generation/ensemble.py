import cv2
import torch

from os import path
from ray.tune.analysis.experiment_analysis import Analysis

from segmentation.data.dataset import GenerationDataset
from segmentation.evaluation.generate import generate_segmentations
from segmentation.helper import transforms
from segmentation.helper.create_model import create_unet

experiment_path = "data/raytune/ensemble"
def main():
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

    train_paths= [
        "data/masks_cnn",
        "data/masks_gmm",
        "data/masks_unet",
        "data/masks_grabcut",
        "data/masks_backsub",
    ]

    read_flags = [
        None,
        None,
        None,
        cv2.IMREAD_GRAYSCALE,
        cv2.IMREAD_GRAYSCALE,
    ]

    preprocess = [
        lambda x: x,
        lambda x: x,
        lambda x: x,
        transforms.threshold,
        transforms.threshold,
    ]

    transform = transforms.compose(*[
        transforms.copy_images,
        lambda x: transforms.pad_reflect(x, (812, 1372)),
        lambda x: transforms.merge(x, cutoff=5)
    ])


    dataset = GenerationDataset(
        train_paths,
        (720, 1280),
        read_flags,
        preprocess,
        transform=transform
    )

    generate_segmentations(dataset, "data/masks_ensemble", model, True)


if __name__ == "__main__":
    main()
