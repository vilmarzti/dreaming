from segmentation.evaluation.generate import generate_segmentations

from ray.tune import Analysis
from ray.tune.analysis.experiment_analysis import Analysis

from segmentation.helper.create_model import create_cnn

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

    generate_segmentations(
        input_path,
        output_path,
        create_cnn,
        best_config,
        best_cp_path,
        (720, 1280),
        252
    )