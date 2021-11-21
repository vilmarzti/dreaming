import cv2

from segmentation.data.dataset import TestDataset, TrainDataset
from segmentation.helper import transforms, create_ensemble
from segmentation.helper.create_model import create_unet
from segmentation.training import create_train

from ray import tune
from ray.tune.schedulers import  HyperBandScheduler 
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper

def trial_str_creator(trial):
    return f"trial_{trial.trial_id}"

def main():
    train_paths= [
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_input/masks_cnn",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_input/masks_gmm",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_input/masks_unet",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_input/masks_grabcut",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_input/masks_backsub",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/train_output"
    ]

    test_paths = [
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_input/masks_cnn",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_input/masks_gmm",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_input/masks_unet",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_input/masks_grabcut",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_input/masks_backsub",
        "/home/martin/Videos/ondrej_et_al/bf/segmentation/ensemble/test_output"
    ]

    read_flags = [
        None,
        None,
        None,
        cv2.IMREAD_GRAYSCALE,
        cv2.IMREAD_GRAYSCALE,
        cv2.IMREAD_GRAYSCALE
    ]

    preprocess = [
        lambda x: x,
        lambda x: x,
        lambda x: x,
        transforms.threshold,
        transforms.threshold,
        transforms.threshold
    ]

    transform = transforms.compose(*[
        transforms.copy_images,
        transforms.random_rotate,
        transforms.random_flip,
        lambda x: transforms.merge(x, cutoff=5)
    ])

    crop_size = 252

    train_set = TrainDataset(
        train_paths,
        crop_size,
        read_flags,
        preprocess,
        transform
    )

    test_set = TestDataset(
        test_paths,
        crop_size,
        read_flags,
        preprocess,
        transform
    )

    train = create_train(
        create_unet,
        transform="scale",
    )

    config = {
        "input_channels": tune.choice([5]),
        "deepness": tune.randint(2, 4),
        "starting_multiplier": tune.randint(3, 8),
        "use_thin": tune.choice([True, False]),
        "positional_encoding": tune.choice([None]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    search_alg = HyperOptSearch(
        metric="val_j_index",
        mode="max",
        n_initial_points=5
    )

    scheduler = HyperBandScheduler(
        max_t=30,
        metric="val_j_index",
        mode="max"
    )

    stopper = TrialPlateauStopper(
        metric="val_j_index",
        std=0.006
    )

    result = tune.run(
        tune.with_parameters(train, train_set=train_set, test_set=test_set),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=50,
        trial_dirname_creator=trial_str_creator,
        scheduler=scheduler,
        local_dir="./data/raytune",
        name="ensemble",
        search_alg=search_alg,
        stop=stopper
    )

    best_trial = result.get_best_trial("val_j_index", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))
 

if __name__ == "__main__":
    main()