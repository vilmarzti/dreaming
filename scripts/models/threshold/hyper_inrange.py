from ray import tune
from ray.tune.schedulers import ASHAScheduler

from segmentation.training import create_test_best, create_train
from segmentation.helper import create_rangeimage

import cv2

def trial_str_creator(trial):
    return f"trial_{trial.trial_id}"

def main(num_samples, max_num_epochs=10, gpus_per_trial=0.5):
    config = {
        "kernel_size": tune.randint(3, 10),
        "padding": tune.choice([False]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    train = create_train(
        create_rangeimage,
        300,
        cv2.COLOR_BGR2HSV,
        False
    )

    test_best_model = create_test_best(
        create_rangeimage,
        5,
        cv2.COLOR_BGR2HSV,
        False 
    )

    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        metric="accuracy",
        mode="max",
        num_samples=num_samples,
        trial_dirname_creator=trial_str_creator,
        scheduler=scheduler,
        local_dir="./data/threshold_raytune",
        name="threshold"
    )

    best_trial = result.get_best_trial("loss", "min", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    test_best_model(best_trial)

if __name__ == "__main__":
    main(30)