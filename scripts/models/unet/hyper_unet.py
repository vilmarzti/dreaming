from ray import tune
from ray.tune.schedulers import ASHAScheduler

from segmentation.training import create_test_best, create_train
from segmentation.helper import create_unet

def trial_str_creator(trial):
    return f"trial_{trial.trial_id}"

def main(num_samples, max_num_epochs=45, gpus_per_trial=0.5):
    config = {
        "deepness": tune.randint(2, 5),
        "starting_multiplier": tune.randint(3, 11),
        "use_thin": tune.choice([True, False]),
        "positional_encoding": tune.choice(["sin", "linear", None]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    train = create_train(
        create_unet,
        300,
        None,
        True,
        False
    )

    test_best_model = create_test_best(
        create_unet,
        5,
        None,
        True
    )

    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        metric="val_accuracy",
        mode="max",
        num_samples=num_samples,
        trial_dirname_creator=trial_str_creator,
        scheduler=scheduler,
        local_dir="./data/raytune",
        name="unet"
    )

    best_trial = result.get_best_trial("val_loss", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_accuracy"]))
    
    test_best_model(best_trial)

if __name__ == "__main__":
    main(50)