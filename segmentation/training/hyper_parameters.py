from ray import tune
from ray.tune.schedulers import ASHAScheduler

from segmentation.training import create_test_best, create_train
from segmentation.helper import create_cnn

def trial_str_creator(trial):
    return f"trial_{trial.trial_id}"

def main(num_samples, max_num_epochs=10, gpus_per_trial=0.5):
    config = {
        "kernel_size": tune.randint(3, 8),
        "intermidiate_channels": tune.randint(1, 5),
        "num_layers": tune.randint(2, 5),
        "thin": tune.choice([True, False]),
        "positional_encoding": tune.choice(["sin", "linear", None]),
        "padding": tune.choice(["reflection", "zero", "replication", None]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    train = create_train(
        create_cnn,
        300,
        None,
        True
    )

    test_best_model = create_test_best(
        create_cnn,
        300,
        None,
        True
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
        local_dir="./data/cnn_raytune",
        name="cnn_segment"
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