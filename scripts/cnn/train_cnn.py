from segmentation.helper import create_cnn
from segmentation.training import create_train

train = create_train(
    create_cnn,
    300,
    None,
    True,
    False
)

config = {
    "kernel_size": 3,
    "intermidiate_channels": 1,
    "num_layers": 2,
    "thin": True,
    "positional_encoding": None,
    "padding": None,
    "learning_rate": 0.01,
    "batch_size": 32,
}

train(config)