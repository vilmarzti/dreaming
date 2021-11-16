from segmentation.helper import create_cnn
from segmentation.training import create_train

train = create_train(
    create_cnn,
    300,
    True,
    False,
    "crop"
)

config = {
  "batch_size": 8,
  "intermidiate_channels": 2,
  "kernel_size": 5,
  "learning_rate": 0.0014633545069953977,
  "num_layers": 4,
  "padding": None,
  "positional_encoding": "linear",
  "thin": False
}

train(config)