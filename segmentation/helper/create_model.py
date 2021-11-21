from segmentation.models import CNNSegmentation, UNet
from segmentation.models.ensemble import Ensemble


def create_cnn(config):
    # CNN params
    input_channels = config["input_channels"]
    kernel_size = config["kernel_size"]
    intermidiate_channels = config["intermidiate_channels"]
    num_layers = config["num_layers"]
    thin =  config["thin"]
    positional_encoding = config["positional_encoding"]
    padding = config["padding"]

    net = CNNSegmentation(kernel_size, input_channels, intermidiate_channels, num_layers, thin, positional_encoding, padding)

    return net

def create_unet(config):
    input_channels = config["input_channels"]
    deepness = config["deepness"]
    starting_multiplier = config["starting_multiplier"]
    use_thin = config["use_thin"]
    positional_encoding = config["positional_encoding"]

    net = UNet(
        input_channels,
        deepness,
        starting_multiplier,
        use_thin,
        positional_encoding
    )

    return net

def create_ensemble(config):
    input_channels = config["input_channels"]
    kernel_size = config["kernel_size"]

    ensemble = Ensemble(input_channels, kernel_size)
    return ensemble