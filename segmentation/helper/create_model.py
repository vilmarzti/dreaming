from segmentation.models import CNNSegmentation, RangeImage, UNet


def create_cnn(config):
    # CNN params
    input_channels = 3
    kernel_size = config["kernel_size"]
    intermidiate_channels = config["intermidiate_channels"]
    num_layers = config["num_layers"]
    thin =  config["thin"]
    positional_encoding = config["positional_encoding"]
    padding = config["padding"]

    net = CNNSegmentation(kernel_size, input_channels, intermidiate_channels, num_layers, thin, positional_encoding, padding)

    return net

def create_rangeimage(config):
    input_channels = 3
    kernel_size = config["kernel_size"]

    net = RangeImage(input_channels, kernel_size)
    return net

def create_unet(config):
    input_channels = 3
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