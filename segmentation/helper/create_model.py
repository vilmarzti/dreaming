from segmentation.cnn import CNNSegmentation

def create_cnn(config):
    # CNN params
    kernel_size = config["kernel_size"]
    input_channels = 3
    intermidiate_channels = config["intermidiate_channels"]
    num_layers = config["num_layers"]
    thin =  config["thin"]
    positional_encoding = config["positional_encoding"]
    padding = config["padding"]

    net = CNNSegmentation(kernel_size, input_channels, intermidiate_channels, num_layers, thin, positional_encoding, padding)

    return net