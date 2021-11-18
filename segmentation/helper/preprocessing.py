import math
import torch
import functools
import numpy as np

from ..constants import BGR_MEAN


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def pad_reflect(images, pad_size):
    """ Pad with reflection to a certain size

    Args:
        images (numpy.ndarray): Has the shape (n, c, h, w) where n is the number of images, c are the channels, h is the height and w is the width
            Reflect_pads along the last two axis.
        pad_size (tuple[int, int]): To which size to pad. Pad should be greater than (w, h)
    Returns:
        A numpy array with the shape (n, c, pad_y, pad_x)
    """
    pad_right = pad_size[0] - images.shape[3]
    pad_bot = pad_size[1] - images.shape[2]
    padded = np.pad(images, ((0, 0), (0, 0), (0, pad_bot), (0, pad_right)), mode="reflect")
    return padded


def subtract_mean(images):
    """Subtract the mean some images

    The mean values that have been provided are from scripts/preprocessing/get_mean_rgb.py. 
    Execute it and substitute the values here with the values the script generated.

    Args:
        images (numpy.ndarray): Has the shape (n, h, w), where n is the number of images, h is the height and w is the width.

    Returns:
        An np.array with the same dimensions as <images> where the mean bgr values have been subtracted.
    """
    bgr_mean = BGR_MEAN
    bgr_mean = np.expand_dims(bgr_mean, axis=(0, 2, 3))

    # Center the BGR values
    images = images - bgr_mean
    return images

def threshold(images, thres=128):
    thresh_images = np.where(images < thres, 0, 1)
    return thresh_images

def add_encoding(images):
    # get encodings
    num_images = images.shape[0]
    image_size_x = images.shape[2]
    image_size_y = images.shape[1]

    # Create linear encoding with channels at the second position
    lin_encoding = positionalencoding2d_linear(image_size_x, image_size_y)
    lin_encoding = np.repeat([lin_encoding], num_images, axis=0)
    lin_encoding = np.transpose(lin_encoding, (0, 3, 1, 2))

    # Create sinus encoding with channels at the second position
    sin_encoding = positionalencoding2d_sin(4, image_size_x, image_size_y)
    sin_encoding = np.repeat([sin_encoding], num_images, axis=0)

    # Add encoding
    images_with_encoding = np.concatenate([images, lin_encoding, sin_encoding], axis=1)
    return images_with_encoding

def positionalencoding2d_sin(d_model, width, height):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.detach().numpy()

    return pe

def positionalencoding2d_linear(width, height):
    x_encoding = np.linspace(-0.5, 0.5, width)
    x_encoding = np.repeat([x_encoding], height, axis=0)

    y_encoding = np.linspace(-0.5, 0.5, height)
    y_encoding = np.transpose([y_encoding])
    y_encoding = np.repeat(y_encoding, width, axis=1)

    encoding = np.stack([x_encoding, y_encoding], axis=2)
    return encoding




