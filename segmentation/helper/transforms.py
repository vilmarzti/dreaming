import math
import torch
import functools
import numpy as np

from ..constants import BGR_MEAN, EXPLAINED_VARIANCE_RATIO, PRINCIPAL_COMPONENTS


def compose(*functions):
    """Composes a list of functions

    Returns:
        A function that is the composition of inputs
    """
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
    pad_horizontal = (pad_size[0] - images.shape[3]) // 2
    pad_vertical = (pad_size[1] - images.shape[2]) // 2
    padded = np.pad(images, ((0, 0), (0, 0), (pad_vertical, pad_vertical), (pad_horizontal, pad_horizontal)), mode="reflect")
    return padded

def subtract_mean(images):
    """Subtract the mean some images

    The mean values that have been provided are from scripts/preprocessing/get_mean_rgb.py. 
    Execute it and substitute the values in constants.py

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
    """Thresholds an image to 0, 1.

    Args:
        images (np.ndarray): Matrix (N, H, W) where n is the number of images, H the height, and W the width
        thres (int, optional): [description]. At which value we threshold. Defaults to 128.

    Returns:
        Returns matrix of the same size but the inputs have been thresholded to 0/1
    """
    thresh_images = np.where(images < thres, 0, 1)
    return thresh_images

def add_encoding(images):
    """Takes in RGB images and adds linear and sinus encoding

    Args:
        images (np.ndarray): Matrix (N, H, W, C) where n is the number of images, H the height, W the width , C the number of channels

    Returns:
        Returns a matrix with additiona positional encodings of size (N, H, W, C+6). 
    """
    # get encodings
    num_images = images.shape[0]
    image_size_x = images.shape[3]
    image_size_y = images.shape[2]

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
    """Adds positional encoding in 2d like it was described in "Attention is all you need"

    Code taken from https://github.com/tatp22/multidim-positional-encoding

    Args:
        d_model (int): width of the positional encoding
        height (int): height of the image
        width (int). width of the image

    Returns:
        A matrix of size (4, H, W) where H is the height and W the width. Encodes position in two dimension
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

def merge(image_set, cutoff=0):
    first_split = image_set[:cutoff]
    second_split = image_set[cutoff:]

    # merge first split
    first_split = np.concatenate(first_split, axis=0)
    return np.array([first_split, *second_split])


def random_transform(image_set):
    """Applies different transformations to an (input, output) tuple.

    Args:
        images_set (list[numpy.ndarray]): A list of sections calculated by __getitem__.

    Returns:
        The transformed sections
    """
    image_set = random_flip(image_set)
    image_set = random_rotate(image_set)
    image_set = random_pertubations(image_set)
    image_set = copy_images(image_set)
    return image_set

def copy_images(image_set):
    new_image_set = [i.copy() for i in image_set]
    return new_image_set


def random_flip(image_set):
    """Randomly flips the data along an axis

    Args:
        image_set (list[numpy.ndarray]): Data sections found by __getitem__

    Returns:
        Returns a randomly flipped version of the inputs
    """
    # flip horizontally
    if np.random.rand() > 0.5:
        image_set  = [np.flip(i, 1) for i in image_set]
    
    # flip vertically
    if np.random.rand() > 0.5:
        image_set = [np.flip(i, 2) for i in image_set]

    return image_set

def random_rotate(image_set):
    """Randomly rotations the images by 0, 90, 180 or 270 degrees.

    For this the section have to have equal height and width.

    Args:
        image_set (numpy.ndarray): Image sectios found by __getitem__.
    Returns:
        A randomly flipped version of the images
    """
    k = np.random.randint(4)
    image_set = [np.rot90(i, k, axes=(1, 2)) for i in image_set]
    return image_set

def random_pertubations(image_set):
    """Perturbs the Images with at least 3 channels (are assumed to be BGR) 
    value along the vectors of the most explained variance.

    Taken from the alexNet paper. Look at the paper for implementation details.

    The pca-components and the explained_variance_ratio have to been set correctly.
    Look at the class-description on how to find those.

    Args:
        image_set (list[numpy.ndarray]): Image sections found by __getitem__.
    Returns:
        A randomly perturbed version of the found sections.
    """

    for i in range(len(image_set)):
        image = image_set[i]
        if image.shape[0] >= 3:
            samples_a = np.random.normal(size=3)
            offset = EXPLAINED_VARIANCE_RATIO * samples_a 
            pixel_pertubation = np.array(PRINCIPAL_COMPONENTS).transpose().dot(offset)
            pixel_pertubation = np.expand_dims(pixel_pertubation, axis=(1, 2))
            image[:3] = image[:3] + pixel_pertubation
        image_set[i] = image
    return image_set

def positionalencoding2d_linear(width, height):
    """Add positional encoding by linearly increasing the values from -0.5 to 0.5

    Args:
        width (int): The width of the image.
        height (int): The height of the image

    Returns:
        A matrix of size (2, H, W) where H is the heigt and W the width. 
    """
    x_encoding = np.linspace(-0.5, 0.5, width)
    x_encoding = np.repeat([x_encoding], height, axis=0)

    y_encoding = np.linspace(-0.5, 0.5, height)
    y_encoding = np.transpose([y_encoding])
    y_encoding = np.repeat(y_encoding, width, axis=1)

    encoding = np.stack([x_encoding, y_encoding], axis=2)
    return encoding
