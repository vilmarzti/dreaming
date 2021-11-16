"""Implements the dataset classes for torch.utils.data.Dataloader.

The dataset are used for training the UNet and ConvNet classes.
The SegmentationDataset is the parent class of TrainDataset and TestDataset.
TrainDataset reads cropped parts of the loaded images.
TestDataset reads non-overlapping cropped parts of the loaded images.

    Typical usage example:

    train_set = TrainDataset(<input_path>, <ouput_path>, <add_encoding>)
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
"""
import cv2
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from segmentation.helper import positionalencoding2d_linear, positionalencoding2d_sin


class SegmentationDataset(Dataset):
    """The parentclass of TrainDataset and TestDataset.

    Implements reading processing the inputs and segmentation outputs 
    and adds the positional encoding if supplied.

    num_crops_x, num_crops_y, total_crops are 

    Attributes:
        input_path: The folder which contains the inputs.
        output_path: The folder which contains the segmentation.
        add_encoding: Bool whether to add positional encoding to the inputs
        images_input: The processed images that have been read from <input_path> 
        images_output: The processed segmentation mask that have been read form <output_path>
        crop_size_x: The width of the returned image sections
        crop_size_y: The height of the returned image sections
        num_crops_x: The number of crops in the x direction
        num_crops_y: The number of crops in the y direction
        total_crops: The number of crops possible per image
    """
    def __init__(self, input_path, output_path, crop_size, add_encoding=True):
        """Initialize the SegmentationDataset. This is called by the children of this class to load images and segmentations.

        There should be a one to one correspondence between the names in the folder <input_path> and <output_path>.
        For example if there is a file input_folder/0001.png there should be a corresponding output_folder/0001.png

        Args:
            input_path (str): Path to the folder containing the input_images.
                Typically this folder contains files in the form of 0001.png.
            output_path (str): Path to the folder containing the output segmentations.
                Typically this folder contains files in the form of 0001.png
            crop_size (int, tuple[int, int]): The size of the returned image sections should be.
                If it is <int> height and width of the sections are the same.
                If it is tuple then the width is the first element and height the second.
            add_encoding (bool, optional): Whether to add positional encodings to the read input images.
                Defaults to True.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.add_encoding = add_encoding

        # Define the number of crops
        if type(crop_size) is int:
            self.x_crop = crop_size
            self.y_crop = crop_size
        else:
            self.x_crop = crop_size[0]
            self.y_crop = crop_size[1]

        # Initialize the num_crops_attribute to zero
        self.num_crops_x = 0
        self.num_crops_x = 0
        self.total_crops = 0

        self.images_input = self.read_images(input_path, cv2.IMREAD_COLOR)
        self.images_output = self.read_images(output_path, cv2.IMREAD_GRAYSCALE)

        # Make sure that mask has values 0 and 1
        self.images_output = np.where(self.images_output < 128, 0, 1)

        #  Prepare for normal segmentation
        self.images_input = self.subtract_mean(self.images_input)

        if add_encoding:
            # get encodings
            num_images = self.images_input.shape[0]
            lin_encoding = positionalencoding2d_linear(1280, 720)
            # Repeat for the number of images
            lin_encoding = np.repeat([lin_encoding], num_images, axis=0)

            sin_encoding = positionalencoding2d_sin(4, 1280, 720)
            # Put Channels at back
            sin_encoding = np.transpose([sin_encoding], [0, 2, 3, 1])
            # Repeat for the number of images
            sin_encoding = np.repeat(sin_encoding, num_images, axis=0)

            # Add encoding
            self.images_input = np.concatenate([self.images_input, lin_encoding, sin_encoding], axis=3)
        
        # Put channel at second place
        self.images_input = np.transpose(self.images_input, [0, 3, 1, 2])

    def __len__(self):
        """Should be implemented by child calsses
        """
        raise NotImplementedError("SegmentationDataset does not implment a __len__ function. Use one of it's child classes instead")

    def __getitem__(self):
        """Should be implemented by child calsses
        """
        raise NotImplementedError("SegmentationDataset does not implment a __getitem__ function. Use one of it's child classes instead")

    def read_images(self, path, flag):
        """ Reads all the images contained in a folder and converts them to a numpy array.

        Args:
            path (str): Path to the folder which contains images.
            flag (int): Flag that is provided to the cv2.imread function. 
                See the opencv docs for details.

        Returns:
            Returns numpy.ndarray of the shape (n, h, w), where n is the number of read images, h is the height and w is the width.
        """
        image_names = os.listdir(path)
        image_names.sort()
        images = [cv2.imread(os.path.join(path, name), flag) for name in image_names]
        return np.array(images)

    def subtract_mean(self, images):
        """Subtract the mean from the loaded images

        The mean values that have been provided are from scripts/preprocessing/get_mean_rgb.py. 
        Execute it and substitute the values here with the values the script generated.

        Args:
            images (numpy.ndarray): Has the shape (n, h, w), where n is the number of images, h is the height and w is the width.

        Returns:
            An np.array with the same dimensions as <images> where the mean bgr values have been subtracted.
        """
        bgr_mean = [16.10248639, 22.22626978, 27.72004287]
        bgr_mean = np.expand_dims(bgr_mean, axis=(0, 1, 2))

        # Center the BGR values
        images = images - bgr_mean
        return images


class TrainDataset(SegmentationDataset):
    """Implements SegmenationDataset where the outputs are overlapping sections of the original images.

    The Image size is assumed to be (720x1280). Change this number accordingly to your use case.

    This class performs pertubations based on the principal components of the image-dataset.
    Use the script in scripts/preprocessing/compute_pca.py to get the right values for your use-case.

    Attributes:
        random_transform: Bool that indicates whether to add random transformations to the output.
        explained_variance: The eigennumbers of the covariance matrix.
        explained_variance_ratio: <explained_variance_ratio> but scaled such that it sums up to one.
        principal_components: The principal components found with PCA.
    """

    def __init__(self, input_path, output_path, crop_size, add_encoding=True, random_transforms=True):
        """Initializes the Traindataset.

        Args:
            input_path (str): See SegmentationDataset
            output_path (str): See SegmentationDataset
            crop_size (int, tuple[int, int]): See SegmentationDataset
            add_encoding (bool, optional): See SegmentationDataset. Defaults to True.
            random_transforms (bool, optional): [description]. Defaults to True.
        """
        super().__init__(input_path, output_path, crop_size, add_encoding)

        self.random_transforms = random_transforms

        self.num_crops_x = 720 - self.x_crop + 1
        self.num_crops_y = 1280 - self.y_crop + 1

        self.total_crops = self.num_crops_x * self.num_crops_y

        if self.random_transforms:
            # These are the values you get from transforming the rgb pixels into PCA space
            # I used the script in scripts/preprocessing/compute_pca.py to get them
            self.explained_variance_ratio = [0.97247924, 0.02025822, 0.00726254]
            self.explained_variance = [4696.1014082, 97.82694237, 35.07080625]
            self.principal_compoments = [
                [ 0.49318574,  0.58025981,  0.64812529],
                [ 0.73493345,  0.12070754, -0.66730991],
                [-0.46544673,  0.80543668, -0.3669211 ]
            ]

            self.explained_variance = np.array(self.explained_variance)
            self.explained_variance_ratio = np.array(self.explained_variance_ratio)
            self.principal_compoments = np.array(self.principal_compoments)

    def __len__(self):
        """Returns the total number of possible crops in the dataset

        Returns:
            An int with the total number of crops in the dataset
        """
        return self.total_crops* len(self.images_input)
    
    def __getitem__(self, idx):
        """Gets an generic sample (input, output) of the dataset and might perform transformations if <self.transform> is True.

        The sections are continously indexed such that the first (idx==0) section in in the upper left corner of the first image
        in self.input_image. The next section (i.e. idx==1) is on the same height but shifted to right by one pixel.

        Args:
            idx (int): idx in the range [0, total_crops * len(images_input)] maps to a section of the images.

        Returns:
            Returns a tuple (input, output) based on the idx and transforms it based on <transform>
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get Image numbers
        img_num = idx // self.total_crops

        # Compute the upper left position of the crops
        x_value = idx % self.num_crops_x
        y_value = (idx // self.num_crops_x) % self.num_crops_y

        # Crop images
        cropped_input = self.images_input[img_num, :, y_value: y_value + self.y_crop, x_value: x_value + self.x_crop]
        cropped_output = self.images_output[img_num, y_value: y_value + self.y_crop, x_value: x_value + self.x_crop]

        # Convert to single precision float
        cropped_input = np.single(cropped_input)
        cropped_output = np.single(cropped_output)

        # Apply transformations
        if self.random_transforms:
            cropped_input, cropped_output = self.random_transform(cropped_input, cropped_output)

        return cropped_input, cropped_output
    
    def random_transform(self, inputs, outputs):
        """Applies different transformations to an (input, output) tuple.

        Args:
            inputs (numpy.ndarray): An input section calculated by __getitem__.
            outputs (numpy.ndarray): An output section calculated by __getitem__.

        Returns:
            The transformed (input, output) tuple
        """
        inputs, outputs = self.random_flip(inputs, outputs)
        inputs, outputs = self.random_rotate(inputs, outputs)
        inputs, outputs = self.random_pertubations(inputs, outputs)
        inputs, outputs = inputs.copy(), outputs.copy()
        return inputs, outputs
    
    def random_flip(self, inputs, outputs):
        """Randomly flips the inputs and outputs

        Args:
            inputs (numpy.ndarray): Image section found by __getitem__
            outputs (numpy.ndarray): Image section found by __getitem__

        Returns:
            Returns a randomly flipped version of input and output
        """
        # flip horizontally
        if np.random.rand() > 0.5:
            inputs = np.flip(inputs, 1)
            outputs = np.flip(outputs, 0)
        
        # flip vertically
        if np.random.rand() > 0.5:
            inputs = np.flip(inputs, 2)
            outputs = np.flip(outputs, 1)
        return inputs, outputs

    def random_rotate(self, inputs, outputs):
        """Randomly rotations the images by 0, 90, 180 or 270 degrees.

        For this the section have to have equal height and width.

        Args:
            inputs (numpy.ndarray): Image section found by __getitem__.
            outputs (numpy.ndarray): Image section found by __getitem__.

        Returns:
            A randomly flipped version of inputs and outputs 
        """
        k = np.random.randint(4)
        inputs = np.rot90(inputs, k, axes=(1, 2))
        outputs = np.rot90(outputs, k, axes=(0, 1))
        return inputs, outputs

    def random_pertubations(self, inputs, outputs):
        """Perturbs the pixel value along the vectors of the most explained variance.

        Taken from the alexNet paper. Look at the paper for implementation details.

        The pca-components and the explained_variance_ratio have to been set correctly.
        Look at the class-description on how to find those.

        Args:
            inputs (numpy.ndarray): Image section found by __getitem__.
            outputs (numpy.ndarray): Image section found by __getitem__.

        Returns:
            A randomly perturbed version of the found sections.
        """
        samples_a = np.random.normal(size=3)
        offset = self.explained_variance_ratio * samples_a 
        pixel_pertubation = self.principal_compoments.transpose().dot(offset)
        pixel_pertubation = np.expand_dims(pixel_pertubation, axis=(1, 2))
        inputs[:3] = inputs[:3] + pixel_pertubation
        return inputs, outputs

class TestDataset(SegmentationDataset):
    """Implements SegmenationDataset where the outputs are non-overlapping sections of the original images.

    The Image size is assumed to be (720x1280). Change this number accordingly to your use case.
    """
    def __init__(self, input_path, output_path, crop_size, add_encoding=True):
        """Initializes the TestDataset by calling its parent class and setting the appropriate number of crops

        Args:
            input_path (str): See SegmentationDataset
            output_path (str): See SegmentationDataset
            crop_size (int, tuple[int, int]): See SegmentationDataset
            add_encoding (bool, optional): See SegmentationDataset. Defaults to True.
        """
        super().__init__(input_path, output_path, crop_size, add_encoding=add_encoding)

        self.num_crops_x = 720 // self.x_crop
        self.num_crops_y = 1280 // self.y_crop

        self.total_crops = self.num_crops_x * self.num_crops_y
    
    def __len__(self):
        """Returns the number of total possible crops in the dataset
        """
        return self.total_crops * len(self.images_input)
    
    def __getitem__(self, idx):
        """Returns a section of an image based on idx.

        The section is non-overlapping with other sections that get accessed with idx. 
        
        The section are index in such a way that the first section (idx=0) is in the upper left in the first image.
        The next section (idx=1) is shifted to the right by the <x_crop> until there is no space in the image left
        where we wrap around to the next row of sections.

        Args:
            idx (int): Int in range [0, len(self)]

        Returns:
            A tuple (inputs, outputs) that are cut from an image in the dataset. There is a one-to-one correspondence between these two.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_num = idx // self.total_crops
        index_x = (idx % self.num_crops_x) * self.x_crop
        index_y = ((idx % self.total_crops) // self.num_crops_x) * self.y_crop

        inputs = self.images_input[img_num, :, index_y: index_y + self.y_crop, index_x : index_x + self.y_crop]
        outputs = self.images_output[img_num, index_y: index_y + self.y_crop, index_x : index_x + self.y_crop]

        inputs = np.single(inputs)
        outputs = np.single(outputs)

        return inputs, outputs
