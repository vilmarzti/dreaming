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

from ..constants import EXPLAINED_VARIANCE, EXPLAINED_VARIANCE_RATIO, IMAGE_SIZE_X, IMAGE_SIZE_Y, PRINCIPAL_COMPONENTS


class SegmentationDataset(Dataset):
    """The parentclass of TrainDataset and TestDataset.

    Implements reading processing the inputs and segmentation outputs 
    and adds the positional encoding if supplied.

    num_crops_x, num_crops_y, total_crops are 

    Attributes:
        paths: List of Paths to the folders with the data
        images_list: The processed images read from the paths. Elements 
            should have size (N, C, H, W), 
                where N is the index of the list
                C is any number of channels
                H is the height of the images
                W is the width
        add_encoding: Bool whether to add positional encoding to the inputs
        crop_size_x: The width of the returned image sections
        crop_size_y: The height of the returned image sections
        num_crops_x: The number of crops in the x direction
        num_crops_y: The number of crops in the y direction
        total_crops: The number of crops possible per image
    """
    def __init__(self, paths, crop_size, read_flags=[], preprocess=[]):
        """Initialize the SegmentationDataset. This is called by the children of this class to load images and segmentations.

        There should be a one to one correspondence between the names in the folder <input_path> and <output_path>.
        For example if there is a file input_folder/0001.png there should be a corresponding output_folder/0001.png

        Args:
            paths (list(str)): List with the paths to the data
                The file-names in each path sould have a one-to-one correspondence.
                i.e. for a filename "0001.png" in a path there should be a corresponding "000.1.png" in the other folders
            crop_size (int, tuple[int, int]): The size of the returned image sections should be.
                If it is <int> height and width of the sections are the same.
                If it is tuple then the width is the first element and height the second.
            read_flags (int): A list of cv2.imread flags with which to read the files in the corresponding paths.
            preprocess (list(function), optional): A list of functions that should be applied to each path loading them
                Defaults to [].
        """
        self.paths = paths

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

        # Read all images with the appropriate flag
        self.images_list = []
        for i in self.paths:
            read_flag = read_flags[i] if len(read_flags) == len(self.paths) else None
            images_path = self.paths[i]
            self.images_list.append(self.read_images(images_path, read_flag))

        # Apply preprocessing
        if len(preprocess) == len(self.paths):
            for i in range(self.images_list):
                self.images_list[i] = preprocess[i](self.images_list[i])

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
            Returns a numpy.ndarray of the shape (n, c, h, w), where n is the number of read images, c is the channels, h is the height and w is the width.
        """
        image_names = os.listdir(path)
        image_names.sort()
        # Read all the images
        images = [cv2.imread(os.path.join(path, name), flag) for name in image_names]
        images = np.array(images)

        # Place the channels at the appropriate decision
        if len(images.shape) == 3:
            images = np.expand_dims(images, 1)
        elif len(images.shape) == 4:
            images = np.transpose(images, [0, 3, 1, 2])

        return images
    
    def crop_position(self, img_idx, upper_left):
        """Crops images from a given position.

        Args:
            img_idx (int): Int that tells us the position of the image we want to crop
            upper_left (tuple[int, int]): The upper left corner given in (x, y) coordinates
        Returns:
            Return a list with images that have been cropped
        """
        cropped = []
        for i in range(self.images_list):
            images = self.images_list[i]
            cropped = images[img_idx, ..., upper_left[1]: upper_left[1] + self.y_crop, upper_left[0]: upper_left[0] + self.x_crop]
        return cropped

       


class TrainDataset(SegmentationDataset):
    """Implements SegmenationDataset where the outputs are overlapping sections of the original images.

    The Image size is assumed to be (720x1280). Change this number accordingly to your use case.

    This class performs pertubations based on the principal components of the image-dataset.
    Use the script in scripts/preprocessing/compute_pca.py to get the right values for your use-case.

    Attributes:
        random_transforms: List that indicates which imagessets to transfroms
        explained_variance: The eigennumbers of the covariance matrix.
        explained_variance_ratio: <explained_variance_ratio> but scaled such that it sums up to one.
        principal_components: The principal components found with PCA.
    """

    def __init__(self, paths, crop_size, preprocess=[], random_transforms=True):
        """Initializes the Traindataset.

        Args:
            paths (list[str]): See SegmentationDataset
            crop_size (int, tuple[int, int]): See SegmentationDataset
            preprocess (list[function], optional): See SegmentationDataset. Defaults to True.
            random_transforms (bool, optional): Bool whether to apply random transformations
            Defaults to True.
        """
        super().__init__(paths, crop_size, preprocess)

        self.random_transforms = random_transforms

        self.num_crops_x = IMAGE_SIZE_X - self.x_crop + 1
        self.num_crops_y = IMAGE_SIZE_Y - self.y_crop + 1

        self.total_crops = self.num_crops_x * self.num_crops_y

        # These are the values you get from transforming the rgb pixels into PCA space
        # I used the script in scripts/preprocessing/compute_pca.py to get them
        self.explained_variance = EXPLAINED_VARIANCE
        self.explained_variance_ratio = EXPLAINED_VARIANCE_RATIO
        self.principal_compoments = PRINCIPAL_COMPONENTS

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
        """Gets an generic sample from <self.images> dataset and might perform transformations if <self.transform> is True.

        The sections are continously indexed such that the first (idx==0) section in in the upper left corner of an image. 
        The next section (i.e. idx==1) is on the same height but shifted to right by one pixel.

        Args:
            idx (int): idx in the range [0, total_crops * len(images_input)] maps to a section of the images.

        Returns:
            Returns a list of cropped images based on the idx and <self.transforms>
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get Image numbers
        img_num = idx // self.total_crops

        # Compute the upper left position of the crops
        x_value = idx % self.num_crops_x
        y_value = (idx // self.num_crops_x) % self.num_crops_y

        # Crop images
        cropped = self.crop_position(img_num, (x_value, y_value))

        # Convert to single precision float
        cropped = [np.single(c) for c in cropped]

        if self.random_transforms:
            transformed = self.random_transform(cropped)

        return transformed
    
    def random_transform(self, image_set):
        """Applies different transformations to an (input, output) tuple.

        Args:
            images_set (list[numpy.ndarray]): A list of sections calculated by __getitem__.

        Returns:
            The transformed sections
        """
        image_set = self.random_flip(image_set)
        image_set = self.random_rotate(image_set)
        image_set = self.random_pertubations(image_set)
        image_set = [i.copy() for i in image_set]
        return image_set
    
    def random_flip(self, image_set):
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

    def random_rotate(self, image_set):
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

    def random_pertubations(self, image_set):
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

        self.num_crops_x = IMAGE_SIZE_X // self.x_crop
        self.num_crops_y = IMAGE_SIZE_Y // self.y_crop

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
            A list of cropped sections
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Compute position of non-overlapping section
        img_num = idx // self.total_crops
        index_x = (idx % self.num_crops_x) * self.x_crop
        index_y = ((idx % self.total_crops) // self.num_crops_x) * self.y_crop

        # Crop at position
        cropped = self.crop_position(img_num, (index_x, index_y))

        # Convert to single float
        cropped = [np.single(i) for i in cropped]

        return cropped
