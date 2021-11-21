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
        preprocess: List of (composed) functions that are applied after reading a set
        crop_size_x: The width of the returned image sections
        crop_size_y: The height of the returned image sections
        num_images: The number of images tuples we have
        num_crops_x: The number of crops in the x direction
        num_crops_y: The number of crops in the y direction
        image_size: Tuple (width, height) of the images
        total_crops: The number of crops possible per image
    """
    def __init__(self, paths, crop_size, read_flags=[], preprocess=[], transform=None):
        """Initialize the SegmentationDataset. This is called by the children of this class to load images and segmentations.

        There should be a one to one correspondence between the names in the folder <input_path> and <output_path>.
        For example if there is a file input_folder/0001.png there should be a corresponding output_folder/0001.png.

        Args:
            paths (list(str)): List with the paths to the data
                The file-names in each path sould have a one-to-one correspondence.
                i.e. for a filename "0001.png" in a path there should be a corresponding "000.1.png" in the other folders.
            crop_size (int, tuple[int, int]): The size of the returned image sections should be.
                If it is <int> height and width of the sections are the same.
                If it is tuple then the width is the first element and height the second.
            read_flags (list[int]): A list of cv2.imread flags with which to read the files in the corresponding paths.
            preprocess (list(function), optional): A list of functions that should be applied to each path loading them.
            transform (function, optional): A transformation that is applied after getting the input. Defaults to None.
                Defaults to [].
        """
        self.paths = paths
        self.preprocess = preprocess
        self.transform = transform

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
        for i in range(len(self.paths)):
            read_flag = read_flags[i] if len(read_flags) == len(self.paths) else None
            images_path = self.paths[i]
            self.images_list.append(self.read_images(images_path, read_flag))
        
        self.num_images = self.images_list[0].shape[0]

        # Apply preprocessing
        if len(preprocess) == len(self.paths):
            for i in range(len(self.images_list)):
                self.images_list[i] = preprocess[i](self.images_list[i])
        
        # Set the image size (x, y) after preprocessing
        # It's assumed that all images have the same dimensions
        self.image_size = (self.images_list[0].shape[3], self.images_list[0].shape[2])

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
        if all(".png" in image for image in image_names):
            images = [cv2.imread(os.path.join(path, name), flag) for name in image_names]
        elif all(".npy" in image for image in image_names):
            images = [np.load(os.path.join(path, name)) for name in image_names]
        
        images = np.array(images)

        # Place the channels at the appropriate position for torch
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
        cropped_set = []
        for i in range(len(self.images_list)):
            images = self.images_list[i]
            cropped = images[img_idx, ..., upper_left[1]: upper_left[1] + self.y_crop, upper_left[0]: upper_left[0] + self.x_crop]
            cropped_set.append(cropped)
        return cropped_set

class TrainDataset(SegmentationDataset):
    """Implements SegmenationDataset where the outputs are overlapping sections of the original images.

    This class performs pertubations based on the principal components of the image-dataset.
    Use the script in scripts/preprocessing/compute_pca.py to get the right values for your use-case
    and replace the ones in constants.py
    """

    def __init__(self, paths, crop_size, read_flags=[], preprocess=[], transform=[]):
        """Initializes the Traindataset.

        Args:
            paths (list[str]): See SegmentationDataset
            crop_size (int, tuple[int, int]): See SegmentationDataset
            read_flags (list[int]): See SegmentationDataset
            preprocess (list[function], optional): See SegmentationDataset. Defaults to [].
            Defaults to True.
        """
        super().__init__(paths, crop_size, read_flags, preprocess, transform)

        self.num_crops_x = self.image_size[0] - self.x_crop + 1
        self.num_crops_y = self.image_size[1] - self.y_crop + 1

        self.total_crops = self.num_crops_x * self.num_crops_y

    def __len__(self):
        """Returns the total number of possible crops in the dataset

        Returns:
            An int with the total number of crops in the dataset
        """
        return self.total_crops * self.num_images
    
    def __getitem__(self, idx):
        """Gets an generic sample from <self.images> dataset and might perform transformations if <self.transform> is True.

        The sections are continously indexed such that the first (idx==0) section in in the upper left corner of an image. 
        The next section (i.e. idx==1) is on the same height but shifted to right by one pixel.

        Args:
            idx (int): idx in the range [0, total_crops * num_images] maps to a section of the images.

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

        # Apply the transforms to each output
        if self.transform:
            transformed = self.transform(cropped)

        return transformed
    

class TestDataset(SegmentationDataset):
    """Implements SegmenationDataset where the outputs are non-overlapping sections of the original images.
    """
    def __init__(self, paths, crop_size, read_flags=[], preprocess=[], transform=[]):
        """Initializes the TestDataset by calling its parent class and setting the appropriate number of crops

        Args:
            paths (list(str)): See SegmentationDataset
            crop_size (int, tuple[int, int]): See SegmentationDataset
            read_flags (list[int]): See SegmentationDataset
            preprocess (list(function), optional): See SegmentationDataset

        """
        super().__init__(paths, crop_size, read_flags, preprocess, transform)

        self.num_crops_x = self.image_size[0] // self.x_crop
        self.num_crops_y = self.image_size[1] // self.y_crop

        self.total_crops = self.num_crops_x * self.num_crops_y
    
    def __len__(self):
        """Returns the number of total possible crops in the dataset
        """
        return self.total_crops * self.num_images
    
    def __getitem__(self, idx):
        """Returns a section of an image based on idx.

        The accessed section is non-overlapping with other sections that could get accessed 
        
        The section are indexed in such a way that the first section (idx=0) is in the upper left in the first image.
        The next section (idx=1) is shifted to the right by <x_crop> until there is no space in the image left.
        Then we wrap around (one pixel down) to the next row of sections.

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

        # Apply the transforms to each output
        if self.transform:
            transformed = self.transform(cropped)

        return transformed 
