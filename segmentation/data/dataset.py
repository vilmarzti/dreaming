import cv2
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from sklearn.decomposition import PCA

from segmentation.helper import positionalencoding2d_linear, positionalencoding2d_sin

def normalize(image_channel):
    max_size = np.amax(image_channel, axis=(1, 2))
    max_size = np.expand_dims(max_size, axis=(1, 2))
    image_channel = image_channel / max_size
    return image_channel

def scale_for_sigmoid(image_channel):
    image_channel = (image_channel / 8.0) - 4.0
    return image_channel

def subtract_mean(images):
    # Assumes that the last dimension are the channesl that represent BGR values
    bgr_mean = np.mean(images, axis=(0, 1, 2))
    bgr_mean = np.expand_dims(bgr_mean, axis=(0, 1, 2))

    # Center the BGR values
    images = images - bgr_mean
    return images

class CNNDataset(Dataset):
    def __init__(self, input_path, output_path, cvt_flag=None, add_encoding=True):
        self.input_path = input_path
        self.output_path = output_path
        self.add_encoding = add_encoding
        self.cvt_flag = cvt_flag

        self.images_input = self.read_images(input_path, cv2.IMREAD_COLOR)
        self.images_output = self.read_images(output_path, cv2.IMREAD_GRAYSCALE)

        # Make sure that mask has values 0 and 1
        self.images_output = np.where(self.images_output < 128, 0, 1)

        # Convert to HSV or Gray
        if cvt_flag:
            self.images_input = [cv2.cvtColor(image, cvt_flag) for image in self.images_input]
            self.images_input = np.array(self.images_input)
        
        # Preprocessing for threshold net
        if cvt_flag == cv2.COLOR_BGR2GRAY:
            # Prepare for threshold net
            self.images_input = scale_for_sigmoid(normalize(self.images_input))
            self.images_input = np.expand_dims(self.images_input, axis=3)

        # Prepare for threshold net
        elif cvt_flag == cv2.COLOR_BGR2HSV:
            self.images_input[:,:,:,0] = scale_for_sigmoid(normalize(self.images_input[:,:,:,0]))
            self.images_input[:,:,:,1] = scale_for_sigmoid(normalize(self.images_input[:,:,:,1]))
            self.images_input[:,:,:,2] = scale_for_sigmoid(normalize(self.images_input[:,:,:,2]))

        #  Prepare for normal segmentation
        else:  
            self.images_input = subtract_mean(self.images_input)

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
        pass

    def __getitem__(self):
        pass

    def read_images(self, path, flag):
        image_names = os.listdir(path)
        image_names.sort()
        images = [cv2.imread(os.path.join(path, name), flag) for name in image_names]
        return np.array(images)

class TrainDataset(CNNDataset):
    def __init__(self, input_path, output_path, crop_size, cvt_flag=None, add_encoding=True, random_transforms=True):
        super().__init__(input_path, output_path, cvt_flag, add_encoding)

        self.random_transforms = random_transforms and not cvt_flag

        # Define the number of crops
        if type(crop_size) is int:
            self.x_crop = crop_size
            self.y_crop = crop_size
        else:
            self.x_crop = crop_size[0]
            self.y_crop = crop_size[1]
        
        self.num_crops_x = 720 - self.x_crop + 1
        self.num_crops_y = 1280 - self.y_crop + 1

        self.total_crops = self.num_crops_x * self.num_crops_y

        if self.random_transforms:
            # Prepare PCA for the random pixel pertubations
            b_values = self.images_input[:,0].flatten()
            g_values = self.images_input[:,1].flatten()
            r_values = self.images_input[:,2].flatten()

            bgr_values = np.vstack([b_values, g_values, r_values]).transpose()
            pca = PCA(n_components=3)
            pca.fit(bgr_values)

            self.pca_components = pca.components_
            self.eigen_values = pca.explained_variance_

    def __len__(self):
        return self.total_crops* len(self.images_input)
    
    def __getitem__(self, idx):
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

        if self.random_transforms:
            cropped_input, cropped_output = self.random_transform(cropped_input, cropped_output)

        return cropped_input, cropped_output
    
    def random_transform(self, inputs, outputs):
        inputs, outputs = self.random_flip(inputs, outputs)
        inputs, outputs = self.random_pertubations(inputs, outputs)
        return inputs, outputs
    
    def random_flip(self, inputs, outputs):
        # flip horizontally
        if np.random.rand() > 0.5:
            inputs = np.flip(inputs, 1)
            outputs = np.flip(outputs, 0)
        
        # flip vertically
        if np.random.rand() > 0.5:
            inputs = np.flip(inputs, 2)
            outputs = np.flip(outputs, 1)
        return inputs, outputs

    # Taken from the alexNet paper
    def random_pertubations(self, inputs, outputs):
        samples_a = np.random.normal(size=3)
        offset = self.eigen_values * samples_a 
        pixel_pertubation = self.pca_components.transpose().dot(offset)
        pixel_pertubation = np.expand_dims(pixel_pertubation, axis=(1, 2))
        inputs[:3] = inputs[:3] + pixel_pertubation
        return inputs, outputs

class TestDataset(CNNDataset):
    def __init__(self, input_path, output_path, split_factor, cvt_flag=None, add_encoding=True):
        super().__init__(input_path, output_path, cvt_flag=cvt_flag, add_encoding=add_encoding)

        self.split_factor = split_factor
        self.splits_per_image = split_factor ** 2

        self.split_x_size = 720 // split_factor
        self.split_y_size = 1280 // split_factor
    
    def __len__(self):
        return self.splits_per_image * len(self.images_input)
    
    def __getitem__(self, idx):

        img_num = idx // self.splits_per_image
        index_x = (idx % self.split_factor) * self.split_x_size
        index_y = ((idx // self.split_factor) % self.split_factor) * self.split_y_size

        inputs = self.images_input[img_num, :, index_y: index_y + self.split_y_size, index_x : index_x + self.split_x_size]
        outputs = self.images_output[img_num, index_y: index_y + self.split_y_size, index_x : index_x + self.split_x_size]

        inputs = np.single(inputs)
        outputs = np.single(outputs)

        return inputs, outputs

