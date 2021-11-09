import cv2
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from segmentation.helper import positionalencoding2d_linear, positionalencoding2d_sin

def normalize(image_channel):
    max_size = np.amax(image_channel, axis=(1, 2))
    max_size = np.expand_dims(max_size, axis=(1, 2))
    image_channel = image_channel / max_size
    return image_channel

def scale_for_sigmoid(image_channel):
    image_channel = (image_channel / 8.0) - 4.0
    return image_channel

class SegmentationDataset(Dataset):
    def __init__(self, input_path, output_path, crop_size, cvt_flag=None, add_encoding=True):
        self.crop_size = crop_size
        self.number_of_crops = (1280 - self.crop_size) * (720 - self.crop_size)

        self.images_input = self.read_images(input_path, cv2.IMREAD_COLOR)
        self.images_output = self.read_images(output_path, cv2.IMREAD_GRAYSCALE)

        # Convert to HSV or Gray
        if cvt_flag:
            self.images_input = [cv2.cvtColor(image, cvt_flag) for image in self.images_input]
            self.images_input = np.array(self.images_input)
        
        # Preprocessing input
        if cvt_flag == cv2.COLOR_BGR2GRAY:
            # Prepare for threshold net
            self.images_input = scale_for_sigmoid(normalize(self.images_input))
            self.images_input = np.expand_dims(self.images_input, axis=3)
        elif cvt_flag == cv2.COLOR_BGR2HSV:
            # Prepare for threshold net
            self.images_input[:,:,:,0] = scale_for_sigmoid(normalize(self.images_input[:,:,:,0]))
            self.images_input[:,:,:,1] = scale_for_sigmoid(normalize(self.images_input[:,:,:,1]))
            self.images_input[:,:,:,2] = scale_for_sigmoid(normalize(self.images_input[:,:,:,2]))
        else:  
            self.images_input = normalize(self.images_input)

        # Prprocessing Output
        self.images_output = np.where(self.images_output < 128, 0, 1)

        if add_encoding:
            # get encodings
            num_images = self.images_input.shape[0]
            lin_encoding = positionalencoding2d_linear(1280, 720)
            lin_encoding = np.repeat([lin_encoding], num_images, axis=0)

            sin_encoding = positionalencoding2d_sin(4, 1280, 720)
            sin_encoding = np.transpose([sin_encoding], [0, 2, 3, 1])
            sin_encoding = np.repeat(sin_encoding, num_images, axis=0)

            # Add encoding
            self.images_input = np.concatenate([self.images_input, lin_encoding, sin_encoding], axis=3)

        self.images_input = np.transpose(self.images_input, [0, 3, 1, 2])

    def __len__(self):
        return self.number_of_crops * len(self.images_input)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_num = idx // self.number_of_crops

        x_value = idx % (720 - self.crop_size)
        y_value = (idx // (720 - self.crop_size)) % (1280 - self.crop_size)

        cropped_input = self.images_input[img_num, :, y_value: y_value + self.crop_size, x_value: x_value + self.crop_size]
        cropped_output = self.images_output[img_num, y_value: y_value + self.crop_size, x_value: x_value + self.crop_size]

        cropped_input = np.single(cropped_input)
        cropped_output = np.single(cropped_output)

        return  cropped_input, cropped_output

    def read_images(self, path, flag):
        image_names = os.listdir(path)
        image_names.sort()
        images = [cv2.imread(os.path.join(path, name), flag) for name in image_names]
        return np.array(images)