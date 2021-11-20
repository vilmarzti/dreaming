import argparse
import os
import cv2

import numpy as np
import multiprocessing as mp

from itertools import repeat
from os import path

# Steps referenced from "Evaluation of Background Subtraction Algorithms with Post-Processing"
def postprocessing(foreground_mask):
    foreground_mask = noise_removal_parallel(foreground_mask)
    foreground_mask = morphological_closing(foreground_mask)
    foreground_mask = area_thresholding(foreground_mask)
    return foreground_mask

# Does noise removal in parallel (splitting the image into smaller sub images)
# see the corresponding <noise_removal> procedure below
def noise_removal_parallel(foreground_mask, threshold=3):
    # num_split must divide x and y  without overlap
    num_splits = 8
    split_size_x = foreground_mask.shape[1] // num_splits
    split_size_y = foreground_mask.shape[0] // num_splits

    # Split images in x and y direction
    split_y_images = np.split(foreground_mask, num_splits, axis=0)
    split_images = np.array([np.split(split, num_splits, axis=1) for split in split_y_images])
    split_images = np.reshape(split_images, (-1, split_images.shape[2], split_images.shape[3]))
    
    # Apply noise removal to each of split
    pool = mp.Pool(8)
    new_masks = pool.starmap(noise_removal, zip(split_images, repeat(threshold)))
    pool.close()
    pool.join()

    # put it all together
    new_mask = np.zeros_like(foreground_mask)
    for y in range(num_splits):
        for x in range(num_splits):
            idx = y * num_splits + x
            new_mask[y * split_size_y: (y+1) * split_size_y, x * split_size_x: (x + 1) * split_size_x] = new_masks[idx]
    return new_mask


# Removes noise from a mask by computing the number of active pixel in the 8-connected neighbourhood
# of each pixel. If there are more than <threshold> active pixels in the neighbourhood it keeps that pixel
def noise_removal(foreground_mask, threshold=3):
    new_mask = np.zeros_like(foreground_mask)
    for y in range(foreground_mask.shape[0]):
        for x in range(foreground_mask.shape[1]):
            # Dont process if pixel is not foreground
            if foreground_mask[y, x] == 0:
                continue

            # find 8 connected neighbour indices
            neighbour_left_x = max(0, x - 1)
            neighbour_right_x = min(foreground_mask.shape[1] - 1, x + 1)

            neighbour_top_y = max(0, y - 1)
            neighbour_bot_y = min(foreground_mask.shape[0] - 1, y + 1)

            # count neighbours
            count_forground = 0
            num_neighbours = 0
            for i in range(neighbour_top_y, neighbour_bot_y + 1):
                for j in range(neighbour_left_x, neighbour_right_x + 1):
                    count_forground = count_forground + 1 if foreground_mask[i, j] == 255 else count_forground
                    num_neighbours += 1
            
            # the pixel itself is counted as well
            count_forground += -1

            if count_forground > threshold * (num_neighbours / 8):
                new_mask[y, x] = 255
    
    return new_mask 

# Apply morphological closing with a kernel of size <kernel> with
def morphological_closing(foreground_mask, kernel_width=19):
    new_mask = foreground_mask.copy()

    kernel = np.ones((kernel_width, kernel_width))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)

    return new_mask 

# Apply area threshold where all blobs with less than <alpha> pixel are discarded
def area_thresholding(foreground_mask, alpha=0.01):
    min_num_pixels = foreground_mask.shape[0] * foreground_mask.shape[1] * alpha
    _, contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = []

    # Find blobs and save those that have the appropriate size
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_num_pixels:
            big_contours.append(contour)
    
    img = np.zeros_like(foreground_mask)
    img = cv2.drawContours(img, big_contours, -1, 255, -1)
    return img 

# This gives a feedback of the mask to the GMM-backsub algorithm
def perturb_image(img, mask):
    # Perturb foreground with random pixels
    random_pixels = np.random.randint(0, 256, size=(img.shape), dtype=np.uint8)
    foreground_perturbed = cv2.bitwise_or(random_pixels, random_pixels, mask=mask)

    # Get normal backgground
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_or(img, img, mask=background_mask)

    # Combine random foreground with background
    combined = cv2.bitwise_or(foreground_perturbed, background)
    return combined


def background_substract(image_paths, first_background):
    # Set up MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    bg_subtractor.setHistory(len(image_paths))

    # Read first image
    img = cv2.imread(image_paths[0])
    bg_subtractor.apply(img)
    masks = []

    # Add a first background image to bootstrap the feedback loop
    fg_mask_post = cv2.imread(first_background, cv2.IMREAD_GRAYSCALE)
    fg_mask_post = np.where(fg_mask_post < 127, 0, 255).astype(np.uint8)
    masks.append(fg_mask_post)

    # Go through the remaining images
    for i, path in enumerate(image_paths[1:]):
        # MOG2 has no way to give feedback with the previous mask
        # So I replace the foreground with pixels with random colors
        # This way the GMM cant learn the rather static foreground as background
        # I also erode the foreground with a 10x10 kernel to adjust for any movement that happens
        fg_mask_post = cv2.erode(fg_mask_post, np.ones((10, 10)))
        img = cv2.imread(path)
        img_per = perturb_image(img, fg_mask_post)
        
        # apply backsub algorithm
        fg_mask = bg_subtractor.apply(img_per)

        # Do some post-processing of the foreground mask
        # See the comment in the function <postporcessing> for the referenced papaer
        fg_mask_post = postprocessing(fg_mask)

        masks.append(fg_mask_post)
        print(f"Processed image {i + 2} from {len(image_paths)}")

    return masks


def process_folder(input_path, reference_mask_path, output_path):
    # Get all the images from input_path
    images = os.listdir(input_path)
    images.sort()

    # get all masks that are supplied
    reference_masks = os.listdir(reference_mask_path)
    reference_masks.sort()

    # Set-up paths for reading and writing
    image_paths = [path.join(input_path, x) for x in images]

    # create slices with reference mask as start and end 
    # with the position before the next reference mask as the end
    start_indices = []
    found_references = []
    for reference_mask in reference_masks:
        # Get index of refernce mask in images array
        idx = np.asarray(np.array(images) == reference_mask).nonzero()[0]
        if len(idx) == 1:
            start_indices.append(idx[0])
            found_references.append(reference_mask)

    # Get the indices with which each segment ends
    end_indices = start_indices[1:]
    end_indices.append(len(images))

    # Start with the next image
    start_indices[1:] = np.array(start_indices)[1:] + 1

     # Create output folder if does not exists
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    # Go through each segment that starts with a reference mask 
    # and ends before the next reference mask
    for i, (start, end, r_mask) in enumerate(zip(start_indices, end_indices, found_references)):
        print(f"Starting with segment {i + 1} ({start}:{end}) of {len(start_indices)}")

        r_path = path.join(reference_mask_path, r_mask)
        images_segment = image_paths[start: end]
        masks = background_substract(images_segment, r_path)

        # save calculated masks
        for i, image_name in enumerate(images_segment):
            mask_path = path.join(output_path, path.basename(image_name))
            cv2.imwrite(mask_path, masks[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Creates masks using Background-Subtraction with GMM
        This algorithm reads a set of input images specified by the input parameter
        and an inital foreground mask that corresponds to the first image. 
        It then propagates the calculated/given foreground mask through the timeseries.
        Finally writes computed masks to the specified output folder.
    """)

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
 
    parser.add_argument("--background-images", "-b",
        help="Path to the background segments. This should include the first image as well.",
        required=True)   

    parser.add_argument("--output-path", "-o",
        help="The folder the masks are writen to",
        required=True
    )

    args = parser.parse_args()
    process_folder(args.input_path, args.background_images, args.output_path)