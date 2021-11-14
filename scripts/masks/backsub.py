import argparse
import os
import cv2


import numpy as np
import multiprocessing as mp

from itertools import repeat
from os import path

# Steps referenced from Evaluation of Background Subtraction Algorithms with Post-Processing
def postprocessing(foreground_mask):
    foreground_mask = noise_removal_parallel(foreground_mask)
    foreground_mask = noise_removal(foreground_mask)
    foreground_mask = morphological_closing(foreground_mask)
    foreground_mask = area_thresholding(foreground_mask)
    return foreground_mask

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

    # put it all together
    new_mask = np.zeros_like(foreground_mask)
    for y in range(num_splits):
        for x in range(num_splits):
            idx = y * num_splits + x
            new_mask[y * split_size_y: (y+1) * split_size_y, x * split_size_x: (x + 1) * split_size_x] = new_masks[idx]
    return new_mask


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
    contours, hier = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    combined = cv2.bitwise_or(foreground_perturbed, background)
    return combined


def background_substract(image_paths):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    bg_subtractor.setHistory(1500)

    img = cv2.imread(image_paths[0])

    bg_subtractor.apply(img)
    fg_mask = cv2.imread("/home/martin/Pictures/bf/segmentation/train_output/0001.png", cv2.IMREAD_GRAYSCALE)
    fg_mask = np.where(fg_mask < 127, 0, 255).astype(np.uint8)

    masks = []
    for path in image_paths[1:]:
        fg_mask_post = postprocessing(fg_mask)
        masks.append(fg_mask_post)

        idx_fore = fg_mask_post != 0
        idx_back = fg_mask_post == 0

        img_with_mask = np.zeros_like(img)
        img_with_mask[idx_fore] = np.minimum(img[idx_fore] + [0, 0, 255], [255, 255, 255]).astype(np.uint8)
        img_with_mask[idx_back] = img[idx_back]
       
        img = cv2.imread(path)

        cv2.imshow("FG Mask", img_with_mask)
        cv2.imshow("FG Mask processed", fg_mask_post)
        k = cv2.waitKey(26)
        if k == 27:
            break

        fg_mask_post = cv2.erode(fg_mask_post, np.ones((10, 10)))
        img_per = perturb_image(img, fg_mask_post)
        
        fg_mask = bg_subtractor.apply(img_per)


    return masks


def process_folder(input_path, output_path):
    images = os.listdir(input_path)
    images.sort()

    images = images[:1418]
    image_paths = [path.join(input_path, x) for x in images]
    mask_paths = list(map(lambda x: str(path.join(output_path, x)), images))

    masks_forward = background_substract(image_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""" Creates masks for body""")

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
    
    parser.add_argument("--output-path", "-o",
        help="The folder the masks are writen to",
        required=True
    )

    args = parser.parse_args()
    process_folder(args.input_path, args.output_path)