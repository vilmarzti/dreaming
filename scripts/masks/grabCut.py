import cv2
import argparse
import os

import numpy as np
import multiprocessing as mp

from os import path

from segmentation.helper import get_image_paths, get_output_paths

import sys
sys.path.append("../Few-Shot-Patch-Based-Training/_tools/")
import config


def check_threshold(img):
    #canny = cv2.Canny(img, 85, 255) 
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_threshold = cv2.inRange(hsv_image, (10, 0, 8), (39, 170, 255))
    kernel = np.ones((7, 7), np.uint8)
    dilate = cv2.dilate(hsv_threshold, kernel, iterations=5)


    cv2.namedWindow('image') # make a window with name 'image'
    cv2.createTrackbar('Lu', 'image', 0, 255, print) #lower threshold trackbar for window 'image
    cv2.createTrackbar('Ll', 'image', 0, 255, print) #lower threshold trackbar for window 'image
    cv2.createTrackbar('Uu', 'image', 0, 255, print) #upper threshold trackbar for window 'image
    cv2.createTrackbar('Ul', 'image', 0, 255, print) #upper threshold trackbar for window 'image
    cv2.createTrackbar('Mu', 'image', 0, 255, print) #upper threshold trackbar for window 'image
    cv2.createTrackbar('Ml', 'image', 0, 255, print) #upper threshold trackbar for window 'image

    while(1):
        cv2.imshow('image', dilate)
        cv2.imshow("original", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: #escape key
            break

        lu = cv2.getTrackbarPos('Lu', 'image')
        ll = cv2.getTrackbarPos('Ll', 'image')

        uu = cv2.getTrackbarPos('Uu', 'image')
        ul = cv2.getTrackbarPos('Ul', 'image')

        mu = cv2.getTrackbarPos('Mu', 'image')
        ml = cv2.getTrackbarPos('Ml', 'image')

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_threshold = cv2.inRange(hsv_image, (ll, ul, ml), (lu, uu, mu))
        kernel = np.ones((7, 7), np.uint8)
        dilate = cv2.dilate(hsv_threshold, kernel, iterations=5)

    cv2.destroyAllWindows()


def create_graph_cut(image):
    img = cv2.imread(image)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Mask obtained through threshold
    hsv_threshold = cv2.inRange(hsv_image, (0, 0, 65), (32, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    dilate = cv2.dilate(hsv_threshold, kernel, iterations=5)

    # mask to remove head
    head_mask = np.ones((1280, 720), np.uint8)
    head_mask = head_mask * 255
    cv2.rectangle(head_mask, (0, 0), (720, 100), (0), -1)

    # mask to include lower half
    lower_mask = np.zeros_like(head_mask)
    cv2.circle(lower_mask, (180, 1280), 360, (255), -1)

    # compine masks
    mask = cv2.bitwise_and(head_mask, dilate)
    mask = cv2.bitwise_or(mask, lower_mask)

    # Fill empty space
    _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(mask)
    cv2.drawContours(fill, contours, -1, 255, -1, 4, hier)
    mask = fill 

    # Prepare GraphCut
    mask[mask >  0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD

    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the the mask segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(img, mask, None, bgModel, fgModel, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)

    # Get foreground and background mask
    bgd_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 255, 0).astype("uint8")
    fgd_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    # get bounding box of lower left half
    # and fill it
    _, contours, hier = cv2.findContours(bgd_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    get_rect = [i for i, x in enumerate(boundingBoxes) if x[0] >= 0 and x[2] < 1000 and x[3] < 500]

    if len(get_rect) > 0:
        cv2.drawContours(fgd_mask, contours, get_rect[0], 255, -1)

    return fgd_mask
    

def create_and_save(image_path, mask_path):
    mask = create_graph_cut(image_path)
    cv2.imwrite(mask_path, mask)


def process_folder(input_path, output_path):
    pool = mp.Pool(mp.cpu_count())

    # Generate all necesary paths
    image_paths = get_image_paths(input_path)
    mask_paths = get_output_paths(image_paths, output_path)

    if not path.isdir(output_path):
        os.mkdir(output_path)

    pool.starmap(create_and_save, zip(image_paths, mask_paths))


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
