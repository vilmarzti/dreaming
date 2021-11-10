from segmentation.helper import get_image_paths
from sklearn.decomposition import PCA

import cv2
import numpy as np

import argparse

def main(input_path):
    image_paths = get_image_paths(input_path)
    imgs = np.array([cv2.imread(image_path) for image_path in image_paths])

    b_values = np.flatten(imgs[:,:,:,0])
    g_values = np.flatten(imgs[:,:,:,1])
    r_values = np.flatten(imgs[:,:,:,2])
    stacked = np.vstack([b_values, g_values, r_values]).transpose()

    pca = PCA(3)
    pca.fit(stacked)

    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Computes the first 3 components of pca""")

    parser.add_argument("--input-path", "-i",
        help="The path to the folder with the images",
        required=True)
    
    args = parser.parse_args()
    main(args.input_path)