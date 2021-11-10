from segmentation.helper import get_image_paths
from sklearn.decomposition import IncrementalPCA

import cv2
import numpy as np

import argparse

def main(input_path):
    image_paths = get_image_paths(input_path)
    imgs = np.array([cv2.imread(image_path) for image_path in image_paths])

    b_values = imgs[:,:,:,0].flatten()
    g_values = imgs[:,:,:,1].flatten()
    r_values = imgs[:,:,:,2].flatten()
    stacked = np.vstack([b_values, g_values, r_values]).transpose()
    num_components = 3
    chunk_size = 1000
    m, _ = stacked.shape

    pca = IncrementalPCA(num_components, copy=False)
    for i in range(m // chunk_size):
        pca.partial_fit(stacked[i * chunk_size: (i + 1) * chunk_size])

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