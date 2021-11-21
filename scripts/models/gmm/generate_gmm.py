import os
import cv2
import numpy as np

from os import path

from sklearn.mixture import GaussianMixture
from sklearn.metrics import recall_score

num_components = 2 # The optimal value of num_components is found with scripts/masks/generate_gmm
num_sampled_imgs = 5
image_folder = "/home/martin/Videos/ondrej_et_al/bf/bf_gen/input_filtered/"
segmentation_folder = "/home/martin/Videos/ondrej_et_al/bf/segmentation/backsub/"
output_path = "data/masks_gmm"
save_png = False

random_image_names = np.random.choice(np.arange(1,1418), size=num_sampled_imgs, replace=False)

# read sampled images
pixel_values = np.empty((num_sampled_imgs * 1280 * 720, 3))
for i, x in enumerate(random_image_names):
    image_path = path.join(image_folder, f"{x:04}.png")
    img = cv2.imread(image_path)
    img = np.reshape(img, (-1, 3))
    pixel_values[i * 1280 * 720: (i + 1) * 1280 * 720] = img

# Fit GMM on those pixel values
gmm = GaussianMixture(num_components)
gmm.fit(pixel_values)

# Select Components of gaussians
segment_names = os.listdir(segmentation_folder)
segment_names.sort()

# Measure true positive rate
recall_scores_sum = np.array([0 for _ in range(num_components)])
for name in segment_names:
    # Read image
    image_path = path.join(image_folder, name)
    image_original = cv2.imread(image_path)
    image = np.reshape(image_original, (-1, 3))

    # Read target segmentation
    target_path = path.join(segmentation_folder, name)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    # reshape into 1-D image and put class labels on it
    target = np.where(target.reshape((-1)) < 127.5, 0, 1)

    # Get segmentation from gmm
    segmentation = gmm.predict(image)

    # Compute the jaccard indices for each component
    segmentation_masks = [np.where(segmentation == component, 1, 0) for component in range(num_components)]
    
    recall_scores_sum = recall_scores_sum + [recall_score(mask, target) for mask in segmentation_masks]

# Find selected components
recall_scores_mean = recall_scores_sum / len(segment_names)
selected_components = np.asarray(recall_scores_mean> 0.75).nonzero()[0]

# Set Segmentation based on found components
image_paths = os.listdir(image_folder)
image_paths.sort()
for image_name in image_paths:
    # Read image
    image_path = path.join(image_folder, image_name)
    image_original = cv2.imread(image_path)
    image = np.reshape(image_original, (-1, 3))

    # Set shape for segmentation mask
    segmentation_shape = (image_original.shape[0], image_original.shape[1], 1)

    # Get labels
    class_proba = gmm.predict_proba(image)

    # Create segmentations
    segmentation = np.zeros(segmentation_shape, dtype=np.int)
    for component in selected_components:
        class_proba = class_proba[:,component].reshape(segmentation_shape)
        segmentation = segmentation + class_proba
    
    segmentation = segmentation / np.max(segmentation)

    # prepare for saving
    if not path.isdir(output_path):
        os.mkdir(output_path)

    # Save
    if save_png:
        segmentation = np.array(segmentation * 255, np.uint8)
        cv2.imwrite(path.join(output_path, image_name), segmentation)
    else:
        np.save(path.join(output_path, image_name.replace(".png", ".npy")), segmentation)