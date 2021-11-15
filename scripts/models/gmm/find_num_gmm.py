import cv2
import numpy as np

from os import path

from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt


num_sampled_imgs = 5
image_folder = "/home/martin/Videos/ondrej_et_al/bf/bf_gen/input_filtered/"
random_image_names = np.random.choice(np.arange(1,1418), size=num_sampled_imgs, replace=False)


# read all images
pixel_values = np.empty((num_sampled_imgs * 1280 * 720, 3))
for i, x in enumerate(random_image_names):
    image_path = path.join(image_folder, f"{x:04}.png")
    img = cv2.imread(image_path)
    img = np.reshape(img, (-1, 3))
    pixel_values[i * 1280 * 720: (i + 1) * 1280 * 720] = img


# sample 10000 pixel values for the computation of the silhouette score
sample_indices = np.random.choice(range(pixel_values.shape[0]), 10000, replace=False)
pixel_values_sampled = pixel_values[sample_indices]

# compute shiloutte scores
distances = pairwise_distances(pixel_values_sampled)
silh_scores = []
max_num_components = 10
for num_components in range(2, max_num_components):
    component_shil_score = []
    gmm = GaussianMixture(num_components)
    for x in range(5):
        # Fit GMM
        labels = gmm.fit_predict(pixel_values)

        # Use a sample of 10000 to compute the silhouette score
        labels_sampled = labels[sample_indices]
        score = silhouette_score(distances, labels_sampled, metric="precomputed")
        component_shil_score.append(score)

    print(f"Mean score {np.mean(component_shil_score)} for {num_components} components")
    silh_scores.append(component_shil_score)

score_means = np.mean(silh_scores, axis=1)
scores_std = np.std(silh_scores, axis=1)

plt.errorbar(range(2, max_num_components), score_means, yerr=scores_std, fmt="o")
plt.xlabel("num_components")
plt.ylabel("Silhoutte Scores mean")
plt.savefig("Silhoutte_scores.png")




"""
random_image_names = np.random.choice(np.arange(1,1418), size=num_sampled_imgs, replace=False)
label_colors = np.random.randint(0, 256, (num_components, 3), dtype=np.uint8)
for x in random_image_names:
    image_path = path.join(image_folder, f"{x:04}.png")
    img = cv2.imread(image_path)
    img = np.reshape(img, (-1, 3))
    labels = gmm.predict(img)
    labels = np.reshape(labels, (1280, 720))

    label_img = np.zeros((1280, 720, 3), dtype=np.uint8)

    for y in range(1280):
        for x in range(720):
            label_img[y, x] = label_colors[labels[y, x]]

    cv2.imshow("classes", label_img)
    cv2.waitKey()
"""
