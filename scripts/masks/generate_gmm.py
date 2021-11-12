import cv2
import numpy as np

from os import path

from sklearn.mixture import GaussianMixture

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

gmm = GaussianMixture(2)
gmm.fit(pixel_values)
