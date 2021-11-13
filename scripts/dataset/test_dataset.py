from segmentation.data import TrainDataset, TestDataset
from torch.utils.data import DataLoader

import cv2
import numpy as np

crop_size = 300
cvt_flag = None
add_encoding = False

train_set = TrainDataset(
    "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input",
    "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output",
    crop_size,
    cvt_flag,
    add_encoding,
    True
)

valid_set = TestDataset(
    "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input",
    "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output",
    252,
    cvt_flag,
    add_encoding
)

train_loader = DataLoader(train_set, batch_size=1)
valid_loader = DataLoader(valid_set, batch_size=1)

len(valid_set)

for img, seg in  valid_loader:
    """
    img = img - img.min()
    img = img / img.max()
    """

    img = img.detach().cpu().numpy().astype(np.uint8)
    seg = seg.detach().cpu().numpy()

    seg = seg[0]
    img = np.transpose(img[0], axes=(1, 2, 0))

    cv2.imshow("Image", img)
    cv2.imshow("Seg", seg)
    cv2.waitKey(0)