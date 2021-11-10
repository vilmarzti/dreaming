from segmentation.data import TrainDataset
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

train_loader = DataLoader(train_set, batch_size=1)

for img, seg in  train_loader:
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