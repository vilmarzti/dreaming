from segmentation.data.dataset import SegmentationDataset
from segmentation.threshold.threshold import RangeImage

def train():
    crop_size = 100
    train_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output", crop_size)

if __name__ == "__main__":
    train()