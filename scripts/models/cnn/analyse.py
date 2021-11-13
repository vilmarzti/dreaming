from ray.tune import Analysis, ExperimentAnalysis
from segmentation.training import create_test_best
from segmentation.helper import create_cnn

analysis = Analysis("data/raytune/cnn_run1/", default_metric="val_accuracy", default_mode="max")

breakpoint()
test_best = create_test_best(create_cnn, 5,     )