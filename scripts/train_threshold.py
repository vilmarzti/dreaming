from segmentation.training.threshold import train

# Train to get GRAYSCALE Thresholds
print("Training Grayscale range")
train(True)

# Train to get HSV Thresholds
print("Training HSV Range")
train(False)