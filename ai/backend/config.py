import os

DATASET_PATH = "dataset"

# initialize the class labels in the dataset
CLASSES = ["cat", "dog"]

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.25
TEST_SPLIT = 0.1

BATCH_SIZE = 128
EPOCHS = 50

# set the path to the serialized model after training
CHECKPOINT_PATH = os.path.join('checkpoint', 'checkpoint.pth')