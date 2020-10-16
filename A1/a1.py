# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# todo reduce imports as much as possible like import seaborn to from seaborn import...
# Import packages
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, Conv2D
from tensorflow.keras import regularizers, optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sn
# from sklearn.metrics import confusion_matrix

# Loading the csv file
# The sep parameter used according to the delimiter adopted.
celebA_labels = pd.read_csv('./Datasets/celeba/labels.csv', sep='\t', dtype='str')
celebA_labels = celebA_labels[['img_name', 'gender']]  # Choose the label of interest
num_examples = celebA_labels.shape[0]

# todo Divide data in two sets: one for training and one for testing
# Division will be made only once for both the tasks assigned on the Dataset
# [delete] The following operations useful only if the dataset are not already divided in the test and training
# Create the Test dataset folder
Path("./Datasets/celeba_test/img").mkdir(parents=True, exist_ok=True)
# [delete] Choose the percentage of data to reserve for the training Dataset
train_size = 0.8
train_examples = round(num_examples * train_size)
source_dir = './Datasets/celeba/img'
target_dir = './Datasets/celeba_test/img'
files = sorted(os.listdir(source_dir), key=lambda x: int(x.split(".")[0]))
files = files[train_examples:num_examples]
# [delete] We want to transfer only the last part (num_examples * train_size) to the test dataset
for file in files:
    shutil.move(os.path.join(source_dir, file), target_dir)

# todo Before we have to do some image preprocessing (for A1 and/or A2)
# todo associate labels to the examples in the training set
# todo add some comment
train_celebA_labels = celebA_labels[:train_examples]
test_celebA_labels = celebA_labels[train_examples:]
batches_size = 20
validation_split = 0.25
image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
# image_generator.flow_from_dataframe() is a directory iterator.
# It produces batches of images when everytime it is required
train_batches = image_generator.flow_from_dataframe(dataframe=train_celebA_labels, directory="./Datasets/celeba/img",
                                                    x_col="img_name", y_col="gender", subset="training",
                                                    batch_size=batches_size, seed=42, shuffle=True, classes=['-1', '1'],
                                                    target_size=(178, 218))
valid_batches = image_generator.flow_from_dataframe(dataframe=train_celebA_labels, directory="./Datasets/celeba/img",
                                                    x_col="img_name", y_col="gender", subset="validation",
                                                    batch_size=batches_size, seed=42, shuffle=True, classes=['-1', '1'],
                                                    target_size=(178, 218))
test_batches = image_generator.flow_from_dataframe(dataframe=test_celebA_labels, directory="./Datasets/celeba_test/img",
                                                   x_col="img_name", y_col="gender", batch_size=batches_size,
                                                   shuffle=False, classes=['-1', '1'], target_size=(178, 218))

# # todo [delete all the paragraph] Visualize the data
# def plot_images(images_arr):
#     fig, axes = plt.subplots(1, 10, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# images, labels = next(train_batches)
# plot_images(images)
# print(labels[0:10])

# todo make from scratch the network
# parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
# is a infinitely repeating dataset
model = Sequential([

])
# model.summary()
model.compile()
