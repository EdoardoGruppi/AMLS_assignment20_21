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
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import TensorBoard


# todo move Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
def data_preprocessing(data_directory, filename_column, target_column, training_percentage_size=0.8, batches_size=10,
                       validation_split=0.25):
    # Loading the csv file
    # The sep parameter chosen according to the delimiter adopted in labels.csv
    path = './Datasets/{}'.format(data_directory)
    dataset_labels = pd.read_csv('{}/labels.csv'.format(path), sep='\t', dtype='str')
    num_examples = dataset_labels.shape[0]

    # Divide data in two sets: one for training and one for testing
    # Division will be made only once for both the tasks (A1 and A2) assigned on the Dataset
    # [DELETE] The following operations useful only if the dataset are not already divided in the test and training
    # Create the Test dataset folder
    Path('{}_test/img'.format(path)).mkdir(parents=True, exist_ok=True)
    # Compute the numbers of examples reserved for the training Dataset
    training_examples = round(num_examples * training_percentage_size)
    training_dir = '{}/img'.format(path)
    testing_dir = '{}_test/img'.format(path)
    # List of all the images available
    files = sorted(os.listdir(training_dir), key=lambda x: int(x.split(".")[0]))
    # Images shape is expected to be the same for each one of them
    # img_size must have only the first two dimensions. By default the third dimension is equal to 3
    img_size = plt.imread(os.path.join(training_dir, files[0])).shape[:2][::-1]
    files = files[training_examples:num_examples]
    # [DELETE] We want to transfer only the last part (num_examples * training_percentage_size) to the test dataset
    for file in files:
        shutil.move(os.path.join(training_dir, file), testing_dir)

    # todo Before we have to do some image preprocessing (for A1 and/or A2)
    training_labels = dataset_labels[:training_examples]
    testing_labels = dataset_labels[training_examples:]
    image_generator = ImageDataGenerator(rescale=1./255., validation_split=validation_split)
    # image_generator.flow_from_dataframe() is a directory iterator.
    # It produces batches of images whe n everytime it is required
    train_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                        x_col=filename_column, y_col=target_column, subset="training",
                                                        batch_size=batches_size, seed=42, shuffle=True,
                                                        target_size=img_size)
    valid_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                        x_col=filename_column, y_col=target_column, subset="validation",
                                                        batch_size=batches_size, seed=42, shuffle=True,
                                                        target_size=img_size)
    test_batches = image_generator.flow_from_dataframe(dataframe=testing_labels, directory=testing_dir,
                                                       x_col=filename_column, y_col=target_column,
                                                       batch_size=batches_size, shuffle=False, target_size=img_size)
    return train_batches, valid_batches, test_batches

#
# # todo make from scratch the network
# # parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
# # is a infinitely repeating dataset
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(178, 218, 3)),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=2),
#     Flatten(),
#     Dense(units=2, activation='softmax')
# ])
# # model.summary()
# model.compile()
