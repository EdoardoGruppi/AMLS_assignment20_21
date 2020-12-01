# Import packages
import pandas as pd
from pathlib import Path
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import sample


def glasses_data_preprocessing(data_directory, filename_column, target_column, training_percentage_size=0.85,
                               batches_size=16, validation_split=0.15, img_size=(96, 96), color_mode='rgb',
                               horizontal_flip=True):
    """
        Whether a test dataset folder does not already exist, it divides the dataset in training and test sets .
        In the second part it prepares the training, validation and test batches.

        :param data_directory: name (not path) of the images folder.
        :param filename_column: name of the column in the csv file where all the related filenames are declared.
        :param target_column: name of the column in the csv file where the labels are declared.
        :param training_percentage_size: percentage size of the entire dataset dedicated to the training dataset.
            default_value=0.85.
        :param batches_size: dimension of every batch. default_value=16.
        :param validation_split: percentage size of the entire dataset dedicated to the validation set.
            default_value=0.15.
        :param img_size: size of images after the reshaping. default_value=(96,96).
        :param color_mode: state 'grayscale' if the images have only one channel. default_value='rgb'.
        :param horizontal_flip: state False if it is not desired images are randomly flipped. default_value=True.
        :return: the training, validation and test batches.
    """
    # Loading the csv file
    path = os.path.join('../Datasets', data_directory)
    # The sep parameter chosen according to the delimiter adopted in labels.csv
    # Take the first 600 examples
    dataset_labels = pd.read_csv(os.path.join(path, 'labels.csv'), sep='\t', dtype='str').head(600)
    dataset_labels[target_column] = list(map(str, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                                                   0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                                                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                   0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                   1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                                   0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                                                   0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                                   1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
                                                   # Others 300
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                                   0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                                   1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
                                                   ]))
    # Divide data in two sets: one for training and one for testing
    train_folder = os.path.join(path, 'img')
    test_folder = os.path.join(path + '_test', 'img')
    # Division will be made only if the testing directory does not already exist
    if not os.path.isdir(test_folder):
        # Create the Test dataset folder
        # If parents is True, any missing parents of the folder will be created
        # If exist_ok is True, an Error is raised if the directory already exists
        Path(test_folder).mkdir(parents=True, exist_ok=True)
        # Sorted list of all the images available
        files = sorted(os.listdir(train_folder), key=lambda x: int(x.split(".")[0]))
        # Simple random sampling to select examples for the test dataset
        images_testing = sorted(sample(files, round(len(files) * (1 - training_percentage_size))))
        # Move all the test images into the related folder
        for file in images_testing:
            shutil.move(os.path.join(train_folder, file), test_folder)
    # List of all the images within the training and testing folders
    random_test_list = sorted([dataset_labels[dataset_labels[filename_column] == i].index[0]
                               for i in os.listdir(test_folder)])
    random_training_list = sorted([dataset_labels[dataset_labels[filename_column] == i].index[0]
                                   for i in os.listdir(train_folder)])

    # Prepare the training, test and validation batches
    # Select labels associated to the images inside the training and test folders
    training_labels = dataset_labels.iloc[[i for i in random_training_list]]
    test_labels = dataset_labels.iloc[[i for i in random_test_list]]
    # With the following line the validation_split passed as argument becomes equal to the percentage of the total
    # dataset (and not anymore to the percentage of the training dataset dedicated to the validation dataset)
    validation_split = validation_split / training_percentage_size
    # ImageDataGenerator generates batches of images with real-time data augmentation
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split,
                                         horizontal_flip=horizontal_flip)
    # It produces batches of images everytime it is called
    training_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=train_folder,
                                                           x_col=filename_column, y_col=target_column,
                                                           subset="training", batch_size=batches_size, seed=42,
                                                           color_mode=color_mode, shuffle=True, target_size=img_size)
    # No data augmentation applied for validation and test data
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
    valid_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=train_folder,
                                                        x_col=filename_column, y_col=target_column, subset="validation",
                                                        batch_size=batches_size, seed=42, shuffle=True,
                                                        color_mode=color_mode, target_size=img_size)
    test_batches = image_generator.flow_from_dataframe(dataframe=test_labels, directory=test_folder,
                                                       x_col=filename_column, y_col=target_column,
                                                       color_mode=color_mode, batch_size=batches_size,
                                                       shuffle=False, target_size=img_size)
    return training_batches, valid_batches, test_batches
