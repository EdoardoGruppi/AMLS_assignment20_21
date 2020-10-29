# todo reduce imports as much as possible like import seaborn to from seaborn import...
# Import packages
from comet_ml import Experiment
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv2D, Activation
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sn
from sklearn.metrics import accuracy_score
from random import sample


# todo move Data preprocessing or Data preparation
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
    training_dir = '{}/img'.format(path)
    test_dir = '{}_test/img'.format(path)
    # List of all the images available
    files = sorted(os.listdir(training_dir), key=lambda x: int(x.split(".")[0]))
    # Images shape is expected to be the same for each one of them
    # img_size must have only the first two dimensions. By default the third dimension is equal to 3
    img_size = plt.imread(os.path.join(training_dir, files[0])).shape[:2][::-1]
    # If num_examples != len(files) the dataset division has been already accomplished
    if num_examples == len(files):
        random_test_list = sorted(sample(range(0, num_examples), round(num_examples * (1 - training_percentage_size))))
        for index in random_test_list:
            shutil.move(os.path.join(training_dir, files[index]), test_dir)
    else:
        random_test_list = sorted([dataset_labels[dataset_labels[filename_column] == i].index[0]
                                   for i in os.listdir(test_dir)])

    # todo Before we have to do some image preprocessing (for A1 and/or A2)
    training_labels = dataset_labels.iloc[[i for i in range(0, num_examples) if i not in random_test_list]]
    test_labels = dataset_labels.iloc[[i for i in random_test_list]]
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split,
                                         horizontal_flip=True)
    # image_generator.flow_from_dataframe() is a directory iterator.
    # It produces batches of images everytime it is required
    training_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                           x_col=filename_column, y_col=target_column,
                                                           subset="training", batch_size=batches_size, seed=42,
                                                           shuffle=True, target_size=img_size)
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
    valid_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                        x_col=filename_column, y_col=target_column, subset="validation",
                                                        batch_size=batches_size, seed=42, shuffle=True,
                                                        target_size=img_size)
    test_batches = image_generator.flow_from_dataframe(dataframe=test_labels, directory=test_dir,
                                                       x_col=filename_column, y_col=target_column,
                                                       batch_size=batches_size, shuffle=False, target_size=img_size)
    return training_batches, valid_batches, test_batches


class A1:
    def __init__(self, input_shape):
        # parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
        # is a infinitely repeating dataset
        self.model = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dropout(rate=0.5),
            Dense(units=2, activation='softmax')
        ])
        self.model.summary()
        # Using a 'binary_crossentropy' we would obtain one output, rather than two
        # In that case the activation of the last layer must be a 'sigmoid'
        # Alternatively it is possible to use a 'categorical_crossentropy' with a 'softmax' in the last layer
        # In that case it is compulsory to insert class_mode='binary' in flow_from_dataframe() functions ...
        # And to insert predicted_labels = np.array(predictions).astype(int).flatten() instead of...
        # ...predicted_labels = np.array(np.argmax(predictions, axis=-1))
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.experiment = Experiment(api_key="hn5we8X3ThjkDumjfdoP2t3rH", project_name="convnet",
                                     workspace="edoardogruppi")

    def train(self, training_batches, valid_batches, epochs=10, verbose=2):
        # Training phase
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=len(training_batches),
                                 validation_data=valid_batches,
                                 validation_steps=len(valid_batches),
                                 epochs=epochs,
                                 verbose=verbose
                                 )
        # return accuracy on the train and validation dataset
        return history.history['val_accuracy'][-1]

    def test(self, test_batches, verbose=1, confusion_mesh=False, class_labels=True):
        # steps parameter indicates on how many batches are necessary to work on each data on the Testing dataset
        predictions = self.model.predict(x=test_batches, steps=len(test_batches), verbose=verbose)
        predictions = np.round(predictions)
        predicted_labels = np.array(np.argmax(predictions, axis=-1))
        true_labels = np.array(test_batches.classes)
        if confusion_mesh:
            confusion_grid = pd.crosstab(true_labels, predicted_labels, normalize=True)
            # Generate a custom diverging colormap
            color_map = sn.diverging_palette(355, 250, as_cmap=True)
            sn.heatmap(confusion_grid, cmap=color_map, vmax=0.5, vmin=0, center=0, xticklabels=class_labels,
                       yticklabels=class_labels, square=True, linewidths=2, cbar_kws={"shrink": .5}, annot=True)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()
        self.experiment.log_confusion_matrix(true_labels, predicted_labels)
        self.experiment.end()
        # return accuracy on the test dataset
        return accuracy_score(true_labels, predicted_labels)

    # todo remove this function
    def evaluate(self, test_batches, verbose=1):
        score = self.model.evaluate(x=test_batches, verbose=verbose)
        print(score)
