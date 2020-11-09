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
from sklearn.metrics import accuracy_score, classification_report
from random import sample
# todo where to insert it? a1.py?
from face_extraction import faces_recognition


def data_preprocessing(data_directory, filename_column, target_column, training_percentage_size=0.8, batches_size=10,
                       validation_split=0.25, img_size=(100, 100), face_extraction=False, color_mode='grayscale'):
    # Loading the csv file
    # The sep parameter chosen according to the delimiter adopted in labels.csv
    path = './Datasets/{}'.format(data_directory)
    dataset_labels = pd.read_csv('{}/labels.csv'.format(path), sep='\t', dtype='str')
    num_examples = dataset_labels.shape[0]
    if face_extraction:
        num_examples, faces_not_detected = faces_recognition(data_directory)
    path = '{}_face_rec'.format(path)
    # Divide data in two sets: one for training and one for testing
    # [DELETE] The lines below are useful only if the datasets has not already been divided between test and training
    # Create the Test dataset folder
    training_dir = '{}/img'.format(path)
    test_dir = '{}_test/img'.format(path)
    # todo Put lines below inside the if?
    # If parents is True, any missing parents of the folder will be created
    # If exist_ok is True, an Error is raised if the directory already exists
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    # List of all the images available
    files = sorted(os.listdir(training_dir), key=lambda x: int(x.split(".")[0]))
    # Division will be made only once for both the tasks (A1 and A2) assigned on the Dataset
    # If num_examples != len(files) the dataset division has been already accomplished
    if num_examples == len(files):
        images_detected = sorted(sample(files, round(num_examples * (1 - training_percentage_size))))
        for file in images_detected:
            shutil.move(os.path.join(training_dir, file), test_dir)

    random_test_list = sorted([dataset_labels[dataset_labels[filename_column] == i].index[0]
                               for i in os.listdir(test_dir)])
    random_training_list = sorted([dataset_labels[dataset_labels[filename_column] == i].index[0]
                                   for i in os.listdir(training_dir)])

    # Prepare the training, test and validation batches
    training_labels = dataset_labels.iloc[[i for i in random_training_list]]
    test_labels = dataset_labels.iloc[[i for i in random_test_list]]
    # With the following line the validation_split passed as argument corresponds to the percentage of the total
    # dataset (and not anymore to the percentage of the training dataset) dedicated to the validation dataset
    validation_split = validation_split / training_percentage_size
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split,
                                         horizontal_flip=True)
    # image_generator.flow_from_dataframe() is a directory iterator.
    # It produces batches of images everytime it is required
    training_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                           x_col=filename_column, y_col=target_column,
                                                           subset="training", batch_size=batches_size, seed=42,
                                                           color_mode=color_mode, shuffle=True, target_size=img_size)
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
    valid_batches = image_generator.flow_from_dataframe(dataframe=training_labels, directory=training_dir,
                                                        x_col=filename_column, y_col=target_column, subset="validation",
                                                        batch_size=batches_size, seed=42, shuffle=True,
                                                        color_mode=color_mode, target_size=img_size)
    test_batches = image_generator.flow_from_dataframe(dataframe=test_labels, directory=test_dir,
                                                       x_col=filename_column, y_col=target_column,
                                                       color_mode=color_mode, batch_size=batches_size,
                                                       shuffle=False, target_size=img_size)
    return training_batches, valid_batches, test_batches


class A1:
    def __init__(self, input_shape):
        # Parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
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
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            # Fraction of the input units dropped
            Dropout(rate=0.5),
            Dense(units=2, activation='softmax')
        ])
        self.model.summary()
        # Using a 'binary_crossentropy' we would obtain one output, rather than two
        # In that case the activation of the last layer must be a 'sigmoid'
        # Alternatively it is possible to use a 'categorical_crossentropy' with a 'softmax' in the last layer
        # In that case it is compulsory to insert class_mode='binary' in flow_from_dataframe() functions ...
        # ...and predicted_labels = np.array(predictions).astype(int).flatten() instead of...
        # ...predicted_labels = np.array(np.argmax(predictions, axis=-1))
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.experiment = Experiment(api_key="hn5we8X3ThjkDumjfdoP2t3rH", project_name="convnet",
                                     workspace="edoardogruppi")

    def train(self, training_batches, valid_batches, epochs=35, verbose=2):
        # Training phase
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=len(training_batches),
                                 validation_data=valid_batches,
                                 validation_steps=len(valid_batches),
                                 epochs=epochs,
                                 verbose=verbose
                                 )
        # Return accuracy on the train and validation dataset
        return history.history['val_accuracy'][-1]

    def test(self, test_batches, verbose=1, confusion_mesh=False, class_labels=None):
        # Steps parameter indicates on how many batches are necessary to work on each data on the Testing dataset
        # model.predict returns the predictions made on the input
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
        print(classification_report(true_labels, predicted_labels))
        self.experiment.log_confusion_matrix(true_labels, predicted_labels)
        self.experiment.end()
        # Return accuracy on the test dataset
        return accuracy_score(true_labels, predicted_labels)

    def evaluate(self, test_batches, verbose=1):
        # model.evaluate predicts the output and returns the metrics function specified in model.compile()
        score = self.model.evaluate(x=test_batches, verbose=verbose)
        print(score)

