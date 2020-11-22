# import packages
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os
from pathlib import Path


def delete_glasses(dataset_name, img_size=(224, 224)):
    """
        Moves avatar images where eyes are hidden behind black sunglasses in a new dedicated folder.
        It leverages a pretrained CNN model

        :param img_size: size of the input image expected by the model. default_value=(224,224)
        :param dataset_name: name (not path) of the folder where to remove avatars.
    """
    # Load the pre trained model in the Modules folder
    model_ModelGlasses = models.load_model('./Modules/model_glasses')
    training_dir = './Datasets/{}/img'.format(dataset_name)
    # List of all the files in the training directory
    files = os.listdir(training_dir)
    remove_images = './Datasets/{}_removed'.format(dataset_name)
    # If parents is True, any missing parents of the folder will be created
    # If exist_ok is True, an Error is raised if the directory already exists
    Path(remove_images).mkdir(parents=True, exist_ok=True)
    # ImageDataGenerator create batches of images everytime it is called
    training_images = ImageDataGenerator().flow_from_directory(directory='./Datasets/{}'.format(dataset_name),
                                                               target_size=img_size, batch_size=16,
                                                               classes=None, class_mode=None, shuffle=False)
    # Distinguish normal avatar from the ones with sunglasses
    predictions = model_ModelGlasses.predict(x=training_images, steps=len(training_images), verbose=1)
    predictions = np.round(predictions)
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    # The images to delete are in the class == 1
    images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
    print('\nFROM TRAINING DIRECTORY:\nThere are {} to delete.'.format(len(images_to_delete)))
    # Move all the files in a dedicated folder
    for i in images_to_delete:
        shutil.move(os.path.join(training_dir, files[i]), remove_images)

    # Same procedure applied to the test directory
    test_dir = './Datasets/{}_testing/img'.format(dataset_name)
    files = os.listdir(test_dir)
    test_images = ImageDataGenerator().flow_from_directory(directory='./Datasets/{}_testing'.format(dataset_name),
                                                           target_size=img_size, batch_size=16,
                                                           classes=None, class_mode=None, shuffle=False)
    predictions = model_ModelGlasses.predict(x=test_images, steps=len(test_images), verbose=1)
    predictions = np.round(predictions)
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
    print('\nFROM TEST DIRECTORY:\nThere are {} to delete.'.format(len(images_to_delete)))
    for i in images_to_delete:
        shutil.move(os.path.join(test_dir, files[i]), remove_images)
