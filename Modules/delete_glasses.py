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
    model_ModelGlasses = models.load_model(os.path.join('./Modules', 'model_glasses'))
    dataset_path = os.path.join('./Datasets', dataset_name)
    removed_images = os.path.join(dataset_path + '_removed')
    # If parents is True, any missing parents of the folder will be created
    # If exist_ok is True, an Error is not raised if the directory already exists
    Path(removed_images).mkdir(parents=True, exist_ok=True)

    folders = ['', '_testing', '_test']
    for folder in folders:
        dataset_folder = dataset_path + folder
        images_folder = os.path.join(dataset_folder, 'img')
        if os.path.isdir(images_folder):
            # List of all the files in the training directory
            files = os.listdir(images_folder)
            training_images = ImageDataGenerator().flow_from_directory(directory=dataset_folder,
                                                                       target_size=img_size, batch_size=16,
                                                                       classes=None, class_mode=None, shuffle=False)
            # Distinguish normal avatar from the ones with sunglasses
            predictions = model_ModelGlasses.predict(x=training_images, steps=len(training_images), verbose=1)
            predictions = np.round(predictions)
            predicted_labels = np.array(np.argmax(predictions, axis=-1))
            # The images to delete are in the class == 1
            images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
            print('\nIn {} there are {} images to delete.'.format(dataset_folder.split(os.sep)[-1],
                                                                  len(images_to_delete)))
            # Move all the files in a dedicated folder
            for i in images_to_delete:
                shutil.move(os.path.join(images_folder, files[i]), removed_images)
