# import packages
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import shutil
import os


def delete_glasses(dataset_name):
    # Load the pre trained model in the PreProcessing folder
    model_ModelGlasses = models.load_model('./PreProcessing/glasses_model')
    training_dir = './Datasets/{}/img'.format(dataset_name)
    # List of all the files in the training directory
    files = os.listdir(training_dir)
    # Get the image size. The number of channels are not considered.
    img_size = plt.imread(os.path.join(training_dir, files[0])).shape[:2][::-1]
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
        shutil.move(os.path.join(training_dir, files[i]), './Datasets/{}_removed'.format(dataset_name))

    # Same procedure applied to the test directory
    test_dir = './Datasets/{}_test/img'.format(dataset_name)
    files = os.listdir(test_dir)
    test_images = ImageDataGenerator().flow_from_directory(directory='./Datasets/{}_test'.format(dataset_name),
                                                           target_size=img_size, batch_size=16,
                                                           classes=None, class_mode=None, shuffle=False)
    predictions = model_ModelGlasses.predict(x=test_images, steps=len(test_images), verbose=1)
    predictions = np.round(predictions)
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
    print('\nFROM TEST DIRECTORY:\nThere are {} to delete.'.format(len(images_to_delete)))
    for i in images_to_delete:
        shutil.move(os.path.join(test_dir, files[i]), './Datasets/{}_removed'.format(dataset_name))
