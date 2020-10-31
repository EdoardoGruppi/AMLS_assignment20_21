import os
import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image


def delete_glasses(dataset_name):
    # Avatar with black glasses to delete in training directory
    model_ModelGlasses = models.load_model('./B2/glasses_model')
    training_dir = './Datasets/{}/img'.format(dataset_name)
    files = sorted(os.listdir(training_dir), key=lambda x: int(x.split(".")[0]))
    img_size = plt.imread(os.path.join(training_dir, files[0])).shape[:2][::-1]
    batch_holder = np.zeros((len(files), img_size[0], img_size[1], 3))
    for i, img in enumerate(files):
        img = image.load_img(os.path.join(training_dir, img), target_size=img_size)
        batch_holder[i, :] = img
    predictions = model_ModelGlasses.predict(batch_holder, verbose=1)
    predictions = np.round(predictions)
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
    print(images_to_delete)
    print(len(images_to_delete))
    for i in images_to_delete:
        os.remove(os.path.join(training_dir, files[i]))
    # # Avatar with black glasses to delete in test directory
    test_dir = './Datasets/{}_test/img'.format(dataset_name)
    files = sorted(os.listdir(test_dir), key=lambda x: int(x.split(".")[0]))
    img_size = plt.imread(os.path.join(test_dir, files[0])).shape[:2][::-1]
    batch_holder = np.zeros((len(files), img_size[0], img_size[1], 3))
    for i, img in enumerate(files):
        img = image.load_img(os.path.join(test_dir, img), target_size=img_size)
        batch_holder[i, :] = img
    predictions = model_ModelGlasses.predict(batch_holder, verbose=1)
    predictions = np.round(predictions)
    predicted_labels = np.array(np.argmax(predictions, axis=-1))
    images_to_delete = np.array(np.where(predicted_labels == 1)).flatten()
    print(images_to_delete)
    print(len(images_to_delete))
    for i in images_to_delete:
        os.remove(os.path.join(test_dir, files[i]))
