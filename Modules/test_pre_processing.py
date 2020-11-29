# Import packages
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from skimage.feature import hog


def test_data_preparation(data_directory, filename_column, target_column, batches_size=16, img_size=(96, 96),
                          color_mode='rgb'):
    """
    It returns test batches prepared from the images found in data_directory.

    :param data_directory: name (not path) of the images folder.
    :param filename_column: name of the column in the csv file where all the related filenames are declared.
    :param target_column: name of the column in the csv file where the labels are declared.
    :param batches_size: dimension of every batch. default_value=16.
    :param img_size: size of images after the reshaping. default_value=(96,96).
    :param color_mode: state 'grayscale' if the images have only one channel. default_value='rgb'.
    :return: the test batches.
    """
    # Loading the csv file
    # The sep parameter chosen according to the delimiter adopted in labels.csv
    path = './Datasets/{}_test'.format(data_directory)
    test_info = pd.read_csv('{}/labels.csv'.format(path), sep='\t', dtype='str')
    # Retrieve folder where images are located
    test_dir = '{}/img'.format(path)
    # ImageDataGenerator generates batches of images in real-time
    image_generator = ImageDataGenerator(rescale=1. / 255.)
    test_batches = image_generator.flow_from_dataframe(dataframe=test_info, directory=test_dir,
                                                       x_col=filename_column, y_col=target_column,
                                                       color_mode=color_mode, batch_size=batches_size,
                                                       shuffle=False, target_size=img_size)
    return test_batches


def test_hog_pca_preprocessing(dataset_name, pca, standard_scaler, img_size=(96, 48), target_column='smiling'):
    """
    Given a dataset it extracts HOG features from each image. Data dimensionality is then further reduced applying
    PCA algorithm.

    :param dataset_name: name (not path) of the folder that contains the images.
    :param pca: PCA fitted on training data.
    :param standard_scaler: Standard Scaler fitted on training data.
    :param img_size: image dimension after reshaping. default_value=(96,48).
    :param target_column: name of the column in the csv file where the labels are declared. default_value='smiling'.
    :return: dataset and labels divided of the test dataset. To each image is associated a reduced vector of
        features by the means of to HOG and PCA algorithms.
    """
    # Create path to access to all the images
    path = './Datasets/{}_test'.format(dataset_name)
    images_dir = '{}/img'.format(path)
    # List all the images within the folder
    files = sorted(os.listdir(images_dir), key=lambda x: int(x.split(".")[0]))
    test_labels = pd.read_csv('{}/labels.csv'.format(path), sep='\t', dtype='str')[target_column]
    feature_matrix = []
    # Counter inserted to display the execution status
    counter = 0
    print('\nExtracting features from test folder...')
    for file in files:
        counter += 1
        img = cv2.imread(images_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        # Resize the image in case it has a different size than the expected
        img = cv2.resize(img, img_size)
        hog_feature = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          multichannel=False, feature_vector=True)
        # Append the HOG features vector to the feature map
        feature_matrix.append(hog_feature)
        if counter % 1000 == 0:
            print('Images processed: {}'.format(counter))

    # Labels have to be transformed in int from the string format
    y_test = [int(label) for label in test_labels]
    x_test = np.array(feature_matrix)
    print('Data dimensionality before PCA: {}'.format(len(feature_matrix[0])))
    # Normalize values before using PCA
    x_test = standard_scaler.transform(x_test)
    # Data is now projected on the first principal components previously extracted from the training set
    x_test = pca.transform(x_test)
    print('Data dimensionality after PCA: {}'.format(pca.n_components_))
    return x_test, y_test
