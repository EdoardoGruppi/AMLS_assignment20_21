# Import packages
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import pandas as pd
from sklearn.preprocessing import StandardScaler


def svm_model(dataset_name, img_size=(96, 48), validation_split=0.15, variance=0.95, multichannel=False,
              training_size=0.85, target_column='smiling'):
    # Create path to access to all the images
    path = './Datasets/{}'.format(dataset_name)
    images_dir = '{}/img'.format(path)
    # List all the images within the folder
    files = sorted(os.listdir(images_dir), key=lambda x: int(x.split(".")[0]))
    dataset_labels = pd.read_csv('{}/labels.csv'.format(path), sep='\t', dtype='str')[target_column]
    feature_matrix = []
    # Counter inserted to display the execution status
    counter = 0
    print('\nExtracting features...')
    for file in files:
        counter += 1
        if multichannel:
            img = cv2.imread(images_dir + '/' + file)
        else:
            img = cv2.imread(images_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        # Resize the image in case it has a different size than the expected
        img = cv2.resize(img, img_size)
        hog_feature = hog(img, orientations=6, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                          multichannel=multichannel, feature_vector=True)
        # Append the HOG features vector to the feature map
        feature_matrix.append(hog_feature)
        if counter % 1000 == 0:
            print('Images processed: {}'.format(counter))

    print('Computing PCA...')
    print('Data dimensionality before PCA: {}'.format(len(feature_matrix[0])))
    pca = PCA(n_components=variance)
    # Retrieve labels of all the image processed
    # Recall: in some images faces, i.e. smiles, are not detected
    files = [file.split('.')[0] for file in files]
    dataset_labels = dataset_labels.iloc[files]
    # Labels have to be transformed in int from the string format
    y = [int(label) for label in dataset_labels]
    X = np.array(feature_matrix)
    test_size = 1 - training_size
    # Split dataset in training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    # Find the percentage of the training dataset that has to be dedicated to validation
    validation_split = validation_split / training_size
    # Divide training dataset between training and validation dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_split, random_state=0)

    # Normalize values before using PCA
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_valid = sc.transform(X_valid)

    # Fit the model with training data and apply the dimensionality reduction on them
    X_train = pca.fit_transform(X_train)
    # Data is now projected on the first principal components previously extracted from the training set
    X_valid = pca.transform(X_valid)
    X_test = pca.transform(X_test)
    print('Data dimensionality after PCA: {}'.format(pca.n_components_))
    print('Training the Support Vector Machine...')
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    # Return mean accuracy
    print('Train accuracy: {}'.format(clf.score(X_train, y_train)))
    print('Validation accuracy: {}'.format(clf.score(X_valid, y_valid)))
    print('Test accuracy: {}'.format(clf.score(X_test, y_test)))


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# print('Training the Random Forest...')
# classifier = RandomForestClassifier(max_depth=10, random_state=0)
# classifier.fit(X_train, y_train)
# # Predicting the Test set results
# print('Train accuracy: {}'.format(classifier.score(X_train, y_train)))
# print('Validation accuracy: {}'.format(classifier.score(X_valid, y_valid)))
# print('Test accuracy: {}'.format(classifier.score(X_test, y_test)))
