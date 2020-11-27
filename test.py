# todo delete test.py
# Import packages
from Modules.delete_glasses import delete_glasses
from Modules.face_extraction import smiles_extraction
from Modules.pre_processing import data_preprocessing, hog_pca_preprocessing
from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2

# A1 ===================================================================================================================
# Extract smiles for A2 task before dividing all the images in 'celeba' in training and test images.
# data_directory, faces_not_detected = smiles_extraction(dataset_name='celeba')
#
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='celeba', img_size=(96, 96),
#                                                                    filename_column='img_name',
#                                                                    target_column='gender',
#                                                                    training_percentage_size=0.85, batches_size=16,
#                                                                    validation_split=0.15)
# input_shape = training_batches.image_shape
# # Build model object.
# model_A1 = A1(input_shape)
# # Train model based on the training set
# acc_A1_train, acc_A1_valid = model_A1.train(training_batches, valid_batches, epochs=25, verbose=2, plot=True)
# # # Used only in test.py
# # model_A1.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_A1_test = model_A1.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TA1:{},{},{}'.format(acc_A1_train, acc_A1_valid, acc_A1_test))
# #
# A2 SVM ===============================================================================================================
# todo data_directory, variance=0.88
# X_test, X_train, X_valid, y_test, y_train, y_valid = hog_pca_preprocessing(dataset_name=data_directory,
#                                                                            img_size=(96, 48),
#                                                                            validation_split=0.15,
#                                                                            variance=0.90,
#                                                                            training_size=0.85,
#                                                                            target_column='smiling')
# # Build model object.
# # todo 0.001, 10 or 'scale', 1
# model_A2 = A2(kernel='rbf', gamma='scale', c=1, verbose=False)
# # Train model based on the training set
# acc_A2_train, acc_A2_valid = model_A2.train(X_train, X_valid, y_train, y_valid)
# # Test model based on the test set.
# acc_A2_test = model_A2.test(X_test, y_test, confusion_mesh=True)
# # Print out your results with following format:
# print('TA2:{},{},{}'.format(acc_A2_train, acc_A2_valid, acc_A2_test))
#
# B1 ===================================================================================================================
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='cartoon_set',
#                                                                    filename_column='file_name',
#                                                                    target_column='face_shape', img_size=(224, 224),
#                                                                    training_percentage_size=0.8,
#                                                                    horizontal_flip=False,
#                                                                    batches_size=16, validation_split=0.2)
# input_shape = training_batches.image_shape
# # Build model object.
# model_B1 = B1(input_shape)
# # Train model based on the training set
# acc_B1_train, acc_B1_valid = model_B1.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # # Used only in test.py
# # model_B1.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_B1_test = model_B1.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TB1:{},{},{}'.format(acc_B1_train, acc_B1_valid, acc_B1_test))
#
# B2 ===================================================================================================================
# To execute after the B1 Task!
#
# delete_glasses(dataset_name='cartoon_set', img_size=(224, 224))
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='cartoon_set',
#                                                                    filename_column='file_name',
#                                                                    target_column='eye_color',
#                                                                    training_percentage_size=0.8,
#                                                                    horizontal_flip=False,
#                                                                    batches_size=16, validation_split=0.2)
# input_shape = training_batches.image_shape
# # Build model object.
# model_B2 = B2(input_shape)
# # Train model based on the training set
# acc_B2_train, acc_B2_valid = model_B2.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # # Used only in test.py
# # model_B2.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_B2_test = model_B2.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TB2:{},{},{}'.format(acc_B2_train, acc_B2_valid, acc_B2_test))
#
# ======================================================================================================================
# delete_glasses_hog_svm('cartoon_set', img_size=(100, 100), n_components=50, orientations=6, pixels_per_cell=(8, 8),
#                        cells_per_block=(3, 3), multichannel=True)
#
# from sklearn.ensemble import BaggingClassifier
# n_estimators = 10
# self.model = BaggingClassifier(svm.SVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
#
# from skimage.feature import hog, local_binary_pattern
# lbp_feature = local_binary_pattern(img, P=8, R=1).flatten()
# hog_feature = [*hog_feature, *lbp_feature]
#
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.001, 'scale'], 'kernel': ['rbf']}
# self.model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# print(self.model.best_estimator_)
# ======================================================================================================================
# A2 CNN ===============================================================================================================
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory=data_directory, img_size=(96, 96),
#                                                                    filename_column='img_name',
#                                                                    target_column='smiling',
#                                                                    training_percentage_size=0.85, batches_size=10,
#                                                                    validation_split=0.15)
#
# # CHANGE A1 to A2
# input_shape = training_batches.image_shape
# # Build model object.
# model_A2 = A1(input_shape)
# # Train model based on the training set
# acc_A2_train = model_A2.train(training_batches, valid_batches, epochs=35, verbose=2)
# # Used only in test.py
# # model_A2.evaluate(test_batches=test_batches, verbose=0)
# # Test model based on the test set.
# acc_A2_test = model_A2.test(test_batches, verbose=0, confusion_mesh=False)
# # Print out your results with following format:
# print('TA2:{},{}'.format(acc_A2_train, acc_A2_test))
#
