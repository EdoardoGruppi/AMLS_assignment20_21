# todo move test.py into _additional_code
# Import packages
from Modules.delete_glasses import delete_glasses
from Modules.face_extraction import smiles_extraction
from Modules.pre_processing import data_preprocessing, hog_pca_preprocessing
from Modules.test_pre_processing import test_data_preparation, test_hog_pca_preprocessing
from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2
import tensorflow as tf

# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# A1 ===================================================================================================================
# # Extract smiles for A2 task before dividing all the images in 'celeba' in training and test images.
# data_directory, faces_not_detected = smiles_extraction(dataset_name='celeba')
# test_directory, faces_not_detected1 = smiles_extraction(dataset_name='celeba_test')
#
# todo delete
acc_A1_test2, faces_not_detected1, faces_not_detected, acc_A2_test2, data_directory, acc_B1_test2 = 0, 0, 0, 0, 0, 0
# #
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='celeba', img_size=(96, 96),
#                                                                    filename_column='img_name', target_column='gender',
#                                                                    training_percentage_size=0.85, batches_size=16,
#                                                                    validation_split=0.15)
# input_shape = training_batches.image_shape
# # Build model object.
# model_A1 = A1(input_shape)
# # Train model based on the training set
# acc_A1_train, acc_A1_valid = model_A1.train(training_batches, valid_batches, epochs=25, verbose=2, plot=True)
# # Test model based on the test set.
# acc_A1_test = model_A1.test(test_batches, verbose=1, confusion_mesh=True)
# # # Test the model on the second larger test dataset provided
# # test_batches = test_data_preparation('celeba_test', filename_column='img_name', target_column='gender',
# #                                      batches_size=16, img_size=(96, 96))
# # acc_A1_test2 = model_A1.test(test_batches, verbose=1, confusion_mesh=False)
# # Print out your results with following format:
# print('TA1: {}, {}, {}, {}'.format(acc_A1_train, acc_A1_valid, acc_A1_test, acc_A1_test2))
# # Clean up memory
# del acc_A1_train, acc_A1_valid, acc_A1_test, model_A1, physical_devices, faces_not_detected, faces_not_detected1, \
#     acc_A1_test2
#
# A2 SVM ===============================================================================================================
# todo data_directory
X_test, X_train, X_valid, y_test, y_train, y_valid, pca, sc = hog_pca_preprocessing(dataset_name='celeba_smiles',
                                                                                    img_size=(96, 48),
                                                                                    validation_split=0.15,
                                                                                    variance=0.90, training_size=0.85,
                                                                                    target_column='smiling')
# Build model object.
# todo 0.001, 10 or 'scale',1
model_A2 = A2(kernel='rbf', gamma=0.001, c=10, verbose=False)
# Train model based on the training set
acc_A2_train, acc_A2_valid = model_A2.train(X_train, X_valid, y_train, y_valid)
# Test model based on the test set.
acc_A2_test = model_A2.test(X_test, y_test, confusion_mesh=True)
# # Test the model on the second larger test dataset provided
# x_test, y_test = test_hog_pca_preprocessing(test_directory, pca, sc, img_size=(96, 48), target_column='smiling')
# acc_A2_test2 = model_A2.test(X_test, y_test, confusion_mesh=False)
# Print out your results with following format:
print('TA2: {}, {}, {}, {}'.format(acc_A2_train, acc_A2_valid, acc_A2_test, acc_A2_test2))
# Clean up memory
del acc_A2_train, acc_A2_valid, acc_A2_test, X_test, X_train, X_valid, y_test, y_train, y_valid, data_directory, \
    model_A2, pca, sc, acc_A2_test2
#
# B1 ===================================================================================================================
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='cartoon_set',
#                                                                    filename_column='file_name',
#                                                                    target_column='face_shape', img_size=(224, 224),
#                                                                    training_percentage_size=0.8, horizontal_flip=False,
#                                                                    batches_size=16, validation_split=0.2)
# input_shape = training_batches.image_shape
# # Build model object.
# model_B1 = B1(input_shape)
# # Train model based on the training set
# acc_B1_train, acc_B1_valid = model_B1.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # Test model based on the test set.
# acc_B1_test = model_B1.test(test_batches, verbose=1, confusion_mesh=True)
# # Test the model on the second larger test dataset provided
# # test_batches = test_data_preparation('cartoon_set_test', filename_column='file_name', target_column='face_shape',
# #                                      batches_size=16, img_size=(224, 224))
# # acc_B1_test2 = model_B1.test(test_batches, verbose=1, confusion_mesh=False)
# # Print out your results with following format:
# print('TA1: {}, {}, {}, {}'.format(acc_B1_train, acc_B1_valid, acc_B1_test, acc_B1_test2))
# # Clean up memory
# del acc_B1_train, acc_B1_valid, acc_B1_test, model_B1, acc_B1_test2
#
# B2 ===================================================================================================================
# To execute after the B1 Task!
# delete_glasses(dataset_name='cartoon_set', img_size=(224, 224))
# training_batches, valid_batches, test_batches = data_preprocessing(data_directory='cartoon_set',
#                                                                    filename_column='file_name',
#                                                                    target_column='eye_color',
#                                                                    training_percentage_size=0.8, batches_size=16,
#                                                                    horizontal_flip=False, validation_split=0.2)
# input_shape = training_batches.image_shape
# # Build model object.
# model_B2 = B2(input_shape)
# # Train model based on the training set
# acc_B2_train, acc_B2_valid = model_B2.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # Test model based on the test set.
# acc_B2_test = model_B2.test(test_batches, verbose=1, confusion_mesh=True)
# Test the model on the second larger test dataset provided
# test_batches = test_data_preparation('cartoon_set_test', filename_column='file_name', target_column='face_shape',
#                                      batches_size=16, img_size=(224, 224))
# acc_B2_test2 = model_B2.test(test_batches, verbose=1, confusion_mesh=False)
# Print out your results with following format:
# print('TA1: {}, {}, {}, {}'.format(acc_B2_train, acc_B2_valid, acc_B2_test, acc_B2_test2))
#
# ======================================================================================================================
#
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.001, 'scale'], 'kernel': ['rbf']}
# self.model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# print(self.model.best_estimator_)
# ======================================================================================================================
