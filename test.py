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
# acc_A1_train = model_A1.train(training_batches, valid_batches, epochs=25, verbose=2, plot=True)
# # # Used only in test.py
# # model_A1.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_A1_test = model_A1.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TA1:{},{}'.format(acc_A1_train, acc_A1_test))
# #
# A2 SVM ===============================================================================================================
# todo data_directory
# X_test, X_train, X_valid, y_test, y_train, y_valid = hog_pca_preprocessing(dataset_name='celeba_smiles',
#                                                                            img_size=(96, 48),
#                                                                            validation_split=0.15, variance=0.88,
#                                                                            training_size=0.85,
#                                                                            target_column='smiling')
# # Build model object.
# model_A2 = A2()
# # # Train model based on the training set
# acc_A2_train = model_A2.train(X_train, X_valid, y_train, y_valid, plot=True)
# # Test model based on the test set.
# acc_A2_test = model_A2.test(X_test, y_test, confusion_mesh=True)
# # Print out your results with following format:
# print('TA1:{},{}'.format(acc_A2_train, acc_A2_test))
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
# acc_B1_train = model_B1.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # # Used only in test.py
# # model_B1.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_B1_test = model_B1.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TB1:{},{}'.format(acc_B1_train, acc_B1_test))
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
# acc_B2_train = model_B2.train(training_batches, valid_batches, epochs=10, verbose=2, plot=True)
# # # Used only in test.py
# # model_B2.evaluate(test_batches=test_batches, verbose=1)
# # Test model based on the test set.
# acc_B2_test = model_B2.test(test_batches, verbose=1, confusion_mesh=True)
# # Print out your results with following format:
# print('TB2:{},{}'.format(acc_B2_train, acc_B2_test))
#
# ======================================================================================================================
# DATA VISUALIZATION
# def plot_images(images_arr):
#     fig, axes = plt.subplots(1, 10, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# images, labels = next(train_batches)
# plot_images(images)
# print(labels[0:10])
#
# TENSORBOARD GRAPHICS
# # tensorboard --logdir logs
# from tensorflow.keras.callbacks import TensorBoard
# shutil.rmtree(r'.\logs', ignore_errors=True)
# tensorboard_callback = TensorBoard(log_dir=".\logs", histogram_freq=1)
# callbacks=[tensorboard_callback] in the fit function
#
#
# COMET_ML
# from comet_ml import Experiment
# self.experiment = Experiment(api_key="hn5we8X3ThjkDumjfdoP2t3rH", project_name="convnetb2",
#                              workspace="edoardogruppi")
# self.experiment.log_confusion_matrix(true_labels, predicted_labels)
# self.experiment.end()
#
#
# batch_holder = np.zeros((len(files), img_size[0], img_size[1], 3))
# for i, img in enumerate(files):
#     img = image.load_img(os.path.join(training_dir, img), target_size=img_size)
#     batch_holder[i, :] = img
# predictions = model_ModelGlasses.predict(batch_holder, verbose=1)
#
#
# Images shape is expected to be the same for each one of them
# # img_size must have only the first two dimensions. By default the third dimension is equal to 3
# img_size = plt.imread(os.path.join(training_dir, files[0])).shape[:2][::-1]
#
#
# delete_glasses_hog_svm('cartoon_set', img_size=(100, 100), n_components=50, orientations=6, pixels_per_cell=(8, 8),
#                        cells_per_block=(3, 3), multichannel=True)
#
#
# from sklearn.ensemble import BaggingClassifier
# # n_estimators = 10
# # self.model = BaggingClassifier(svm.SVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
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
