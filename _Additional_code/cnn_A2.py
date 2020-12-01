# Import packages
from Modules.pre_processing import data_preprocessing
from Modules.test_pre_processing import test_data_preparation
from A1.a1 import A1
import tensorflow as tf

# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# A2 CNN ===============================================================================================================
training_batches, valid_batches, test_batches = data_preprocessing(data_directory='celeba', img_size=(96, 96),
                                                                   filename_column='img_name',
                                                                   target_column='smiling',
                                                                   training_percentage_size=0.85, batches_size=10,
                                                                   validation_split=0.15)
input_shape = training_batches.image_shape
# Build model object.
model_A2 = A1(input_shape)
# Train model based on the training set
acc_A2_train, acc_A2_valid = model_A2.train(training_batches, valid_batches, epochs=25, verbose=2)
# Test model based on the test set.
acc_A2_test = model_A2.test(test_batches, verbose=1, confusion_mesh=False)
# Test the model on the second larger test dataset provided
test_batches = test_data_preparation('celeba_test', filename_column='img_name', target_column='gender', batches_size=16,
                                     img_size=(96, 96))
acc_A2_test2 = model_A2.test(test_batches, verbose=1, confusion_mesh=False)
# Print out your results with following format:
print('TA1: {}, {}, {}, {}'.format(acc_A2_train, acc_A2_valid, acc_A2_test, acc_A2_test2))




