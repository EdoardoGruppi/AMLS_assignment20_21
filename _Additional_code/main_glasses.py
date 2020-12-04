# Import packages
from _Additional_code.glasses_data_preparation import glasses_data_preprocessing
from _Additional_code.model_glasses import ModelGlasses
from shutil import move, rmtree
import os

# Model Glasses ========================================================================================================
# Remove model already saved in the Modules folder
# rmtree(os.path.join('../Modules', 'model_glasses'), ignore_errors=True)
training_batches, valid_batches, test_batches = glasses_data_preprocessing(data_directory='cartoon_glasses',
                                                                           filename_column='file_name',
                                                                           target_column='glasses', img_size=(224, 224),
                                                                           training_percentage_size=0.85,
                                                                           horizontal_flip=False,
                                                                           batches_size=16, validation_split=0.15)

input_shape = training_batches.image_shape
# Build model object.
model_ModelGlasses = ModelGlasses(input_shape)
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_ModelGlasses_train, acc_ModelGlasses_valid = model_ModelGlasses.train(training_batches, valid_batches, epochs=10,
                                                                          verbose=2, plot=True)
# Used only in test.py
model_ModelGlasses.evaluate(test_batches=test_batches, verbose=1)
# Test model based on the test set.
acc_ModelGlasses_test = model_ModelGlasses.test(test_batches, verbose=1, confusion_mesh=True)
# Print out your results with following format:
print('TA1:{},{}'.format(acc_ModelGlasses_train, acc_ModelGlasses_test))
# Move model saved in the Modules folder
move('model_glasses', '../Modules')
