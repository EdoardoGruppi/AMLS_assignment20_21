# Import packages
from A2.a2 import A2GridSearch
from Modules.pre_processing import hog_pca_preprocessing


X_test, X_train, X_valid, y_test, y_train, y_valid, pca, sc = hog_pca_preprocessing(dataset_name='celeba_smiles',
                                                                                    img_size=(96, 48),
                                                                                    validation_split=0.15,
                                                                                    variance=0.90,
                                                                                    training_size=0.85,
                                                                                    target_column='smiling')
# List of parameters to evaluate in each possible combination
parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1e-2, 1e-3, 1e-4, 'scale'], 'C': [0.1, 1, 5, 10, 100]}]
model_A2 = A2GridSearch(tol=0.01, parameters=parameters, n_jobs=-1)
# Train model based on the training set
acc_A2_train, acc_A2_valid = model_A2.train(X_train, X_valid, y_train, y_valid)
# Test model based on the test set.
acc_A2_test = model_A2.test(X_test, y_test, confusion_mesh=False)
# Print results
print('TA2: {:.4f}, {:.4f}, {:.4f}'.format(acc_A2_train, acc_A2_valid, acc_A2_test))
