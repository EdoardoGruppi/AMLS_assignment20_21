# import packages
from Modules.pre_processing import hog_pca_preprocessing
from A2.a2 import A2
from matplotlib import pyplot as plt
import seaborn as sn
from numpy import arange

X_test, X_train, X_valid, y_test, y_train, y_valid, pca, sc = hog_pca_preprocessing(dataset_name='celeba_smiles',
                                                                                    img_size=(96, 48),
                                                                                    validation_split=0.15,
                                                                                    variance=0.90,
                                                                                    training_size=0.85,
                                                                                    target_column='smiling')
# Note: to run this code you have to move outside the folder.
# The following code plots the accuracy obtained in different moments of the training phase.
# to find the correct tol run the code multiple times with different values and compare the plots.
# The good tol is the one that stops the training when the validation and training accuracy do not increase anymore.
# For instance, tol=0.1 interrupts the learning phase before than tol=0.01 but at the same time it permits to arrive at
# convergence. Therefore, choosing tol=0.1 could be a better solution to save time.
train_acc, valid_acc = [], []
for i in arange(0, 3500, 100):
    # # Build model object.
    model = A2(kernel='rbf', gamma=0.001, c=5, verbose=2, tol=0.1, max_iter=i)
    # # Train model based on the training set
    acc_train, acc_valid = model.train(X_train, X_valid, y_train, y_valid)
    train_acc.append(acc_train)
    valid_acc.append(acc_valid)
    acc_test = model.test(X_test, y_test, confusion_mesh=False)
    print(acc_test)
sn.set()
fig = plt.figure()
plt.plot(train_acc, marker='.')
plt.plot(valid_acc, marker='.')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()

# Print out your results with following format:
print('Train Accuracy: ', train_acc, '\nValid Accuracy: ', valid_acc)
