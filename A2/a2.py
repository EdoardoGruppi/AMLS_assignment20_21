# Import packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Modules.results_visualization import plot_confusion_matrix
from numpy import array

from sklearn.model_selection import GridSearchCV


class A2:
    def __init__(self, kernel='rbf', gamma='scale', c=1, verbose=False, tol=0.001, max_iter=8000):
        """
        Defines and configures the SVM model.

        :param kernel: kernel type adopted in the algorithm.
        :param gamma: coefficient used for non-linear kernels. default_value='scale' means
            1/(n_features * x_train.var())
        :param c: regularization parameter that introduces a squared l2 penalty. default_value=1
        :param verbose: verbosity level. default_value=False
        """
        self.model = SVC(kernel=kernel, gamma=gamma, C=c, verbose=verbose, tol=tol, max_iter=max_iter)

    def train(self, x_train, x_valid, y_train, y_valid):
        """
        Trains the model until the stopping criterion is met.

        :param x_train: matrix composed by n_examples training vectors of dimension n_features.
        :param x_valid: matrix composed by n_examples vectors of n_features used for validation.
        :param y_train: target values of all the training vectors.
        :param y_valid: target values of all the validation vectors.
        :return: the accuracies measured on the training and validation sets.
        """
        print('Training the Support Vector Machine...')
        self.model.fit(x_train, y_train)
        train_accuracy = self.model.score(x_train, y_train)
        validation_accuracy = self.model.score(x_valid, y_valid)
        return train_accuracy, validation_accuracy

    def test(self, x_test, y_test, confusion_mesh=True, class_labels='auto'):
        """
        Generates output predictions for the input examples and compares them with the true labels returning
        the accuracy gained.

        :param x_test: matrix composed by n_examples vectors of n_features used for test.
        :param y_test: target values of all the test vectors.
        :param confusion_mesh: if True it plots the confusion matrix. default_value=True
        :param class_labels: list of the class names used in the confusion matrix. default_value='auto'
        :return: the test accuracy score
        """
        # Predict labels
        # In svm the predictions are not probabilities of belonging in each class.
        # For each image it returns a single element (e.g. 1 or 0)
        predicted_labels = self.model.predict(x_test)
        y_test = array(y_test)
        # Plot results through a confusion matrix
        if confusion_mesh:
            plot_confusion_matrix(class_labels, predicted_labels, y_test)
        return accuracy_score(y_test, predicted_labels)
