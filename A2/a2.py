# Import packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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
        :param tol: tolerance to stop the training phase. default_value=0.001
        :param max_iter: maximum number of iterations to execute. It allows to stop the learning phase before
            tol reaches the value required. default_value=8000
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


# Class useful to select the optimal (c, gamma) values. To have more information see the file grid_search.py
# inside the _Additional_code folder
class A2GridSearch:
    def __init__(self, parameters, tol=0.001, n_jobs=None, cv=None):
        """
        Class that allows to search the optimal values of c and gamma from a dictionary of parameters.

        :param tol: tolerance to stop the training phase. default_value=0.001
        :param parameters: dictionary of parameters evaluated by the grid search algorithm if prod=False.
            default_value=None
        :param n_jobs: number of jobs to run in parallel. default_value=None means 1
        :param cv: cross-validation splitting strategy. default_value=None means 5 folds.
        :returns it displays the best parameters and the scores reached during the evaluation.
        """
        self.model = GridSearchCV(SVC(tol=tol), parameters, n_jobs=n_jobs, cv=cv, return_train_score=True)

    def train(self, x_train, x_valid, y_train, y_valid):
        """
        Trains the model until the stopping criterion is met.

        :param x_train: matrix composed by n_examples training vectors of dimension n_features.
        :param x_valid: matrix composed by n_examples vectors of n_features used for validation.
        :param y_train: target values of all the training vectors.
        :param y_valid: target values of all the validation vectors.
        :return: the accuracies measured on the training and validation sets once the model with the best parameters is
            refit.
        """
        print('Training the Support Vector Machine...')
        self.model.fit(x_train, y_train)
        print("Best parameters set found on development set:\n", self.model.best_params_,
              "\n\nGrid scores on development set:")
        means = self.model.cv_results_['mean_test_score']
        stds = self.model.cv_results_['std_test_score']
        means_tr = self.model.cv_results_['mean_train_score']
        stds_tr = self.model.cv_results_['std_train_score']
        for mean_tr, std_tr, mean, std, params in zip(means_tr, stds_tr, means, stds, self.model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) - %0.3f (+/-%0.03f) for %r" % (mean_tr, std_tr * 2, mean, std * 2, params))
        y_pred = self.model.predict(x_valid)
        print("\nDetailed classification report:",
              "\nThe model is trained on the full development set.",
              "\nThe scores are computed on the full evaluation set.\n",
              classification_report(y_valid, y_pred),
              '\nAccuracy scores on the final model:\n')
        return accuracy_score(y_train, self.model.predict(x_train)), accuracy_score(y_valid, y_pred)

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
