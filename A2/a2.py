# Import packages
from sklearn import svm
from sklearn.metrics import accuracy_score
from Modules.results_visualization import plot_history, plot_confusion_matrix
from numpy import array


class A2:
    def __init__(self):
        print('Training the Support Vector Machine...')
        self.model = svm.SVC()

    def train(self, x_train, x_valid, y_train, y_valid, plot=False):
        self.model.fit(x_train, y_train)
        print('Train accuracy: {}'.format(self.model.score(x_train, y_train)))
        validation_accuracy = self.model.score(x_valid, y_valid)
        print('Validation accuracy: {}'.format(validation_accuracy))
        # todo
        # if plot:
        #     # Plot loss and accuracy achieved on training and validation dataset
        #     plot_history()
        # Return accuracy on validation dataset
        return validation_accuracy

    def test(self, x_test, y_test, confusion_mesh=False, class_labels='auto'):
        # Predict labels
        # In svm the predictions are not probabilities of belonging in each class.
        # For each image it returns a single element (e.g. 1 or 0)
        predicted_labels = self.model.predict(x_test)
        y_test = array(y_test)
        # Plot results through a confusion matrix
        if confusion_mesh:
            plot_confusion_matrix(class_labels, predicted_labels, y_test)
        return accuracy_score(y_test, predicted_labels)
