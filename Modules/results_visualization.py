# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import classification_report


def plot_history(accuracy, val_accuracy, loss, val_loss, title=None):
    """
    Plots the history of the training phase and validation phase. It compares in two different subplots the accuracy
    and the loss of the model.

    :param accuracy: list of values for every epoch.
    :param val_accuracy: list of values for every epoch.
    :param loss: list of values for every epoch.
    :param val_loss: list of values for every epoch.
    :param title: tile of the figure printed. default_value=None
    :return:
    """
    x_axis = [i for i in range(1, len(accuracy)+1)]
    sn.set()
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    # First subplot
    plt.subplot(211)
    plt.plot(x_axis, accuracy)
    plt.plot(x_axis, val_accuracy)
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'], loc='lower right')
    # Second subplot
    plt.subplot(212)
    plt.plot(x_axis, loss)
    plt.plot(x_axis, val_loss)
    plt.ylabel('Loss')
    plt.ylim(top=0.7)
    plt.xlabel('Epoch')
    # Legend
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.show()


def plot_confusion_matrix(class_labels, predicted_labels, true_labels, title=None):
    """
    Plots the confusion matrix given both the true and predicted results.

    :param class_labels: list of the names of the labels.
    :param predicted_labels: list of the predicted labels.
    :param true_labels: list of the true labels.
    :param title: tile of the figure printed
    :return:
    """
    sn.set()
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    confusion_grid = pd.crosstab(true_labels, predicted_labels, normalize=True)
    # Generate a custom diverging colormap
    color_map = sn.diverging_palette(355, 250, as_cmap=True)
    sn.heatmap(confusion_grid, cmap=color_map, vmax=0.5, vmin=0, center=0, xticklabels=class_labels,
               yticklabels=class_labels, square=True, linewidths=2, cbar_kws={"shrink": .5}, annot=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    # Print a detailed report on the classification results
    print('\nClassification Report:\n')
    print(classification_report(true_labels, predicted_labels))





