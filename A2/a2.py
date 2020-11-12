# todo reduce imports as much as possible like import seaborn to from seaborn import...
# Import packages
from comet_ml import Experiment
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv2D, Activation
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, classification_report


class A2:
    def __init__(self, input_shape):
        # Parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
        # is a infinitely repeating dataset
        self.model = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            # Fraction of the input units dropped
            Dropout(rate=0.5),
            Dense(units=2, activation='softmax')
        ])
        self.model.summary()
        # Using a 'binary_crossentropy' we would obtain one output, rather than two
        # In that case the activation of the last layer must be a 'sigmoid'
        # Alternatively it is possible to use a 'categorical_crossentropy' with a 'softmax' in the last layer
        # In that case it is compulsory to insert class_mode='binary' in flow_from_dataframe() functions ...
        # ...and predicted_labels = np.array(predictions).astype(int).flatten() instead of...
        # ...predicted_labels = np.array(np.argmax(predictions, axis=-1))
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        # # todo comet_ml
        # self.experiment = Experiment(api_key="hn5we8X3ThjkDumjfdoP2t3rH", project_name="convnet",
        #                              workspace="edoardogruppi")

    def train(self, training_batches, valid_batches, epochs=35, verbose=2):
        # Training phase
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=len(training_batches),
                                 validation_data=valid_batches,
                                 validation_steps=len(valid_batches),
                                 epochs=epochs,
                                 verbose=verbose
                                 )
        # Return accuracy on the train and validation dataset
        return history.history['val_accuracy'][-1]

    def test(self, test_batches, verbose=1, confusion_mesh=False, class_labels=None):
        # Steps parameter indicates on how many batches are necessary to work on each data on the Testing dataset
        # model.predict returns the predictions made on the input
        predictions = self.model.predict(x=test_batches, steps=len(test_batches), verbose=verbose)
        predictions = np.round(predictions)
        predicted_labels = np.array(np.argmax(predictions, axis=-1))
        true_labels = np.array(test_batches.classes)
        if confusion_mesh:
            confusion_grid = pd.crosstab(true_labels, predicted_labels, normalize=True)
            # Generate a custom diverging colormap
            color_map = sn.diverging_palette(355, 250, as_cmap=True)
            sn.heatmap(confusion_grid, cmap=color_map, vmax=0.5, vmin=0, center=0, xticklabels=class_labels,
                       yticklabels=class_labels, square=True, linewidths=2, cbar_kws={"shrink": .5}, annot=True)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()
        print(classification_report(true_labels, predicted_labels))
        # # todo comet_ml
        # self.experiment.log_confusion_matrix(true_labels, predicted_labels)
        # self.experiment.end()
        # Return accuracy on the test dataset
        return accuracy_score(true_labels, predicted_labels)

    def evaluate(self, test_batches, verbose=1):
        # model.evaluate predicts the output and returns the metrics function specified in model.compile()
        score = self.model.evaluate(x=test_batches, verbose=verbose)
        print(score)