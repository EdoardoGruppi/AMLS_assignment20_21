# Import packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score
from Modules.results_visualization import plot_history, plot_confusion_matrix


class ModelGlasses:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=2, activation='softmax')
        ])
        self.model.summary()
        # Alternatively it is possible to use a 'binary_crossentropy' obtaining one output rather than two.
        # In that case the activation of the last layer must be a 'sigmoid', class_mode must be binary in
        # flow_from_dataframe() functions and in the test function below there will be ...
        # ...predicted_labels = np.array(predictions).astype(int).flatten() instead of...
        # ...predicted_labels = np.array(np.argmax(predictions, axis=-1))
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, training_batches, valid_batches, epochs=35, verbose=2, plot=False):
        # Parameters needed because fit() will run forever since image_generator.flow_from_dataframe()
        # is a infinitely repeating dataset
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=len(training_batches),
                                 validation_data=valid_batches,
                                 validation_steps=len(valid_batches),
                                 epochs=epochs,
                                 verbose=verbose
                                 )
        if plot:
            # Plot loss and accuracy achieved on training and validation dataset
            plot_history(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'],
                         history.history['val_loss'])
        # Return accuracy on validation dataset
        return history.history['val_accuracy'][-1]

    def test(self, test_batches, verbose=1, confusion_mesh=False, class_labels='auto'):
        # Steps parameter indicates how many batches are necessary to work on each data in the testing dataset
        # model.predict returns the predictions made on the input given
        # It returns the probabilities that each image belongs to the existing classes
        predictions = self.model.predict(x=test_batches, steps=len(test_batches), verbose=verbose)
        # Transform each prediction to an hot-encoding vector
        predictions = np.round(predictions)
        # The image is associated to the class with the highest probability
        predicted_labels = np.array(np.argmax(predictions, axis=-1))
        # Retrieve the true labels of the input
        true_labels = np.array(test_batches.classes)
        # Plot results through a confusion matrix
        if confusion_mesh:
            plot_confusion_matrix(class_labels, predicted_labels, true_labels)
        self.model.save('model_glasses')
        return accuracy_score(true_labels, predicted_labels)

    def evaluate(self, test_batches, verbose=1):
        # model.evaluate predicts the output and returns the metrics function specified in model.compile()
        score = self.model.evaluate(x=test_batches, verbose=verbose)
        print(score)
