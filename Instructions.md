# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in the [README.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/README.md) file.
2. To install the face_recognition packet may be necessary to install dlib and cmake before. [At this link](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508) a complete guide is provided.
   Hence, run this command to install the package.
   ```python
   pip install face-recognition
   ```
3. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/AMLS_assignment20_21).
4. Tensorflow enables to work directly on GPU without requiring explicity additional code. The only hardware requirement is having a Nvidia GPU card with Cuda enabled. To see if Tensorflow has detected a GPU run the following few lines (see main.py).

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If not, there are lots of guides on the web to install everything you need. For instance, you can take a look at
   [this](https://deeplizard.com/learn/video/IubEtS2JAiY).

5. Finally, it is crucial to run the code below since Tensorflow tends to allocate directly all the GPU memory even if is not entirely needed. With these lines instead, it will allocate gradually the memory required by the program (see main.py).

   ```python
   if len(physical_devices) is not 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

   **Note:** during the several experiments conducted, each project execution has needed only 0.9 GB of GPU memory. Furthermore, the instruments provided by Tensorflow make the GPU management during the execution significantly difficult. Therefore, differently from the main memory, no instructions has been written to deallocate dinamically the GPU memory. Nevertheless, if it is necessary a possible solution is proposed in the Issues Section below.

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal.

**Note:** To follow step by step the main execution take a look on the dedicated Section below.

```
python main.py
```

## Datasets

### CelebA dataset

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset provided contains 5000 celebrity images. Each one of them is associated to two labels that describe the celebrity gender and whether they are smiling. The table below depicts how the two categories for each label are divided.

| Class | Gender | Smiling |
| ----- | ------ | ------- |
| -1    | 2500   | 2500    |
| 1     | 2500   | 2500    |

### Cartoon_set dataset

Cartoon_set dataset is made up of 10000 avatar images. They are obtained by choosing randomly the avatar traits between 10 artworks, 4 colors and 4 proportions. The following table summarizes how the examples are distributed amongst the five possibilities for both: eye color and face shape.

| Class | Eye color | Face shape | Eye color (no black sunglasses) |
| ----- | --------- | ---------- | ------------------------------- |
| 0     | 2004      | 2000       | 1643                            |
| 1     | 2018      | 2000       | 1654                            |
| 2     | 1969      | 2000       | 1593                            |
| 3     | 1992      | 2000       | 1621                            |
| 4     | 2017      | 2000       | 1635                            |

### Dataset division

The rule of thumb followed throughout the division of both the datasets consists in assigning 80\% of the images to the training and validation sets. The remaining part is reserved to the test set. This rule is usually related to the Pareto principle: 20\% of causes produce 80\% of effects. However, since the celebA dataset is slightly small, I have opted to move its ratio from 60:20:20 to 70:15:15 dedicating therefore more pictures to the training phase.

## Models

In this project, several methodologies are proposed to deal with various tasks. Firstly, a CNN has been designed from scratch maintaining as lower as possible the simpleness of the network along with the memory requirements and the computational time. This architecture has been then adopted to perform the gender detection task (A1) as well as, with minor amendments, the eye-color (B2) and face-shape (B1) recognition problems. Finally, a different direction has been undertaken for smile detection (A2) leveraging on the efficacy of HOG and PCA algorithms along with the simpleness of a SVM as a classifier.

## Main execution

Before the code execution, the Datasets folder must have the following structure.

![image](https://user-images.githubusercontent.com/48513387/100546886-065feb80-3264-11eb-97a5-fc698833878b.png)

The celeba and cartoon_set folders contain the starting datasets from which training, validation and test sets are generated. The others folders instead contain a second larger test dataset provided subsequently. At the beginning the `smiles_extraction` function extracts smiles from the original celeba dataset and the final celeba test dataset. The images generated are saved in the celeba_smiles and celeba_test_smiles folders. They will be used in the Task A2.

```python
data_directory, faces_not_detected = smiles_extraction(dataset_name='celeba')
test_directory, faces_not_detected1 = smiles_extraction(dataset_name='celeba_test')
```

![image](https://user-images.githubusercontent.com/48513387/100548200-66a65b80-326b-11eb-9453-c8e02ce3042e.png)

Then, the Task A1 execution starts. Batches are prepared through the `data_preprocessing` function and the original dataset is divided in train, validation and test sets. A new folder (celeba_testing) is created to contain the first test dataset.

```python
training_batches, valid_batches, test_batches = data_preprocessing(...)
```

![image](https://user-images.githubusercontent.com/48513387/100548250-aa00ca00-326b-11eb-8bff-6b60d54a45ad.png)

The model is therefore trained, validated and tested on the original dataset.

```python
# Build model object.
model_A1 = A1(...)
# Train model based on the training set
acc_A1_train, acc_A1_valid = model_A1.train(...)
# Test model based on the test set.
acc_A1_test = model_A1.test(...)
```

It is also tested on the second test set. Finally, memory is cleaned.

```python
# Test the model on the second larger test dataset provided
test_batches = test_data_preparation(...)
acc_A1_test2 = model_A1.test(...)
# Print out your results with following format:
print('TA1: {}, {}, {}, {}'.format(acc_A1_train, acc_A1_valid, acc_A1_test, acc_A1_test2))
# Clean up memory/GPU etc
del ...
```

Task A2 starts. Examples in the celeba_smiles folder and their related labels are divided in three parts by means of the `hog_pca_preprocessing` function. Then the model is trained, validated and tested on them. Finally, it is also tested on the second test set and memory is cleaned.

```python
X_test, X_train, X_valid, y_test, y_train, y_valid, pca, sc = hog_pca_preprocessing(...)
# Build model object.
model_A2 = A2(...)
# Train model based on the training set
acc_A2_train, acc_A2_valid = model_A2.train(...)
# Test model based on the test set.
acc_A2_test = model_A2.test(...)
# Test the model on the second larger test dataset provided
x_test, y_test = test_hog_pca_preprocessing(...)
acc_A2_test2 = model_A2.test(...)
# Print out your results with following format:
print('TA2: {}, {}, {}, {}'.format(acc_A2_train, acc_A2_valid, acc_A2_test, acc_A2_test2))
# Clean up memory/GPU etc
del ...
```

At the begnning of Task B1 the `data_preprocessing` splits the dataset and create a new folder (cartoon_set_testing) for the first dataset of the cartoon_set.

```python
training_batches, valid_batches, test_batches = data_preprocessing(...)
```

![image](https://user-images.githubusercontent.com/48513387/100548294-e7655780-326b-11eb-82cf-223a889ad98b.png)

The dedicated model follows the same procedure of the previous ones.

```python
# Build model object.
model_B1 = B1(input_shape)
# Train model based on the training set
acc_B1_train, acc_B1_valid = model_B1.train(...)
# Test model based on the test set.
acc_B1_test = model_B1.test(...)
# Test the model on the second larger test dataset provided
test_batches = test_data_preparation(...)
acc_B1_test2 = model_B1.test(...)
# Print out your results with following format:
print('TB1: {}, {}, {}, {}'.format(acc_B1_train, acc_B1_valid, acc_B1_test, acc_B1_test2))
# Clean up memory/GPU etc
del ...
```

Once Task B1 is performed the `delete_glasses` function removes all the avatars in the various cartoon_set folders that wear balck sunglasses and move them into a dedicated folder called cartoon_set_removed.

![image](https://user-images.githubusercontent.com/48513387/100548328-1a0f5000-326c-11eb-9a08-b94832b642f2.png)

Again the B2 model follows the same process.

```python
# Build model object.
model_B2 = B2(input_shape)
# Train model based on the training set
acc_B2_train, acc_B2_valid = model_B2.train(...)
# Test model based on the test set.
acc_B2_test = model_B2.test(...)
# Test the model on the second larger test dataset provided
test_batches = test_data_preparation(...)
acc_B2_test2 = model_B2.test(...)
# Print out your results with following format:
print('TB2: {}, {}, {}, {}'.format(acc_B2_train, acc_B2_valid, acc_B2_test, acc_B2_test2))
# Clean up memory/GPU etc
del ...
```

## Issues

- If the device's GPU memory is not enough to run the code, it is possible to execute the training of each model inside a dedicated subprocess. Tensorflow then will release the part of GPU memory used by each subprocess as soon as it ends.
