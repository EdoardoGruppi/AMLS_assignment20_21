# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in the README.md file.
2. To install the face_recognition packet may be necessary to install dlib and cmake before. [At this link](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508) a complete guide is provided.
3. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/AMLS_assignment20_21).
4. Tensorflow enables to work directly on GPU without requiring explicity additional code. The only hardware requirement is having a Nvidia GPU card with Cuda enabled. To see if Tensorflow has detected a GPU run the following few lines.

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If not, there are lots of guides on the web to install everything you need. For instance, you can take a look at
   [this](https://deeplizard.com/learn/video/IubEtS2JAiY).

5. Finally, it is crucial to run the code below since Tensorflow tends to allocate directly all the GPU memory even if is not entirely needed. With these lines instead, it will allocate gradually the memory needed by the program.

   ```python
   if len(physical_devices) is not 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

   **Note:** during the several experiments conducted, each execution has needed only 0.9 GB of GPU memory. Furthermore, the instruments provided by Tensorflow make the GPU management during the execution significantly difficult. Therefore, differently from the main memory, no instructions has been written to deallocate dinamically the GPU memory. Nevertheless, if it is necessary a possible solution is proposed in the Issues Section below.

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal:

```
python main.py
```

**Note:** To follow step by step the main execution take a look on the dedicated Section below.

## Datasets

### CelebA dataset

The CelebA dataset provided contains 5000 celebrity images. Each one of them is associated to two labels that describe the celebrity gender and whether they are smiling. The table below depicts how the two categories for each label are divided.

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

## Model

In this project, several methodologies are proposed to deal with various tasks. Firstly, a CNN has been designed from scratch maintaining as lower as possible the simpleness of the network along with the memory requirements and the computational time. This architecture has been then adopted with minor amendments to solve the gender detection task (A1) as well as the eye-color (B2) and face-shape (B1) recognition problems. Finally, a different direction has been undertaken for smile detection (A2) leveraging on the efficacy of HOG and PCA algorithms along with the simpleness of a SVM as a classifier.

## Main execution

```python
data_directory, faces_not_detected = smiles_extraction(dataset_name='celeba')
```

## Issues

- If the device's GPU memory is not enough to run the code, it is possible to execute the training of each model inside a dedicated subprocess. Tensorflow then will release the part of GPU memory used by each subprocess as soon as it ends.
