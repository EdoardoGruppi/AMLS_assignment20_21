# Description of the project

[Project](https://github.com/EdoardoGruppi/AMLS_assignment20_21.git) ~ [Guide](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/Instructions.md)

In this report, four distinct challenging scopes are addressed under the supervised machine learning paradigm. They comprise binary classification tasks for gender (A1) and smile detection (A2) along with multi-categorical classification tasks concerning eye-colour (B2) and face-shape (B1) recognition. Most notably, several methodologies are proposed to deal with these duties (see Models Section in [Instruction.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/Instructions.md) to have more details).

|                                       |                   Task A1                    |                                                Task A2                                                 |             Task B1              |                                       Task B2                                       |
| ------------------------------------- | :------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------: | :---------------------------------------------------------------------------------: |
| Dataset                               |                    CelebA                    |                                                 CelebA                                                 |           Cartoon Set            |                                     Cartoon Set                                     |
| Dataset division                      |                   70:15:15                   |                                                70:15:15                                                |             60:20:20             |                                      60:20:20                                       |
| Original examples                              |                 5.000 images                 |                                              5.000 images                                              |          10.000 images           |                                    10.000 images                                    |
| Size of each image                    |                  178x218x3                   |                                               178x218x3                                                |            500x500x3             |                                      500x500x3                                      |
| First operations                      |                     None                     | Smiles are extracted by means of face_recognition models from images previously converted in grayscale |               None               | Harmful images are removed with the pre-trained model_glasses specifically designed |
| Examples                              |                  Unchanged                   |                                              4990 images                                               |            Unchanged             |                                     8146 images                                     |
| New image size                        |                  Unchanged                   |                                                96x48x1                                                 |            Unchanged             |                                      Unchanged                                      |
| Pre-processing                        |       Images are rescaled and reshaped       |         HOG features extracted from smile images are standardised before being reduced by PCA          | Images are rescaled and reshaped |                          Images are rescaled and reshaped                           |
| Data augmentation on training dataset | Images are randomly and horizontally flipped |                                                  None                                                  |               None               |                                        None                                         |
| Input example shape                   |                   96x96x3                    |                                                 360x1                                                  |            224x224x3             |                                      224x224x3                                      |
| Model                                 |                     CNN                      |                                                  SVM                                                   |               CNN2               |                                        CNN2                                         |
| Batch size                            |                      16                      |                                                  None                                                  |                16                |                                         16                                          |
| Epoch                                 |                      25                      |                                                  None                                                  |                10                |                                         10                                          |

## How to start

A comprehensive guide concerning how to run the code along with additional information is provided in the file [Instruction.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/Instructions.md).

The packages required for the execution of the code along with the role of each file and the software used are described in the Sections below.

## Packages required

The following lists gather all the packages needed to run the project code.
Please note that the descriptions provided in this subsection are taken directly from the package source pages. For more details it is reccomended to directly reference to their official sites.

**Compulsory :**

- **Pandas** provides fast, flexible, and expressive data structures designed to make working with structured and time series data both easy and intuitive.

- **Numpy** is the fundamental package for array computing with Python.

- **Tensorflow** is an open source software library for high performance numerical computation. Its allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been completely wrapped within Tensorflow.

- **Pathlib** offers a set of classes to handle filesystem paths.

- **Shutil** provides a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **Sklearn** offers simple and efficient tools for predictive data analysis.

- **Skimage** is a collection of algorithms for image processing.

- **Random** implements pseudo-random number generators for various distributions.

- **Seaborn** is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

- **Cv2** is an open-source library that includes several hundreds of computer vision algorithms.

- **Face_recognition** is useful to recognize and manipulate faces with the world’s simplest face recognition library. Built from dlib’s state-of-the-art deep learning library.

**Optional :**

- **Comet_ml** helps to manage and track machine learning experiments.

- **Vprof** is a Python package providing rich and interactive visualizations for various Python program characteristics such as running time and memory usage.

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images and instantiate, train and test models.

**a1.py** contains the class A1 from which to instantiate the CNN model for gender detection. Once the model is created, it provides functions in order to be trained and evaluated and also to predict the class membership of unlabelled examples.

**a2.py** contains the class A2 from which to instantiate the HOG+SVM model for smiles detection. Once the model is created, it provides functions in order to be trained and to predict the class membership of unlabelled examples.

**b1.py** contains the class B1 from which to instantiate the CNN model used for face-shape recognition. Once the model is created, it provides functions in order to be trained and evaluated and also to predict the class membership of unlabelled examples.

**b2.py** contains the class B2 from which to instantiate the CNN model used for eye-color recognition. Once the model is created, it provides functions in order to be trained and evaluated and also to predict the class membership of unlabelled examples.

**config.py** makes available all the global variables used in the project.

**pre_processing.py** provides crucial functions related to the data preparation. `data_preprocessing`: splits the original dataset into three different parts for training, validation and testing; rescales and reshapes images; performs data augmentation; and prepares batches to feed the models. It is called in the main.py file for Tasks A1, B1 and B2. `hog_pca_preprocessing` is instead called exclusively for Task A2. Primarily, it extracts meaningful features from images by means of the Histograms of Oriented Gradients (HOG) descriptor. Secondarily, it separates datasets in three parts. Then, it standardizes features before reducing data dimensionality via the Principal Component Analysis (PCA) algorithm. The last function `hog_pca_augmentation_preprocessing` follows the process pipeline just described for Task A2 but allowing to perform data_augmentation on training images.

**delete_glasses.py** includes the homonymous function to delete avatars that wear black sunglasses making unfeasible eye-color detection in task B2. It employs a pre trained model created specifically and saved in the model_glasses directory within the Modules folder.

**face_extraction.py** leverages the external face_recognition package to extract faces or smiles through the `face_extraction` and `smiles_extraction` functions respectively. The latter is adopted during the images pre-processing of Task A2.

**results_visualization.py** exploits the seaborn and matplotlib libraries to plot the performance and learning curves of the training phase and to generate confusion matrices summarizing the models results.

**test_pre_processing.py** contains functions to prepare the test batches starting from the test datasets provided subsequently.

**\_Additional_code folder** includes some .py files useful for the code devolepment as well as to report the most noteworthy experiments conducted during the project. In particular, `model_glasses.py`, `main_glasses.py` and `glasses_data_preparation.py` show the code employed to create from scratch the pre-trained model used to remove avatars with black glasses in Task B2 . `grid_search.py` allowed to select the optimal pair of c and gamma values for the SVM model. `training_A2_plot.py` was used to plot the training phase of the SVM. `face_net.py` and `viola_jones.py` are some alternatives taken into account to extract smiles in Task A2. `normalizing.py` could help to normalize and standardize images (featurewise) before training the models. It returns the mean and the standard deviation computed on all the images in a given folder. Finally, `test.py` was **exclusively** used as main file during the development of the code in order to preserve the structure of the official `main.py` throughout this phase. **Note:** to run one of the files within this folder (although it is not necessary for the execution of the project), it may be required to move it outside the folder to work properly.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the most advanced working environments and for its ease of use.

> <img src="https://camo.githubusercontent.com/9e56fd69605928b657fcc0996cebf32d5bb73c46/68747470733a2f2f7777772e636f6d65742e6d6c2f696d616765732f6c6f676f5f636f6d65745f6c696768742e706e67" width="140" alt="comet">

Comet is a cloud-based machine learning platform that allows data scientists to track, compare and analyse experiments and models.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular machine learning libraries and it offers GPUs where you can execute the code as well.

<!---
## References

```
@article{citation-example,
  title={Image-to-Image Translation},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={CVPR},
  year={2017}
}
```
--->
