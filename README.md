# Description of the project

[Project](https://github.com/EdoardoGruppi/AMLS_assignment20_21.git)

SHORT DESCRIPTION OF THE PROJECT

CHANGE REFERENCES

## Before starting

### Packages required

The following lists gather all the packages needed to run the project code.
Please note that the descriptions provided in this subsection are taken directly from the package source pages. In order to have more details on them it is reccomended to directly reference to their official sites.

Compulsory

- `Pandas` provides fast, flexible, and expressive data structures designed to make working with structured and time series data both easy and intuitive.

- `Numpy` is the fundamental package for array computing with Python.

- `Tensorflow` is an open source software library for high performance numerical computation. Its allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been completely wrapped within Tensorflow.

- `Pathlib` offers a set of classes to handle filesystem paths.

- `Shutil` provides a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

- `Os` provides a portable way of using operating system dependent functionality.

- `Matplotlib` is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- `Sklearn` offers simple and efficient tools for predictive data analysis.

- `Skimage` is a collection of algorithms for image processing.

- `Random` implements pseudo-random number generators for various distributions.

- `seaborn` is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

- `cv2` is an open-source library that includes several hundreds of computer vision algorithms.

- `face_recognition` Recognize and manipulate faces with the world’s simplest face recognition library. Built from dlib’s state-of-the-art deep learning library.

Optional

- `comet_ml` helps to manage and track machine learning experiments.

- `vprof` is a Python package providing rich and interactive visualizations for various Python program characteristics such as running time and memory usage.

### Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is a cross platform integrated development environment (IDE) for Python programmers. The choice
fell on it because of its ease of use while remaining one of the most advanced working environments.

> <img src="https://camo.githubusercontent.com/9e56fd69605928b657fcc0996cebf32d5bb73c46/68747470733a2f2f7777772e636f6d65742e6d6c2f696d616765732f6c6f676f5f636f6d65745f6c696768742e706e67" width="140" alt="comet">

Comet is a cloud-based machine learning platform that allows data scientists to track, compare and
analyse experiments and models.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is a environment that enables to run python notebook entirely in the cloud. It supports many
popular machine learning libraries and it offers GPUs where you can execute the code as well.

### Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images and instantiate, train and test models.

**a1.py** contains the class A1 from which to instantiate the CNN model for gender detection. Once the model is created, it provides functions to be trained, evaluated and also to predict the class membership of unlabelled examples.

**a2.py** contains the class A2 from which to instantiate the HOG+SVM model for smiles detection. Once the model is created, it provides functions to be trained and to predict the class membership of unlabelled examples.

**b1.py** contains the class B1 from which to instantiate the CNN model used for face-shape recognition. Once the model is created, it provides functions to be trained, evaluated and also to predict the class membership of unlabelled examples.

> **b2.py** contains the class B2 from which to instantiate the CNN model used for eye-color recognition. Once the model is created, it provides functions to be trained, evaluated and also to predict the class membership of unlabelled examples.

> **pre_processing.py** provides crucial functions related to data preparation. _data_preprocessing_ splits the original dataset in three different parts for training, validation and testing; rescales and reshapes images; applies data augmentation; and prepares batches to feed the models. It is called in the main.py file for Tasks A1, B1 and B2. _hog_pca_preprocessing_ instead is called exclusively for Task A2. Primarily, it extracts meaningful features with the Histogram of Oriented Gradients (HOG) descriptor from a given image. Secondly, it separates datasets in three parts. Then, it normalizes features before reducing data dimensionality via the Principal Component Analysis (PCA) algorithm. The last function it provides (_hog_pca_augmentation_preprocessing_) follows the process pipeline just described for Task A2 but allowing to apply data_augmentation on the training images.

> **delete_glasses.py** includes the homonymous function to delete avatars that wear black sunglasses making not feasible the eye-color detection in the task B2. It employs a pre trained model created specifically and saved in the model_glasses directory within the Modules folder.

> **face_extraction.py** leverages the face*recognition package to extract faces or smiles through the \_face_extraction* and _smiles_extraction_ functions respectively. The latter is adopted during the images pre-processing of Task A2.

> **results_visualization.py** exploits the seaborn and matplotlib libraries to plot the performance and learning curves of the training phase and to generate confusion matrices summarizing the testing results.

> **test_pre_processing.py** contains functions to prepare the test batches from the test datasets provided subsequently.

> **\_Additional_code folder** includes some .py files of the most noteworthy experiments conducted during the project. In particular, _model_glasses.py_, _main_glasses.py_ and _glasses_data_preparation.py_ show the code employed to create from scratch the pre-trained model used to remove avatars with black glasses in Task B2 . _face_net.py_ and _viola_jones.py_ are some alternatives taken into account to extract smiles in Task A2. _normalizing.py_ instead has been considered to normalize images (featurewise) before training the model. It returns the mean and the std computed on all the images in a given folder.

## How to start

A comprehensive guide is provided in the file [Instruction.md](https://github.com/EdoardoGruppi/AMLS_assignment20_21/blob/main/Instructions.md).

## References

```
@article{citation-example,
  title={Image-to-Image Translation},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={CVPR},
  year={2017}
}
```
