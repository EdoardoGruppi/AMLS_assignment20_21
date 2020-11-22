# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in the README.md file.
2. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/AMLS_assignment20_21)
3. (Optional) Tensorflow enables to work directly on GPU without requiring explicity additional code.
   The only hardware requirement is having a Nvidia GPU card with Cuda enabled.

   To see if Tensorflow has detected a GPU run the following few lines.

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If not, there are lots of guides on the web to install everything you need. For instance take a look at
   [this](https://deeplizard.com/learn/video/IubEtS2JAiY).

## How to compile

Once all the necessary packages have been installed you can run the code by typing this line
on the terminal:

```
python main.py
```

## Datasets

### CelebA dataset

The CelebA dataset provided contains 5000 celebrity images. Each one of them is associated
to two labels that describe the celebrity gender and whether they are smiling.

The table below depicts how the two categories for each label are divided.

| Class | Gender | Smiling |
| ----- | ------ | ------- |
| -1    | 2500   | 2500    |
| 1     | 2500   | 2500    |

### Cartoon_set dataset

Cartoon_set dataset is made up of 10000 avatar images. They are obtained by choosing
randomly the avatar traits between 10 artworks, 4 colors and 4 proportions.

The following table summarizes how the examples are distributed amongst the five
possibilities for both: eye color and face shape.

| Class | Eye color | Face shape | Eye color (no black sunglasses) |
| ----- | --------- | ---------- | ------------------------------  |
| 0     | 2004      | 2000       | 1643                            |
| 1     | 2018      | 2000       | 1654                            |
| 2     | 1969      | 2000       | 1593                            |
| 3     | 1992      | 2000       | 1621                            |
| 4     | 2017      | 2000       | 1635                            |

### Dataset division

## Data Processing

The random algorithm in `data_processing()` allows to achieve a fair division in both training
and test dataset.

## Model

## References

```
@article{citation-example,
  title={Image-to-Image Translation},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={CVPR},
  year={2017}
}
```
