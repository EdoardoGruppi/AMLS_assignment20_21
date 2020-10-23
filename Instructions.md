# Instructions

## Setup

1. Install Tensorflow and/or Tensorflow GPU
2. Install Libraries as Cuda/cudnn [to work directly on GPU](https://deeplizard.com/learn/video/IubEtS2JAiY). Control if it is needed whether the [set_memory_growth](https://deeplizard.com/learn/video/Boo6SmgmHuM) is setted.

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

## How to compile

## Dataset Preparation

### CelebA dataset

| Group | Gender | Smiling |
| ----- | ------ | ------- |
| 1°    | 2500   | 2500    |
| 2°    | 2500   | 2500    |

The random algorithm in `data_processing()` allows to achieve a fair division in both training and
test dataset.

### Cartoon_set dataset

## Data Processing

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
