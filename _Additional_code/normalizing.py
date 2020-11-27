# import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def calculate_statistic(data_path, image_size, batch_size=32, color_mode='grayscale'):
    """
    It computes the mean and std on a series of batch_size images taken from the data_path folder.

    :param data_path: path to the folder where images are hosted.
    :param image_size: dimension of each image during the subsequent process wherein batches are prepared.
    :param batch_size: number of images used to calculate the mean and standard deviation. To compute the mean or std on
        the entire training set pass the path of the folder and put batch_size equal to the number of images in the
        folder. Note that too much images will provoke memory issues. default_value=32.
    :param color_mode: 'rgb' if images involved have 3 channels. Otherwise 'grayscale'. default_value='grayscale'.
    :return: the mean and the standard deviation calculated on a batch of batch_size images.
    """
    # Rescale images so that each pixel value is inside the interval [0,1]
    data_gen = ImageDataGenerator(rescale=1. / 255.)
    # Load the batch generator
    gen_dataset = data_gen.flow_from_directory(data_path, target_size=image_size, shuffle=False, batch_size=batch_size,
                                               color_mode=color_mode, class_mode=None)
    # Instantiate required batch of batch_size dimension
    dataset = next(gen_dataset)
    print(dataset.shape)
    # Calculate the mean and the std of the batch
    mean_image = dataset.mean(axis=0)
    std_image = dataset.std(axis=0)
    print(mean_image.shape)
    print(std_image.shape)
    return mean_image, std_image

# In data_preprocessing ================================================================================================
#
# from centering import calculate_mean
#
# image_mean, image_std = calculate_statistic(data_directory, img_size, batch_size=num_examples)
# image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split,
#                                      horizontal_flip=True, featurewise_center=True,
#                                      featurewise_std_normalization=True)
# image_generator.mean = image_mean
# image_generator.std = image_std
#
