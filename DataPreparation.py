# Import necessary modules
from configparser import ConfigParser

import tensorflow as tf
from keras.datasets import mnist
from numpy import concatenate

# Read configurations from the 'configs.ini' file
parser = ConfigParser()
parser.read("configs.ini")

# Set configurations based on the values in 'configs.ini'
NOISE_SIZE: int = int(parser["DEFAULT"]["noise_size"])
BATCH_SIZE: int = int(parser["DEFAULT"]["batch_size"])
EPOCHS: int = int(parser["DEFAULT"]["epochs"])
LEARNING_RATE: float = float(parser["DEFAULT"]["learning_rate"])

# Load MNIST dataset and filter images labeled as '3' (images of triplets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_data = concatenate((x_train, x_test), axis=0)
y_data = concatenate((y_train, y_test), axis=0)
x_data = x_data[y_data == 3]


# Create a TensorFlow Dataset from the filtered data
ds: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices(x_data).shuffle(1000).batch(BATCH_SIZE)
)

# Normalize the dataset by scaling pixel values between 0 and 1
ds = ds.map(lambda x: x / 255)
