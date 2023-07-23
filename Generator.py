# Import necessary modules
from keras import Model, initializers, layers

"""
The Generator model in GANs is responsible for synthesizing new data instances by
converting random noise into realistic data. It competes with the Discriminator, aiming to
generate high-quality synthetic data that is indistinguishable from real data. Through this
adversarial process, the Generator learns to capture the underlying data distribution and
produce diverse and novel data, making it a powerful tool for various generative tasks.
"""


# Class definition for the Generator model
# TODO: Experiment with model hyperparameters
class Generator(Model):
    def __init__(self, noise_length: int):
        super().__init__()
        self.noise_length = noise_length

        self.d1 = layers.Dense(
            7 * 7 * 256, activation="relu"
        )  # Fully connected layer for mapping noise to initial shape
        self.reshape = layers.Reshape((7, 7, 256))  # Reshape to 4D tensor

        self.conv1t = layers.Conv2DTranspose(
            128,
            (5, 5),
            (2, 2),
            padding="same",
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
        )  # Deconvolution layer
        self.norm1 = (
            layers.BatchNormalization()
        )  # Batch normalization for stable training
        self.act1 = layers.LeakyReLU()  # LeakyReLU activation function

        self.red_conv = layers.Conv2DTranspose(
            1,
            (1, 1),
            (2, 2),
            padding="same",
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
        )  # Reduce the number of channels to 1
        self.add = layers.Add()  # Element-wise addition for the skip connection

        self.conv2t = layers.Conv2DTranspose(
            64,
            (5, 5),
            (2, 2),
            padding="same",
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
        )  # Deconvolution layer
        self.norm2 = (
            layers.BatchNormalization()
        )  # Batch normalization for stable training
        self.act2 = layers.LeakyReLU()  # LeakyReLU activation function

        self.conv3t = layers.Conv2DTranspose(
            1,
            (5, 5),
            (1, 1),
            padding="same",
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
            activation="sigmoid",
        )  # Final deconvolution layer with sigmoid activation to produce output image

    def call(self, inputs):
        # Ensure the input noise has the correct shape
        assert inputs.shape[1] == self.noise_length

        x = self.d1(inputs)  # Fully connected layer for initial transformation
        x = self.reshape(x)  # Reshape to 4D tensor
        red = self.red_conv(x)  # Reduce the number of channels to 1

        x = self.conv1t(x)  # Deconvolution layer
        x = self.norm1(x)  # Batch normalization
        x = self.act1(x)  # LeakyReLU activation

        x = self.add(
            [red, x]
        )  # Skip connection: Add the reduced and deconvoluted tensors

        x = self.conv2t(x)  # Deconvolution layer
        x = self.norm2(x)  # Batch normalization
        x = self.act2(x)  # LeakyReLU activation

        generated_image = self.conv3t(
            x
        )  # Final deconvolution layer with sigmoid activation

        return generated_image
