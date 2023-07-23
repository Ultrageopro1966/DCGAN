# Import necessary modules
from keras import Model, layers

"""
The Discriminator model in GANs is responsible for binary classification, distinguishing
between real and synthetic data. It serves as the adversary of the Generator, aiming to
accurately classify data instances as genuine or fake. Through iterative training, the
Discriminator becomes more discerning, guiding the Generator to produce high-quality
synthetic data that resembles real data.
"""


# Class definition for the Discriminator model
class Discriminator(Model):
    def __init__(self):
        super().__init__()

        # Define the first convolutional layer
        self.conv1 = layers.Conv2D(
            8,
            (5, 5),
            (2, 2),
            padding="same",
            activation="elu",
        )
        self.drop1 = layers.Dropout(0.3)  # Dropout layer to prevent overfitting

        # Define the second convolutional layer
        self.conv2 = layers.Conv2D(
            16,
            (5, 5),
            (2, 2),
            padding="same",
            activation="elu",
        )
        self.drop2 = layers.Dropout(0.3)  # Dropout layer to prevent overfitting

        # Define the third convolutional layer
        self.conv3 = layers.Conv2D(
            32,
            (5, 5),
            (1, 1),
            padding="same",
            activation="elu",
        )
        self.drop3 = layers.Dropout(0.3)  # Dropout layer to prevent overfitting

        # Flatten the data to prepare for the fully connected layers
        self.fl = layers.Flatten()

        # First Dense
        self.d1 = layers.Dense(128, activation="relu")

        # Final dense (fully connected) layer with sigmoid activation for binary classification
        self.d2 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        # Forward pass through the discriminator model
        x = self.conv1(inputs)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        x = self.fl(x)
        x = self.d1(x)

        return self.d2(x)
