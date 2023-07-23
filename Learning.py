# Import necessary modules
import tensorflow as tf

from DataPreparation import EPOCHS, LEARNING_RATE, NOISE_SIZE, ds
from Discriminator import Discriminator
from Generator import Generator

# Initialize the Generator and Discriminator models
generator = Generator(NOISE_SIZE)
discriminator = Discriminator()

# Define Adam optimizers for Generator and Discriminator models
gen_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
discr_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)

# Create metrics to track loss history
gen_loss_history = tf.keras.metrics.Mean()
discr_loss_history = tf.keras.metrics.Mean()
discr_accuracy = tf.keras.metrics.BinaryAccuracy()


# Generator loss function using binary cross-entropy
def gen_loss(fake_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the binary cross-entropy loss for the Generator.

    Args:
        fake_pred (tf.Tensor): Predictions from the Discriminator for generated images.

    Returns:
        tf.Tensor: Generator loss value.
    """
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_pred), fake_pred)


# Discriminator loss function using binary cross-entropy
def discr_loss(fake_pred: tf.Tensor, real_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the binary cross-entropy loss for the Discriminator.

    Args:
        fake_pred (tf.Tensor): Predictions from the Discriminator for generated images.
        real_pred (tf.Tensor): Predictions from the Discriminator for real images.

    Returns:
        tf.Tensor: Discriminator loss value.
    """
    fake_loss: tf.Tensor = tf.keras.losses.binary_crossentropy(
        tf.zeros_like(fake_pred), fake_pred
    )
    real_loss: tf.Tensor = tf.keras.losses.binary_crossentropy(
        tf.ones_like(real_pred), real_pred
    )

    discr_accuracy(tf.zeros_like(fake_pred), fake_pred)
    discr_accuracy(tf.ones_like(real_pred), real_pred)

    return real_loss + fake_loss


# Training step function decorated with tf.function for performance optimization
@tf.function
def train_step(images):
    """
    Performs one training step for the GAN.

    Args:
        images (tf.Tensor): Batch of real images for training.

    """
    images: tf.Tensor = tf.cast(images, tf.float32)
    images: tf.Tensor = tf.expand_dims(images, axis=-1)
    noise: tf.Tensor = tf.random.normal((images.shape[0], NOISE_SIZE))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
        generated_image: tf.Tensor = generator(noise, training=True)
        fake_pred: tf.Tensor = discriminator(generated_image, training=True)
        real_pred: tf.Tensor = discriminator(images, training=True)
        # Calculate losses for Generator and Discriminator
        dloss: tf.Tensor = discr_loss(fake_pred, real_pred)
        gloss: tf.Tensor = gen_loss(fake_pred)

    # Record losses in the history metrics
    gen_loss_history(gloss)
    discr_loss_history(dloss)

    # Calculate gradients and apply optimization steps
    gen_grads = gen_tape.gradient(gloss, generator.trainable_variables)
    discr_grads = discr_tape.gradient(dloss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discr_optimizer.apply_gradients(zip(discr_grads, discriminator.trainable_variables))


# Lists to store the generated loss data during training
gen_plot_data: list = []
discr_plot_data: list = []
discr_accuracy_data: list = []

# Noise for generating a single example image to visualize during training
example_noise: tf.Tensor = tf.random.normal((1, NOISE_SIZE))

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a 2x2 subplot for displaying the loss graph and the generated image side by side
fig = plt.figure(figsize=(12, 6))
grid = GridSpec(2, 2, figure=fig)

ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[1, 0])
ax2 = fig.add_subplot(grid[:, 1])
fig.subplots_adjust(hspace=0.4, wspace=0.4)


# Training loop over epochs and batches
for epoch in range(EPOCHS):
    for batch_num, sample in enumerate(ds):
        # Train Step
        train_step(sample)

        # Store the loss/accuracy data for plotting later
        gen_plot_data.append(round(float(gen_loss_history.result().numpy()), 3))
        discr_plot_data.append(round(float(discr_loss_history.result().numpy()), 3))
        discr_accuracy_data.append(round(float(discr_accuracy.result().numpy()) * 100))

        # Clear the current axis and display the generated image/loss graph
        ax0.cla()
        ax1.cla()
        ax2.cla()

        ax0.grid()
        ax0.set_ylabel("Loss")
        ax0.plot(gen_plot_data[-50:], label="gen_loss", c="b", linewidth=3)
        ax0.plot(discr_plot_data[-50:], label="discr_loss", c="y", linewidth=3)
        ax0.legend()

        ax1.grid()
        ax1.set_ylabel("Discriminator Accuracy (%)")
        ax1.set_xlabel("Iteration")
        ax1.plot(discr_accuracy_data, c="g", linewidth=3)

        x_annotate_point = len(discr_accuracy_data) - 1
        y_annotate_point = discr_accuracy_data[-1]

        ax1.scatter(x_annotate_point, y_annotate_point, c="g", s=100, edgecolor="black")
        ax1.annotate(
            f"{y_annotate_point}%",
            (x_annotate_point, y_annotate_point),
            xycoords="data",
            xytext=(-15, -20),
            textcoords="offset points",
        )

        ax2.set_title("Example of a generated image")
        ax2.imshow(generator(example_noise).numpy()[0], cmap="gray")
        ax2.set_xlabel(
            f"BATCH {batch_num + 1}/{tf.data.experimental.cardinality(ds)} EPOCH {epoch + 1}/{EPOCHS} ({round((epoch+1)/EPOCHS * (batch_num + 1)/float(tf.data.experimental.cardinality(ds).numpy()) * 100, 2)}%)"
        )
        plt.pause(0.1)  # Pause for a short time to visualize the generated image

# Save the weights of the Generator and Discriminator models
generator.save_weights("model_results/gen_weights.h5")
discriminator.save_weights("model_results/discr_weights.h5")

# Generate and save 4x4 grid of generated images using the trained generator
fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        ax[i][j].imshow(
            generator(tf.random.normal((1, NOISE_SIZE)))[0].numpy(), cmap="gray"
        )
plt.savefig("model_results/gen_examples.png")

# Plot and save the loss graph during training
fig, ax = plt.subplots(1, 1)
ax.cla()
ax.plot(gen_plot_data, label="gen_loss")
ax.plot(discr_plot_data, label="discr_loss")
ax.grid()
ax.legend()
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
plt.savefig("model_results/loss_graph.png")
