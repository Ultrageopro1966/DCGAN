# DCGAN Project

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow for generating images of numbers from the [mnist](https://en.wikipedia.org/wiki/MNIST_database) dataset. The DCGAN consists of a Generator and a Discriminator that compete against each other during training. The goal is for the Generator to produce realistic-looking images that can fool the Discriminator.

## Project Structure

The project is organized as follows:

- `configs.ini`: Configuration file containing hyperparameters for the GAN.
- `DataPreparation.py`: Module responsible for preparing and loading the dataset for training.
- `Discriminator.py`: Module defining the Discriminator model.
- `Generator.py`: Module defining the Generator model.
- `Learning.py`: Main script that trains the GAN.
- `requirements.txt`: A list of Python packages required to run the project.

## DCGAN Overview

The DCGAN consists of two models:
1. **Generator**: This model takes random noise as input and generates fake images.
2. **Discriminator**: This model tries to distinguish between real images from the dataset and fake images generated by the Generator.

During training, the Generator tries to produce realistic images to fool the Discriminator, while the Discriminator tries to correctly classify real and fake images. The competition between the two models helps the Generator to improve over time, generating increasingly realistic images.

## Results and Visualizations

After training the DCGAN using the provided code, the Generator model becomes capable of generating images that resemble the original dataset. The quality of the generated images improves as the training progresses through epochs. The trained Generator and Discriminator models also have their learned parameters (weights) saved for future use. Saving the model weights allows you to reload the trained models later without having to retrain them from scratch. The saved weights are stored in the following files:

- `model_results/gen_weights.h5`: Saved weights of the trained Generator model.
- `model_results/discr_weights.h5`: Saved weights of the trained Discriminator model.

With the saved model weights, you can use the trained models for various tasks, such as generating more images or fine-tuning the models on related datasets.

### Loss Graph

The following graph shows the loss of the Generator and Discriminator during the training process:

![loss_graph](https://github.com/Ultrageopro1966/DCGAN/assets/120571667/3cffbe32-7383-48a7-b541-762721dca911)

The plot illustrates how the loss of both models changes over training iterations (batches). As the training proceeds, the Generator's loss decreases, indicating that it is getting better at generating realistic images. Simultaneously, the Discriminator's loss decreases as it learns to distinguish real images from the fake ones generated by the Generator.

## Visualization of Generated Images during Training

During training, the code provides a visual representation of the generated images and the loss/accuracy graph. The images are updated for each batch iteration, allowing you to observe the progress of the DCGAN in real-time.

Please refer to the main script `Learning.py` for the implementation details on how the visualizations are carried out.

### Sample Generated Images

Below are some examples of images generated by the trained Generator model:

![gen_examples](https://github.com/Ultrageopro1966/DCGAN/assets/120571667/2b3d65f4-cdb0-4672-8b97-56981a19b471)


The images show a 4x4 grid of randomly generated images using the trained DCGAN. As training continues, the Generator can produce images with increasingly realistic features, reflecting the characteristics of the original dataset.

## Tips for Better Results

To achieve better results with the GAN, consider the following tips:

- **Adjust Hyperparameters:** Experiment with different values for learning rate, noise size, and training epochs to optimize your GAN's performance.
- **Model Architecture:** Modify the Generator and Discriminator models to better suit your image generation task. Explore new architectural elements for improved results.
- **Diverse Dataset:** Ensure a representative and diverse dataset to enhance the GAN's ability to generate realistic images.
- **Experiment with `configs.ini`:** Use a configuration file to store settings, streamlining the experimentation process for better GAN performance.

Remember that GAN training can be a time-consuming process and may require several iterations and parameter tuning to achieve satisfactory results.

## License

This project is licensed under the **MIT License**.

## Author

This GAN project was created by [UltraGeoPro](https://github.com/Ultrageopro1966).

Enjoy experimenting with the GAN and generating fascinating images!
