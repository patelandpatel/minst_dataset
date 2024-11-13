# MNIST and Fashion MNIST Classification

This repository contains code and files for training and evaluating deep learning models on the MNIST and Fashion MNIST datasets. The project includes data loading, model training, and model evaluation scripts, as well as saved model weights for quick testing.

## Repository Contents

- **`Untitled-Copy1.ipynb`**: Jupyter notebook that includes code for loading data, building models, training, and evaluating performance on the MNIST or Fashion MNIST datasets.
- **`fashion_mnist.h5`**: Saved model file for Fashion MNIST dataset.
- **`fashion_mnist_weights.h5`**: Saved weights file for Fashion MNIST model.
- **`README.md`**: This file, providing an overview of the project and instructions for use.

## Getting Started

### Prerequisites

To run this project, you'll need the following packages installed:
- `numpy`
- `tensorflow` or `keras`
- `matplotlib`
- `pandas`
- `sklearn`

You can install them using pip:
```bash
pip install numpy tensorflow matplotlib pandas sklearn
```

### Dataset
This project uses the Fashion MNIST dataset, which is a collection of 60,000 training images and 10,000 test images of clothing items. The dataset is similar in structure to the classic MNIST dataset (handwritten digits) but includes fashion items instead.

### File Structure

- **Training and Evaluation Notebook (`Untitled-Copy1.ipynb`)**: This notebook covers the following steps:
  1. Loading the Fashion MNIST dataset.
  2. Preprocessing the data for training.
  3. Building and compiling a neural network model.
  4. Training the model and saving the results.
  5. Evaluating model performance and visualizing predictions.

- **Saved Model and Weights (`fashion_mnist.h5` and `fashion_mnist_weights.h5`)**:
  - The `fashion_mnist.h5` file is the complete saved model, which can be loaded directly for inference.
  - The `fashion_mnist_weights.h5` file contains only the trained model weights, which can be loaded into an identical model architecture.

## Usage

1. **Run the Notebook**:
   Open the `Untitled-Copy1.ipynb` notebook and execute the cells to train or test the model on the Fashion MNIST dataset. This notebook allows you to customize parameters, visualize data, and see performance metrics.

2. **Load the Pretrained Model**:
   To skip training and directly test the pretrained model, load `fashion_mnist.h5` in the notebook or script:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('fashion_mnist.h5')
   ```

3. **Using Model Weights**:
   If you prefer to load only the weights, first initialize the model architecture and then load weights:
   ```python
   model.load_weights('fashion_mnist_weights.h5')
   ```

## Example Results

After running the notebook, you should see performance metrics like accuracy and loss for training and test datasets. Visualization of sample predictions can also be found in the notebook, showing the model’s effectiveness in classifying Fashion MNIST images.

## License

This project is open-source and available for modification and redistribution.
