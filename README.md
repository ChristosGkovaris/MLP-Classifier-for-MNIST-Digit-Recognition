# Multi-Layer Perceptron (MLP) Implementation
Welcome to "Multi-Layer Perceptron (MLP) Implementation" repository. This repository contains the implementation of a fully connected MLP neural network using Python and PyTorch. The MLP model is designed to classify images from the MNIST dataset. The final grade of the project is 2.5 (excluding the additional  bonus points) out of 3.



## Implementation Details
- Helper Functions: `weight_variable`: Initializes weight tensors for the layers using Xavier
  initialization. `bias_variable`: Initializes bias tensors with small positive values.
- Forward Propagation: Computes the outputs of the hidden and output layers using ReLU and linear
  transformations.
- Training Process: The model is trained using mini-batches and updated using backpropagation. Loss values
  are printed after each epoch for monitoring.
- Visualization: Filters from the first layer are reshaped to 28x28 and visualized to understand their learning.



## How to Run
- Clone the Repository:
  ```bash
     git clone https://github.com/ChristosGkovaris/MLP-Classifier-for-MNIST-Digit-Recognition.git
     cd MLP-Classifier-for-MNIST-Digit-Recognition
- Python: Ensure that Python 3.8 or higher is installed with PyTorch and Matplotlib:
- Run the Notebook: Open and run the Jupyter Notebook file provided in the repository. The notebook includes all the necessary code for training and evaluating
  the MLP model.



## Notes
The MLP model occasionally shows lower accuracy if initialization or hyperparameters deviate. Ensure proper learning rates and initialization methods are used. Filters and confusion matrix visualization help debug and fine-tune the model for better results.
