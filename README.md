# Multi-Layer Perceptron (MLP) Implementation
Welcome to the "Multi-Layer Perceptron (MLP) Implementation" repository. This project showcases the implementation of a fully connected MLP neural network using Python and PyTorch. The MLP model is designed to classify images from the MNIST dataset. The final grade of the project is  (excluding the additional  bonus points) out of .


### MLP Implementation
- Objective: Build a two-layer MLP neural network to classify handwritten digits from the MNIST dataset.
- Architecture: Input Layer: Processes 28x28 grayscale images (784 features). Hidden Layer: Consists
  of 50 neurons with ReLU activation. Output Layer: Produces logits for 10 classes (digits 0-9).
- Training: The model uses CrossEntropyLoss for classification and Stochastic Gradient Descent (SGD)
  for optimization. The dataset is divided into batches of size 50 for training. The training loop
  runs for 10 epochs.
- Evaluation: Displays the confusion matrix to analyze predictions. Computes accuracy, expected
  to be approximately 95-97% for the given setup.


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
     git clone https://github.com/YourUsername/MLP-Classifier-for-MNIST-Digit-Recognition.git
     cd MLP-Classifier-for-MNIST-Digit-Recognition
- Install Required Libraries:
   Ensure that Python 3.8 or higher is installed with PyTorch and Matplotlib:
   ```bash
      pip install torch torchvision matplotlib
- Run the Notebook:
   Open and run the Jupyter Notebook file provided in the repository. The notebook includes all the necessary code for training and evaluating the MLP model.


## Collaboration
This project was a collaborative effort. Special thanks to [SpanouMaria](https://github.com/SpanouMaria), for their significant contributions to the development and improvement of the project.


## Notes
The MLP model occasionally shows lower accuracy if initialization or hyperparameters deviate. Ensure proper learning rates and initialization methods are used.
Filters and confusion matrix visualization help debug and fine-tune the model for better results.
