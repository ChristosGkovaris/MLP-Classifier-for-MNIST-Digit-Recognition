# MLP CLASSIFIER FOR MNIST DIGIT RECOGNITION

This project implements **Machine Learning** and **Deep Learning** pipelines for the **MNIST handwritten digit dataset**, including classification using **Logistic Regression, kNN, PCA**, as well as advanced models using **PyTorch (MLP & CNN)**. It was developed as part of the **ΜΥΕ046 – Computer Vision** course at the **University of Ioannina**.

---

## TABLE OF CONTENTS
1. [Overview](#overview)
2. [Features](#features)
3. [Input Data](#input-data)
4. [Algorithms Implemented](#algorithms-implemented)
5. [Installation](#installation)
6. [Usage](#usage)
7. [License](#license)
8. [Contact](#contact)

---

## OVERVIEW

The **MNIST Classification Project** is a two-part assignment exploring both  
**traditional machine learning techniques** and **deep learning neural networks**.

Students implement, evaluate, and compare:

- Classical ML models (Logistic Regression, kNN, PCA-based kNN)  
- Dimensionality reduction techniques  
- Metrics such as accuracy and confusion matrices  
- Fully-connected neural networks in PyTorch  
- (Optional) Convolutional Neural Networks for improved accuracy  

The full analysis, code execution, and results are contained in the provided Jupyter Notebook.

---

## FEATURES

- **Classical Machine Learning Models**
  - Logistic Regression classifier  
  - k-Nearest Neighbors (kNN)  
  - PCA dimensionality reduction + kNN on reduced space  

- **Performance Evaluation**
  - Accuracy calculation  
  - Confusion matrix visualization  
  - Comparison between PCA-based and original-feature classifiers  

- **Deep Learning Models (PyTorch)**
  - Fully connected MLP with ReLU activations  
  - Training on MNIST with minibatch SGD  
  - Evaluation on test set (loss & accuracy)

- **(Optional) CNN Model**
  - Higher-performance convolutional neural network  
  - Flexible architecture (Conv2D → ReLU → Pooling → FC layers)  

- **Clean, Modular Structure**
  - Organized pipeline  
  - Clear separation between ML and DL sections  
  - Notebook-ready code blocks  

---

## INPUT DATA

- **Dataset:** MNIST Handwritten Digits  
- **Training Set:** 60,000 grayscale 28×28 images  
- **Test Set:** 10,000 images  
- **Format:** Images flattened to vectors for ML, tensors (1×28×28) for PyTorch models  
- **Labels:** Digits 0–9  

Data are loaded automatically through:
- **scikit-learn** (for ML tasks)  
- **torchvision.datasets.MNIST** (for PyTorch tasks)

---

## ALGORITHMS IMPLEMENTED

1. **Logistic Regression**
   - Multiclass classifier using the one-vs-rest scheme  
   - Trains on flattened MNIST features  
   - Evaluates accuracy on test set  

2. **kNN Classifier**
   - Uses Euclidean distance  
   - Performs classification based on k nearest neighbors  
   - Achieves baseline performance for comparison  

3. **PCA Dimensionality Reduction**
   - Reduces MNIST dimensionality to *k* components  
   - Reconstruction and variance analysis  
   - kNN applied on PCA-transformed data  

4. **Confusion Matrix**
   - Visual comparison between predicted and true labels  
   - Highlights class-level performance  

5. **PyTorch MLP**
   - Architecture: Linear → ReLU → Linear → ReLU → Output layer  
   - Trained using mini-batch SGD  
   - Computes loss and accuracy after each epoch  

6. **(Optional) CNN**
   - Convolution → Activation → Pooling → Dense layers  
   - Significantly improved accuracy on MNIST  

---

## INSTALLATION

1. **Clone the repository:**
```bash
git clone https://github.com/YourUsername/MNIST-ML-DL-Project.git
cd MNIST-ML-DL-Project
```
2. Install Python (>= 3.9)
3. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Open the notebook:
```bash
jupyter notebook
```

---

## USAGE

1. Open the notebook file:
```bash
MNIST_Assignment.ipynb
```
2. Run the Machine Learning section (Exercise 1):
   - Load MNIST  
   - Train Logistic Regression  
   - Run kNN  
   - Apply PCA and evaluate PCA+kNN  
   - Generate confusion matrix  
3. Run the Deep Learning section (Exercise 2):
   - Train the PyTorch MLP  
   - Evaluate accuracy & loss  
   - (Optional) Train CNN model  
4. Compare results between ML and DL models based on:
   - Accuracy  
   - Model complexity  
   - Training time  

---

## LICENSE

This project was developed as part of the **ΜΥΕ046 – Computer Vision** course at the
University of Ioannina.

Original academic material based on the provided assignment instructions.
Implementation, analysis, and code by the project author.

---

## CONTACT

**Christos-Grigorios Gkovaris**  
University of Ioannina – Computer Science and Engineering  
[GitHub](https://github.com/ChristosGkovaris)
