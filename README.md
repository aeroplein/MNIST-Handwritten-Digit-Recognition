# MNIST Handwritten Digit Recognition with Keras & MLP

This project is a **Multi-Layer Perceptron(MLP)** model built using TensorFlow/Keras. The goal is to classify 28x28 pixel images of handwritten digits from the standard MNIST dataset into 10 respective classes (0-9).
This small project includes the complete lifecycle of a model: data loading, preprocessing, model design, training, evaluation, and visualization.

## How to Run This Project

1. Clone this repository on your local machine.
2. Install the necessary libraries:
    ```bash
   pip install -r requirements.txt
    ```
3. Run the main Python script:
   ```bash
   python main.py
   ```
   
## Technologies Used
* **TensorFlow / Keras:** For building, compiling, and training the model.
* **NumPy:** For numerical operations and data manipulation(reshaping, type conversion).
* **Matplotlib:** For visualizing the training results (loss and accuracy).

---

## Project Pipeline

### 1. Data Loading
This dataset was loaded directly from Keras using the `keras.datasets.mmnist.load_data()` function.
* **Training Set:** 60,000 images (`X_train`) and their corresponding labels (`y_train`).
* **Test Set:** 10,000 images (`X_test`) and their corresponding labels (`y_test`).

### 2. Data Preprocessing
Before being fed to the neural networks, the raw data underwent three critical preprocessing steps:

1. **Flattening:**
   * **Problem:** Standard `Dense` layers cannot process 2D images (`28x28`).They expect a 1D vector.
   * **Solutions:** Each `(28, 28)` image matrix was "flattened" (reshaped) into a `(784,)` 1D vector.
   
2. **Normalization:**
   * **Problem:** Pixel values range from `[0, 255]`. Feeding large integer values into the network can slow down the optimization process and cause instability.
   * **Solutions:** All pixel values were divided by `255.0` to scale them into a `[0, 1]` floating point range.

3. **Categorical Conversion (One-Hot Encoding):**
   * **Problem:** The labels are integers (e.g., `5`, `0`, `2` ). The model has a `softmax` output which gives a probability vector. (e.g., `[0.1, 0.05, ...]`). The `categorical_crossentropy` loss function cannot directly compare these two formats.
   * **Solutions:** The `to_categorical` utility was used to convert each integer label into a "one-hot" vector (e.g., `5` becomes `[0,0,0,0,0,1,0,0,0,0]`).

### 3. Model Architecture
A simple 3-layer Multi-Layer Perceptron (MLP) was designed:

* **Optimizer:** `adam`
* **Loss Function:** `categorical_crossentropy`

---

## Results and Analysis

The model was trained for `10 epochs` with a `batch_size` of `128`.

### Test Performance
The final results evaluated on the unseen test dataset.
* **Test Kaybı (Loss):** `0.0732`
* **Test Başarısı (Accuracy):** `0.9781`

---
The next logical improvement would be to implement this project with **Convolutional Neural Network (CNN)**, an architecture which is specific for 2D spacial information. 

