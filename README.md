# Deep Learning Fundamentals with TensorFlow & Keras

## Overview

This repository is part of the **Deep Learning using TensorFlow and Keras** course and focuses on the **core concepts required to build neural networks**.  
The notebooks combine **data preprocessing**, **manual implementations**, and **Keras-based models** to build strong intuition before moving to deeper architectures.

---


## 1) Auto MPG Data Processing
**Notebook:** `AutoMPG_DataProcessing.ipynb`

This notebook focuses on **loading and preprocessing a real-world regression dataset**.

### Code snippet
```python
raw_dataset = pd.read_csv(url, na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
```

### Key concepts
- Loading structured data
- Handling missing values
- Preparing features for regression models

---

## 2) Linear Regression from Scratch
**Notebook:** `LinearRegression.ipynb`

This notebook implements **linear regression manually** to understand parameter updates.

### Key concepts
- Weight and bias updates
- Learning rate
- Training loop mechanics

---

## 3) Linear Regression using Keras
**Notebook:** `LinearRegressionWithKeras.ipynb`

This notebook solves the same regression problem using **Keras high-level APIs**.

### Code snippet
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
```

### Key concepts
- Sequential models
- Dense layers
- Comparing manual vs framework-based approaches

---

## 4) Regression Loss Functions
**Notebook:** `RegressionLossFunctions.ipynb`

This notebook explores **loss functions used in regression**.

### Code snippet
```python
def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.math.square(y_true - y_pred))
    return mse_loss
```

### Key concepts
- Mean Squared Error (MSE)
- Sensitivity to large errors
- Optimization behavior

---

## 5) Binary Classification with Keras
**Notebook:** `BinaryClassificationWithKeras.ipynb`

This notebook introduces **binary classification models** using neural networks.

### Code snippet
```python
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
```

### Key concepts
- Binary labels
- Classification loss functions
- Model evaluation metrics

---

## 6) MNIST Digit Classification
**Notebook:** `Classification_MNIST_dataset.ipynb`

This notebook demonstrates **multi-class image classification** using the MNIST dataset.

### Code snippet
```python
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

### Key concepts
- Image reshaping
- Multi-class outputs
- Training neural networks on images

---

## 7) Classification Loss Functions
**Notebook:** `ClassificationLossFunctions.ipynb`

This notebook explains **loss functions used for classification**.

### Code snippet
```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce_loss = bce(y_true, y_pred)
```

### Key concepts
- Binary cross-entropy
- Relationship between outputs and loss
- Model convergence

---

## 8) Backpropagation using GradientTape
**Notebook:** `BackPropagation_using_GradientTape.ipynb`

This notebook demonstrates **manual backpropagation** using TensorFlow.

### Code snippet
```python
with tf.GradientTape() as tape:
    y_pred = W * x + b
    loss = tf.reduce_mean(tf.square(y - y_pred))
```

### Key concepts
- Automatic differentiation
- Gradient computation
- Parameter updates

---


## Requirements

```bash
pip install tensorflow numpy pandas matplotlib
```

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

