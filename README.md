# Cat vs. Dog Image Classifier 🐱🐶

An end-to-end computer vision pipeline built with TensorFlow and Keras to classify images of cats and dogs. This project features a custom Convolutional Neural Network (CNN) trained directly on Kaggle data using Google Colab.

## 🚀 Project Overview

The goal of this project was to build a computer vision model from scratch capable of distinguishing between cats and dogs. The model processes 150x150 RGB images and outputs a definitive binary prediction. It also includes an interactive testing script that allows users to upload custom images from the internet or their local machine to test the model's predictive capabilities in real-time.

### Tech Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Data Provisioning:** `kagglehub`
* **Visualization:** Matplotlib
* **Environment:** Google Colab (T4 GPU)

## 🧠 Model Architecture

The CNN was built from scratch and relies on the following structure to maximize accuracy and prevent data memorization:
* **Preprocessing:** Automated rescaling of RGB values (0-255 mapped to 0-1) for neural network stability.
* **Data Augmentation:** Real-time application of `RandomFlip`, `RandomRotation`, and `RandomZoom` to artificially increase dataset variance.
* **Convolutional Base:** Three distinct blocks of `Conv2D` layers (32, 64, and 128 filters) paired with `MaxPooling2D` to extract hierarchical spatial features.
* **Regularization:** Integration of `BatchNormalization` layers to stabilize learning, and a `Dropout(0.5)` layer prior to the dense layers to force redundant feature learning.
* **Decision Layers:** A fully connected `Dense(128)` layer followed by a final `Dense(1)` sigmoid output for binary classification.

## 🛠️ Challenges & Solutions

Building this model required navigating several technical hurdles, specifically regarding environment setup and model generalization:

1. **API Authentication & Data Provisioning:** 
   Initial attempts to download the Kaggle dataset resulted in `403 Forbidden` errors. This was resolved by pivoting to the modern `kagglehub` library, fixing dataset slug typos, and utilizing interactive notebook logins to securely mount the dataset directly into Google Colab's high-speed cache.
2. **Combating Severe Overfitting:** 
   The initial 3-layer CNN suffered from massive overfitting—memorizing the training set while validation accuracy stagnated around 55-60%. This was fixed by completely overhauling the architecture to include Data Augmentation, Dropout layers, and Batch Normalization.
3. **Training Stability:** 
   Early training runs produced highly erratic, zigzagging loss graphs. This was mitigated by extending the training loop to 30 epochs and implementing Keras Callbacks (`EarlyStopping` and `ReduceLROnPlateau`) to smooth out gradient descent and restore the best model weights automatically.

## 💻 How to Run This Project

Because this project is built for Google Colab and uses `kagglehub`, you do not need to download any massive datasets to your local machine!

1. Open the `Cat_vs_Dog_Classifier.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Turn on the free GPU (**Runtime** > **Change runtime type** > **T4 GPU**).
3. Run the cells in order. The dataset will automatically download to the Colab temporary storage.
4. When you reach the final cell, you can paste the URL of *any* dog or cat image from Google Images into the code to see the AI make a live prediction!
