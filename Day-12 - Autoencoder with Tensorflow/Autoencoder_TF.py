# Avoid Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

from utils import displayData, displayLoss

# Hyperparameters
batch_size = 128
epochs = 50
noise_factor = 0.4

def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

# Load the Mnist dataset
(train_data, _), (test_data, _) = mnist.load_data()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Add noise
noisy_train_data = train_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_data.shape)
noisy_test_data =  test_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)

# Pixel values can be out of range [0,1], so we need to clip the values.
noisy_train_data = np.clip(noisy_train_data, 0.0, 1.0)
noisy_test_data = np.clip(noisy_test_data, 0.0, 1.0)

# Display the train data and noisy data
displayData(train_data, noisy_train_data)

# MODEL CREATION
input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# Train the Autoencoder
history = autoencoder.fit(x=noisy_train_data, y=train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(noisy_test_data, test_data))

# Display the Loss curve
displayLoss(history)

# Display Noisy and Denoised Images
predictions = autoencoder.predict(noisy_test_data)
displayData(noisy_test_data, predictions)

