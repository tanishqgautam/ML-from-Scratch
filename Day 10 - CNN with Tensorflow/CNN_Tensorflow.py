# Avoid Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import Tensorflow, Keras and Matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 3e-4
epochs = 5
num_classes = 10

# Load Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize Data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Total Train and Test Data
print(x_train.shape[0], "training samples")
print(x_test.shape[0], "testing samples \n")

# Define Model
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape = (32, 32, 3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes),
    ]
)

# Architecture of Model
print(model.summary(), "\n")

# Define Loss, Optimizer and Metrics
model.compile(

    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=["accuracy"],
)

# Train Model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#Evaluate Model
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("\nTest Loss: ", score[0])
print("Test Accuracy: ", score[1])