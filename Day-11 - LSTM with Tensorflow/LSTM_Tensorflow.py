# Avoid Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import Tensorflow, Keras and Matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
learning_rate = 3e-4
epochs = 1
num_classes = 10
max_features = 20000  
maxlen = 200  

# Load Dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), "Training sequences")
print(len(x_test), "Validation sequences")


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define Functional Model
inputs = keras.Input(shape=(None,), dtype="int32")

x = layers.Embedding(max_features, 128)(inputs)

x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# Architecture of Model
print(model.summary(), "\n")

# Define Loss, Optimizer and Metrics
model.compile(

    loss=keras.losses.BinaryCrossentropy(from_logits=True),
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