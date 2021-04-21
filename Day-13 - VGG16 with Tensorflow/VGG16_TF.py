# Avoid Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

learning_rate = 0.00001
epochs = 100
num_classes = 2

train = ImageDataGenerator()
traindata = train.flow_from_directory(directory="./dataset/train",target_size=(224,224))

test = ImageDataGenerator()
testdata = test.flow_from_directory(directory="./dataset/test", target_size=(224,224))

model = Sequential(
    [
        Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),

        Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),

        Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),

        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),

        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),

        Flatten(),
        Dense(units = 4096, activation = "relu"),
        Dense(units = 4096, activation = "relu"),
        Dense(units = num_classes, activation = "softmax"),   
    ]
)

# Architecture of Model
print(model.summary(), "\n")

# Define Loss, Optimizer and Metrics
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=["accuracy"],
)

checkpoint = ModelCheckpoint(".model/vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='max')

history = model.fit(
    traindata,
    validation_data= testdata, 
    epochs=epochs,
    callbacks=[checkpoint,early]
)

# Plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

