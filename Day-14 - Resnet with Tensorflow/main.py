import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from model import resnet50

learning_rate = 0.0001
epochs = 100
num_classes = 2

train = ImageDataGenerator()
traindata = train.flow_from_directory(directory="./dataset/train",target_size=(224,224))

test = ImageDataGenerator()
testdata = test.flow_from_directory(directory="./dataset/test", target_size=(224,224))

model = resnet50(num_classes)

print(model.summary(), "\n")

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=Adam(lr=learning_rate),
    metrics=["accuracy"],
)

checkpoint = ModelCheckpoint(".model/resnet.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='max')

history = model.fit(
    traindata,
    validation_data= testdata, 
    epochs=epochs,
    callbacks=[checkpoint,early]
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

