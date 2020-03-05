#!/usr/bin/env python3

import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt




mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,  activation=tf.nn.relu,
                                input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256,  activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 20
model_history = model.fit(x_train,
                          y_train,
                          epochs=epochs,

                          validation_data=(x_test, y_test),
                          verbose=2)

'''
predictions = model.predict([x_test])
value = [np.argmax(predictions[i]) for i in range(10)]
print(value)
'''

model.save('numreader.model')
tfjs.converters.save_keras_model(model, './model')



def plot_history(history,epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


plot_history(model_history, epochs)