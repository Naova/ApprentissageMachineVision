from Dataset_Loader import create_dataset

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, concatenate
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import random
import numpy as np

import convert

import sys
sys.path.insert(0,'..')
import config as cfg

def stride(x):
    return (x, x)
def kernel(x):
    return (x, x)

def add_conv_2d(x, n_filters=16, kernel=kernel(3), stride=stride(1), batch_normalization=False, leaky_relu=False, kernel_initializer='he_uniform'):
    x = Conv2D(n_filters, kernel, stride)(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if leaky_relu:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def create_model(shape:tuple, categories:int):
    inputs = keras.layers.Input(shape=shape)
    x = add_conv_2d(inputs, 16, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 32, kernel(3), stride(2), True, True)
    x = add_conv_2d(x, 16, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 32, kernel(3), stride(1), True, True)
    x = MaxPool2D()(x)

    x = add_conv_2d(x, 64, kernel(3), stride(2), True, True)
    x = add_conv_2d(x, 32, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = MaxPool2D()(x)

    #x = add_conv_2d(x, categories, kernel(1), stride(1), False, False)
    x = Conv2D(categories, kernel(1), activation='sigmoid')(x)

    return keras.models.Model(inputs, x)

def train_model(modele, x_train, y_train, x_validation, y_validation):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    modele.fit(x_train, y_train, batch_size=200, epochs=1, validation_data=(x_validation, y_validation), callbacks=[es])
    return modele

def display_model_prediction(prediction, x_val):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(x_val)
    plt.colorbar(orientation='horizontal')
    print('max min :')
    print(prediction.max())
    print(prediction.min())
    print(np.where(prediction == prediction.max()))
    fig.add_subplot(1, 2, 2)
    plt.imshow(prediction)
    plt.colorbar(orientation='horizontal')
    plt.show()

x_train, y_train, x_validation, y_validation, x_test, y_test = create_dataset()
shape = (cfg.image_height, cfg.image_width, 3)
modele = create_model(shape, cfg.yolo_categories)
modele.summary()
modele = train_model(modele, x_train, y_train, x_validation, y_validation)

for i in range(len(x_test)):
    v = [[val for val in x_val] for x_val in x_test[i]]
    prediction = modele.predict([[v]])[0][:,:,1]
    display_model_prediction(prediction, x_test[i])

input_path = 'yolo_modele.h5'
output_path = 'yolo_modele.json'

modele.save(input_path, include_optimizer=False)

sys.argv = ['', input_path, output_path]

convert.main()

breakpoint()