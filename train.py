#!/usr/bin/env python
# coding: utf-8

#importe les dependances
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

import input_loader

def create_model():
    stride = (2, 2)

    inputs = keras.layers.Input(shape=(64, 64, 1))
    x = Conv2D(128, (3, 3), stride, activation=keras.layers.LeakyReLU(alpha=0.3))(inputs)
    x = Conv2D(128, (3, 3), stride, activation=keras.layers.LeakyReLU(alpha=0.3))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation=keras.layers.LeakyReLU(alpha=0.3))(x)
    x = Flatten()(x)
    x = Dense(128, activation=keras.layers.LeakyReLU(alpha=0.3))(x)
    x = Dense(1, activation='sigmoid')(x)
    return keras.models.Model(inputs, x)

def train_model(modele, x_train, y_train, x_val, y_val):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    modele.fit(x_train, y_train, batch_size=1100, epochs=30, validation_data=(x_val, y_val), callbacks=[es])
    return modele

def predict(modele, path):
    img = input_loader.load_image(path)
    print(modele.predict([[img]])[0][0])

if __name__ == '__main__':
    images_paths = input_loader.get_file_paths("dataset")
    images_neg, images_pos = input_loader.load_images(images_paths)
    x_train, y_train, x_val, y_val = input_loader.create_dataset(images_neg, images_pos)
    modele = create_model()
    modele.summary()
    modele = train_model(modele, x_train, y_train, x_val, y_val)
    predict(modele, "dataset/neg/bottom_log_59793.png")
    predict(modele, "dataset/pos/top_64805_neg_up_log_1_crop_36_38_55_57.png")
    breakpoint()