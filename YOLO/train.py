from Dataset_Loader import create_dataset

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, concatenate
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import random
import numpy as np

from skimage.draw import rectangle_perimeter

from time import process_time

import convert

import sys
sys.path.insert(0,'..')
import config as cfg

def custom_loss(y_true, y_pred):
    obj_mask  = y_true[:,:,:,4]
    mask_shape = tf.shape(obj_mask)
    noobj_mask = tf.ones(mask_shape) - obj_mask

    lambda_1 = 5
    lambda_2 = 0.5

    #classification
    classification_loss = K.square(y_true[:,:,:,5] - y_pred[:,:,:,5]) * obj_mask \
                        + K.square(y_true[:,:,:,6] - y_pred[:,:,:,6]) * obj_mask
    #localization
    position_loss = K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]) * obj_mask \
                    + K.square(y_true[:,:,:,1] - y_pred[:,:,:,1]) * obj_mask
    position_loss = position_loss * lambda_1
    size_loss = K.square(K.sqrt(y_true[:,:,:,2]) - K.sqrt(y_pred[:,:,:,2])) * obj_mask \
                + K.square(K.sqrt(y_true[:,:,:,3]) - K.sqrt(y_pred[:,:,:,3])) * obj_mask
    size_loss = size_loss * lambda_1
    #confidence
    confidence_loss = K.square(y_true[:,:,:,4] - y_pred[:,:,:,4]) * obj_mask \
                      + K.square(y_true[:,:,:,4] - y_pred[:,:,:,4]) * noobj_mask * lambda_2
    return classification_loss + position_loss + size_loss + confidence_loss

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

def create_model(shape:tuple, nb_categories:int):
    inputs = keras.layers.Input(shape=shape)
    x = add_conv_2d(inputs, 16, kernel(5), stride(2), True, True)
    x = MaxPool2D(stride(2))(x)
    x = add_conv_2d(x, 32, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 32, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 96, kernel(3), stride(1), True, True)
    x = MaxPool2D(stride(2))(x)
    
    x = add_conv_2d(x, 64, kernel(3), stride(2), True, True)
    x = add_conv_2d(x, 96, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(1), stride(1), True, True)

    x = Conv2D(5 + nb_categories, kernel(1), activation='sigmoid')(x)
    return keras.models.Model(inputs, x)

def train_model(modele, x_train, y_train, x_validation, y_validation):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,
              metrics=['accuracy', 'binary_crossentropy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    modele.fit(x_train, y_train, batch_size=5, epochs=20, validation_data=(x_validation, y_validation), callbacks=[es])
    return modele

def display_model_prediction(input_image, prediction, x_test, y_test):
    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(input_image)
    plt.title('input_image')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 2)
    plt.imshow(prediction)
    plt.title('model output')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 3)
    plt.imshow(x_test)
    plt.title('model output on image')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 4)
    plt.imshow(y_test)
    plt.title('ground truth')
    plt.colorbar(orientation='horizontal')
    plt.show()

def generate_prediction_image(prediction, x_test, y_test):
    coord = np.where(prediction[:,:,4] > 0.5)
    ratio_x = cfg.image_width / cfg.yolo_width
    ratio_y = cfg.image_height / cfg.yolo_height
    ground_truth = x_test.copy()
    wanted_output = x_test.copy()
    coord_2 = np.where(y_test[:,:,4] > 0.5)
    for i, obj in enumerate(y_test[coord_2]):
        center_x = (coord_2[1][i] + obj[1]) * ratio_x
        center_y = (coord_2[0][i] + obj[0]) * ratio_y
        width = obj[2] * cfg.image_width
        height = obj[3] * cfg.image_height
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        right = int(center_x + width / 2)
        bottom = int(center_y + height / 2)
        rect = rectangle_perimeter((top, left), (bottom, right), shape=(cfg.image_height, cfg.image_width), clip=True)
        wanted_output[rect] = 1
    for i, obj in enumerate(prediction[coord]):
        center_x = (coord[1][i] + obj[1]) * ratio_x
        center_y = (coord[0][i] + obj[0]) * ratio_y
        width = obj[2] * cfg.image_width
        height = obj[3] * cfg.image_height
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        right = int(center_x + width / 2)
        bottom = int(center_y + height / 2)
        rect = rectangle_perimeter((top, left), (bottom, right), shape=(cfg.image_height, cfg.image_width), clip=True)
        x_test[rect] = 1
    display_model_prediction(ground_truth, prediction[:,:,4], x_test, wanted_output)

x_train, y_train, x_validation, y_validation, x_test, y_test = create_dataset()
shape = (cfg.image_height, cfg.image_width, 3)
if cfg.retrain:
    modele = create_model(shape, cfg.nb_categories)
    modele.summary()
    modele = train_model(modele, x_train, y_train, x_validation, y_validation)
else:
    modele = keras.models.load_model(cfg.model_path, custom_objects={'custom_loss': custom_loss})


for i in range(len(x_test)):
    v = [[val for val in x_val] for x_val in x_test[i]]
    start = process_time()
    prediction = modele.predict([[v]])[0]
    stop = process_time()
    print('temps execution : ', stop - start)
    image = generate_prediction_image(prediction, x_test[i], y_test[i])

modele.save(cfg.model_path, include_optimizer=False)

sys.argv = ['', cfg.model_path, cfg.model_path_fdeep]

convert.main()

breakpoint()