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
import utils
import convert

import sys
sys.path.insert(0,'..')
import config as cfg

def custom_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true[:,:,:,0], K.round(y_pred[:,:,:,0])))

def custom_loss(y_true, y_pred):
    obj_mask  = y_true[:,:,:,0]
    mask_shape = tf.shape(obj_mask)
    noobj_mask = tf.ones(mask_shape) - obj_mask

    lambda_1 = 10
    lambda_2 = 0.5

    #localization
    position_loss = K.square(y_true[:,:,:,1] - y_pred[:,:,:,1]) * obj_mask \
                    + K.square(y_true[:,:,:,2] - y_pred[:,:,:,2]) * obj_mask
    position_loss = position_loss * lambda_1
    #box size
    size_loss = K.square(K.sqrt(y_true[:,:,:,3]) - K.sqrt(y_pred[:,:,:,3])) * obj_mask
    for i in range(cfg.yolo_nb_anchors - 1):
        size_loss += K.square(K.sqrt(y_true[:,:,:,4 + i]) - K.sqrt(y_pred[:,:,:,4 + i])) * obj_mask
    #confidence
    confidence_loss = K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]) * obj_mask \
                      + K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]) * noobj_mask * lambda_2
    return position_loss + size_loss + confidence_loss

def stride(x):
    return (x, x)
def kernel(x):
    return (x, x)

def add_conv_2d(x, n_filters=16, kernel=kernel(3), stride=stride(1), batch_normalization=False, leaky_relu=False, kernel_initializer='he_uniform'):
    x = Conv2D(n_filters, kernel, stride)(x)
    if leaky_relu:
        x = LeakyReLU(alpha=0.1)(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    return x

def create_model(shape:tuple, nb_anchors:int):
    inputs = keras.layers.Input(shape=shape)
    x = add_conv_2d(inputs, 75, kernel(5), stride(2), True, True)
    x = MaxPool2D(stride(2))(x)
    x = add_conv_2d(x, 32, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 32, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = MaxPool2D(stride(2))(x)
    x = add_conv_2d(x, 32, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = MaxPool2D(stride(2))(x)
    
    x = add_conv_2d(x, 32, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 96, kernel(1), stride(1), True, True)

    x = Conv2D(3, kernel(1), activation='sigmoid')(x)
    y = Conv2D(nb_anchors, kernel(1), activation='softmax')(x)
    x = concatenate([x, y])
    return keras.models.Model(inputs, x)

def train_model(modele, x_train, y_train, x_validation, y_validation):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,
              metrics=[custom_accuracy, 'binary_crossentropy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    modele.fit(x_train, y_train, batch_size=5, epochs=20, validation_data=(x_validation, y_validation), callbacks=[es])
    return modele

def display_model_prediction(prediction, wanted_prediction, prediction_on_image, wanted_output):
    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(prediction)
    plt.title('model output')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 2)
    plt.imshow(wanted_prediction)
    plt.title('wanted output')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 3)
    plt.imshow(prediction_on_image)
    plt.title('model output on image')
    plt.colorbar(orientation='horizontal')
    fig.add_subplot(1, 4, 4)
    plt.imshow(wanted_output)
    plt.title('wanted output on image')
    plt.colorbar(orientation='horizontal')
    plt.show()

def generate_prediction_image(prediction, x_test, y_test):
    ratio_x = cfg.image_width / cfg.yolo_width
    ratio_y = cfg.image_height / cfg.yolo_height
    coords = utils.n_max_coord(prediction[:,:,0], 3)
    prediction_on_image = utils.draw_rectangle_on_image(x_test.copy(), prediction, coords)
    coords = utils.treshold_coord(y_test[:,:,0])
    wanted_output = utils.draw_rectangle_on_image(x_test.copy(), y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output)

x_train, y_train, x_validation, y_validation, x_test, y_test = create_dataset()
shape = (cfg.image_height, cfg.image_width, 3)
if cfg.retrain:
    modele = create_model(shape, cfg.yolo_nb_anchors)
    modele.summary()
    modele = train_model(modele, x_train, y_train, x_validation, y_validation)
else:
    modele = keras.models.load_model(cfg.model_path_keras, custom_objects={'custom_loss': custom_loss, 'custom_accuracy':custom_accuracy})

modele.save(cfg.model_path_keras, include_optimizer=False)
for i in range(len(x_test)):
    v = [[val for val in x_val] for x_val in x_test[i]]
    start = process_time()
    prediction = modele.predict([[v]])[0]
    stop = process_time()
    print('temps execution : ', stop - start)
    image = generate_prediction_image(prediction, x_test[i], y_test[i])

#modele.save(cfg.model_path_keras, include_optimizer=False)

sys.argv = ['', cfg.model_path_keras, cfg.model_path_fdeep]

convert.main()

breakpoint()