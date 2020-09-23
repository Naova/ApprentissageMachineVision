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

    anchors = tf.constant(cfg.get_anchors())

    lambda_1 = 10
    lambda_2 = 0.4

    #position
    true_center_x = y_true[:,:,:,1]
    true_center_y = y_true[:,:,:,2]

    pred_center_x = y_pred[:,:,:,1]
    pred_center_y = y_pred[:,:,:,2]

    #get rayons
    anchor_index_true = K.argmax(y_true[:,:,:,3:])
    true_rayons = K.gather(anchors, anchor_index_true)

    anchor_index_pred = K.argmax(y_pred[:,:,:,3:])
    pred_rayons = K.gather(anchors, anchor_index_pred)

    #boxes
    scale = cfg.image_width / cfg.yolo_width
    true_left = true_center_x - true_rayons * scale
    true_right = true_center_x + true_rayons * scale
    true_top = true_center_y - true_rayons * scale
    true_bottom = true_center_y + true_rayons * scale

    pred_left = pred_center_x - pred_rayons * scale
    pred_right = pred_center_x + pred_rayons * scale
    pred_top = pred_center_y - pred_rayons * scale
    pred_bottom = pred_center_y + pred_rayons * scale

    #intersection
    i_left = K.maximum(true_left, pred_left)
    i_top = K.maximum(true_top, pred_top)
    i_right = K.minimum(true_right, pred_right)
    i_bottom = K.minimum(true_bottom, pred_bottom)

    intersection_area = (i_right - i_left) * (i_bottom - i_top)

    #areas
    true_area = (true_right - true_left) * (true_bottom - true_top)
    pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

    #union
    union_area = true_area + pred_area - intersection_area

    #iou
    iou = intersection_area / union_area
    iou_loss = (1 - iou) * obj_mask * lambda_1

    #confidence
    confidence_loss = K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]) * obj_mask \
                      + K.square(y_true[:,:,:,0] - y_pred[:,:,:,0]) * noobj_mask * lambda_2
    return iou_loss + confidence_loss

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
    x = add_conv_2d(inputs, 64, kernel(5), stride(2), True, True)
    x = MaxPool2D(stride(2))(x)
    
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 32, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(3), stride(1), True, True)
    x = MaxPool2D(stride(2))(x)
    
    x = add_conv_2d(x, 96, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 64, kernel(1), stride(1), True, True)
    x = add_conv_2d(x, 96, kernel(3), stride(1), True, True)
    x = MaxPool2D(stride(2))(x)
    
    x = add_conv_2d(x, 128, kernel(3), stride(1), True, True)
    x = add_conv_2d(x, 96, kernel(1), stride(1), True, True)
    x = Conv2D(3, kernel(1), activation='sigmoid')(x)
    y = Conv2D(nb_anchors, kernel(1), activation='softmax')(x)
    x = concatenate([x, y])
    return keras.models.Model(inputs, x)

def train_model(modele, x_train, y_train, x_validation, y_validation):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,
              metrics=[custom_accuracy, 'binary_crossentropy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    modele.fit(x_train, y_train, batch_size=5, epochs=20, validation_data=(x_validation, y_validation), callbacks=[es])
    return modele

def display_model_prediction(prediction, wanted_prediction, prediction_on_image, wanted_output, save_to_file_name = None):
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
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if save_to_file_name:
        plt.savefig(save_to_file_name, dpi=300)
    plt.show()

def generate_prediction_image(prediction, x_test, y_test, prediction_number = None):
    ratio_x = cfg.image_width / cfg.yolo_width
    ratio_y = cfg.image_height / cfg.yolo_height
    coords = utils.n_max_coord(prediction[:,:,0], 3)
    prediction_on_image = utils.draw_rectangle_on_image(x_test.copy(), prediction, coords)
    coords = utils.treshold_coord(y_test[:,:,0])
    wanted_output = utils.draw_rectangle_on_image(x_test.copy(), y_test, coords)
    display_model_prediction(prediction[:,:,0], y_test[:,:,0], prediction_on_image, wanted_output, 'prediction_' + str(prediction_number) + '.png')

def train():
    x_train, y_train, x_validation, y_validation, x_test, y_test = create_dataset()
    shape = (cfg.image_height, cfg.image_width, 3)
    if cfg.retrain:
        modele = create_model(shape, cfg.yolo_nb_anchors)
        modele.summary()
        modele = train_model(modele, x_train, y_train, x_validation, y_validation)
        modele.save(cfg.model_path_keras, include_optimizer=False)
    else:
        modele = keras.models.load_model(cfg.model_path_keras, custom_objects={'custom_loss': custom_loss, 'custom_accuracy':custom_accuracy})
    
    for i in range(len(x_test)):
        v = [[val for val in x_val] for x_val in x_test[i]]
        start = process_time()
        prediction = modele.predict([[v]])[0]
        stop = process_time()
        print('temps execution : ', stop - start)
        generate_prediction_image(prediction, x_test[i], y_test[i], i)

    sys.argv = ['', cfg.model_path_keras, cfg.model_path_fdeep]

    convert.main()

    breakpoint()

if __name__ == '__main__':
    train()