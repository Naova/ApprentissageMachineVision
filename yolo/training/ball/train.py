import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.dataset_loader import load_train_val_set
import yolo.utils.args_parser as args_parser

from focal_loss import BinaryFocalLoss

import tensorflow as tf
import tensorflow.keras.backend as K

def load_model(modele_path = None, env = None):
    if modele_path is None:
        modele_path = cfg_prov.get_config().get_modele_path(env)
    modele = keras.models.load_model(modele_path, custom_objects={'custom_loss':custom_loss, 'custom_activation':custom_activation})
    return modele

def custom_loss(y_true, y_pred, lambda_1 = 5, lambda_2 = 0.5):
    obj_mask = y_true[:,:,:,0:1]
    mask_shape = tf.shape(obj_mask)
    noobj_mask = tf.ones(mask_shape) - obj_mask

    bfl = BinaryFocalLoss(gamma=3)

    #confidence
    confidence_loss = bfl(y_true[...,0:1], y_pred[...,0:1]) * obj_mask \
                        + bfl(y_true[...,0:1], y_pred[...,0:1]) * noobj_mask * lambda_2
    
    #box
    box_loss = bfl(y_true[..., 1:], y_pred[..., 1:]) * obj_mask * lambda_1

    #global
    global_loss = tf.concat([confidence_loss, box_loss], axis=-1)
    return global_loss

def custom_activation(x):
    return tf.concat(
        (
            K.sigmoid(x[..., 0:1]),
            K.softmax(x[..., 1:3]),
            K.softmax(x[..., 3:5]),
            K.softmax(x[..., 5:])
        ), axis=-1)

def kernel(x):
    return (x, x)

def create_model_upper_simulation():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(inputs)
    x = LeakyReLU()(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, kernel(1), kernel(1), bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation=custom_activation, bias_initializer='random_normal')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower_simulation():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2))(inputs)
    x = LeakyReLU()(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, kernel(1), kernel(1))(x)
    x = LeakyReLU()(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation=custom_activation)(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_upper_robot():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(16, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(24, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, kernel(1), kernel(1), bias_initializer='random_normal')(x)
    x = LeakyReLU()(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation=custom_activation, bias_initializer='random_normal')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower_robot():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(32, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, kernel(1), kernel(1))(x)
    x = LeakyReLU()(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation=custom_activation)(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model(env):
    if env == 'Genere' or env == 'Kaggle':
        if cfg_prov.get_config().camera == 'upper':
            return create_model_upper_robot()
        else:
            return create_model_lower_robot()
    else:
        if cfg_prov.get_config().camera == 'upper':
            return create_model_upper_simulation()
        else:
            return create_model_lower_simulation()

def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
    camera = cfg_prov.get_config().camera
    mc = keras.callbacks.ModelCheckpoint('modeles/modele_balles_robot_' + camera + '_{epoch:02d}.h5', monitor='val_loss')
    modele.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[es, mc])
    return modele

def train(train_generator, validation_generator, modele_path, env):
    modele = create_model(env)
    modele.summary()
    cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])
    modele = train_model(modele, train_generator, validation_generator)
    modele.save(modele_path, include_optimizer=False)
    print('sauvegarde du modele : ' + modele_path)
        
def main():
    args = args_parser.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    args.detect_balls = True
    env = args_parser.set_config(args, use_robot=False)

    labels = cfg_prov.get_config().get_labels_path(env)
    dossier_ycbcr = cfg_prov.get_config().get_dossier(env, 'YCbCr')
    modele_path = cfg_prov.get_config().get_modele_path(env)
    train_generator, validation_generator = load_train_val_set(0.8, 16, labels, dossier_ycbcr, env)
    train(train_generator, validation_generator, modele_path, env)


if __name__ == '__main__':
    main()
