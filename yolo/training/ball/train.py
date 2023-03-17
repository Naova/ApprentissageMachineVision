import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.dataset_loader import create_dataset
import yolo.utils.args_parser as args_parser

from focal_loss import BinaryFocalLoss

from tensorflow.keras import backend as K
import tensorflow as tf

def custom_activation(x):
    return tf.concat(
        (
            K.sigmoid(x[...,0:1]),
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
    x = SeparableConv2D(64, kernel(3), kernel(2))(inputs)
    x = LeakyReLU()(x)
    x = SeparableConv2D(48, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU()(x)
    x = SeparableConv2D(48, kernel(3), kernel(2), padding='same')(x)
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
    loss = BinaryFocalLoss(gamma=3)
    modele.compile(optimizer=keras.optimizers.Adam(), loss=loss)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, restore_best_weights=True)
    mc = keras.callbacks.ModelCheckpoint('modeles/modele_balles_robot_upper_{epoch:02d}.h5', monitor='val_loss')
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
    train_generator, validation_generator = create_dataset(0.8, 16, labels, dossier_ycbcr, env)
    train(train_generator, validation_generator, modele_path, env)


if __name__ == '__main__':
    main()
