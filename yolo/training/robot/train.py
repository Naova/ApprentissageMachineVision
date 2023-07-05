import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

from yolo.training.ball.train import custom_loss, custom_activation
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.dataset_loader import load_train_val_set
import yolo.utils.args_parser as args_parser

from focal_loss import BinaryFocalLoss

def kernel(x):
    return (x, x)

def create_model_upper():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 1))
    x = Conv2D(16, kernel(3), kernel(1), padding='same')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = SeparableConv2D(32, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(2), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation=custom_activation)(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model_lower():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, kernel(1), kernel(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_model(env):
    if cfg_prov.get_config().camera == 'upper':
        return create_model_upper()
    else:
        return create_model_lower()

def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = keras.callbacks.ModelCheckpoint('modeles/modele_robots_robot_upper_{epoch:02d}.h5', monitor='val_loss')
    modele.fit(train_generator, validation_data=validation_generator, epochs=1000, callbacks=[es, mc])
    return modele

def train(train_generator, validation_generator, modele_path, env):
    modele = create_model(env)
    modele.summary()
    cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])
    modele = train_model(modele, train_generator, validation_generator)
    modele.save(modele_path, include_optimizer=False)
    print('sauvegarde du modele : ' + modele_path)

def main():
    args = args_parser.parse_args_env_cam('Train a yolo model to detect robots on an image.')
    args.detect_balls = False
    env = args_parser.set_config(args, use_robot=False)

    labels = cfg_prov.get_config().get_labels_path(env)
    dossier_ycbcr = cfg_prov.get_config().get_dossier(env, 'YCbCr')
    modele_path = cfg_prov.get_config().get_modele_path(env)
    train_generator, validation_generator = load_train_val_set(0.8, 16, labels, dossier_ycbcr, env)
    train(train_generator, validation_generator, modele_path, env)


if __name__ == '__main__':
    main()
