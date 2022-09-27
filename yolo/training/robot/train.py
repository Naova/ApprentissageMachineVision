import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, SeparableConv2D, LeakyReLU

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.dataset_loader import create_dataset, lire_entrees
import yolo.utils.image_processing as image_processing
import yolo.utils.args_parser as args_parser


def kernel(x):
    return (x, x)

#a modifier
def create_model_upper():
    inputs = keras.Input(shape=(*cfg_prov.get_config().get_model_input_resolution(), 3))
    x = SeparableConv2D(16, kernel(3), kernel(2), padding='same', bias_initializer='random_normal')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = SeparableConv2D(24, kernel(3), kernel(1), padding='same', bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, kernel(1), kernel(1), bias_initializer='random_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(5 + len(cfg_prov.get_config().get_anchors()), kernel(1), kernel(1), activation='sigmoid', bias_initializer='random_normal')(x)
    return keras.Model(inputs=inputs, outputs=x)

#a modifier
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

#a modifier
def train_model(modele, train_generator, validation_generator):
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, restore_best_weights=True)
    mc = keras.callbacks.ModelCheckpoint('modele_robots_robot_upper_{epoch:02d}.h5', monitor='val_loss')
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
    args = args_parser.parse_args_env_cam('Train a yolo model to detect robots on an image.', choosedetector=False)
    args.detect_balls = False
    env = args_parser.set_config(args, use_robot=False, use_kaggle=True)

    labels = cfg_prov.get_config().get_labels_path(env)
    dossier_ycbcr = cfg_prov.get_config().get_dossier(env, 'YCbCr')
    modele_path = cfg_prov.get_config().get_modele_path(env)
    train_generator, validation_generator = create_dataset(0.9, 16, labels, dossier_ycbcr, env)
    train(train_generator, validation_generator, modele_path, env, True)


if __name__ == '__main__':
    main()
