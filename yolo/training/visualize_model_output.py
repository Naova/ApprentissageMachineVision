import tensorflow.keras as keras

from time import process_time
import numpy as np


from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.image_processing as image_processing
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import load_test_set
from yolo.training.ball.train import custom_activation, custom_loss

import random


def main():
    args = args_parser.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    env = args_parser.set_config(args, use_robot=True)
    
    modele_path = cfg_prov.get_config().get_modele_path(env)
    
    if env == 'Kaggle':
        env = 'Robot'
    #test_data = lire_entrees(cfg_prov.get_config().get_labels_path(env), cfg_prov.get_config().get_dossier(env), env)
    #test_data = lire_toutes_les_images(cfg_prov.get_config().get_dossier('RobotSansBalle'))

    test_data_negative, test_data_positive = load_test_set()
    test_data = test_data_negative + test_data_positive
    random.shuffle(test_data)
    
    modele = keras.models.load_model(modele_path,custom_objects={'custom_loss':custom_loss, 'custom_activation':custom_activation})
    modele.summary()
    cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])

    for i, entree in enumerate(test_data):
        entree_x = entree.x()
        start = process_time()
        prediction = modele.predict(np.array([entree_x]))[0]
        stop = process_time()
        print(entree.nom + ' : ', stop - start)
        nom = entree.nom.split('/')[-1]
        flip = 'flipped' if entree.flipper else 'original'
        filename = f'{nom}_{flip}.png'
        image_processing.generate_prediction_image(prediction, entree_x, entree.y(), filename)

if __name__ == '__main__':
    main()
