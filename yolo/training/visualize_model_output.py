import tensorflow.keras as keras

from time import process_time
import numpy as np


from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.dataset_loader import create_dataset, lire_entrees
import yolo.utils.image_processing as image_processing
import yolo.utils.args_parser as args_parser


def main():
    args = args_parser.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    env = args_parser.set_config(args, use_robot=True)
    
    modele_path = cfg_prov.get_config().get_modele_path(env)
    
    if env == 'Kaggle':
        env = 'Robot'
    test_data = lire_entrees(cfg_prov.get_config().get_labels_path(env), cfg_prov.get_config().get_dossier(env), env)
    #test_data = lire_toutes_les_images(cfg_prov.get_config().get_dossier('RobotSansBalle'))
    
    modele = keras.models.load_model(modele_path)
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
