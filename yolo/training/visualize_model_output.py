import tensorflow.keras as keras

from time import process_time
import numpy as np


import yolo.training.configuration_provider as cfg_prov
cfg = cfg_prov.ConfigurationProvider().get_config()
from yolo.training.dataset_loader import create_dataset, lire_entrees
import yolo.utils.image_processing as image_processing
import yolo.utils.args_parser as args_parser


def main():
    args = args_parser.parse_args_env_cam('Train a yolo model to detect balls on an image.')
    env = args_parser.set_config(args, use_robot=False)

    labels = cfg_prov.get_config().get_labels_path(env)
    dossier_ycbcr = cfg_prov.get_config().get_dossier(env, 'YCbCr')
    modele_path = cfg_prov.get_config().get_modele_path(env)

    train_generator, validation_generator, test_data = create_dataset(16, labels, dossier_ycbcr, env)
    if not args.simulation:
        test_data = lire_entrees(cfg_prov.get_config().get_labels_path('Robot'), cfg_prov.get_config().get_dossier('Robot'), 'Robot')
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
        image_processing.generate_prediction_image(prediction, entree_x, entree.y(), i)

if __name__ == '__main__':
    main()
