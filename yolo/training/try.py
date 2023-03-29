from pathlib import Path

import tensorflow.keras as keras

from yolo.training.test_robot_model import make_predictions, save_stats
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import lire_toutes_les_images, lire_entrees
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

import os
import pickle as pkl

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)

    test_data_positive = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobotPositive'))
    test_data_negative = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobot'))
    test_data_positive += lire_entrees(cfg_prov.get_config().get_labels_path('Robot'), cfg_prov.get_config().get_dossier('Robot'), env='Robot')

    fichiers = Path('modeles/').glob('*.h5')

    fichiers = sorted([f.as_posix() for f in fichiers])

    for fichier in fichiers:
        print(f'\n\ntest du modele : {fichier}\n')
        modele = keras.models.load_model(fichier)
        modele.summary()
        cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])



        data = {}
        if Path(f'{fichier}.pkl').exists():
            with open(f'{fichier}.pkl', 'rb') as f:
                data = pkl.load(f)
                max_confidences_negative = data['n']
                max_confidences_positive = data['p']
                iou = data['i']
        else:
            max_confidences_negative = make_predictions(modele, test_data_negative)
            max_confidences_positive = make_predictions(modele, test_data_positive)
            ious = [m[-1] for m in max_confidences_positive if m[-1] is not None]
            iou = 100 * sum(ious) / len(ious)
            with open(f'{fichier}.pkl', 'wb') as f:
                data['p'] = max_confidences_positive
                data['n'] = max_confidences_negative
                data['i'] = iou
                pkl.dump(data, f)

        save_stats(max_confidences_negative, max_confidences_positive, fichier, iou)

    #fichiers = Path('modeles/').glob('*')
    #for fichier in fichiers:
    #    os.remove(fichier.as_posix())


if __name__ == '__main__':
    main()
