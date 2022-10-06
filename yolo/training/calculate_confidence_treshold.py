import tensorflow.keras as keras

from pathlib import Path

import json

import tqdm

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import lire_toutes_les_images, lire_entrees
from yolo.training.test_robot_model import test_model

def test_datapoints(y_neg, y_pos):
    nb_points = 1000
    min_reached1, min_reached2 = False, False

    stats = {}

    for i in range(0, nb_points):
        #pos = len([y for y in y_pos if y > i / (nb_points*2)]) / len(y_pos)
        neg = len([y for y in y_neg if y > i / (nb_points)]) / len(y_neg)
        pourcent_1 = 0.02
        pourcent_2 = 0.015
        pourcent_3 = 0.01

        if neg < pourcent_1 and not min_reached1:
            print(f'Seuil de détection pour {pourcent_1*100}% faux positifs : ' + str(i/nb_points))
            print('Pourcentage de vrais positifs acceptés : ' + str(len([y for y in y_pos if y > i/nb_points]) / len(y_pos)))
            print('Pourcentage de faux positifs acceptés : ' + str(len([y for y in y_neg if y > i/nb_points]) / len(y_neg)))
            stats[pourcent_1] = {
                'seuil_detection':i/nb_points,
                'vrai_positifs_acceptes':len([y for y in y_pos if y > i/nb_points]) / len(y_pos),
                'faux_positifs_acceptes':len([y for y in y_neg if y > i/nb_points]) / len(y_neg),
            }
            min_reached1 = True
        if neg < pourcent_2 and not min_reached2:
            print(f'Seuil de détection pour {pourcent_2*100}% faux positifs : ' + str(i/nb_points))
            print('Pourcentage de vrais positifs acceptés : ' + str(len([y for y in y_pos if y > i/nb_points]) / len(y_pos)))
            print('Pourcentage de faux positifs acceptés : ' + str(len([y for y in y_neg if y > i/nb_points]) / len(y_neg)))
            stats[pourcent_2] = {
                'seuil_detection':i/nb_points,
                'vrai_positifs_acceptes':len([y for y in y_pos if y > i/nb_points]) / len(y_pos),
                'faux_positifs_acceptes':len([y for y in y_neg if y > i/nb_points]) / len(y_neg),
            }
            min_reached2 = True
        if neg < pourcent_3:
            print(f'Seuil de détection pour {pourcent_3*100}% faux positifs : ' + str(i/nb_points))
            print('Pourcentage de vrais positifs acceptés : ' + str(len([y for y in y_pos if y > i/nb_points]) / len(y_pos)))
            print('Pourcentage de faux positifs acceptés : ' + str(len([y for y in y_neg if y > i/nb_points]) / len(y_neg)))
            stats[pourcent_3] = {
                'seuil_detection':i/nb_points,
                'vrai_positifs_acceptes':len([y for y in y_pos if y > i/nb_points]) / len(y_pos),
                'faux_positifs_acceptes':len([y for y in y_neg if y > i/nb_points]) / len(y_neg),
            }
            return stats
    return stats

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)

    test_data_negative = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobot'))
    test_data_positive = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobotPositive'))
    test_data_positive += lire_entrees(cfg_prov.get_config().get_labels_path('Robot'), cfg_prov.get_config().get_dossier('Robot'), env='Robot')

    modeles = Path(f'tests/{cfg_prov.get_config().camera}/').glob('*.h5')
    modeles = [m for m in modeles]
    for modele_path in tqdm.tqdm(modeles):
        print(f'loading {modele_path}')
        modele = keras.models.load_model(str(modele_path))
        #modele.summary()

        max_confidences_negative = test_model(modele, test_data_negative)
        max_confidences_positive = test_model(modele, test_data_positive)

        y_neg = [x[2] for x in max_confidences_negative]
        y_pos = [x[2] for x in max_confidences_positive]

        stats_brutes = {
            'y_neg':y_neg,
            'y_pos':y_pos,
        }
        
        with open(str(modele_path).replace('.h5', '.json'), 'w') as f:
            json.dump(stats_brutes, f)

        new_stats = test_datapoints(y_neg, y_pos)

        with open(f'stats_modeles_confidence_{cfg_prov.get_config().camera}.json', 'r') as f:
            stats = json.load(f)
        stats[str(modele_path).replace('\\', '/').split('/')[-1]] = new_stats
        with open(f'stats_modeles_confidence_{cfg_prov.get_config().camera}.json', 'w') as f:
            json.dump(stats, f)


if __name__ == '__main__':
    main()
