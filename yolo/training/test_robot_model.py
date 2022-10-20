import tensorflow.keras as keras

import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import lire_toutes_les_images, lire_entrees
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov


def make_predictions(modele, test_data):
    max_confidences = []

    for entree in tqdm.tqdm(test_data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        max_confidences.append((entree.nom, entree.flipper, prediction[:,:,0].max().astype('float')))
    max_confidences.sort(key=lambda x: x[2])
    return max_confidences

def calculate_confidence_treshold(y_neg, y_pos):
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

def save_stats_brutes(y_neg, y_pos, filepath:str):
    stats_brutes = {
        'y_neg':y_neg,
        'y_pos':y_pos,
    }
    with open(filepath, 'w') as f:
        json.dump(stats_brutes, f)

def copy_model_file(modele_path, time):
    destination = f'tests/{cfg_prov.get_config().camera}/{time}.h5'
    shutil.copy(modele_path, destination)

def save_stats(confidences_negative, confidences_positive, modele_path):
    time = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

    y_neg = [x[2] for x in confidences_negative]
    y_pos = [x[2] for x in confidences_positive]

    save_stats_brutes(y_neg, y_pos, f'tests/{cfg_prov.get_config().camera}/{time}.json')

    new_stats = calculate_confidence_treshold(y_neg, y_pos)

    with open(f'stats_modeles_confidence_{cfg_prov.get_config().camera}.json', 'r') as f:
        stats = json.load(f)

    treshold = new_stats[0.01]['seuil_detection']

    false_negative = [x for x in y_pos if x <= treshold]
    true_negative = [x for x in y_neg if x <= treshold]
    false_positive = [x for x in y_neg if x > treshold]
    true_positive = [x for x in y_pos if x > treshold]

    precision = 100 * len(true_positive) / (len(true_positive) + len(false_positive))
    recall = 100 * len(true_positive) / (len(true_positive) + len(false_negative))
    f1_score = 2 * (recall * precision) / (recall + precision)
    
    fp = 100 * len(false_positive) / (len(true_negative) + len(false_positive))
    fn = 100 * len(true_positive) / (len(false_negative) + len(true_positive))
    print(f'False positive : {fp:.2f}%')
    print(f'True positive : {fn:.2f}%')
    
    print(f'Precision : {precision:.2f}%')
    print(f'Recall : {recall:.2f}%')
    print(f'F1 score : {f1_score}%')

    if fn < 65:
        print('Score trop bas, ne sauvegarde pas.')
        return
    else:
        print('Score bon, on sauvegarde')

    stats[cfg_prov.get_config().get_modele_path(env).replace('\\', '/').split('/')[-1]] = new_stats
    with open(f'stats_modeles_confidence_{cfg_prov.get_config().camera}.json', 'w') as f:
        json.dump(stats, f)

    plt.scatter(range(len(false_negative)), false_negative, s=10, color='orange')
    plt.scatter(range(len(false_negative), len(true_positive)+len(false_negative)), true_positive, s=10, color='blue')
    plt.scatter(range(len(true_negative)), true_negative, s=10, color='green')
    plt.scatter(range(len(true_negative), len(false_positive)+len(true_negative)), false_positive, s=10, color='red')
    
    plt.text(0,0.65, f'True positive : {fn:.2f}%')
    plt.text(0,0.6, f'False positive : {fp:.2f}%')
    plt.text(0,0.55,f'Precision : {precision:.2f}%')
    plt.text(0,0.5,f'Recall : {recall:.2f}%')
    plt.text(0,0.45,f'F1 Score : {f1_score:.2f}%')

    plt.axhline(y = treshold, color = 'b', linestyle = ':')
    plt.rcParams["axes.titlesize"] = 10
    plt.title(f'Maximal confidence level per image from the test dataset.\n{cfg_prov.get_config().camera.capitalize()} camera.')
    plt.xlabel('Images (sorted)')
    plt.ylabel('Max confidence level')
    plt.ylim(-0.05, 1.)
    plt.legend(['False negatives', 'True positives', 'True negatives', 'False positives', 'Detection treshold'])
    plt.grid()

    plt.savefig(f'tests/{cfg_prov.get_config().camera}/{time}.png')

    plt.clf()

    somme_neg = sum(y_neg)
    print(somme_neg)
    somme_pos = sum(y_pos)
    print(somme_pos)
    
    copy_model_file(modele_path, time)

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)
    modele_path = cfg_prov.get_config().get_modele_path(env)
    print(modele_path)
    modele = keras.models.load_model(modele_path)
    modele.summary()
    test_data_positive = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobotPositive'))
    test_data_negative = lire_toutes_les_images(cfg_prov.get_config().get_dossier('TestRobot'))
    test_data_positive += lire_entrees(cfg_prov.get_config().get_labels_path('Robot'), cfg_prov.get_config().get_dossier('Robot'), env='Robot')

    max_confidences_negative = make_predictions(modele, test_data_negative)
    max_confidences_positive = make_predictions(modele, test_data_positive)
    
    save_stats(max_confidences_negative, max_confidences_positive, modele_path)


if __name__ == '__main__':
    main()
