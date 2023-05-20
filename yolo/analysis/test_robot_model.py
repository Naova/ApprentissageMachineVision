import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import load_test_set
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.utils.metrics import iou_balles
from yolo.training.ball.train import load_model


def make_predictions(modele, test_data):
    max_confidences = []

    for entree in tqdm.tqdm(test_data):
        entree_x = entree.x()
        prediction = modele.predict(np.array([entree_x]))[0]
        if entree.balles:
            iou = iou_balles(prediction, entree.y())
        else:
            iou = None
        max_confidences.append((entree, prediction[:,:,0].max().astype('float'), iou))
    max_confidences.sort(key=lambda x: x[1])
    return max_confidences

def calculate_confidence_treshold(y_neg, y_pos):
    nb_points = 1000

    stats = {}

    for i in range(0, nb_points):
        neg = len([y for y in y_neg if y > i / (nb_points)]) / len(y_neg)
        seuil = 0.01

        if neg < seuil:
            print(f'Seuil de détection pour {seuil*100}% faux positifs : ' + str(i/nb_points))
            print('Pourcentage de vrais positifs acceptés : ' + str(len([y for y in y_pos if y > i/nb_points]) / len(y_pos)))
            print('Pourcentage de faux positifs acceptés : ' + str(len([y for y in y_neg if y > i/nb_points]) / len(y_neg)))
            stats[seuil] = {
                'seuil_detection':i/nb_points
            }
            return stats
    return stats

def copy_model_file(modele_path, time):
    destination = f'tests/{cfg_prov.get_config().camera}/{time}.h5'
    print(f'copy from {modele_path} to {destination}')
    shutil.copy(modele_path, destination)

def save_fp_fn(confidences_negative, confidences_positive, treshold, modele_path):
    camera = cfg_prov.get_config().camera
    
    try:
        with open(f'images_fn_{camera}.json', 'r') as fn:
            fn_images_global = json.load(fn)
        with open(f'images_fp_{camera}.json', 'r') as fp:
            fp_images_global = json.load(fp)
    except:
        fn_images_global = {}
        fp_images_global = {}

    fp_images = set()
    fn_images = set()

    for entree, confiance, _ in confidences_positive:
        if confiance < treshold: # faux negatif
            fn_images.add(entree.image_path)
    fn_images_global[modele_path] = list(fn_images)

    for entree, confiance, _ in confidences_negative:
        if confiance > treshold: # faux positif
            fp_images.add(entree.image_path)
    fp_images_global[modele_path] = list(fp_images)
    

    with open(f'images_fn_{camera}.json', 'w') as fn:
        json.dump(fn_images_global, fn)
    with open(f'images_fp_{camera}.json', 'w') as fp:
        json.dump(fp_images_global, fp)

def create_image(false_negative, true_negative, false_positive, true_positive, precision, recall, f1_score, iou, treshold, time):
    plt.scatter(range(len(false_negative)), false_negative, s=10, color='orange')
    plt.scatter(range(len(false_negative), len(true_positive)+len(false_negative)), true_positive, s=10, color='blue')
    plt.scatter(range(len(true_negative)), true_negative, s=10, color='green')
    plt.scatter(range(len(true_negative), len(false_positive)+len(true_negative)), false_positive, s=10, color='red')
    
    plt.text(0,0.55,f'Precision : {precision:.2f}%')
    plt.text(0,0.5,f'Recall : {recall:.2f}%')
    plt.text(0,0.45,f'F1 Score : {f1_score:.2f}%')
    plt.text(0,0.4,f'Avg. IoU : {iou:.2f}%')

    plt.axhline(y = treshold, color = 'b', linestyle = ':')
    plt.rcParams["axes.titlesize"] = 10
    plt.title(f'Maximal confidence level per image from the test dataset.\n{cfg_prov.get_config().camera.capitalize()} camera.')
    plt.xlabel('Images (sorted)')
    plt.ylabel('Max confidence level')
    plt.ylim(-0.05, 1.)
    plt.legend(['Detection treshold', 'False negatives', 'True positives', 'True negatives', 'False positives'])
    plt.grid()

    plt.savefig(f'tests/{cfg_prov.get_config().camera}/{time}.png')

    plt.clf()

def save_stats(confidences_negative, confidences_positive, modele_path, iou):
    time = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

    y_neg = [x[1] for x in confidences_negative]
    y_pos = [x[1] for x in confidences_positive]

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
    
    new_stats[0.01]['iou'] = iou
    new_stats[0.01]['recall'] = recall
    new_stats[0.01]['precision'] = precision
    new_stats[0.01]['f1_score'] = f1_score
    
    print(f'Precision : {precision:.2f}%')
    print(f'Recall : {recall:.2f}%')
    print(f'F1 score : {f1_score:.2f}%')
    print(f'Average IoU: {iou:.2f}%')

    if recall < cfg_prov.get_config().get_min_recall() or iou < 50:
        print('Score trop bas, ne sauvegarde pas.')
        return
    else:
        print('Score bon, on sauvegarde')

    stats[f'{time}.h5'] = new_stats
    with open(f'stats_modeles_confidence_{cfg_prov.get_config().camera}.json', 'w') as f:
        json.dump(stats, f)

    create_image(false_negative, true_negative, false_positive, true_positive, precision, recall, f1_score, iou, treshold, time)
    save_fp_fn(confidences_negative, confidences_positive, treshold, time)

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
    modele = load_model(env=env)
    modele.summary()
    cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])
    
    test_data_negative, test_data_positive = load_test_set()

    max_confidences_positive = make_predictions(modele, test_data_positive)
    max_confidences_negative = make_predictions(modele, test_data_negative)
    ious = [m[-1] for m in max_confidences_positive if m[-1] is not None]
    iou = 100 * sum(ious) / len(ious)
    
    save_stats(max_confidences_negative, max_confidences_positive, modele_path, iou)


if __name__ == '__main__':
    main()
