import numpy as np
from pathlib import Path
import json
import utils
import random

import sys
sys.path.insert(0,'..')
import config as cfg
from KerasSequence import KerasSequence

def best_anchor(anchors, rayon):
    distances = [abs(a - rayon) for a in anchors]
    return distances.index(min(distances))

class Entree:
    def __init__(self, nom, labels, image_path):
        self.nom = nom
        #self.robots = [label for label in labels if label['categorie'] == 2]
        self.balles = [label for label in labels if label['categorie'] == 1]
        self.image_path = image_path
    def x(self):
        image = np.fromfile(self.image_path, dtype=np.float32)
        image = np.reshape(image, (cfg.image_height, cfg.image_width, 3))
        return image
    def y(self):
        anchors = cfg.get_anchors()
        value = np.zeros((cfg.yolo_height, cfg.yolo_width, 3 + len(anchors)))
        for balle in self.balles:
            width = balle['right'] - balle['left']
            height = balle['bottom'] - balle['top']
            x = balle['left'] + width / 2 #centre geometrique de la boite
            y = balle['top'] + height / 2 #centre geometrique de la boite
            center_x = int(x / cfg.image_width * cfg.yolo_width)
            center_y = int(y / cfg.image_height * cfg.yolo_height)
            center = (center_y, center_x)

            value[center][0] = 1                                                     #presence d'objet
            value[center][1] = x / cfg.image_width * cfg.yolo_width - center_x       #center_x
            value[center][2] = y / cfg.image_height * cfg.yolo_height - center_y     #center_y
            rayon = max(width, height) / cfg.image_width / 2
            
            best_anchor_index = best_anchor(anchors, rayon)
            value[center][3 + best_anchor_index] = 1                                 #rayon anchor

        return value

def lire_entrees():
    entrees = []
    with open(cfg.json_etiquettes) as fichier:
        labels = json.loads(fichier.read())
        for image_label in labels:
            fichier_image = cfg.dossier_brut + image_label
            entrees.append(Entree(image_label, labels[image_label], fichier_image))
    return entrees

def split_dataset(entrees, batch_size=16):
    random.shuffle(entrees)
    ratio_train = 0.9
    ratio_validation = 0.09

    i = int(len(entrees) * ratio_train)
    j = int(len(entrees) * (ratio_train + ratio_validation))
    k = int(len(entrees) - i - j)

    train = KerasSequence(entrees[:i], batch_size, Entree.x, Entree.y)
    validation = KerasSequence(entrees[i:j], batch_size, Entree.x, Entree.y)
    test = entrees[j:]

    return train, validation, test

def create_dataset(batch_size):
    entrees = lire_entrees()
    train, validation, test = split_dataset(entrees, batch_size)
    return train, validation, test