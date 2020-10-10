import numpy as np
from pathlib import Path
import json
import utils

import sys
sys.path.insert(0,'..')
import config as cfg

def best_anchor(anchors, rayon):
    distances = [abs(a - rayon) for a in anchors]
    return distances.index(min(distances))

class Entree:
    def __init__(self, nom, labels, image):
        self.nom = nom
        #self.robots = [label for label in labels if label['categorie'] == 2]
        self.balles = [label for label in labels if label['categorie'] == 1]
        self.image = image
    def train_data(self):
        return self.image
    def value_data_yolo(self):
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
            image = np.fromfile(fichier_image, dtype=np.float32)
            image = np.reshape(image, (cfg.image_height, cfg.image_width, 3))
            entrees.append(Entree(image_label, labels[image_label], image))
    return entrees

def split_dataset(x, y):
    ratio_train = 0.89
    ratio_validation = 0.1
    ratio_test = 1.0 - (ratio_train + ratio_validation)

    i = int(len(x) * ratio_train)
    j = int(len(x) * ratio_validation)
    k = int(len(x) * ratio_test)

    t = np.zeros(i) #0 = train
    u = np.ones(j) #1 = validation
    v = np.add(np.ones(k), np.ones(k)) #2 = test

    arr = np.concatenate([t, u, v])

    while len(arr) > len(x):
        arr = arr[1:]
    while len(arr) < len(x):
        arr = np.append(arr, 0)
    np.random.shuffle(arr)

    x_train = [x[i] for i in np.where(arr == 0)[0]]
    y_train = [y[i] for i in np.where(arr == 0)[0]]
    x_val = [x[i] for i in np.where(arr == 1)[0]]
    y_val = [y[i] for i in np.where(arr == 1)[0]]
    x_test = [x[i] for i in np.where(arr == 2)[0]]
    y_test = [y[i] for i in np.where(arr == 2)[0]]

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_dataset():
    entrees = lire_entrees()
    x = [e.train_data() for e in entrees]
    y = [e.value_data_yolo() for e in entrees]
    x_train, y_train, x_validation, y_validation, x_test, y_test = split_dataset(x, y)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_validation, y_validation, x_test, y_test