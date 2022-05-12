import numpy as np
from pathlib import Path
import json
import utils
import random
import math

from PIL import Image

import sys
sys.path.insert(0,'..')
import config as cfg
from KerasSequence import KerasSequence

def best_anchor(anchors, rayon):
    distances = [abs(a - rayon) for a in anchors]
    return distances.index(min(distances))

class Entree:
    def __init__(self, nom:str, labels:dict, image_path:str, flipper:bool, env:str):
        self.nom = nom
        self.balles = [label for label in labels if label['categorie'] == 1]
        self.image_path = image_path
        self.flipper = flipper
        self.env = env
    def nearest(self, image, resized_image_width, resized_image_height):
        """
        Exact same function as executed in C++ on the robot.
        """
        new_image = np.zeros((resized_image_height, resized_image_width, 3))
        x_scale = (image.shape[1] * 1.0) / resized_image_width
        y_scale = (image.shape[0] * 1.0) / resized_image_height
        x = 0
        y = 0
        for i in range(resized_image_height):
            x = math.floor(1.0 + (i * x_scale))
            for j in range(resized_image_width):
                y = math.floor(1.0 + (j * y_scale))
                new_image[i][j][0] = image[x][y][0]
                new_image[i][j][1] = image[x][y][1]
                new_image[i][j][2] = image[x][y][2]
        return new_image
    def x(self):
        image_height, image_width = cfg.get_image_resolution(self.env)
        resized_image_height, resized_image_width = cfg.get_resized_image_resolution()
        image = np.fromfile(self.image_path, dtype=np.float32)
        image = np.reshape(image, (image_height, image_width, 3))
        if resized_image_height != image_height or resized_image_width != image_width:
            img = Image.fromarray((image*255).astype(np.uint8))
            img = img.resize((resized_image_width, resized_image_height), Image.NEAREST)
            image = np.array(img) / 255.
        if self.flipper:
            return np.fliplr(image)
        return image
    def y(self):
        image_height, image_width = cfg.get_image_resolution()
        yolo_height, yolo_width = cfg.get_yolo_resolution()
        anchors = cfg.get_anchors()
        value = np.zeros((yolo_height, yolo_width, 3 + len(anchors)))
        for balle in self.balles:
            width = balle['right'] - balle['left']
            height = balle['bottom'] - balle['top']
            if self.flipper:
                x = image_width - balle['right'] + width / 2 #centre geometrique de la boite
            else:
                x = balle['left'] + width / 2 #centre geometrique de la boite
            y = balle['top'] + height / 2 #centre geometrique de la boite
            center_x = int(x / image_width * yolo_width)
            center_y = int(y / image_height * yolo_height)
            center = (center_y, center_x)

            value[center][0] = 1                                             #presence d'objet
            value[center][1] = x / image_width * yolo_width - center_x       #center_x
            value[center][2] = y / image_height * yolo_height - center_y     #center_y
            rayon = max(width, height) / image_width / 2
            
            best_anchor_index = best_anchor(anchors, rayon)
            value[center][3 + best_anchor_index] = 1                         #rayon anchor

        return value

def lire_entrees(labels_path:str, brut_path:str, env:str = 'Simulation'):
    entrees = []
    with open(labels_path) as fichier:
        labels = json.loads(fichier.read())
    for image_label in labels:
        fichier_image = brut_path + image_label
        if cfg.flipper_images:
            entrees.append(Entree(image_label, labels[image_label], fichier_image, True, env))
        entrees.append(Entree(image_label, labels[image_label], fichier_image, False, env))
    return entrees

def lire_toutes_les_images(path:str):
    dossier = Path(path).glob('*')
    fichiers = [str(f) for f in dossier]
    entrees = [Entree(f.split('/')[-1], {}, f, False, 'Robot') for f in fichiers]
    entrees += [Entree(f.split('/')[-1], {}, f, True, 'Robot') for f in fichiers]
    return entrees

def split_dataset(entrees, batch_size=16, test=True):
    random.shuffle(entrees)
    ratio_train = 0.95 #95%
    ratio_test = 20 / len(entrees) #nombre fixe, pas besoin de plus
    #ratio_validation = 10% - 20

    i = int(len(entrees) * ratio_train)#train
    if test:
        j = int(len(entrees) * (ratio_train + ratio_test))#test
    else:
        j = i

    train = KerasSequence(entrees[:i], batch_size, Entree.x, Entree.y)
    validation = KerasSequence(entrees[j:], batch_size, Entree.x, Entree.y)
    test_data = entrees[i:j] if test else None

    return train, validation, test_data

def create_dataset(batch_size, labels_path:str, images_path:str, env:str):
    entrees = lire_entrees(labels_path, images_path, env)
    if env == 'Genere':
        path = '../' + cfg.get_dossier('HardNegative', 'Brut')
        entrees += lire_toutes_les_images(path)
    train, validation, test = split_dataset(entrees, batch_size, env == 'Simulation')
    return train, validation, test
