import numpy as np
from pathlib import Path
import json
import random
import math
import os

from PIL import Image

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.keras_sequence import KerasSequence

def best_anchor(anchors, rayon):
    distances = [abs(a - rayon) for a in anchors]
    return distances.index(min(distances))

class Entree:
    def __init__(self, nom:str, labels:dict, image_path:str, flipper:bool, env:str):
        self.nom = nom
        self.balles = [label for label in labels if label['categorie'] == 1]
        self.robots = [label for label in labels if label['categorie'] == 2]
        self.image_path = image_path
        self.flipper = flipper
        self.env = env
    def x(self):
        image = Image.open(self.image_path)
        resized_image_height, resized_image_width = cfg_prov.get_config().get_model_input_resolution()
        if resized_image_height != image.size[1] or resized_image_width != image.size[0]:
            image = image.resize((resized_image_width, resized_image_height), Image.NEAREST)
        image = np.array(image) / 255.
            
        if self.flipper:
            return np.fliplr(image)
        return image
    def y(self):
        image_height, image_width = cfg_prov.get_config().get_image_resolution(self.env if self.env != 'Genere' else 'Robot')
        yolo_height, yolo_width = cfg_prov.get_config().get_model_output_resolution()
        anchors = cfg_prov.get_config().get_anchors()
        value = np.zeros((yolo_height, yolo_width, 5 + len(anchors)))
        for obj in self.balles:
            width = obj['right'] - obj['left']
            height = obj['bottom'] - obj['top']
            if self.flipper:
                x = image_width - obj['right'] + width / 2 #centre geometrique de la boite
            else:
                x = obj['left'] + width / 2 #centre geometrique de la boite
            y = obj['top'] + height / 2 #centre geometrique de la boite
            center_x = int(x / image_width * yolo_width)
            center_y = int(y / image_height * yolo_height)
            center = (center_y, center_x)

            value[center][0] = 1 #presence d'objet
            #classification pour position x-y
            if x / image_width * yolo_width - center_x < 0.5:
                value[center][1] = 1
            else:
                value[center][2] = 1
            if y / image_height * yolo_height - center_y < 0.5:
                value[center][3] = 1
            else:
                value[center][4] = 1
            #classification pour la taille de l'objet
            rayon = max(width, height) / image_width / 2
            best_anchor_index = best_anchor(anchors, rayon)
            value[center][5 + best_anchor_index] = 1 #rayon anchor
        return value

def lire_entrees(labels_path:str, data_path:str, env:str = 'Simulation'):
    entrees = []
    with open(labels_path) as fichier:
        labels = json.loads(fichier.read())
    for image_label in labels:
        fichier_image = data_path + image_label
        if os.path.exists(fichier_image):
            entrees.append(Entree(image_label, labels[image_label], fichier_image, True, env))
            entrees.append(Entree(image_label, labels[image_label], fichier_image, False, env))
    return entrees

def lire_toutes_les_images(path:str):
    dossier = Path(path).glob('*/*')
    fichiers = [str(f) for f in dossier]
    entrees = [Entree(f.split('/')[-1], {}, f, False, 'Robot') for f in fichiers]
    entrees += [Entree(f.split('/')[-1], {}, f, True, 'Robot') for f in fichiers]
    return entrees

def split_dataset(entrees, ratio_train=0.9, batch_size=16):
    random.shuffle(entrees)

    i = int(len(entrees) * ratio_train)

    train = KerasSequence(entrees[:i], batch_size, Entree.x, Entree.y)
    validation = KerasSequence(entrees[i:], batch_size, Entree.x, Entree.y)

    return train, validation

def create_dataset(ratio_train, batch_size, labels_path:str, images_path:str, env:str):
    entrees = lire_entrees(labels_path, images_path, env)
    #entrees = [e for e in entrees if e.balles] ? (faut tester)
    if env == 'Genere':
        path = cfg_prov.get_config().get_dossier('HardNegative', 'YCbCr')
        entrees += lire_toutes_les_images(path)
    train, validation = split_dataset(entrees, ratio_train, batch_size)
    return train, validation
