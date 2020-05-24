import numpy as np
from pathlib import Path
import csv

import sys
sys.path.insert(0,'..')
import config as cfg


fichier_etiquetes_csv = cfg.csv_etiquettes

def normaliser_image(image):
    return image/255

class Entree:
    def __init__(self, nom, centre_x, centre_y, radius, image, containsBall):
        self.nom = nom
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.radius = radius * 1.3
        self.image = normaliser_image(image)
        self.containsBall = containsBall
    def train_data(self):
        return self.image
    def value_data(self):
        value = np.zeros((cfg.yolo_height, cfg.yolo_width, cfg.yolo_categories))
        x = int(self.centre_x / cfg.image_width * cfg.yolo_width)
        x = x if x < cfg.yolo_width else cfg.yolo_width - 1
        y = int(self.centre_y / cfg.image_height * cfg.yolo_height)
        y = y if y < cfg.yolo_height else cfg.yolo_height - 1
        value[:,:,0] = 1
        if self.containsBall:
            value[y][x][1] = 1
            value[y][x][0] = 0
        return value

def lire_entrees():
    entrees = []
    with open(fichier_etiquetes_csv) as fichier:
        reader = csv.DictReader(fichier)
        for row in reader:
            containsBall = False
            if row['bContainsBall'] == 'True':
                containsBall = True
            nom = row['imgFile']
            x_center = int(row['xCenter'])
            y_center = int(row['yCenter'])
            radius = int(row['radius'])
            fichier_image = cfg.dossier_brut + nom
            image = np.fromfile(fichier_image, dtype=np.int32)
            image = np.reshape(image, (cfg.image_height, cfg.image_width, 3))
            entrees.append(Entree(nom, x_center, y_center, radius, image, containsBall))
    return entrees

def split_dataset(x, y):
    ratio = 0.9
    l = int(len(x) * ratio)
    return x[:l], y[:l], x[l:], y[l:]

def create_dataset():
    entrees = lire_entrees()
    x = [e.train_data() for e in entrees]
    y = [e.value_data() for e in entrees]
    x_train, y_train, x_validation, y_validation = split_dataset(x, y)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    return x_train, y_train, x_validation, y_validation