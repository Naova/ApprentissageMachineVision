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
    ratio_train = 0.88
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
    y = [e.value_data() for e in entrees]
    x_train, y_train, x_validation, y_validation, x_test, y_test = split_dataset(x, y)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_validation, y_validation, x_test, y_test