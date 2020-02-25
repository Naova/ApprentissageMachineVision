
import os
from PIL import Image
import pathlib
import platform
import numpy as np
import matplotlib as plt


def get_file_paths(dossier):
    #va chercher les paths vers les images
    dossier = pathlib.Path(dossier)
    images_paths = list(dossier.glob('*/*.png'))
    image_count = len(images_paths)
    print("image count : " + str(image_count))
    return images_paths

def load_images(images_paths):
    images_neg = []
    images_pos = []
    for path in images_paths:
        img = Image.open(path)
        img = img.convert('L') #converti en noir et blanc
        img = img.resize((64, 64))
        img = np.array(img)
        
        img = np.expand_dims(img, axis=2) #garde la shape en 3D (64, 64, 1)
        if str(path).split('/')[1] == 'neg':
            images_neg.append(img)
        else:
            images_pos.append(img)
    return images_neg, images_pos

def load_image(image_path):
    path = pathlib.Path(image_path)
    img = Image.open(path)
    img = img.convert('L') #converti en noir et blanc
    img = img.resize((64, 64))
    img = np.array(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=2) #garde la shape en 3D (64, 64, 1)
    return img

def create_dataset(images_neg, images_pos):
    l_pos = len(images_pos)
    l_neg = len(images_neg)
    l_total = l_pos + l_neg
    ratio_test_validation = 0.8
    l_train = int(l_total * ratio_test_validation)
    l_validation = l_total - l_train
    mask = np.ones(l_train, dtype=bool)
    mask = np.concatenate((mask, np.zeros(l_validation, dtype=bool)))

    np.random.shuffle(mask) #selectionne aleatoirement le dataset de test et de validation

    x = [i for i in images_neg]
    x += [i for i in images_pos]
    y = [0 for i in images_neg]
    y += [1 for i in images_pos]

    x = np.array(x)
    y = np.array(y)

    x_train = x[mask]
    x_val = x[~mask]
    y_train = y[mask]
    y_val = y[~mask]

    return x_train, y_train, x_val, y_val