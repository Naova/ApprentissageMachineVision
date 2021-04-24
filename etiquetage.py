from tkinter import *
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import sys
import config as cfg
import os

class Entree:
    def __init__(self, image_nom:str):
        self.image_nom = image_nom
        self.reset_label()
        self.label_nom = image_nom + '_label'

    def append_label(self, left, right, top, bottom, categorie):
        d = {'left':left,
             'right':right,
             'top':top,
             'bottom':bottom,
             'categorie':categorie}
        self.json_labels.append(d)

    def reset_label(self):
        self.json_labels = []
    
    def as_json_dict(self):
        return self.json_labels

    def __repr__(self):
        return self.image_nom

class ClickInfo:
    def __init__(self, x:int, y:int):
        self.x_init = x
        self.y_init = y

#retourne une liste de PIL.Image a partir d'une liste de chemins d'acces
def get_actual_images(fichiers):
    with open(cfg.labels_simulation, 'r') as f:
        etiquettes = json.load(f)
    images = []
    for fichier_image in fichiers:
        f = np.fromfile(str(fichier_image), dtype=np.float32)
        f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
        image = Image.fromarray((f*255).astype('uint8'))
        image_nom = str(fichier_image).replace('\\', '/').split('/')[-1]
        images.append((image_nom, ImageTk.PhotoImage(image=image), etiquettes[image_nom] if image_nom in etiquettes else None))
    return images

entrees = []
entree_courante = None
index_courant = -1
images = []
categorie_courante = None
mode = 'verification' #ou 'etiquetage'

current_click = None

#avance d'une image
def afficher_prochaine_image():
    global index_courant, entree_courante
    index_courant += 1
    #si on arrive a la fin du dataset, sauvegarde et quitte
    if index_courant == len(images):
        window.destroy()
        return
    canvas.create_image(0, 0, anchor=NW, image=images[index_courant][1])
    if mode == 'verification':
        if images[index_courant][2]:
            labels = images[index_courant][2]
            for label in labels:
                canvas.create_rectangle(label['left'], label['top'], label['right'], label['bottom'])
    entree_courante = Entree(images[index_courant][0])
    if mode == 'verification':
        entree_courante.json_labels = images[index_courant][2]
    print(images[index_courant][0])

def set_categorie(categorie:int):
    global categorie_courante
    categorie_courante = categorie
    print('categorie ' + str(categorie) + ' selectionnee')

def key_pressed_callback(event):
    try:
        numero = int(event.char)
        set_categorie(numero)
    except:
        print('touche ' + event.char + ' pas prise en compte!')

#revient une image en arriere
def previous_image(event=None):
    global index_courant
    entrees.pop(-1)
    index_courant -= 1
    canvas.create_image(0, 0, anchor=NW, image=images[index_courant][1])
    if mode == 'verification':
        if images[index_courant][2]:
            labels = images[index_courant][2]
            for label in labels:
                canvas.create_rectangle(label['left'], label['top'], label['right'], label['bottom'])

#prend en note le centre de l'image, la bordure et calcule le rayon, et passe a l'image suivante
def mouse_pressed_callback(event):
    print('pressed')
    global current_click
    current_click = ClickInfo(event.x, event.y)

def mouse_released_callback(event):
    print('released')
    global current_click
    left = current_click.x_init
    top = current_click.y_init
    current_click = None
    right = event.x
    bottom = event.y
    if left > right:
        swap = right
        right = left
        left = swap
    if top > bottom:
        swap = top
        top = bottom
        bottom = swap
    if top < 0:
        top = 0
    if bottom >= cfg.image_height:
        bottom = cfg.image_height - 1
    if left < 0:
        left = 0
    if right >= cfg.image_width:
        right = cfg.image_width - 1
    #dessine un rectange. A modifier pour une strategie de dessin (ajouter d'autres formes)
    entree_courante.append_label(left, right, top, bottom, categorie_courante)
    canvas.create_rectangle(left, top, right, bottom)


def mouse_move(event):
    if current_click is not None:
        pass

#Avance a la prochaine image
def next_image(event=None):
    global entree_courante
    global label_courant
    entrees.append(entree_courante)
    label_courant = None
    save_label()
    entree_courante = None
    afficher_prochaine_image()

#sauvegarde le label courant
def save_label(event=None):
    previous_labels = {}
    try:
        with open(cfg.labels_simulation) as input_file:
            previous_labels = json.loads(input_file.read())
    except:
        breakpoint()
    with open(cfg.labels_simulation, 'w') as output_file:
        previous_labels[entree_courante.image_nom] = entree_courante.as_json_dict()
        output_file.write(json.dumps(previous_labels))
    print('label saved : ' + entree_courante.label_nom)

def delete_current_labels(event=None):
    canvas.create_image(0, 0, anchor=NW, image=images[index_courant][1])
    entree_courante.reset_label()

#affiche les commandes dans la console
def helpe(event=None):
    print('''Liste des commandes :

    #controle
    'd', next_image
    's', save_label
    'h', help
    'a', previous_image
    '<Number>', set categorie

    #dessin
    '<ButtonPress>', commence a dessiner
    '<ButtonRelease>', termine de dessiner
    '<Motion>', dessine

    #deboguage
    'p', pause
''')

#renvoie a la console dans le debogueur
def pause(event=None):
    breakpoint()

def delete_image(event=None):
    global entree_courante, label_courant, index_courant
    os.remove(cfg.dossier_brut_simulation + entree_courante.image_nom)
    previous_labels = {}
    try:
        with open(cfg.labels_simulation) as input_file:
            previous_labels = json.loads(input_file.read())
    except:
        breakpoint()
    if entree_courante.image_nom in previous_labels:
        with open(cfg.labels_simulation, 'w') as output_file:
            del previous_labels[entree_courante.image_nom]
            output_file.write(json.dumps(previous_labels))
    print('image and label deleted : ' + entree_courante.label_nom)
    label_courant = None
    entree_courante = None
    index_courant -= 1
    afficher_prochaine_image()
    

window = Tk()
canvas = Canvas(window, width = cfg.image_width, height = cfg.image_height)
canvas.focus_set()
canvas.pack()
canvas.bind('<Key>', key_pressed_callback)
canvas.bind('a', previous_image)
canvas.bind('<ButtonPress-1>', mouse_pressed_callback)
canvas.bind('<ButtonRelease-1>', mouse_released_callback)
canvas.bind('d', next_image)
canvas.bind('s', save_label)
canvas.bind('x', delete_current_labels)
canvas.bind('h', helpe)
canvas.bind('p', pause)
canvas.bind('<Motion>', mouse_move)
canvas.bind('f', delete_image)

def main():
    global images
    global index_courant

    set_categorie(1)

    dossier_images = Path(cfg.dossier_brut_simulation).glob('**/*')
    #charge le fichier s'il est deja rempli
    with open(cfg.labels_simulation, 'r') as f:
        etiquettes = json.load(f)
        etiquettes = list(etiquettes.keys())
    
    if mode == 'verification':
        images_paths = [x for x in dossier_images if x.is_file() and 'batch_10' in str(x)]
    else:
        images_paths = [x for x in dossier_images if x.is_file() and str(x).split('\\')[-1] not in etiquettes]
    images = get_actual_images(images_paths)

    afficher_prochaine_image()

    window.mainloop()


if __name__ == '__main__':
    main()