from tkinter import *
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from skimage.draw import polygon
import math
import numpy as np
import json
import sys
sys.path.insert(0,'..')
import config as cfg

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
        self.label = np.zeros((cfg.image_height, cfg.image_width, cfg.nb_categories))
        self.label[:,:,0] = 1.0
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
    images = []
    for fichier_image in fichiers:
        f = np.fromfile(str(fichier_image), dtype=np.int32)
        f = np.reshape(f, (cfg.image_height,cfg.image_width,3))
        image = Image.fromarray(f.astype('uint8'))
        #image = Image.open(str(fichier_image))
        images.append((str(fichier_image).split('\\')[-1], ImageTk.PhotoImage(image=image)))
    return images

entrees = []
entree_courante = None
index_courant = -1
images = []
categorie_courante = None
current_tool = 'rectangle'
current_polygon = ([], [])

input_directory = cfg.dossier_brut
output_directory = cfg.dossier_etiquettes

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
    entree_courante = Entree(images[index_courant][0])
    print(images[index_courant][0])

def set_categorie(categorie:int):
    global categorie_courante
    categorie_courante = categorie
    print(str(categorie) + ' : ' + cfg.yolo_categories[categorie] + ' selectionne')

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

#prend en note le centre de l'image, la bordure et calcule le rayon, et passe a l'image suivante
def mouse_pressed_callback(event):
    print('pressed')
    global current_click
    current_click = ClickInfo(event.x, event.y)

def mouse_released_callback(event):
    print('released')
    global current_click, current_tool
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
    if current_tool == 'rectangle':
        entree_courante.label[top:bottom, left:right, :] = 0.0
        entree_courante.label[top:bottom, left:right, categorie_courante] = 1.0
        entree_courante.append_label(left, right, top, bottom, categorie_courante)
        canvas.create_rectangle(left, top, right, bottom)
    if current_tool == 'polygon':
        current_polygon[0].append(event.y)
        current_polygon[1].append(event.x)


def mouse_move(event):
    if current_click is not None:
        pass

def change_tool(event=None):
    global current_tool, current_polygon
    if current_tool == 'rectangle':
        current_tool = 'polygon'
        current_polygon = ([], [])
    elif current_tool == 'polygon':
        current_tool = 'rectangle'
    print('switch to ' + current_tool)

def display_label(event=None):
    plt.imshow(entree_courante.label[:,:,categorie_courante])
    plt.show()

#Avance a la prochaine image
def next_image(event=None):
    global entree_courante
    global label_courant
    entrees.append(entree_courante)
    label_courant = None
    save_label()
    entree_courante = None
    afficher_prochaine_image()

def confirm_polygon(event=None):
    global current_polygon
    rr, cc = polygon(current_polygon[0], current_polygon[1])
    current_polygon = ([], [])
    entree_courante.label[rr, cc] = 0.0
    entree_courante.label[rr, cc, categorie_courante] = 1.0

#sauvegarde le label courant
def save_label(event=None):
    np.save(cfg.dossier_etiquettes + entree_courante.label_nom, entree_courante.label)
    previous_labels = {}
    try:
        with open(cfg.json_etiquettes) as input_file:
            previous_labels = json.loads(input_file.read())
    except:
        pass
    with open(cfg.json_etiquettes, 'w') as output_file:
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
    '<Button-3>', change tool

    #dessin
    '<ButtonPress>', commence a dessiner
    '<ButtonRelease>', termine de dessiner
    '<Motion>', dessine

    #deboguage
    'w', display_label
    'p', pause
''')

#renvoie a la console dans le debogueur
def pause(event=None):
    breakpoint()

window = Tk()
canvas = Canvas(window, width = cfg.image_width, height = cfg.image_height)
canvas.focus_set()
canvas.pack()
canvas.bind('<Key>', key_pressed_callback)
canvas.bind('a', previous_image)
canvas.bind('<ButtonPress-1>', mouse_pressed_callback)
canvas.bind('<ButtonRelease-1>', mouse_released_callback)
canvas.bind('<Button-3>', change_tool)
canvas.bind('w', display_label)
canvas.bind('d', next_image)
canvas.bind('s', save_label)
canvas.bind('e', confirm_polygon)
canvas.bind('x', delete_current_labels)
canvas.bind('h', helpe)
canvas.bind('p', pause)
canvas.bind('<Motion>', mouse_move)

def main():
    global images
    global index_courant

    set_categorie(1)

    dossier_images = Path(input_directory).glob('**/*')
    #charge le fichier s'il est deja rempli
    fichiers_deja_etiquetes = []
    etiquettes = [str(p).split('\\')[-1].split('.')[0][:-6] for p in Path(cfg.dossier_etiquettes).glob('**/*')]
    
    images_paths = [x for x in dossier_images if x.is_file() and str(x).split('\\')[-1] not in etiquettes]
    images = get_actual_images(images_paths)

    afficher_prochaine_image()

    window.mainloop()


if __name__ == '__main__':
    main()