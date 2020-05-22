from tkinter import *
from pathlib import Path
from PIL import Image, ImageTk
import math
import numpy as np
import csv
import ast
import sys
sys.path.insert(0,'..')
import config as cfg

class Entree:
    def __init__(self, image_nom:str, x:int, y:int, balle_presente:bool, radius = -1):
        self.image_nom = image_nom
        self.center_x = x
        self.center_y = y
        self.balle_presente = balle_presente
        self.radius = radius

    def to_csv_line(self):
        return str(self.center_x) + "," + str(self.center_y) + "," + str(self.radius) + "," + self.image_nom

    def as_csv_dict(self):
        d = {}
        d['xCenter'] = self.center_x
        d['yCenter'] = self.center_y
        d['radius'] = self.radius
        d['imgFile'] = self.image_nom
        d['bContainsBall'] = self.balle_presente
        return d

    def __repr__(self):
        return self.image_nom + " " + str(self.center_x) + " " + str(self.center_y) + " " + str(self.radius) + " " + str(self.balle_presente)

#retourne une liste de PIL.Image a partir d'une liste de chemins d'acces
def get_actual_images(fichiers):
    images = []
    for fichier_image in fichiers:
        image = Image.open(str(fichier_image))
        images.append((str(fichier_image).split('\\')[-1], ImageTk.PhotoImage(image=image)))
    return images

entrees = []
entree_courante = None
index_courant = -1
images = []

input_directory = cfg.dossier_PNG
csv_file = cfg.csv_etiquettes

#avance d'une image
def afficher_prochaine_image():
    global index_courant
    index_courant += 1
    #si on arrive a la fin du dataset, sauvegarde et quitte
    if index_courant == len(images):
        save_to_csv()
        window.destroy()
        return
    canvas.create_image(0, 0, anchor=NW, image=images[index_courant][1])
    print(images[index_courant][0])

#revient une image en arriere
def left_arrow_callback(event):
    global index_courant
    entrees.pop(-1)
    index_courant -= 1
    canvas.create_image(0, 0, anchor=NW, image=images[index_courant][1])

#prend en note le centre de l'image, la bordure et calcule le rayon, et passe a l'image suivante
def left_click_callback(event):
    global entree_courante
    x = event.x
    y = event.y
    if entree_courante is None:
        entree_courante = Entree(images[index_courant][0], x, y, True)
    else:
        entree_courante.radius = int(math.sqrt(math.pow(x - entree_courante.center_x, 2) + math.pow(y - entree_courante.center_y, 2)))
        entrees.append(entree_courante)
        entree_courante = None

        afficher_prochaine_image()

#l'image n'a pas de balle, passe a la suivante
def right_click_callback(event):
    global entree_courante
    entrees.append(Entree(images[index_courant][0], -1, -1, False))
    entree_courante = None
    afficher_prochaine_image()


def save_to_csv(event=None):
    with open(csv_file, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, ['xCenter','yCenter','radius','imgFile','bContainsBall'])
        writer.writeheader()
        for e in entrees:
            writer.writerow(e.as_csv_dict())

window = Tk()
canvas = Canvas(window, width = 240, height = 320)
canvas.focus_set()
canvas.pack()
canvas.bind("a", left_arrow_callback)
canvas.bind("<Button-1>", left_click_callback)
canvas.bind("<Button-3>", right_click_callback)
canvas.bind("s", save_to_csv)

def main():
    global images
    global index_courant
    dossier_images = Path(input_directory).glob('**/*')
    #charge le fichier s'il est deja rempli
    with open(csv_file, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            entrees.append(Entree(line['imgFile'], int(line['xCenter']), int(line['yCenter']), ast.literal_eval(line['bContainsBall']), int(line['radius'])))
    images_paths = [x for x in dossier_images if x.is_file()]
    images = get_actual_images(images_paths)
    index_courant = len(entrees) - 1

    afficher_prochaine_image()

    window.mainloop()


if __name__ == '__main__':
    main()