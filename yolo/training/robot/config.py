import json
import os

import yolo.config as cfg


def get_modele_path(env='Kaggle'):
    env = env.lower()
    return f'modele_balles_{env}_{cfg.camera}.h5'


#resolution de l'image d'entree
resolutions = {
    'Kaggle': {
        'upper': (240, 320), # a verifier
        'lower': (120, 160), # a verifier
    },
}

upper_resized_image_height = 60
upper_resized_image_width = 80

lower_resized_image_height = 30
lower_resized_image_width = 40

def get_model_input_resolution():
    if cfg.camera == 'upper':
        return upper_resized_image_height, upper_resized_image_width
    else:
        return lower_resized_image_height, lower_resized_image_width


#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
upper_yolo_height = None
upper_yolo_width = None
lower_yolo_height = None
lower_yolo_width = None

def set_model_output_resolution(height, width):
    if cfg.camera == 'upper':
        global upper_yolo_height, upper_yolo_width
        upper_yolo_height = height
        upper_yolo_width = width
    else:
        global lower_yolo_height, lower_yolo_width
        lower_yolo_height = height
        lower_yolo_width = width

def get_model_output_resolution():
    if cfg.camera == 'upper':
        return upper_yolo_height, upper_yolo_width
    else:
        return lower_yolo_height, lower_yolo_width

#a modifier
yolo_nb_anchors_upper = 3
yolo_nb_anchors_lower = 3

def get_nb_anchors():
    if cfg.camera == 'upper':
        return yolo_nb_anchors_upper
    else:
        return yolo_nb_anchors_lower

__yolo_anchors = [] #rayon de la balle en pourcentage de la largeur de l'image

#ancres a determiner
def get_anchors_path():
    return f'anchors_robots_{cfg.camera}.json'

def get_anchors():
    global __yolo_anchors
    if __yolo_anchors:
        return __yolo_anchors
    else:
        anchors_path = get_anchors_path()
        if not os.path.exists(anchors_path):
            raise Exception('Anchors file doesn\'t exist. Maybe you should run the clustering.py script before training the model.')
        with open(anchors_path, 'r') as anchors_file:
            __yolo_anchors = json.loads(anchors_file.read())
        return __yolo_anchors

def get_image_resolution(env='Simulation'):
    return resolutions[env][cfg.camera]

def get_dossier(env='Kaggle', type_fichier='YCbCr'):
    return f'Dataset/{env}/{cfg.camera}/{type_fichier}/'
def get_labels_path(env='Kaggle'):
    return f'Dataset/{env}/{cfg.camera}/labels.json'

