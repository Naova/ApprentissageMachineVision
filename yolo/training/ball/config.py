import json
import os

camera = ''

def get_modele_path(env='Simulation'):
    env = env.lower()
    if env == 'genere':
        env = 'robot'
    return f'modele_balles_{env}_{camera}.h5'


#resolution de l'image d'entree
resolutions = {
    'Simulation': {
        'upper': (240, 320),
        'lower': (120, 160),
    },
    'Kaggle': {
        'upper': (480, 640),
        'lower': (120, 160),
    },
    'Genere': {
        'upper': (128, 144),
        'lower': (128, 144),
    },
    'Robot': {
        'upper': (240, 320),
        'lower': (120, 160),
    },
    'HardNegative': {
        'upper': (240, 320),
        'lower': (120, 160),
    },
    'NewNegative': {
        'upper': (240, 320),
        'lower': (120, 160),
    },
    'RobotSansBalle': {
        'upper': (240, 320),
        'lower': (120, 160),
    },
}

upper_resized_image_height = 120
upper_resized_image_width = 160

lower_resized_image_height = 75
lower_resized_image_width = 100

cycle_gan_image_height = 128
cycle_gan_image_width = 144

def get_model_input_resolution():
    if camera == 'upper':
        return upper_resized_image_height, upper_resized_image_width
    else:
        return lower_resized_image_height, lower_resized_image_width


#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
upper_yolo_height = None
upper_yolo_width = None
lower_yolo_height = None
lower_yolo_width = None

def set_model_output_resolution(height, width):
    if camera == 'upper':
        global upper_yolo_height, upper_yolo_width
        upper_yolo_height = height
        upper_yolo_width = width
    else:
        global lower_yolo_height, lower_yolo_width
        lower_yolo_height = height
        lower_yolo_width = width

def get_model_output_resolution():
    if camera == 'upper':
        return upper_yolo_height, upper_yolo_width
    else:
        return lower_yolo_height, lower_yolo_width

yolo_nb_anchors_upper = 4
yolo_nb_anchors_lower = 4

def get_nb_anchors():
    if camera == 'upper':
        return yolo_nb_anchors_upper
    else:
        return yolo_nb_anchors_lower

__yolo_anchors = [] #rayon de la balle en pourcentage de la largeur de l'image

def get_anchors_path():
    return f'anchors_balles_{camera}.json'

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
    return resolutions[env][camera]


"""
env dans {'Simulation', 'Robot', 'Genere'}
type_fichier dans {'RGB', 'YCbCr'}
"""
def get_dossier(env='Simulation', type_fichier='YCbCr'):
    return f'Dataset/{env}/{camera}/{type_fichier}/'
def get_labels_path(env='Simulation'):
    return f'Dataset/{env}/{camera}/labels.json'

