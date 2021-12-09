import json

camera = 'lower' # doit etre dans {'upper', 'lower'}

# env in {'Simulation', 'Robot', 'Genere'}
# type_fichier in {'Brut', 'PNG'}
def get_dossier(env='Simulation', type_fichier='Brut'):
    return f'Dataset/{env}/{camera}/{type_fichier}/'
def get_labels_path(env='Simulation'):
    return f'Dataset/{env}/{camera}/labels.json'

#resolution de l'image d'entree
upper_image_height = 240
upper_image_width = 320

lower_image_height = 120
lower_image_width = 160

def get_image_resolution():
    if camera == 'upper':
        return upper_image_height, upper_image_width
    else:
        return lower_image_height, lower_image_width

upper_resized_image_height = 120
upper_resized_image_width = 160

lower_resized_image_height = 90
lower_resized_image_width = 120

def get_resized_image_resolution():
    if camera == 'upper':
        return upper_resized_image_height, upper_resized_image_width
    else:
        return lower_resized_image_height, lower_resized_image_width

#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
upper_yolo_height = 28
upper_yolo_width = 38

lower_yolo_height = 20
lower_yolo_width = 28

def get_yolo_resolution():
    if camera == 'upper':
        return upper_yolo_height, upper_yolo_width
    else:
        return lower_yolo_height, lower_yolo_width

yolo_nb_anchors_upper = 5 # + 3
yolo_nb_anchors_lower = 4 # + 3

def get_nb_anchors():
    if camera == 'upper':
        return yolo_nb_anchors_upper
    else:
        return yolo_nb_anchors_lower

__yolo_anchors = [] #rayon de la balle en pourcentage de la largeur de l'image

def get_anchors_path():
    return f'anchors_{camera}.json'

def get_anchors():
    global __yolo_anchors
    if __yolo_anchors:
        return __yolo_anchors
    else:
        with open(get_anchors_path(), 'r') as anchors_file:
            __yolo_anchors = json.loads(anchors_file.read())
        return __yolo_anchors

flipper_images = True
retrain = True
model_path_simulation = 'yolo_modele_simulation.h5'
model_path_robot = 'yolo_modele_robot.h5'

def get_modele_path(env='Simulation'):
    env = env.lower()
    if camera == 'upper':
        return f'yolo_modele_{env}_upper.h5'
    else:
        return f'yolo_modele_{env}_lower.h5'

