import json

#chemins d'acces vers le dataset (robot)
dataset_robot_root = 'Dataset/Robot/'
dossier_brut_robot = dataset_robot_root + 'Brut/'
dossier_PNG_robot = dataset_robot_root + 'PNG/'
dossier_tempo_robot = dataset_robot_root + 'tempo/'
labels_robot = dataset_robot_root + 'labels.json'

#chemins d'acces vers le dataset (simulation)
dataset_simulation_root = 'Dataset/Simulation/'
dossier_brut_simulation = dataset_simulation_root + 'Brut/'
dossier_PNG_simulation = dataset_simulation_root + 'PNG/'
dossier_tempo_simulation = dataset_simulation_root + 'tempo/'
labels_simulation = dataset_simulation_root + 'labels.json'

#resolution de l'image d'entree (elle est retournee a 90 degres)
image_height = 240
image_width = 320

#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
yolo_height = 9
yolo_width = 14

yolo_nb_anchors = 6

__yolo_anchors = [] #rayon de la balle en pourcentage de la largeur de l'image
yolo_anchors_path = 'anchors.json'

def get_anchors():
    global __yolo_anchors
    if __yolo_anchors:
        return __yolo_anchors
    else:
        with open(yolo_anchors_path, 'r') as anchors_file:
            __yolo_anchors = json.loads(anchors_file.read())
        return __yolo_anchors

flipper_images = True
retrain = True
model_path_keras = 'yolo_modele.h5'
model_path_fdeep = 'yolo_modele.json'
