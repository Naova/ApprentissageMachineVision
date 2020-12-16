import json

#chemins d'acces vers le dataset
dossier_brut_tempo = '../Dataset/Brut_Robot_tempo/'
dossier_brut = '../Dataset/Brut_Robot/images/'
dossier_PNG = '../Dataset/PNG_Robot/'
json_etiquettes = '../Dataset/Brut_Simulation/images_balles_positions.json'

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
