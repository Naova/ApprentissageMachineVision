import json

#chemins d'acces vers le dataset
dossier_brut = 'D:\\_Ecole\\Naova\\Vision\\Dataset_Brut\\'
dossier_PNG = 'D:\\_Ecole\\Naova\\Vision\\Dataset_PNG\\'
json_etiquettes = 'D:\\_Ecole\\Naova\\Vision\\images_balles_positions.json'

#resolution de l'image d'entree (elle est retournee a 90 degres)
image_height = 320
image_width = 240

#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
yolo_height = 14
yolo_width = 9

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

retrain = True
model_path_keras = 'yolo_modele.h5'
model_path_fdeep = 'yolo_modele.json'
