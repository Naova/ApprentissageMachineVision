
#chemins d'acces vers le dataset
dossier_brut = 'D:\\_Ecole\\Naova\\Vision\\Dataset_Brut\\'
dossier_PNG = 'D:\\_Ecole\\Naova\\Vision\\Dataset_PNG\\'
json_etiquettes = 'D:\\_Ecole\\Naova\\Vision\\images_balles_positions.json'
#dossier_etiquettes = 'D:\\_Ecole\\Naova\\Vision\\Dataset_Etiquettes\\'

#resolution de l'image d'entree (elle est retournee a 90 degres)
image_height = 320
image_width = 240

#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
yolo_height = 17
yolo_width = 12

#set la sortie du modele
yolo_categories = {
    1:'balle',
    2:'robot'
}
nb_categories = len(yolo_categories)

retrain = True
model_path = 'yolo_modele.h5'
model_path_fdeep = 'yolo_modele.json'