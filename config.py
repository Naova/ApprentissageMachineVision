
#chemins d'acces vers le dataset
dossier_brut = 'D:\\_Ecole\\Naova\\Vision\\Dataset_Brut\\'
dossier_PNG = 'D:\\_Ecole\\Naova\\Vision\\Dataset_PNG\\'
csv_etiquettes = 'D:\\_Ecole\\Naova\\Vision\\images_balles_positions.csv'

#resolution de l'image d'entree (elle est retournee a 90 degres)
image_height = 320
image_width = 240

#resolution de l'output du modele. Doit concorder avec le modele lui-meme. (voir la derniere couche du summary)
yolo_height = 18
yolo_width = 13
#set la sortie du modele
yolo_categories = 2 #rien = 0, balle = 1