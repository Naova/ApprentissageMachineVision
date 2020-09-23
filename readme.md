# Classificateur de ballons de soccer par apprentissage profond

Code Python permettant d'étiqueter des images de balles et d'entraîner un réseau de neurones (suivant le modèle YOLO) pour en faire la détection.

## Installation et démarrage

Lancer l'entraînement requiert un environnement Python avec certains paquets installés. La liste se trouve dans le fichier `requirements.txt`.

`h5py, matplotlib, numpy, Pillow, tensorboard==2.1.0, tensorflow==2.1.0, tqdm`

Il est recommandé d'installer ces paquets dans un environnement virtuel.

## Utilisation

### Étape 1
On génère des images avec la branche de NaovaCode "Creation_Dataset".

Il y a quelques endroits utiles pour changer des paramètres (au besoin):

`Src\Modules\Perception\DatasetCreator.h` Incrémenter la variable 'batch_number' à chaque fois qu'on génère une nouvelle part de dataset pour éviter les conflits avec les noms de fichiers et pour que tout reste bien clair. La variable 'saveImageMaxCountdown' sert à modifier la fréquence à laquelle on sauvegarde les images. Plus elle est petite, et plus les images seront enregistrées vites, et plus elles seront semblables les unes les autres, et plus la simulation sera lente.

`Src\Representations\Infrastructure\Image.h`. Les variables 'maxKerasResolutionWidth' et 'maxKerasResolutionHeight' servent à déterminer la résolution de l'image de sortie (on fait un rescale de l'image pour réduire la charge de calculs).

### Étape 2

Du dépôt git 'ApprentissageMachineVision', exécuter le script `brut_to_png.py`. Ce script génèrera les images PNG à partir des données provenant du Nao. Ça n'est techniquement pas obligatoire, mais permet de visualiser les données.

`etiquetage.py`, qui lance le programme pour étiqueter les images.

Tous ces script utilisent le fichier `config.py` pour connaître les chemins vers les dossiers. À modifier pour vos chemins d'accès.

### Étape 3

Une fois que le programme d'étiquetage est lancé, une fenêtre apparaît à l'écran avec une image provenant de la caméra du nao.

Si l'image ne contient pas de balle : passer à l'image suivante en appuyant sur la touche 'd'.

Si l'image contient une balle : faire un clic gauche de la souris sur un coin de la balle, puis glisser la souris jusqu'au coin opposé, puis relâcher le bouton. Cela permet de calculer à la fois la position et le rayon de la balle. Passer à l'image suivante en appuyant sur la touche 'd'.

Pour revenir en arrière d'une image, appuyer sur la touche 'a'.

La progression est sauvegardée à chaque changement d'image. Faire un retour en arrière efface l'étiquette pour l'image précédente.

### Étape 4

Lancer le script `clustering.py`. Cela créera un fichier `anchors.json` basé sur les étiquettes du dataset pour sélectionner les rayons de balles optimaux.

### Étape 5

Lancer le script `train.py`. Cela utilisera les fichiers étiquetés et les ancres générées précédemment pour performer un entraînement d'un modèle et le sauvegardera dans un fichier HDF, puis le convertira en json, au format lisible par Frugally Deep, la librairie utilisée par les NAOs pour exécuter les réseaux de neurones.
