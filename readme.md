# Détecteur d'objets par apprentissage profond

Code Python permettant d'entraîner un réseau de neurones (suivant plus ou moins le modèle YOLO) pour faire la détection des balles et de robots.

## Installation et démarrage

Lancer l'entraînement requiert un environnement Python avec certains paquets installés. La liste se trouve dans le fichier `requirements.txt`.

`h5py, matplotlib, numpy, Pillow, tensorflow==2.5.2, tqdm, scikit-image, git+https://github.com/Naova/nncg@NaovaVision, pydot`

Exécuter `pip install -r requirements.txt`

Il est préférable d'installer ces paquets dans un environnement virtuel.

## Utilisation

Tous les scripts utilisent les mêmes trois paramètres : un pour la caméra visée (-u pour upper ou -l pour lower), un pour l'environnement du dataset (-r pour robot, -s pour simulation, -g pour les images générées par CycleGAN), et finalement, le détecteur qui sera utilisé -db pour le détecteur de balles et -dr pour le détecteur de robots.

### Entraînement

Le script `yolo/training/ball/train.py` sert à entraîner le détecteur de balles. Un script différent est utilisé pour le détecteur de robots `yolo/training/robot/train.py`, étant donné que la configuration est différente.

Pour entraîner le modèle détectant les balles sur la caméra du haut sur un robot physique :
```
python yolo/training/ball/train.py -u -r -db
```

### Déploiement

#### Génération, ajustements et déploiement du code

Le script `yolo/code_generator/generate_compile_and_move.py` utilise nncg pour générer un fichier cpp, fixer les erreurs que le générateur de code produit, puis déplacer le fichier au bon endroit dans NaovaCode.

Exemple d'utilisation pour la caméra du haut du robot:
```
python yolo/code_generator/generate_compile_and_move.py -u -r -db
```

#### Dans NaovaCode

On peut alors configurer et compiler NaovaCode.
Pour la configuration, le seul fichier à modifier à la main est `Src/Representations/Configuration/YoloModelDefinitions.h`.
Il définit un certain nombre de macros qui doivent correspondre aux valeurs utilisées lors de l'entraînement du modèle.
Donc si on entraîne un modèle avec une résolution d'entrée ou de sortie différente de la version précédente, il faudra changer les définitions correspondantes dans ce fichier. Même chose si on change le nombre d'ancres utilisées pour la classification de la taille des balles ainsi que le seuil minimal de détection accepté par le modèle.
