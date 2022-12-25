# Classificateur de ballons de soccer par apprentissage profond

Code Python permettant d'entraîner un réseau de neurones (suivant le modèle YOLO) pour faire la détection des balles.

## Installation et démarrage

Lancer l'entraînement requiert un environnement Python avec certains paquets installés. La liste se trouve dans le fichier `requirements.txt`.

`h5py, matplotlib, numpy, Pillow, tensorflow==2.5.2, tqdm, scikit-image, git+https://github.com/Naova/nncg@NaovaVision, pydot`

Exécuter `pip install -r requirements.txt`

Il est préférable d'installer ces paquets dans un environnement virtuel.

Télécharger le dataset à partir du Drive, et le dézipper dans le répertoire racine du dépot.
Vous devriez avoir ApprentissageMachineVision/Dataset/{Robot/, Simulation/, ...}

## Utilisation

Tous les scripts utilisent les mêmes trois paramètres : un pour la caméra visée (-u pour upper ou -l pour lower), un pour l'environnement du dataset (-r pour robot, -s pour simulation, -g pour les images générées par CycleGAN), et finalement, le détecteur qui sera utilisé -db pour le détecteur de balles et -dr pour le détecteur de robots.

### Entraînement

Le script `yolo/training/ball/train.py` sert à entraîner le détecteur de balles. Un script différent est utilisé pour le détecteur de robots `yolo/training/robot/train.py`, étant donné que la configuration est différente.

Pour entraîner le modèle détectant les balles sur la caméra du haut sur un robot physique :
```
python yolo/training/ball/train.py -u -r -db
```

### Déploiement

#### Génération du code

Le script `yolo/code_generator/h5_to_nncg.py` utilise nncg pour générer un fichier cpp. Encore une fois, il faut préciser quel modèle on veut selon l'environnement et la caméra.

Exemple d'utilisation pour la caméra du haut du robot:
```
python yolo/code_generator/h5_to_nncg.py -u -r -db
```

#### Ajustements du code

NNCG ne génère pas du code propre à être déployé dans NaovaCode. Non seulement ça, mais il fait aussi une erreur de compilation (oups). Le script `yolo/code_generator/fix_generated_code.py` règle ces problèmes. Il est bien important de le lancer une seule fois pour un fichier cpp. Autrement, à la seconde exécution, le script brisera le code.
Les modifications apportées au fichier cpp ne sont pas très nombreuses ni très compliquées. NNCG ne donne pas le bon type aux arguments de sa fonction; il déclare l'argument contenant le tableau de sortie comme un tableau à une dimension, mais y accède comme si c'était un tableau à trois dimensions. Le script modifie les deux arguments pour que ce soient des tableaux à une dimension. On va aussi modifier le code pour retirer des aberrations, comme des additions de 0 ou des multiplications par 1 qui se répètent un peu partout.
Éventuellement, on pourra s'arranger pour ne plus avoir de warnings quand on make le projet.

Ce script a la même interface que les autres :
```
python yolo/code_generator/fix_generated_code.py -u -r -db
```

#### Déplacer les fichiers

Pour sauver du temps à copier-coller les fichiers .cpp, on peut simplement lancer le script `yolo/code_generator/move_cpp_file.py`, qui va copier le fichier cpp sélectionné au bon endoit dans NaovaCode. Nécessite d'avoir bien indiqué où se trouve NaovaCode sur votre ordinateur dans le fichier `config.py`.

```
python yolo/code_generator/move_cpp_file.py -u -r -db
```

#### Dans NaovaCode

On peut alors configurer et compiler NaovaCode.
Pour la configuration, le seul fichier à modifier à la main est `Src/Representations/Configuration/YoloModelDefinitions.h`.
Il définit un certain nombre de macros qui doivent correspondre aux valeurs utilisées lors de l'entraînement du modèle.
Donc si on entraîne un modèle avec une résolution d'entrée ou de sortie différente de la version précédente, il faudra changer les définitions correspondantes dans ce fichier. Même chose si on change le nombre d'ancres utilisées pour la classification de la taille des balles ainsi que le seuil minimal de détection accepté par le modèle.
