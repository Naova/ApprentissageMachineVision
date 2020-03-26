import numpy as np
from PIL import Image
from pathlib import Path

dossier_entree = Path("..\\Dataset_Brut\\")
dossier_sortie = "..\\Dataset_PNG\\"
fichiers = [str(x) for x in dossier_entree.glob('**/*') if x.is_file()]

for fichier in fichiers:
    f = np.fromfile(fichier, dtype=np.int32)
    f = np.reshape(f, (320,240,3))
    image = Image.fromarray(f.astype('uint8'))
    image.save(dossier_sortie + fichier.split('\\')[-1] + ".png", "png")

#version en une ligne parce que je m'ennuie :
#[Image.fromarray(np.reshape(np.fromfile(str(x), dtype=np.int32), (320,240,3)).astype('uint8')).save("..\\Dataset_PNG\\" + str(x).split('\\')[-1] + ".png", "png") for x in Path("..\\Dataset_Brut\\").glob('**/*') if x.is_file()]