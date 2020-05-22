import os
from pathlib import Path
import re
import config as cfg

"""
ajoute du zero padding aux numeros de batch et d'images pour conserver l'ordre lorsqu'on itere a travers le dataset
"""
def main():
    dossier_entree = Path(cfg.dossier_brut).glob('**/*')
    fichiers = [str(x) for x in dossier_entree if x.is_file()]

    for f in fichiers:
        index_batch_n = f.find('batch_')
        if index_batch_n != -1:
            index_batch_n += 6
        else:
            print('f' + f)
            continue
        index_image_n = f.find('image_')
        if index_image_n != -1:
            i = f'{int(f[index_batch_n:index_image_n - 1]):02}'
            index_image_n += 6
        else:
            print('f' + f)
            continue
        j = f'{int(f[index_image_n:]):04}'
        f2 = f[:index_batch_n] + i + f[index_batch_n + len(i):index_image_n] + j
        os.rename(f, f2)

if __name__ == '__main__':
    main()