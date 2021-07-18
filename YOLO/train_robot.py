from train import train
from Dataset_Loader import create_dataset

import sys
sys.path.insert(0,'..')
import config as cfg

cfg.retrain = True

def main():
    train_generator, validation_generator, test_generator = create_dataset(16, '../'+cfg.labels_simulation, '../'+cfg.dossier_brut_genere)
    train(train_generator, validation_generator, test_generator, cfg.model_path_robot)

if __name__ == '__main__':
    main()
