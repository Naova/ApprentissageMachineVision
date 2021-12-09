from train import train
from Dataset_Loader import create_dataset

import sys
sys.path.insert(0,'..')
import config as cfg

cfg.retrain = True

def main():
    env = 'Simulation'
    labels = cfg.get_labels_path(env)
    dossier_brut = cfg.get_dossier(env, 'Brut')
    train_generator, validation_generator, test_generator = create_dataset(16, '../'+labels, '../'+dossier_brut)
    train(train_generator, validation_generator, test_generator, cfg.model_path_simulation)

if __name__ == '__main__':
    main()
