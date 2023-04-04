from pathlib import Path

import tensorflow.keras as keras

from yolo.training.test_robot_model import make_predictions, save_stats
import yolo.utils.args_parser as args_parser
from yolo.training.dataset_loader import load_test_set
from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.ball.train import custom_activation

from focal_loss import BinaryFocalLoss
import os

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)

    test_data_negative, test_data_positive = load_test_set()

    fichiers = Path('modeles/').glob(f'{cfg_prov.get_config().get_modele_path(env).split(".")[0]}_*.h5')

    fichiers = sorted([f.as_posix() for f in fichiers])

    for fichier in fichiers[-20:]:
        print(f'\n\ntest du modele : {fichier}\n')
        modele = keras.models.load_model(fichier, custom_objects={'loss':BinaryFocalLoss, 'custom_activation':custom_activation})
        modele.summary()
        cfg_prov.get_config().set_model_output_resolution(modele.output_shape[1], modele.output_shape[2])

        max_confidences_negative = make_predictions(modele, test_data_negative)
        max_confidences_positive = make_predictions(modele, test_data_positive)
        ious = [m[-1] for m in max_confidences_positive if m[-1] is not None]
        iou = 100 * sum(ious) / len(ious)

        save_stats(max_confidences_negative, max_confidences_positive, fichier, iou)

    fichiers = Path('modeles/').glob('*')
    for fichier in fichiers:
        os.remove(fichier.as_posix())


if __name__ == '__main__':
    main()
