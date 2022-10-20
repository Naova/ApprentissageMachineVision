from tensorflow.keras import Model
import tensorflow.keras as keras

from keras_flops import get_flops

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.utils import args_parser

def main():
    args = args_parser.parse_args_env_cam('Test the yolo model on a bunch of test images and output stats.')
    env = args_parser.set_config(args)
    modele = keras.models.load_model(cfg_prov.get_config().get_modele_path(env))
    modele.summary()
    
    flops = get_flops(modele, batch_size=1)
    
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

if __name__ == '__main__':
    main()
