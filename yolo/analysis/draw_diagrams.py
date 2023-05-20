from tensorflow.keras.layers import LeakyReLU
import visualkeras

from yolo.utils.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.ball.train import load_model

def main():
    modele = load_model(env='Robot')
    camera = cfg_prov.get_config().camera
    visualkeras.layered_view(modele, legend=True, type_ignore=[LeakyReLU], scale_xy=1, scale_z=1,max_z=1000).save(f'diagramme_{camera}.png')


if __name__ == '__main__':
    cfg_prov.set_config('balles')
    cfg_prov.get_config().camera = 'upper'
    main()
    cfg_prov.get_config().camera = 'lower'
    main()
