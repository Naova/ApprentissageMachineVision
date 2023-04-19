import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
from PIL import ImageFont
import visualkeras

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov
from yolo.training.ball.train import custom_activation, custom_loss

def main():
    modele = keras.models.load_model(cfg_prov.get_config().get_modele_path('Robot'), custom_objects={'custom_loss':custom_loss, 'custom_activation':custom_activation})
    #font = ImageFont.truetype("arial.ttf", 14)
    camera = cfg_prov.get_config().camera
    visualkeras.layered_view(modele, legend=True, type_ignore=[LeakyReLU], scale_xy=1, scale_z=1,max_z=1000).save(f'diagramme_{camera}.png')


if __name__ == '__main__':
    cfg_prov.set_config('balles')
    cfg_prov.get_config().camera = 'upper'
    main()
    cfg_prov.get_config().camera = 'lower'
    main()
