import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
from PIL import ImageFont
import visualkeras

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

def main():
    modele = keras.models.load_model(cfg_prov.get_config().get_modele_path('Robot'))
    font = ImageFont.truetype("arial.ttf", 14)
    visualkeras.layered_view(modele, legend=True, font=font, type_ignore=[LeakyReLU], scale_xy=1, scale_z=1,max_z=1000).show()


if __name__ == '__main__':
    cfg_prov.get_config().camera = 'upper'
    main()
    cfg_prov.get_config().camera = 'lower'
    main()
