import yolo.training.ball.config as cfg_balles
import yolo.training.robot.config as cfg_robots
try:
    import yolo.config as cfg_global
    naovaCodePath = cfg_global.naovaCodePath
except ModuleNotFoundError:
    with open('yolo/config.py', 'w') as f:
        f.write("naovaCodePath = '../NaovaCode'")
    print("yolo/config.py a ete cree!")
    print("'../NaovaCode' est le chemin d'acces par defaut pour NaovaCode. Il peut etre change dans yolo/config.py.")
    naovaCodePath = '../NaovaCode'
    

class ConfigurationProvider:
    class __ConfigurationProvider:
        def __init__(self, detector:str):
            if detector == 'balles':
                self.config = cfg_balles
            else:
                self.config = cfg_robots
            self.config.detector = detector
            self.config.naovaCodePath = naovaCodePath
    
    instance = None
    
    @staticmethod
    def set_config(detector:str):
        ConfigurationProvider.instance = ConfigurationProvider.__ConfigurationProvider(detector)
    
    @staticmethod
    def get_config():
        return ConfigurationProvider.instance.config
