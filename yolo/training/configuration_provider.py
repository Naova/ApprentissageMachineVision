import yolo.training.ball.config as cfg_balles
import yolo.training.robot.config as cfg_robots

class ConfigurationProvider:
    class __ConfigurationProvider:
        def __init__(self, detector:str):
            if detector == 'balles':
                self.config = cfg_balles
            else:
                self.config = cfg_robots
            self.config.detector = detector
    
    instance = None
    
    @staticmethod
    def set_config(detector:str):
        ConfigurationProvider.instance = ConfigurationProvider.__ConfigurationProvider(detector)
    
    @staticmethod
    def get_config():
        return ConfigurationProvider.instance.config
