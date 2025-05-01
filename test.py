from common.fileManager import Config
import random
import math

class HyperParameterManager:
    def __init__(self, config_path=None):
        if not config_path:
            config_path = "config/Hyperparameters.yaml"

        self.config = Config.load_config(config_path)
    
    def get_data(self, data_name):
        if data_name in vars(self.config):
            print(self._get_random(vars(self.config)[data_name]))

    def _get_random(self, data):
        if '.' in data.value:
            return random.uniform(float(data.min), float(data.max))
        else:
            return random.randint(int(data.min), int(data.max))


class StockParameterManager:
    def __init__(self, config_path=None):
        if not config_path:
            config_path = "config/StockConfig.yaml"
    