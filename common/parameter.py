from common.fileManager import Config

class HyperParameterManager:
    def __init__(self, config_path=None):
        if not config_path:
            config_path = "config/Hyperparameters.yaml"

        self.config = Config.load_config(config_path)
    
    def get_data(self, data_name):
        if data_name in vars(self.config):
            print(data_name)


if __name__ == "__main__":
    HyperParameterManager().get_data("K_epochs")
    