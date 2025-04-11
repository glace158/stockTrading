import os
import yaml
from typing import Dict, Any

from types import SimpleNamespace

class File:
    def __init__(self, dir_path, filename):
        self.dir_path = dir_path

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.filename = self.dir_path + filename
        self.file = open(self.filename,"w+")

    def get_file_path(self):
        return self.filename
    
    def write_flush(self, write_data):
        self.file.write(write_data)
        self.file.flush()
    
    def write(self, write_data):
        self.file.write(write_data)

    def write_append(self, write_data):
        with open(self.filename, "a", encoding="utf-8") as file:
            file.write(write_data)

    def close(self):
        self.file.close()
    
    def __del__(self):
        self.close()

class Config:
    """Configuration class to load, save and update configuration"""

    @staticmethod
    def _convert_dict_to_obj(config_dict: Dict[str, Any]) -> SimpleNamespace:
        """Convert dictionary to an object with dot notation access"""
        namespace = SimpleNamespace()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(namespace, key, Config._convert_dict_to_obj(value))    # Recursively convert nested dicts
            else:
                setattr(namespace, key, value)
        return namespace

    @staticmethod
    def _convert_obj_to_dict(ns: SimpleNamespace) -> Dict:
        """Convert a Config object back to a dictionary"""
        output = {}
        for key, value in ns.__dict__.items():
            if isinstance(value, SimpleNamespace):
                output[key] = Config._convert_obj_to_dict(value)
            else:
                output[key] = value
        return output

    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return Config._convert_dict_to_obj(config_dict)

    @staticmethod
    def save_config(config, save_path: str) -> None:
        dir_path = os.path.dirname(save_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        """Save configuration to YAML file"""
        config_dict = Config._convert_obj_to_dict(config)
        if config_dict:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f,allow_unicode=True)

    @staticmethod
    def update_config(config: SimpleNamespace, updates: Dict) -> SimpleNamespace:
        """Update configuration with new parameters"""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like 'training.learning_rate'
                keys = key.split('.')
                conf = config
                for k in keys[:-1]:
                    if not hasattr(conf, k):
                        setattr(conf, k, SimpleNamespace())
                    conf = getattr(conf, k)
                setattr(conf, keys[-1], value)
            else:
                setattr(config, key, value)
        return config

if __name__ == "__main__":

    config = Config.load_config(os.path.dirname(__file__) +"/"+ "config_cartpole.yaml")
    print(config)
    print(config.env_name)