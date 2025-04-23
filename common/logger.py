from common.fileManager import File
import os

class Logger:
    def __init__(self, root_dir=""):
        self.root_dir = root_dir
        self.file_dict = {}

    def add_file(self, key, dir_path, file_name):
        self.file_dict[key] = File(self.root_dir + dir_path, file_name)
    
    def get_file_path(self, key):
        return self.file_dict[key].get_file_path()

    def write_file(self, key, data:str):
        self.file_dict[key].write_flush(data)

    def list_write_file(self, key, data_list:list):
        str_data = [str(item) for item in data_list]
        data = ','.join(str_data) + '\n'
        self.file_dict[key].write_flush(data)

    def print_wirte_file(self, key, data:str):
        print(data)
        self.file_dict[key].write_append(data)

    def close_all(self):
        for key in self.file_dict.keys():
            self.file_dict[key].close()
    
    def is_exists(self, key):
        return key in self.file_dict