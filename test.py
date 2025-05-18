from common.fileManager import Config
import pandas as pd
import os

# 현재 디렉토리 위치 얻기
current_directory = os.getcwd()
print(current_directory)
stock_config = Config.load_config("config/StockConfig.yaml")
path = str(stock_config.stock_code_path.value) + "/" + " 005930" + ".csv"
pd.read_csv(path, header=0, index_col=0)