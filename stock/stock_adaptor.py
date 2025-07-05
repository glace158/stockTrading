import abc
import pandas as pd
import numpy as np
import os
import random
from common.fileManager import Config

class Adaptor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_info(self):
        pass

    @abc.abstractmethod
    def get_info_len(self):
        pass

class DailyStockAdaptor(Adaptor):
    def __init__(self, data_filter, path):
        super().__init__()
        self.data_filter = data_filter
        self.path = path 

        self.index = 0
        self.extra_index = 0

        self.silce_datas = None

    def load_datas(self, stock_code, count, extra_count = 0): # 데이터 읽어오기
        """
            저장된 주식데이터를 count만큼 가지고 옵니다.
        """
        self.stock_code = stock_code
        self.index = extra_count
        self.extra_index = extra_count
        count += 1
        
        self.stock_datas = pd.read_csv(self.path + "/" + stock_code, header=0, index_col=0)
        
        maxindex = len(self.stock_datas) - count - extra_count# 최대 인덱스 설정
        
        silce_index = random.randint(0, maxindex) # 인덱스 추출
        silce_index = 0

        self.silce_datas = self.stock_datas.iloc[silce_index : silce_index + count + extra_count] # 슬라이싱
        
        self.silce_datas.index = np.arange(len(self.silce_datas.index)) # 인덱스 재정렬
        
        self.filtering_datas = self.silce_datas[list(self.data_filter)]# 데이터 필터링

        return self.filtering_datas

    # 주식 데이터 하나씩 가져오기
    def get_info(self):
        """
            load_datas()로 불러온 주식데이터를 하나씩 반환합니다.
        """
        self.price = float(self.filtering_datas["stck_clpr"].iloc[self.index])# 현재 가격 추출
        next_price = float(self.filtering_datas["stck_clpr"].iloc[self.index + 1])# 다음 날 가격 추출
        current_date = self.silce_datas["stck_bsop_date"][self.index] # 현재 날짜 가져오기

        data = self.filtering_datas.iloc[self.index]
        
        done = len(self.filtering_datas) - 2 == self.index # 환경 종료 여부

        extra_datas = self.filtering_datas.iloc[self.index - self.extra_index + 1 : self.index + 1] # 추가 데이터 슬라이싱
        #print(current_date)
        #print(extra_datas.iloc[0])
        self.index += 1 # 다음 인덱스로 

        return data, extra_datas, done, {
                            "current_date": current_date,
                            "stock_code": self.stock_code,
                            "price": self.price,
                            "next_price": next_price
                            }

    

if __name__ == '__main__':
    stock_codes = [ "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030" ] # "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030"
    data_filter = ["stck_clpr","stck_hgpr","stck_lwpr","acml_vol","prdy_vrss",'5','20','60',"rsi","bb_upper","bb_lower"]
    stock_config = Config.load_config("config/StockConfig.yaml")
    a = DailyStockAdaptor(data_filter,str(stock_config.stock_code_path.value))
    np.set_printoptions(precision=6, suppress=True)
    a.load_datas(" 005930",30,0)
    for i in range(30):
        print(i)
        datas, extra_datas, done, info = a.get_info()
        print(datas)
        print(extra_datas)
        