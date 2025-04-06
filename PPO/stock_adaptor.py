import abc
import pandas as pd
from PPO.stock_data import Stock
import numpy as np
import datetime
from PPO.fileManager import Config

class Adaptor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_info(self):
        pass

    @abc.abstractmethod
    def get_info_len(self):
        pass

class DailyStockAdaptor(Adaptor):
    def __init__(self):
        super().__init__()
        self.stock = Stock("vps")
        self.moving_days = [5,20,60]
        self.rsi_days = 14
        self.bb_days = 20

        self.index = 0
        #self.set_init_datas(itm_no=itm_no, inqr_strt_dt=inqr_strt_dt , count=count, is_remove_date=is_remove_date)
    
    def _compare_datetime(self, standard_datas : pd.DataFrame, target_datas : pd.DataFrame):
        rt_data = pd.DataFrame(columns=target_datas.columns)
        for i in range(standard_datas.shape[0]):
            for j in reversed(range(0, target_datas.shape[0])):
                if datetime.datetime.strptime(standard_datas["stck_bsop_date"][i], "%Y%m%d") > datetime.datetime.strptime(target_datas["stck_bsop_date"][j], "%Y%m%d"):
                    rt_data.loc[i] = target_datas.iloc[j]
                    break
        
        return rt_data
    
    def set_init_datas(self, itm_no=None, inqr_strt_dt=None , count=30, is_remove_date=True):
        self.index = 0
        self.result = pd.DataFrame()

        datas = []
        # 주식정보
        stock_data = self.stock.get_daily_stock_info(itm_no=itm_no, count=count+ max(self.moving_days), inqr_strt_dt=inqr_strt_dt)
        print(stock_data.loc[stock_data.shape[0] - count :].reset_index(drop=True))
        sliced_stock_data = stock_data.loc[stock_data.shape[0] - count :].reset_index(drop=True)

        #datas.append(sliced_stock_data)

        # 이동평균선
        move_line_data = self.stock.get_moving_average_line(stock_data=stock_data, count=count, moving_days=self.moving_days)
        print(move_line_data)
        datas.append(move_line_data)
        
        # RSI
        rsi_data = self.stock.get_rsi(stock_data=stock_data, count=count,days=self.rsi_days)
        print(rsi_data)
        datas.append(rsi_data)
        
        # BollingerBand
        bb_data = self.stock.get_bollinger_band(stock_data=stock_data, moving_average_line=move_line_data, count=count, days=self.bb_days, k=2)
        print(bb_data)
        datas.append(bb_data)
        
        # 코스피
        kospi_data = self.stock.get_daily_index_price(itm_no="0001", count=count, inqr_strt_dt=inqr_strt_dt)
        print(kospi_data)
        datas.append(kospi_data)
        
        # 코스닥
        kosdaq_data = self.stock.get_daily_index_price(itm_no="1001", count=count, inqr_strt_dt=inqr_strt_dt)
        print(kosdaq_data)
        datas.append(kosdaq_data)
        
        # 나스닥
        nasdaq_data = self.stock.get_daily_chartprice(itm_no="COMP", count=count+60, inqr_strt_dt=inqr_strt_dt)
        nasdaq_data = self._compare_datetime(sliced_stock_data,nasdaq_data)
        print(nasdaq_data)
        datas.append(nasdaq_data)
        
        # S&P500
        spx_data = self.stock.get_daily_chartprice(itm_no="SPX", count=count+60, inqr_strt_dt=inqr_strt_dt)
        spx_data = self._compare_datetime(sliced_stock_data,spx_data)
        #del spx_data["acml_vol"]
        print(spx_data)
        datas.append(spx_data)

        if is_remove_date:
            del sliced_stock_data["stck_bsop_date"]
            
        self.result = sliced_stock_data

        for data in datas:
            del data["stck_bsop_date"]
            self.result = pd.concat([self.result, data], axis=1)

        print("================================================")
        return self.result
    
    def save_datas(self, path):

        self.result.to_csv(path, header=True, index=True)
    
    def load_datas(self, path, inqr_strt_dt, count):
        self.index = 0

        self.result = pd.read_csv(path, header=0, index_col=0)
        index_number = []
        
        while not index_number:
            index_number = self.result[self.result["stck_bsop_date"] == np.float64(inqr_strt_dt)].index.to_list() # 해당하는 날짜 인덱스 찾기
            inqr_strt_dt = (datetime.datetime.strptime(inqr_strt_dt, "%Y%m%d") - datetime.timedelta(days=1)).strftime("%Y%m%d")
        #print(index_number)
        
        silce_datas = self.result.loc[index_number[0] - count + 1 : index_number[0]] # 데이터 슬라이싱
        del silce_datas["stck_bsop_date"] # 날짜 값 삭제
        self.result = silce_datas # 슬라이싱 한 데이터로 바꾸기
        self.result.index = np.arange(len(self.result.index)) # 인덱스 재정렬
        
        config = Config.load_config("config/StockConfig.yaml")
        self.result = self.result[list(config.stock_columns)]
        
        return self.result
    
    def get_info(self):
        #print(str(len(self.result)- 1)  + " / " + str(self.index))
        if len(self.result) - 1 == self.index:
            data = self.result.iloc[self.index, :].values
            self.index += 1
            done = True
        else:
            data = self.result.iloc[self.index, :].values
            self.index += 1
            done = False
        
        data = list(map(float, data))
        return data, done

    
if __name__ == '__main__':
    #a = DailyStockAdaptor()
    #a.save_datas()
    min_dt="20190101" 
    max_dt="20250131"
    days = (datetime.datetime.strptime(max_dt, "%Y%m%d")) - (datetime.datetime.strptime(min_dt, "%Y%m%d"))
    print(days)
    stock_codes = [ "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030" ] # "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030"
    """
    for code in stock_codes:
        a = DailyStockAdaptor()
        a.set_init_datas(itm_no=code,inqr_strt_dt=max_dt,count=1500, is_remove_date=False)
        a.save_datas("API\datas\ " + code + ".csv")
    """
    a = DailyStockAdaptor()
    print(a.load_datas("API\datas\ " + "098460" + ".csv", "20190208", 30))

    for i in range(30):
        print(a.get_info())
