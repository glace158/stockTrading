import abc
import pandas as pd
from stock_data import Stock
import numpy as np
import random
import datetime

class Adaptor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_info(self):
        pass

    @abc.abstractmethod
    def get_info_len(self):
        pass

class DailyStockAdaptor(Adaptor):
    def __init__(self, itm_no="005930", inqr_strt_dt="20190226" , count=30):
        super().__init__()
        self.stock = Stock("vps")
        self.moving_days = [5,20,60]
        self.rsi_days = 14
        self.bb_days = 20

        self.index = 0
        self.set_init_datas(itm_no=itm_no, inqr_strt_dt=inqr_strt_dt , count=count)
    
    def _compare_datetime(self, standard_datas : pd.DataFrame, target_datas : pd.DataFrame):
        rt_data = pd.DataFrame(columns=target_datas.columns)
        for i in range(standard_datas.shape[0]):
            for j in reversed(range(0, target_datas.shape[0])):
                if datetime.datetime.strptime(standard_datas["stck_bsop_date"][i], "%Y%m%d") > datetime.datetime.strptime(target_datas["stck_bsop_date"][j], "%Y%m%d"):
                    rt_data.loc[i] = target_datas.iloc[j]
                    break
        
        return rt_data
    
    def set_init_datas(self, itm_no=None, inqr_strt_dt=None , count=30):
        self.index = 0
        self.result = pd.DataFrame()

        datas = []
        # 주식정보
        stock_data = self.stock.get_daily_stock_info(itm_no=itm_no, count=count+ max(self.moving_days), inqr_strt_dt=inqr_strt_dt)
        #print(stock_data.loc[stock_data.shape[0] - count :].reset_index(drop=True))
        sliced_stock_data = stock_data.loc[stock_data.shape[0] - count :].reset_index(drop=True)

        datas.append(sliced_stock_data)

        # 이동평균선
        move_line_data = self.stock.get_moving_average_line(stock_data=stock_data, count=count, moving_days=self.moving_days)
        #print(move_line_data)
        datas.append(move_line_data)
        
        # RSI
        rsi_data = self.stock.get_rsi(stock_data=stock_data, count=count,days=self.rsi_days)
        #print(rsi_data)
        datas.append(rsi_data)
        
        # BollingerBand
        bb_data = self.stock.get_bollinger_band(stock_data=stock_data, moving_average_line=move_line_data, count=count, days=self.bb_days, k=2)
        #print(bb_data)
        datas.append(bb_data)
        
        # 코스피
        kospi_data = self.stock.get_daily_index_price(itm_no="0001", count=count, inqr_strt_dt=inqr_strt_dt)
        #print(kospi_data)
        datas.append(kospi_data)
        
        # 코스닥
        kosdaq_data = self.stock.get_daily_index_price(itm_no="1001", count=count, inqr_strt_dt=inqr_strt_dt)
        #print(kosdaq_data)
        datas.append(kosdaq_data)
        
        # 나스닥
        nasdaq_data = self.stock.get_daily_chartprice(itm_no="COMP", count=count+5, inqr_strt_dt=inqr_strt_dt)
        nasdaq_data = self._compare_datetime(sliced_stock_data,nasdaq_data)
        #print(nasdaq_data)
        datas.append(nasdaq_data)
        
        # S&P500
        spx_data = self.stock.get_daily_chartprice(itm_no="SPX", count=count+5, inqr_strt_dt=inqr_strt_dt)
        spx_data = self._compare_datetime(sliced_stock_data,spx_data)
        del spx_data["acml_vol"]
        #print(spx_data)
        datas.append(spx_data)
        
        for data in datas:
            del data["stck_bsop_date"]
            self.result = pd.concat([self.result, data], axis=1)

        return self.result

    def get_info(self):
        try:
            data = self.result.iloc[self.index, :].values
            self.index += 1
            done = False
        except IndexError:
            data = None
            done = True

        return data, done

    def get_info_len(self):
        return len(self.result.columns)

    
if __name__ == '__main__':
    a = DailyStockAdaptor()
    for i in range(40):
        print(a.get_info())
        print("===========================")
