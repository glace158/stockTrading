import abc
import pandas as pd
from stock.stock_wallet import TrainStockWallet 
import numpy as np
import datetime
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
    def __init__(self, data_filter):
        super().__init__()
        self.wallet = TrainStockWallet()

        self.data_filter = data_filter

        self.index = 0

        self.inqr_strt_dt = ''
        self.init_price = 0
        
        self.bond_yield_data_directory = "API/extra_datas/" + "TreasuryBondYield" + ".csv"
        self.bond_yield_datas = pd.read_csv(self.bond_yield_data_directory)

    def init_balance(self, start_amt):
        self.wallet.init_balance(start_amt)
    
    def load_datas(self, path, inqr_strt_dt, count): # 데이터 읽어오기
        self.index = 0
        self.inqr_strt_dt = inqr_strt_dt

        self.result = pd.read_csv(path, header=0, index_col=0)
        index_number = []
        
        while not index_number:
            index_number = self.result[self.result["stck_bsop_date"] == np.float64(inqr_strt_dt)].index.to_list() # 해당하는 날짜 인덱스 찾기
            inqr_strt_dt = (datetime.datetime.strptime(inqr_strt_dt, "%Y%m%d") - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        silce_datas = self.result.loc[index_number[0] - count + 1 : index_number[0] + 1] # 데이터 슬라이싱

        self.result = silce_datas # 슬라이싱 한 데이터로 바꾸기
        self.result.index = np.arange(len(self.result.index)) # 인덱스 재정렬
        
        self.filtering_datas = self.result[list(self.data_filter)]# 데이터 필터링
        self.price = float(self.filtering_datas["stck_clpr"].iloc[self.index])# 가격 추출
        self.init_price = self.price 
        return self.result 

    # 주식 데이터 하나씩 가져오기
    def get_info(self, stock_code, action):
        self.price = float(self.filtering_datas["stck_clpr"].iloc[self.index])# 가격 추출
        data = self.filtering_datas.iloc[self.index, :].values # 현재 인덱스에 맞춰 슬라이싱
        
        done = len(self.filtering_datas) - 2 == self.index # 환경 종료 여부
        
        data = list(map(float, data)) # 리스트로 변환
        
        current_total_amt, daily_rate, order_qty , qty, is_order = self.wallet.order(stock_code, action, self.price) # 주문 시도
        current_total_amt = self._np_to_float(current_total_amt)
        order_qty = self._np_to_float(order_qty)
        qty = self._np_to_float(qty)

        evlu_rate = self.wallet.get_total_evlu_rate(self.price) # 현재 주식 수익률 계산
        diff_price = self.price - self.init_price # 현재 주식 증감률 계산
        price_rate = (diff_price / self.init_price) * 100
        evlu_rate = self._np_to_float(evlu_rate) - self._np_to_float(price_rate) # 순이익 구하기
        
        
        # 정보 추가
        data.insert(0, evlu_rate) # 주식 수익률
        data.insert(0, qty) # 보유 수량
        data.insert(0, self._np_to_float(self.wallet.get_psbl_qty(self.price))) # 현재 구매 가능한 수량
        
        # 새프 지수, 소르티노 지수 계산
        sharp_data = self.sharpe_ratio()
        sharp_data = self._np_to_float(sharp_data)
        
        sortino_data = self.sortino_ratio()
        sortino_data = self._np_to_float(sortino_data)

        self.index += 1
        
        next_price = float(self.filtering_datas["stck_clpr"].iloc[self.index])# 다음 날 가격 추출
        diff_price = next_price - self.price
        next_ctrt = (diff_price / self.price) * 100 # 다음 날 증감률 추출
        next_ctrt = self._np_to_float(next_ctrt)

        next_day_rate = self.wallet.get_next_day_evlu_rate(next_price)# 다음 날 수익률 계산
        next_day_rate = self._np_to_float(next_day_rate)
        next_day_rate = next_day_rate - next_ctrt # 순수 수익률 구하기
        print(next_day_rate)
        print(next_ctrt)

        data = np.array(data, dtype=np.float32)
        return data, done, {"price": self.price,
                            "current_amt" : current_total_amt, 
                            "total_rate": evlu_rate,
                            "daily_rate": daily_rate,
                            "next_day_rate" : next_day_rate, 
                            "order_qty" : order_qty, 
                            "is_order" : is_order, 
                            "sharp_ratio" : sharp_data, 
                            "sortino_ratio" : sortino_data
                            }
    
    # 라벨 정보 가져오기
    def get_data_label(self):
        return ["psbl_qty", "qty", "evlu_rate"] + list(self.filtering_datas.columns)
    
    def sharpe_ratio(self):

        current_date = self.result["stck_bsop_date"][self.index] # 현재 날짜 가져오기

        index = self.bond_yield_datas[self.bond_yield_datas["stck_bsop_date"] == np.float64(current_date)].index.to_list()[0] # 해당하는 날짜 인덱스 찾기

        i = 0
        while True: # 국채 금리 데이터 가져오기
            bond_yield = self.bond_yield_datas["Treasury_Bond_Yield(10Year)"][index + i]
            
            if bond_yield != -1: # 해당 날짜의 데이터가 -1이 아니면 
                break
            
            i -= 1 # 해당 날짜의 데이터가 -1이면 이전 데이터 확인
        
        total_evlu_rate = self.wallet.get_total_evlu_rate(self.price) # 현재 전체 수익률
        rate_std = np.std(self.wallet.rate_list) # 수익률 표준 편차 구하기
    

        if rate_std != 0:
            ratio = (total_evlu_rate - bond_yield) / rate_std
        else:
            ratio  = 0.0

        return ratio
    
    def sortino_ratio(self):

        current_date = self.result["stck_bsop_date"][self.index] # 현재 날짜 가져오기

        index = self.bond_yield_datas[self.bond_yield_datas["stck_bsop_date"] == np.float64(current_date)].index.to_list()[0] # 해당하는 날짜 인덱스 찾기

        i = 0
        while True: # 국채 금리 데이터 가져오기
            bond_yield = self.bond_yield_datas["Treasury_Bond_Yield(10Year)"][index + i]
            
            if bond_yield != -1: # 해당 날짜의 데이터가 -1이 아니면 
                break
            
            i -= 1 # 해당 날짜의 데이터가 -1이면 이전 데이터 확인

        total_evlu_rate = self.wallet.get_total_evlu_rate(self.price) # 현재 전체 수익률
        minus_rate_list = list(filter(lambda x: x<0, self.wallet.rate_list)) # 마이너스 수익률

        if not minus_rate_list: # 마이너스 수익률이 없는 경우
            rate_std = 0
        else:    
            rate_std = np.std(minus_rate_list) # 마이너스 수익률 표준 편차 구하기
        
        if rate_std != 0:
            ratio = (total_evlu_rate - bond_yield) / rate_std
        else:
            ratio  = 0.0

        return ratio


    def _np_to_float(self, x):
        if isinstance(x, np.ndarray): # numpy 자료형 바꾸기
            return float(x[0])
        return x

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
