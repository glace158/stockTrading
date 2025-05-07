import abc
import pandas as pd
from stock.stock_wallet import TrainStockWallet 
import numpy as np
import datetime
from common.fileManager import Config
from PPO.reward import BuySellReward, ExpReward

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
        self.reward_cls = ExpReward(self.wallet)

        self.data_filter = data_filter

        self.index = 0

        self.inqr_strt_dt = ''
        
        self.bond_yield_data_directory = "API/extra_datas/" + "TreasuryBondYield" + ".csv"
        self.bond_yield_datas = pd.read_csv(self.bond_yield_data_directory)

    
    def load_datas(self, path, inqr_strt_dt, count, start_amt): # 데이터 읽어오기
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

        # 주식 지갑 초기화
        self.wallet.init_balance(start_amt, self.price)
        self.reward_cls.init_datas(self.wallet)

        return self.result 

    # 주식 데이터 하나씩 가져오기
    def get_info(self, stock_code, action):
        self.price = float(self.filtering_datas["stck_clpr"].iloc[self.index])# 가격 추출
        next_price = float(self.filtering_datas["stck_clpr"].iloc[self.index + 1])# 다음 날 가격 추출
        current_date = self.result["stck_bsop_date"][self.index] # 현재 날짜 가져오기

        data = self.filtering_datas.iloc[self.index, :].values # 현재 인덱스에 맞춰 슬라이싱
        data = list(map(float, data)) # 리스트로 변환

        done = len(self.filtering_datas) - 2 == self.index # 환경 종료 여부
        
        # 주문 시도
        # 총 자산, 전날, 주문 수량, 보유 주식 수량, 구매 성공 여부 
        current_total_amt, order_qty , qty, is_order = self.wallet.order(stock_code, action, self.price) 
 
        reward, reward_info = self.reward_cls.get_reward(current_date, is_order, action, self.price, next_price, current_total_amt, self.wallet.get_current_amt(), qty)
        
        # 정보 추가
        data.insert(0, reward_info["evlu_rate"]) # 주식 수익률
        data.insert(0, qty) # 보유 수량
        data.insert(0, self.wallet.get_psbl_qty(self.price)) # 현재 구매 가능한 수량

        self.index += 1 # 다음 인덱스로 
        
        data = np.array(data, dtype=np.float32)
        return data, done, { **{
                            "price": self.price,
                            "current_amt" : current_total_amt,
                            "qty" : qty,
                            "order_qty" : order_qty, 
                            "is_order" : is_order, 
                            "reward" : reward
                            }, **reward_info}
    
    # 라벨 정보 가져오기
    def get_data_label(self):
        return ["psbl_qty", "qty", "evlu_rate"] + list(self.filtering_datas.columns)

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
