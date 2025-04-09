import abc
import gym
import numpy as np
import random
import datetime
import os

from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)

from gym import spaces
from stock.stock_adaptor import DailyStockAdaptor
from stock.stock_wallet import TrainStockWallet

class Environment:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self) -> Tuple[Any, dict]:
        pass
    
    @abc.abstractmethod
    def getObservation(self) -> spaces.Space:
        pass
    
    @abc.abstractmethod
    def getActon(self) -> spaces.Space:
        pass
    
    @abc.abstractmethod
    def getRewardRange(self) -> tuple[float, float]:
        pass
    
    @abc.abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        pass
    
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def seed(self,random_seed):
        pass

    @abc.abstractmethod
    def close(self):
        pass
    
class GymEnvironment(Environment): # OpenAI gym 환경
    def __init__(self, env_name='CartPole-v1', render_mode = ""):
        super().__init__()
        
        if render_mode == "":
            self.env = gym.make(env_name)
        else:
            self.env = gym.make(env_name, render_mode=render_mode)
    
    def reset(self): # observation, info
        return self.env.reset()
        
    def getObservation(self): 
        return self.env.observation_space
    
    def getActon(self):
        return self.env.action_space
    
    def getRewardRange(self):
        return self.env.reward_range
    
    def step(self, action) : # (observation, reward, terminated, truncated, info) 
        return self.env.step(action)
    
    def render(self):
        self.env.render()

    def seed(self,random_seed):
        self.env.seed(random_seed)

    def close(self):
        self.env.close()

class StockEnvironment(Environment): # 주식 환경
    def __init__(self, stock_code_path = "API/datas", #["005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030"], 
                 min_dt="20190214", max_dt="20250131", defult_count=30):
        super().__init__()
        self.stock = DailyStockAdaptor()
        self.wallet = TrainStockWallet()
        
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.stock_code_path = stock_code_path
        self.defult_count = defult_count


        self.observation_space = spaces.Box(low = -np.inf, high= np.inf, shape= (len(self.reset()[0]),), dtype=np.float32)
        self.action_space = spaces.Box(low= -1, high= 1, shape=(1,), dtype=np.float32)

    def print_log(self):
        print("--------------------------------------------------------------------------------------------")
        print("start date : " + self.strt_dt)
        print("stock code : " + self.stock_code)
        print("total count :" + str(self.count))
        print("start balance :" + str(self.wallet.start_amt))
        print("--------------------------------------------------------------------------------------------")

    def reset(self) -> Tuple[Any, dict]: # ndarray, {None}
        self.strt_dt = self._get_random_strt_dt() # 시작 날짜 설정
        self.stock_code = self._get_random_stock_code() # 주식코드 설정
        self.count = self._get_random_count() # 에피소드 크기 설정
        self.wallet.init_balance(self._get_random_balance()) # 자산 초기화
        
        self.current_total_amt = self.wallet.get_balance() # 현재(초기) 자산 가져오기

        self.previous_total_amt = self.current_total_amt # 이전 자산 초기화
        
        self.result = self.stock.load_datas(self.stock_code_path + "/" + self.stock_code, inqr_strt_dt=self.strt_dt, count=self.count) # 주식 파일 로드
        #print(result)
        data, _ = self.stock.get_info() # 주식 정보 가져오기
        
        price = data[0] # 가격 추출
        evlu_rate = self.wallet.get_total_evlu_rate(price) # 현재 증감률 계산

        # 정보 추가
        data.insert(0, evlu_rate) 
        data.insert(0, self.wallet.qty)
        data.insert(0, self.wallet.get_psbl_qty(price))
        
        data = np.array(data, dtype=np.float32)
        data = self.normalize(data) # 데이터 정규화

        return data, {} # 데이터 반환
    
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]: # (nextstate, reward, terminated, truncated, info) 
        self.previous_total_amt = self.current_total_amt # 이전 자산 저장

        nextstate, terminated = self.stock.get_info()# 다음 주식 정보 가져오기

        price = nextstate[0] # 가격 추출
        current_total_amt,order_qty , qty, is_order = self.wallet.order(self.stock_code, action, price) # 주문 시도
        
        current_total_amt = self._np_to_int(current_total_amt) # 자료형 변환
        order_qty = self._np_to_int(order_qty)

        self.current_total_amt = current_total_amt # 현재 자산 저장

        evlu_rate = self.wallet.get_total_evlu_rate(price) # 초기 자산 대비 증감률 계산

        # 정보 추가
        nextstate.insert(0, evlu_rate) 
        nextstate.insert(0, self.wallet.qty)
        nextstate.insert(0, self.wallet.get_psbl_qty(price))
        
        truncated = False
        info = {"current_amt" : self.current_total_amt, "order_qty" : order_qty} # 추가 정보
        
        reward = self._get_reward(evlu_rate, is_order) # 보상 계산

        nextstate = np.array(nextstate, dtype=np.float32)
        nextstate = self.normalize(nextstate) # 정규화 
        
        return nextstate, reward, terminated, truncated, info
    
    def getObservation(self) -> spaces.Space: # box
        return self.observation_space
    
    def getActon(self) -> spaces.Space: # box
        return self.action_space
    
    def normalize(self, data): # 정규화 
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        
        return data / norm

    def render(self): 
        pass

    def seed(self,random_seed):
        random.seed(random_seed)

    def close(self):
        pass
    
    def get_data_label(self): # 데이터 라벨 반환
        return ["psbl_qty", "qty", "evlu_rate"] + list(self.result.columns)

    def _np_to_int(self, x):
        if isinstance(x, np.ndarray): # numpy 자료형 바꾸기
            return x[0]
        return x

    def _get_reward(self, evlu_rate, is_order): # 보상 계산
        if is_order:
            if self.current_total_amt == 0:
                reward = 0
            else:
                amt_diff = self.current_total_amt - self.previous_total_amt 
                current_rate_reward = np.clip(amt_diff / self.previous_total_amt, -1, 1)
                total_rate_reward = self._np_to_int(evlu_rate)
                reward = current_rate_reward + total_rate_reward
        else:
            reward = -1
        return reward

    def _get_random_strt_dt(self):
        days = (datetime.datetime.strptime(self.max_dt, "%Y%m%d") - datetime.datetime.strptime(self.min_dt, "%Y%m%d")).days # 최대 날짜와 최소 날짜를 빼준다
        days = random.randrange(0,days-1) # 일 랜덤 뽑기
        return (datetime.datetime.strptime(self.max_dt, "%Y%m%d") - datetime.timedelta(days=days)).strftime("%Y%m%d") # 최종 시작날짜 계산

    def _get_random_stock_code(self):
        stock_file_list = next(os.walk(self.stock_code_path + '/'))[2]
        return random.choice(stock_file_list) # 주식 코드 랜덤 뽑기

    def _get_random_count(self):
        if random.random() > 0.0001:
            return self.defult_count
        else:
            return random.randint(10,self.defult_count)
        
    def _get_random_balance(self):
        return random.randrange(300000, 100000001,100000)
        
if __name__ == '__main__':
    stock_env= StockEnvironment()

    print(stock_env.reset()[0].shape)
    print(stock_env.step(0)[0].shape)

    
    print(stock_env.get_data_label())