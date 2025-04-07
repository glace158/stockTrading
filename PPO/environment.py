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
        self.strt_dt = self._get_random_strt_dt()
        self.stock_code = self._get_random_stock_code()
        self.count = self._get_random_count()
        self.wallet.init_balance(self._get_random_balance())
        #self.print_log()
        
        #self.stock.set_init_datas(itm_no=self.stock_code, inqr_strt_dt=self.strt_dt, count=self.count)
        self.result = self.stock.load_datas(self.stock_code_path + "/" + self.stock_code, inqr_strt_dt=self.strt_dt, count=self.count)
        #print(result)
        data, _ = self.stock.get_info()
        
        price = data[0]
        evlu_rate = self.wallet.get_total_evlu_rate(price)

        data.insert(0, evlu_rate)
        data.insert(0, self.wallet.qty)
        data.insert(0, self.wallet.get_psbl_qty(price))
        
        data = np.array(data, dtype=np.float32)
        data = self.normalize(data)

        return data, {}

    def getObservation(self) -> spaces.Space: # box
        return self.observation_space
    
    def getActon(self) -> spaces.Space: # box
        return self.action_space
    
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]: # (nextstate, reward, terminated, truncated, info) 
        nextstate, terminated = self.stock.get_info()

        price = nextstate[0]
        current_amt, qty, is_order = self.wallet.order(self.stock_code, action, price)
        evlu_rate = self.wallet.get_total_evlu_rate(price)

        nextstate.insert(0, evlu_rate)
        nextstate.insert(0, qty)
        nextstate.insert(0, self.wallet.get_psbl_qty(price))
        
        truncated = False
        info = {}
        
        if is_order:
            reward = np.clip(evlu_rate, -1, 1)[0]
        else:
            reward = -1
        #print(reward)
        nextstate = np.array(nextstate, dtype=np.float32)
        nextstate = self.normalize(nextstate)
        
        return nextstate, reward, terminated, truncated, info
    
    def normalize(self, data):
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
    
    def get_data_label(self):
        return ["psbl_qty", "qty", "evlu_rate"] + list(self.result.columns)

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