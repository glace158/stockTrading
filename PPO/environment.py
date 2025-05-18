import os
import abc
import gym
import numpy as np
import random
import datetime
from sklearn.preprocessing import StandardScaler
from common.fileManager import Config, File
from PPO.reward import BuySellReward, ExpReward

from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)

from gym import Env
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
    def __init__(self, stock_config=None):
        super().__init__()
        if stock_config == None:
            raise ValueError("stock_config is None")
        
        self.stock_config = stock_config

        self.min_dt = self.stock_config.min_dt.value
        self.max_dt = self.stock_config.max_dt.value
        self.stock_code_path = self.stock_config.stock_code_path.value
        self.defult_count = int(self.stock_config.count.value)

        self.stock = DailyStockAdaptor(self.stock_config.stock_columns, self.stock_code_path)

        self.wallet = TrainStockWallet()
        self.reward_cls = ExpReward()

        self.observation_space = spaces.Box(low = -np.inf, high= np.inf, shape= (len(self.reset()[0]),), dtype=np.float32)
        self.action_space = spaces.Box(low= -1, high= 1, shape=(1,), dtype=np.float32)

    def reset(self) -> Tuple[Any, dict]: # ndarray, {None}
        self.stock_code = self._get_random_stock_code() # 주식코드 설정
        self.count = self._get_random_count() # 에피소드 크기 설정

        self.result = self.stock.load_datas(self.stock_code, count=self.count) # 주식 파일 로드
        #print(result)
        
        data, extra_datas, done, info = self._get_observation_datas() # 주식 정보 가져오기

        start_amt = self._get_random_balance() 
        self.wallet.init_balance(start_amt)
        self.reward_cls.init_datas(self.price, start_amt)

        reward, reward_info = self.reward_cls.get_reward(info["current_date"], 
                                                         True,
                                                         0.0, 
                                                         self.price, 
                                                         info["next_price"], 
                                                         self.wallet.get_total_amt(self.price), 
                                                         self.wallet.get_current_amt(), 
                                                         self.wallet.get_qty()
                                                         )

        data = data.astype(np.float32)
        return (data, {**info, **reward_info}) # 데이터 반환
    
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]: # (nextstate, reward, terminated, truncated, info) 

        total_amt, current_amt, order_qty, qty, is_order = self.wallet.order(self.stock_code, action, self.price)
        nextstate, extra_datas, terminated, info = self._get_observation_datas() # 주식 정보 가져오기
        self.price = info["price"]

        reward, reward_info = self.reward_cls.get_reward(info["current_date"], 
                                                         is_order, 
                                                         action, 
                                                         self.price, 
                                                         info["next_price"], 
                                                         total_amt, 
                                                         current_amt, 
                                                         qty
                                                         ) # 보상

        truncated = False
        
        nextstate = nextstate.astype(np.float32)
        return (nextstate, reward, terminated, truncated, {**info, **reward_info})
    
    def getObservation(self) -> spaces.Space: # box
        return self.observation_space
    
    def getActon(self) -> spaces.Space: # box
        return self.action_space

    def render(self): 
        pass

    def seed(self,random_seed):
        random.seed(random_seed)

    def close(self):
        pass

    def _get_random_stock_code(self):
        stock_file_list = next(os.walk(self.stock_code_path + '/'))[2]
        return random.choice(stock_file_list) # 주식 코드 랜덤 뽑기

    def _get_random_count(self):
        if random.random() > 0.001:
            return self.defult_count
        else:
            return random.randint(10,self.defult_count)
        
    def _get_random_balance(self):
        return random.randrange(300000, 100000001,100000)
    
    def _get_observation_datas(self):
        datas, extra_datas, done, info = self.stock.get_info() # 주식 정보 가져오기
        self.price = info["price"]

        datas = np.insert(datas, 0, self.wallet.get_qty())
        datas = np.insert(datas, 0, self.wallet.get_current_amt())
        datas = np.insert(datas, 0, self.wallet.get_total_amt(self.price))

        return datas, extra_datas, done, info
    
if __name__ == '__main__':
    config_path = "config/Hyperparameters.yaml"

    config = Config.load_config(config_path)
    stock_config = Config.load_config("config/StockConfig.yaml")

    stock_env= StockEnvironment(stock_config=stock_config)

    #print(stock_env.reset()[0])
    print(stock_env.step(0)[0])

    for i, val in enumerate(stock_env.step(0)[0]):
        print(f"[{i:2}] {val:,.4f}")
    
    print(stock_env.get_data_label())