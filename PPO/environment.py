import os
import abc
import gym
import pandas as pd
import numpy as np
import random
from gym import spaces
import torchvision.transforms as transforms
import torch

from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)

from PPO.reward import BuySellReward, ExpReward
from common.fileManager import Config
from common.image_tools import get_time_series_image,get_multiple_time_series_images
from common.data_visualization import VisualGraphGenerator
from stock.stock_adaptor import DailyStockAdaptor
from stock.stock_wallet import TrainStockWallet
from common.data_preprocessing import VectorL2Normalizer

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

        #self.min_dt = self.stock_config.min_dt.value
        #self.max_dt = self.stock_config.max_dt.value
        self.stock_code_path = self.stock_config.parameters.stock_code_path.value
        self.defult_count = int(self.stock_config.parameters.count.value)
        self.extra_count = int(self.stock_config.parameters.extra_datas.value)
        self.visualization_columns = self.stock_config.visualization_columns
        self.visualization_format = self.stock_config.parameters.visualization_format.value

        self.stock = DailyStockAdaptor(self.stock_config.stock_columns, self.stock_code_path)

        self.wallet = TrainStockWallet()
        self.reward_cls = ExpReward()

        self.preprocessing = VectorL2Normalizer()

        observation, _ = self.reset()
        
        if isinstance(observation, dict):
            self.observation_space = spaces.Dict({
                "img": spaces.Box(low = -1, high= 1, shape= observation["img"].shape, dtype=np.float32),
                "num": spaces.Box(low = -np.inf, high= np.inf, shape= observation["num"].shape, dtype=np.float32) 
                })
        else:
            self.observation_space = spaces.Box(low = -np.inf, high= np.inf, shape= (len(observation),), dtype=np.float32)
        
        self.action_space = spaces.Box(low= -1, high= 1, shape=(1,), dtype=np.float32)

    def reset(self) -> Tuple[Any, dict]: # ndarray, {None}
        self.stock_code = self._get_random_stock_code() # 주식코드 설정
        self.count = self.defult_count # 에피소드 크기 설정

        start_amt = self._get_random_balance() 
        self.wallet.init_balance(start_amt)

        self.result = self.stock.load_datas(self.stock_code, count=self.count, extra_count=self.extra_count) # 주식 파일 로드

        data, extra_datas, done, info = self._get_observation_datas() # 주식 정보 가져오기

        total_amt = self.wallet.get_total_amt(self.price)
        current_amt = self.wallet.get_current_amt()
        qty = self.wallet.get_qty()
        order_info = {
            "current_date":self.current_date,
            "price": self.price,
            "next_price": self.next_price,
            "total_amt":total_amt,
            "current_amt":current_amt, 
            "order_qty":0.0, 
            "qty":qty,
            "is_order":True
                      }

        self.reward_cls.init_datas(self.price, start_amt)
        reward, truncated, reward_info = self.reward_cls.get_reward(
                                                         self.current_date, 
                                                         True,
                                                         0.0, 
                                                         self.price, 
                                                         self.next_price, 
                                                         total_amt, 
                                                         current_amt, 
                                                         qty,
                                                         0.0
                                                         )

        return (data, {**{"stock_code" :info["stock_code"]}, **order_info, **reward_info}) # 데이터 반환
    
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]: # (nextstate, reward, terminated, truncated, info) 
        action = np.clip(action, -1, 1)

        total_amt, current_amt, order_qty, qty, is_order = self.wallet.order(self.stock_code, action, self.price)
        
        order_info = {
            "current_date":self.current_date,
            "price": self.price,
            "next_price": self.next_price,
            "total_amt":total_amt,
            "current_amt":current_amt, 
            "order_qty":order_qty, 
            "qty":qty,
            "is_order":is_order
                      }

        reward, truncated, reward_info = self.reward_cls.get_reward(
                                                         self.current_date, 
                                                         is_order, 
                                                         action, 
                                                         self.price, 
                                                         self.next_price, 
                                                         total_amt, 
                                                         current_amt, 
                                                         qty,
                                                         order_qty
                                                         ) # 보상
        
        nextstate, extra_datas, terminated, info = self._get_observation_datas(reward_info["init_total_evlu_rate"]) # 다음 주식 정보 가져오기
        
        
        

        return (nextstate, reward, terminated, truncated, {**{"stock_code" :info["stock_code"]}, **order_info, **reward_info})
    
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
    """
    def _get_random_count(self):
        if random.random() > 0.001:
            return self.defult_count
        else:
            return random.randint(10,self.defult_count)
    """    
    def _get_random_balance(self):
        return random.randrange(300000, 100000001,100000)
    
    def _get_observation_datas(self, init_total_evlu_rate = 0.0):
        datas, extra_datas, done, info = self.stock.get_info() # 다음 주식 정보 가져오기

        if extra_datas.shape[0] != self.extra_count: 
            raise IndexError(f"extra_datas Not Matched datas count {self.extra_count - extra_datas.shape}")
        
        datas = datas.values # 데이터프레임에서 값만 가져오기
        self.price = info["price"]
        self.next_price = info["next_price"]
        self.current_date = info["current_date"]

        qty = self.wallet.get_qty()
        current_amt = self.wallet.get_current_amt()
        total_amt = self.wallet.get_total_amt(info["price"])

        # 상태 데이터 추가
        datas = np.insert(datas, 0, init_total_evlu_rate) # 초기 자산 대비 현재 총자산 증감률
        datas = np.insert(datas, 0, qty) # 현재 보유 수량
        datas = np.insert(datas, 0, current_amt) # 현재 현금 보유 수량
        datas = np.insert(datas, 0, total_amt) # 현재 총 자산

        if not extra_datas.empty:
            time_series_images = self._get_visualization_data(self.visualization_columns, extra_datas) # 시계열 데이터 생성
            datas = self.preprocessing.get_preprocessing(datas) # 데이터 전처리 
            datas = dict({"img": time_series_images, "num": datas}) # 데이터 합치기
        else:
            datas = self.preprocessing.get_preprocessing(datas) # 데이터 전처리
            datas = datas.astype(np.float32)


        return datas, extra_datas, done, {**info, **{"qty": qty, "current_amt": current_amt, "total_amt": total_amt}}
    
    def _get_visualization_data(self, target_column_list : list, extra_datas : pd.DataFrame): # 시계열 데이터로 변환
        if self.visualization_format == "graph":
            transform = transforms.Compose([
                transforms.ToTensor(),                # [0, 255] → [0.0, 1.0] & (HWC → CHW)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] → [-1, 1]
            ])

            graph_generator = VisualGraphGenerator()
            tensor_img = np.array(transform(graph_generator.drawing_graph(extra_datas,resize=(128,128))))
            return tensor_img
        else:
            time_series_image = get_multiple_time_series_images(self.visualization_format, target_column_list, extra_datas)
            return time_series_image 
        

if __name__ == '__main__':
    config_path = "config/Hyperparameters.yaml"

    config = Config.load_config(config_path)
    stock_config = Config.load_config("config/StockConfig.yaml")

    stock_env= StockEnvironment(stock_config=stock_config)

    #print(stock_env.reset()[0])
    print(stock_env.step(0)[0])

    for i, val in enumerate(stock_env.step(0)[0]):
        print(f"[{i:2}] {val:,.4f}")