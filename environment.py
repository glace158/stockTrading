import abc
import gym
import numpy as np
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)

from gym import spaces
from communication.redisCommunication import RedisCom

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
    def __init__(self):
        super().__init__()
        self.red = RedisCom()

        self.observation_space = spaces.Box(low = -100, high= 100, shape= (3,4), dtype=np.float32)
        self.action_space = spaces.Box(low= -1, high= 1, shape=(3,), dtype=np.float32)

    def reset(self) -> Tuple[Any, dict]: # ndarray, {None}
        pass

    def getObservation(self) -> spaces.Space: # box
        return self.observation_space
    
    def getActon(self) -> spaces.Space: # box
        return self.action_space
    
    def getRewardRange(self) -> tuple[float, float]: # -1 ~ 1
        pass
    
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]: # (nextstate, reward, terminated, truncated, info) 
        pass
    
    def render(self):
        pass

    def seed(self,random_seed):
        pass

    def close(self):
        pass

    def stockDataLoad(self):
        self.data = self.red.loadData("stock_data")
    
    def requestStockData(self, stock_code): # 주식 코드 요청
        self.red.sendData("stock_code", stock_code)
    

        

