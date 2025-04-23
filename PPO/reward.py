import numpy as np
import abc
from stock.stock_wallet import Wallet

class Reward:
    __metaclass__ = abc.ABCMeta

    def _np_to_float(self, x):
        if isinstance(x, np.ndarray): # numpy 자료형 바꾸기
            return float(x[0])
        return x
    
    def get_price_rate(self, price, next_price):
        diff = next_price - price
        return (diff / price) * 100
    
    def get_exp_reward(self, alpha ,reward):
        exp_reward = 1 - np.exp(-alpha * (reward ** 2))
        if reward < 0: # 보상이 음수면 0
            return 0
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return exp_reward
        else: # 보상은 0
            return 0
    
    def get_exp_reward2(self, alpha, reward):
        exp_reward = 1 - np.exp(-alpha * (reward ** 2))
        if reward < 0: # 오차가 커질수록 -1에 수렴
            return -exp_reward
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return exp_reward
        else: # 보상은 0
            return 0
    
class ExpReward(Reward):
    def get_reward(self, wallet, is_order, order_percent, price, next_price, next_day_rate, wait_see_rate):
        if not is_order:
            return (-1, 0, 0, 0, 0, "not_order")
    
        rate_reward = (next_day_rate - wait_see_rate)
        rate_reward = self._np_to_float(rate_reward)
        
        rate_exp_reward = self.get_exp_reward(alpha = 5.0 , reward=rate_reward)
        rate_exp_reward = self._np_to_float(rate_exp_reward)        
        
        net_income_rate_reward = wallet.get_net_income_rate(next_price) # 순이익 계산
        net_income_rate_reward = self._np_to_float(net_income_rate_reward)        
        
        net_income_rate_exp_reward = self.get_exp_reward(alpha = 0.01 , reward=net_income_rate_reward)
        net_income_rate_exp_reward = self._np_to_float(net_income_rate_exp_reward) 
 
        
        total_rate_reward = wallet.get_total_evlu_rate(next_price) # 총자산 증감 비율
        total_rate_reward = self._np_to_float(total_rate_reward) 

        total_rate_exp_reward = self.get_exp_reward(alpha = 0.01 , reward=total_rate_reward)
        total_rate_exp_reward = self._np_to_float(total_rate_exp_reward) 

        reward = 0.5 * rate_exp_reward + 0.4 * total_rate_exp_reward + 0.1 * net_income_rate_exp_reward
        reward = self._np_to_float(reward)
        
        if order_percent == 0.0: # 관망
            reward_log = "wait"
        elif order_percent < 0 and order_percent >= -1: # 매도
            reward_log = "sell"
        elif order_percent > 0 and order_percent <= 1: # 매수
            reward_log = "buy"
        else:
            reward_log += "wrong" 
            reward = -1
        
        wallet.get_next_day_evlu_rate(next_price) # 다음날 증감률


        return (reward, rate_reward, rate_exp_reward, total_rate_reward, total_rate_exp_reward, reward_log)

class BuySellReward(Reward):

    def get_reward(self, wallet, is_order, order_percent, price, next_price, next_day_rate, wait_see_rate):
        if not is_order:
            return (-1, 0, 0, 0, 0, "not_order")

        price_rate = self.get_price_rate(price, next_price) # 다음날 주식 증감률 계산
        
        if price_rate > 0:
            rate_reward, reward_log = self.next_day_up_reward(wallet, order_percent, price, next_day_rate, price_rate, wait_see_rate)
        elif price_rate < 0:
            rate_reward, reward_log = self.next_day_down_reward(wallet, order_percent, next_day_rate, wait_see_rate)
        else:
            rate_reward = 0
            reward_log = "price_rate = 0"

        total_rate_reward = wallet.get_net_income_rate(next_price) # 순이익 계산

        # 지수형 거리 기반 보상 (Exponential reward based on distance)
        rate_exp_reward = self.get_exp_reward(alpha = 0.5 , reward=rate_reward)
        total_rate_exp_reward = self.get_exp_reward(alpha = 0.01 , reward=total_rate_reward)

        reward = rate_reward

        reward = self._np_to_float(reward)
        rate_reward = self._np_to_float(rate_reward)
        total_rate_reward = self._np_to_float(total_rate_reward)        
        rate_exp_reward = self._np_to_float(rate_exp_reward)
        total_rate_exp_reward = self._np_to_float(total_rate_exp_reward)
        return (reward, rate_reward, rate_exp_reward, total_rate_reward, total_rate_exp_reward, reward_log)
    

    def next_day_down_reward(self, wallet, order_percent, next_day_rate, wait_see_rate): # 다음날 하락할 때
        #rate = self.wallet.get_next_day_evlu_rate(next_price) # 다음날 수익률
        reward_log = "down-" 

        if order_percent == 0.0: # 관망
            if wallet.get_qty() == 0: # 매도할 물량이 없는 경우 관망 / 0
                reward_log += "wait-no_qty"
                reward = 0.01
            elif next_day_rate < 0: # 관망 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
                reward_log += "wait-n"
                reward = next_day_rate
            elif next_day_rate >= 0: # 관망 했는데 +인 경우(물량이 0인 경우) / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0
                reward_log += "wait-p"
                reward = (next_day_rate - wait_see_rate)
            else:
                raise
        elif order_percent < 0 and order_percent >= -1: # 매도
            if next_day_rate <= 0: # 매도 했는데 -인 경우 / 손해를 막은 만큼 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
                reward_log += "sell-n"
                reward = (next_day_rate - wait_see_rate)
            else:
                raise

        elif order_percent > 0 and order_percent <= 1: # 매수
            if next_day_rate <= 0: # 매수 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
                reward_log += "buy-n"
                reward = next_day_rate
            else:
                raise
        else:
            reward_log += "wrong"
            reward = -1
    
        return reward, reward_log 
    
    def next_day_up_reward(self, wallet, order_percent, price, next_day_rate, price_rate, wait_see_rate): # 다음날 상승할 때
        #rate = self.wallet.get_next_day_evlu_rate(next_price) # 다음날 수익률
        reward_log = "up-" 
        
        if order_percent == 0.0: # 관망
            if wallet.get_psbl_qty(price) == 0: # 매수할 물량이 없는 경우 관망 / 0
                reward_log += "wait-no_qty" 
                reward = 0.01
            elif next_day_rate <= 0: # 관망 했는데 -인 경우(물량이 0 경우) / 가격이 상승한 만큼 페널티 (-price_rate)
                reward_log += "wait-n" 
                reward = -price_rate
            elif next_day_rate > 0: # 관망 했는데 +인 경우 / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0
                reward_log += "wait-p" 
                reward = (next_day_rate - wait_see_rate)
            else:
                raise
        elif order_percent < 0 and order_percent >= -1: # 매도
            if next_day_rate >= 0: # 매도 했는데 +인 경우 / 다음날 수익 낼 수 있었던 만큼 페널티 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 
                reward_log += "sell-p"
                reward = (next_day_rate - wait_see_rate)
            else:
                raise

        elif order_percent > 0 and order_percent <= 1: # 매수
            if next_day_rate >= 0: # 매수 했는데 +인 경우 / 다음날 수익낸 만큼 보상 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
                reward_log += "buy-p"
                reward = (next_day_rate - wait_see_rate)
            else:
                raise
        else:
            reward_log += "wrong" 
            reward = -1

        return reward, reward_log 
    