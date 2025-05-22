import numpy as np
import abc
import pandas as pd

from stock.stock_wallet import Wallet, TrainStockWallet

class Reward:
    __metaclass__ = abc.ABCMeta

    def _np_to_float(self, x):
        if isinstance(x, np.ndarray): # numpy 자료형 바꾸기
            return float(x[0])
        return x
    
    def get_price_rate(self, price, next_price): # 가격 변동률
        diff = next_price - price
        return self._np_to_float((diff / price) * 100)
    
    def get_exp_reward(self, alpha ,reward): # 지수함수 보상 계산
        exp_reward = 1 - np.exp(-alpha * (reward ** 2))
        if reward < 0: # 보상이 음수면 
            return -0.01
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return self._np_to_float(exp_reward)
        else: # 보상은 0
            return 0
    
    def get_exp_reward2(self, alpha ,reward): # 지수함수 보상 계산 2
        exp_reward = 1 - np.exp(-alpha * (reward ** 2))
        if reward < 0: # 보상이 음수면 0
            return self._np_to_float(-exp_reward)
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return self._np_to_float(exp_reward)
        else: # 보상은 0
            return 0
        
    
class ExpReward(Reward):
    def __init__(self, init_price=0, start_amt=0):
        super().__init__()
        self.init_datas(init_price, start_amt)
        self.bond_yield_data_directory = "API/extra_datas/" + "TreasuryBondYield" + ".csv"
        self.bond_yield_datas = pd.read_csv(self.bond_yield_data_directory)

    def init_datas(self, init_price, start_amt):
        self.init_price = init_price
        self.start_amt = start_amt

        self.total_amt_list = [self.start_amt]
        self.rate_list = []
        self.current_amt_list = [self.start_amt]
        self.qty_list = [0]

    def get_reward(self, current_date, is_order, order_percent, price, next_price, current_total_amt, current_amt, qty):
        init_price_rate = self.get_init_price_rate(next_price) # 초기 대비 가격 증감률
        next_day_total_amt = current_amt + (next_price * qty)
        
        evlu_rate = self.get_total_evlu_rate(current_total_amt) # 현재 총자산 증감률
        daily_evlu_rate = self.get_daily_evlu_rate(current_total_amt) # 거래당 자산 증감 비율 (현재 기준)
        wait_see_rate = self.get_wait_see_next_day_evlu_rate(next_price) # 만약 관망 했을 때 수익률
        
        next_total_evlu_rate = self.get_total_evlu_rate(next_day_total_amt) # 다음날 총자산 증감률
        net_income_rate = self.get_net_income_rate(init_price_rate, next_total_evlu_rate) # 다음날 순이익 비율
        next_day_evlu_rate = self.get_next_day_evlu_rate(next_day_total_amt, current_total_amt) # 다음날 자산 증감률
        unrealized_gain_loss = self.get_unrealized_gain_loss(price, next_price, qty) #  미실현 수익 기회 손실 계산
        rate_reward = self.get_rate_reward(order_percent, wait_see_rate, next_day_evlu_rate) # 수익 증감 보상

        next_total_evlu_rate_exp = self.get_exp_reward(0.01, next_total_evlu_rate) # 다음날 총자산 증감률
        net_income_rate_exp = self.get_exp_reward(0.01, net_income_rate) # 다음날 순이익 비율
        next_day_evlu_rate_exp = self.get_exp_reward(0.01, next_day_evlu_rate) # 다음날 자산 증감률
        unrealized_gain_loss_exp = self.get_exp_reward(0.01, unrealized_gain_loss) #  미실현 수익 기회 손실 계산
        rate_reward_exp = self.get_exp_reward(alpha = 5.0 , reward=rate_reward) # 수익 증감 보상
        
        # 현재 데이터 저장
        self.total_amt_list.append(current_total_amt)
        self.rate_list.append(daily_evlu_rate)
        self.current_amt_list.append(current_amt)
        self.qty_list.append(qty)

        sharp_data = self.sharpe_ratio(current_date, evlu_rate) # 샤프 지수
        sortino_data = self.sortino_ratio(current_date, evlu_rate) # 소르티노 지수

        if is_order: # 보상 계산
            reward = 0.5 * rate_reward_exp + 0.3 * next_total_evlu_rate_exp + 0.2 * net_income_rate_exp; 
        else:
            reward = -0.05

        reward_log = self.get_reward_log(order_percent, is_order) # 보상 로그

        return self._np_to_float(reward), {
                                            "wait_see_rate" : wait_see_rate,
                                            "evlu_rate" : evlu_rate,
                                            "daily_evlu_rate" : daily_evlu_rate,
                                            "rate_reward" : rate_reward,
                                            "rate_reward_exp" : rate_reward_exp, 
                                            "next_total_evlu_rate" : next_total_evlu_rate,
                                            "next_total_evlu_rate_exp" : next_total_evlu_rate_exp,
                                            "net_income_rate" : net_income_rate,
                                            "net_income_rate_exp" : net_income_rate_exp,
                                            "next_day_evlu_rate" : next_day_evlu_rate,
                                            "next_day_evlu_rate_exp" : next_day_evlu_rate_exp,
                                            "unrealized_gain_loss" : unrealized_gain_loss,
                                            "unrealized_gain_loss_exp" : unrealized_gain_loss_exp,
                                            "sharp_data" : sharp_data,
                                            "sortino_data" : sortino_data,
                                            "reward_log" : reward_log
                                            }

    def get_reward_log(self, order_percent, is_order):
        reward_log = ''
        if not is_order:
            reward_log = "wrong"
        elif order_percent == 0.0: # 관망
            reward_log = "wait"
        elif order_percent < 0 and order_percent >= -1: # 매도
            reward_log = "sell"
        elif order_percent > 0 and order_percent <= 1: # 매수
            reward_log = "buy"
        else:
            reward_log = "wrong"
        return reward_log

    def get_rate_reward(self, order_percent, wait_see_rate, next_day_evlu_rate):
        if order_percent == 0.0: # 관망
                rate_reward = wait_see_rate
        else:                
            rate_reward = (next_day_evlu_rate - wait_see_rate) 

        return self._np_to_float(rate_reward)

    def get_init_price_rate(self, price): # 초기 대비 가격 증감률
        diff = price - self.init_price
        price_rate = (diff / self.init_price) * 100
        return self._np_to_float(price_rate)
    
    def get_total_evlu_rate(self, total_amt): # 총자산 증감 비율 
        amt_diff = total_amt - self.start_amt # 초기 자산 차이 계산
        evlu_rate = (amt_diff / self.start_amt) * 100 # 비율 계산

        return self._np_to_float(evlu_rate)
    
    def get_net_income_rate(self, price_rate, evlu_rate): # 순이익 비율 계산
        net_income_rate = evlu_rate - price_rate
        return self._np_to_float(net_income_rate)

    def get_daily_evlu_rate(self, current_total_amt): # 거래당 증감 비율
        if len(self.total_amt_list) == 0:
            return 0
        
        amt_diff = current_total_amt - self.total_amt_list[-1]  # 현재 자산 - 이전 자산
        rate = (amt_diff / self.total_amt_list[-1]) * 100 # (현재 자산 - 이전 자산) / 이전 자산
        return self._np_to_float(rate)
    
    def get_next_day_evlu_rate(self, next_day_total_amt, current_total_amt): # 다음날 증감 비율
        amt_diff = next_day_total_amt - current_total_amt  # 예측 다음날 자산 - 현재 자산
        rate = (amt_diff / current_total_amt) * 100 # (예측 다음날 자산 - 현재 자산) / 현재 자산
        return self._np_to_float(rate) 
    
    def get_wait_see_next_day_evlu_rate(self, next_price): # 관망 시 증감비율
        next_day_total_amt = self.current_amt_list[-1] + (next_price * self.qty_list[-1]) # 관망 시 다음날 자산

        amt_diff = next_day_total_amt - self.total_amt_list[-1]  # 예측 다음날 자산 - 거래하기 전의 자산
        rate = (amt_diff / self.total_amt_list[-1]) * 100 # (예측 다음날 자산 - 현재 자산) / 현재 자산
        return self._np_to_float(rate)

    def get_unrealized_gain_loss(self, price, next_price, qty): # 미실현 수익 계산
        unrealized_gain_loss = ((next_price - price) * qty) * 0.00001
        return self._np_to_float(unrealized_gain_loss) 
        
    # 샤프 지수
    def sharpe_ratio(self, current_date, total_evlu_rate):
        bond_yield = self.get_valid_bond_yield(current_date)
        
        if len(self.rate_list) < 2:
            rate_std = 0
        else:
            rate_std = np.std(self.rate_list) # 수익률 표준 편차 구하기
    

        if rate_std != 0:
            ratio = (total_evlu_rate - bond_yield) / rate_std
        else:
            ratio  = 0.0

        return self._np_to_float(ratio)
    
    # 소르티노 지수
    def sortino_ratio(self, current_date, total_evlu_rate):
        bond_yield = self.get_valid_bond_yield(current_date)
        
        minus_rate_list = list(filter(lambda x: x<0, self.rate_list)) # 마이너스 수익률

        if len(minus_rate_list) < 2: # 마이너스 수익률이 없는 경우
            rate_std = 0
        else:    
            rate_std = np.std(minus_rate_list) # 마이너스 수익률 표준 편차 구하기
        
        if rate_std != 0: 
            ratio = (total_evlu_rate - bond_yield) / rate_std
        else:
            ratio  = 0.0

        return self._np_to_float(ratio)

    def get_valid_bond_yield(self, current_date):
        index = self.bond_yield_datas[self.bond_yield_datas["stck_bsop_date"] == np.float64(current_date)].index.to_list()[0] # 해당하는 날짜 인덱스 찾기

        i = 0
        while True: # 국채 금리 데이터 가져오기
            bond_yield = self.bond_yield_datas["Treasury_Bond_Yield(10Year)"][index + i]
            
            if bond_yield != -1: # 해당 날짜의 데이터가 -1이 아니면 
                break
            
            i -= 1 # 해당 날짜의 데이터가 -1이면 이전 데이터 확인
        return bond_yield

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
    