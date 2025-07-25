import numpy as np
import abc
import pandas as pd

from stock.stock_wallet import Wallet, TrainStockWallet

class Reward:
    __metaclass__ = abc.ABCMeta
    def __init__(self, init_price=0, start_amt=0):
        self.init_datas(init_price, start_amt)
        self.bond_yield_data_directory = "API/extra_datas/" + "TreasuryBondYield" + ".csv"
        self.bond_yield_datas = pd.read_csv(self.bond_yield_data_directory)

    def init_datas(self, init_price, start_amt):
        self.init_price = init_price
        self.start_amt = start_amt

        self.total_amt_list = [] # 총 자산 저장
        self.rate_list = [] # 거래 손실 저장
        self.current_money_list = [] # 현재 현금 보유액 저장
        self.qty_list = [] # 현재 보유 개수 저장
        self.step_count = 0 # 현재 스텝

        self.order_list = [] # 주문 내역
        self.total_buy_qty = 0 # 전체 구매 수량

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
        if reward < 0: # 보상이 음수면
            return self._np_to_float(-exp_reward)
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return self._np_to_float(exp_reward)
        else: # 보상은 0
            return 0
        
    def get_exp_reward3(self, alpha ,reward): # 지수함수 보상 계산 3
        exp_reward = 1 - np.exp(-alpha * (reward ** 2))
        if reward < 0: # 보상이 음수면 0
            return 0
        elif reward > 0: # 오차가 커질수록 1에 수렴
            return self._np_to_float(exp_reward)
        else: # 보상은 0
            return 0

     # 행동 로그 출력
    def get_reward_log(self, order_percent, is_order, order_qty):
        reward_log = ''
        if not is_order:
            reward_log = "wrong"
        elif order_percent == 0.0 or order_qty == 0: # 관망
            reward_log = "wait"
        elif order_percent < 0 and order_percent >= -1: # 매도
            reward_log = "sell"
        elif order_percent > 0 and order_percent <= 1: # 매수
            reward_log = "buy"
        else:
            reward_log = "wrong"
        return reward_log

    # 거래 행동 보상
    def get_rate_reward(self, order_percent, wait_see_rate, next_day_evlu_rate): 
        if order_percent == 0.0: # 관망
                rate_reward = wait_see_rate
        else:                
            rate_reward = (next_day_evlu_rate - wait_see_rate) 

        return self._np_to_float(rate_reward)

    # 초기 대비 가격 증감률
    def get_init_price_rate(self, price): 
        diff = price - self.init_price
        price_rate = (diff / self.init_price) * 100
        return self._np_to_float(price_rate)
    
    # 총자산 증감 비율 
    def get_total_evlu_rate(self, total_amt): 
        amt_diff = total_amt - self.start_amt # 초기 자산 차이 계산
        evlu_rate = (amt_diff / self.start_amt) * 100 # 비율 계산

        return self._np_to_float(evlu_rate)
    
     # 순이익 비율 계산
    def get_net_income_rate(self, price_rate, evlu_rate):
        net_income_rate = evlu_rate - price_rate
        return self._np_to_float(net_income_rate)

    # 거래당 증감 비율
    def get_daily_evlu_rate(self, current_total_amt):
        if len(self.total_amt_list) == 0:
            return 0
        
        amt_diff = current_total_amt - self.total_amt_list[-1]  # 현재 자산 - 이전 자산
        rate = (amt_diff / self.total_amt_list[-1]) * 100 # (현재 자산 - 이전 자산) / 이전 자산
        return self._np_to_float(rate)
    
    # 다음날 증감 비율
    def get_next_day_evlu_rate(self, next_day_total_amt, current_total_amt): # 다음날 증감 비율
        amt_diff = next_day_total_amt - current_total_amt  # 예측 다음날 자산 - 현재 자산
        rate = (amt_diff / current_total_amt) * 100 # (예측 다음날 자산 - 현재 자산) / 현재 자산
        return self._np_to_float(rate) 
    
     # 관망 시 증감비율
    def get_wait_see_next_day_evlu_rate(self, price, next_price): # 관망 시 증감비율
        if len(self.current_money_list) == 0 or len(self.qty_list) == 0:
            return 0
        
        next_wait_see_total_amt = self.current_money_list[-1] + (next_price * self.qty_list[-1]) # 관망 시 다음날 자산
        wait_see_total_amt = self.current_money_list[-1] + (price * self.qty_list[-1]) # 관망 시 현재 자산

        amt_diff = next_wait_see_total_amt - wait_see_total_amt  # 관망 시 다음날 자산 - 거래하기 전의 자산
        rate = (amt_diff / wait_see_total_amt) * 100 # (예측 다음날 자산 - 현재 자산) / 현재 자산

        return self._np_to_float(rate)

     # 미실현 수익 계산
    def get_unrealized_gain_loss(self, price, next_price, qty):
        unrealized_gain_loss = ((next_price - price) * qty) * 0.0001
        return self._np_to_float(unrealized_gain_loss) 
    
    # 평단가 계산
    def get_average_price(self, qty, price, order_percent, order_qty, is_order):
        average_price = 0
        if is_order and order_percent > 0:
            self.total_buy_qty += order_qty
            self.order_list.append(order_qty * price) 

        if qty != 0 : # 현재 구매 한 이력이 있으면
            average_price = sum(self.order_list) / self.total_buy_qty # 평단가 구하기
        elif qty == 0: # 현재 수량이 없으면
            self.order_list = []
            self.total_buy_qty = 0

        average_price_rate = 0
        if average_price !=0:
            average_price_rate = self.get_price_rate(average_price, price)

        return self._np_to_float(average_price_rate)
    
    # 자산의 이동 평균 구하기
    def get_average_line(self, count=5):
        if len(self.total_amt_list) < count + 1:
            return 0.0, 0.0

        datas = self.total_amt_list[-count:]
        average = sum(datas) / len(datas)

        pre_datas = self.total_amt_list[-count - 1:-1]
        pre_average = sum(pre_datas) / len(pre_datas)

        slope = ((average - pre_average) / pre_average) * 100 

        return self._np_to_float(average), self._np_to_float(slope)

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

    # 국채 금리 데이터
    def get_valid_bond_yield(self, current_date):
        index = self.bond_yield_datas[self.bond_yield_datas["stck_bsop_date"] == np.float64(current_date)].index.to_list()[0] # 해당하는 날짜 인덱스 찾기

        i = 0
        while True: # 국채 금리 데이터 가져오기
            bond_yield = self.bond_yield_datas["Treasury_Bond_Yield(10Year)"][index + i]
            
            if bond_yield != -1: # 해당 날짜의 데이터가 -1이 아니면 
                break
            
            i -= 1 # 해당 날짜의 데이터가 -1이면 이전 데이터 확인
        return bond_yield
    
class ExpReward(Reward):
    def __init__(self, init_price=0, start_amt=0):
        super().__init__(init_price, start_amt)

    def get_reward(self, current_date, is_order, order_percent, price, next_price, current_total_amt, current_money, qty, order_qty):
        init_price_rate = self.get_init_price_rate(next_price) # 초기 대비 가격 증감률
        price_rate = self.get_price_rate(price, next_price) # 다음날 주식 증감률 계산

        average_price_rate = self.get_average_price(qty, price, order_percent, order_qty, is_order) # 평단가 계산

        next_day_total_amt = current_money + (next_price * qty) # 다음날 예측 총자산

        init_total_evlu_rate = self.get_total_evlu_rate(current_total_amt) # 초기 자산 대비 현재 총자산 증감률
        daily_evlu_rate = self.get_daily_evlu_rate(current_total_amt) # 거래당 자산 증감 비율 (현재 기준)
        wait_see_rate = self.get_wait_see_next_day_evlu_rate(price, next_price) # 만약 관망 했을 때 수익률
        unrealized_gain_loss = self.get_unrealized_gain_loss(price, next_price, qty) # 미실현 수익
        next_total_evlu_rate = self.get_total_evlu_rate(next_day_total_amt) # 다음날 초기 자산 대비 총자산 증감률
        next_day_evlu_rate = self.get_next_day_evlu_rate(next_day_total_amt, current_total_amt) # 다음날 자산 증감률
        rate_reward = self.get_rate_reward(qty, order_percent,order_qty, wait_see_rate, price_rate, next_day_evlu_rate) # 수익 증감 보상
        
        net_income_rate = self.get_net_income_rate(init_price_rate, next_total_evlu_rate) # 순이익 비률

        rate_reward_exp = self.get_exp_reward2(alpha = 10.0 , reward=rate_reward) # 수익 증감 보상
        next_day_evlu_rate_exp = self.get_exp_reward2(5, next_day_evlu_rate) # 다음날 자산 증감률
        
        # 현재 데이터 저장
        self.total_amt_list.append(current_total_amt)
        self.rate_list.append(daily_evlu_rate)
        self.current_money_list.append(current_money)
        self.qty_list.append(qty)

        average_line, average_slope = self.get_average_line(count=14) # 전체 자산의 이동 평균 구하기
        sharp_data = self.sharpe_ratio(current_date, init_total_evlu_rate) # 샤프 지수
        sortino_data = self.sortino_ratio(current_date, init_total_evlu_rate) # 소르티노 지수

        reward = next_day_evlu_rate + average_price_rate + average_slope

        reward_log = self.get_reward_log(order_percent, is_order, order_qty) # 보상 로그

        truncated = False

        reward = np.round(reward, 2)
        self.step_count += 1
        return self._np_to_float(reward),  truncated, {
                                                        "init_price_rate": init_price_rate,
                                                        "wait_see_rate" : wait_see_rate,
                                                        "init_total_evlu_rate" : init_total_evlu_rate,
                                                        "daily_evlu_rate" : daily_evlu_rate,
                                                        "average_price_rate" : average_price_rate,
                                                        "average_line" : average_line,
                                                        "average_slope" : average_slope,
                                                        "rate_reward" : rate_reward,
                                                        "rate_reward_exp" : rate_reward_exp,
                                                        "net_income_rate" : net_income_rate,
                                                        "next_total_evlu_rate" : next_total_evlu_rate,
                                                        "next_day_evlu_rate" : next_day_evlu_rate,
                                                        "next_day_evlu_rate_exp": next_day_evlu_rate_exp,
                                                        "unrealized_gain_loss" : unrealized_gain_loss,
                                                        "sharp_data" : sharp_data,
                                                        "sortino_data" : sortino_data,
                                                        "reward_log" : reward_log
                                                        }
    # 거래 행동 보상
    def get_rate_reward(self, qty, order_percent, order_qty, wait_see_rate, price_rate, next_day_rate):
        rate_reward = 0
        if price_rate > 0: # 주가 상승 예상 (또는 실제 상승)
            if order_percent > 0 and order_qty != 0:
                rate_reward = next_day_rate  # 잘했음 (작은 보너스)
            elif order_percent < 0 and order_qty != 0: # 매도
                rate_reward = (next_day_rate - wait_see_rate) # 잘못했음 (기회비용 손실)
            else: # HOLD는 중립 또는 약간의 보상/페널티
                rate_reward = 0
        elif price_rate < 0: # 주가 하락 예상 (또는 실제 하락)
            if order_percent < 0 and order_qty != 0:
                rate_reward = (next_day_rate - wait_see_rate)  # 잘했음 (손실 회피)
            elif order_percent > 0 and order_qty != 0:
                rate_reward = next_day_rate # 잘못했음 (손실 발생)
            else : # HOLD는 중립 또는 약간의 보상/페널티
                rate_reward = 0
        elif price_rate == 0: # 주가가 그대로 일 때 
            if order_percent != 0 and order_qty != 0: # 매수 또는 매도를 했을 경우
                rate_reward = -1
            else: 
                rate_reward = 1


        return self._np_to_float(rate_reward)

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
    