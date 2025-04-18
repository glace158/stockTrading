import numpy as np
import abc
import pandas as pd

class Wallet:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_balance(self): # 현재 자산
        pass

    @abc.abstractmethod
    def get_psbl_qty(self, price): # 구매가능 개수
        pass

    @abc.abstractmethod
    def get_total_evlu_rate(self, price): # 총자산 증감 비율
        pass 
    
    @abc.abstractmethod
    def order(self, item_no="005930", order_percent=0.0, price=0):
        pass

class TrainStockWallet(Wallet):
    def __init__(self, start_amt=100000000, init_price=0,fee=0.0142, tax=0.15):
        super().__init__()
        self.init_balance(start_amt, init_price)
        
        self.fee = fee
        self.tax = tax
        
    def init_balance(self, start_amt, init_price):
        self.start_amt = start_amt # 처음 자산
        self.current_amt = start_amt # 현재 자산
        
        self.qty = 0.0 # 보유한 주식 개수

        self.total_amt_list = [start_amt]
        self.rate_list = []

        self.init_price = init_price

    def get_net_income_rate(self, price): # 순이익 비율 계산
        diff = price - self.init_price
        price_rate = (diff / self.init_price) * 100
        
        evlu_rate = self.get_total_evlu_rate(price)
        return evlu_rate - price_rate

    def get_total_amt(self, price):
        return self.current_amt + (price * self.qty) # 총 자산 계산

    def get_balance(self): # 현재 현금 자산
        return self.current_amt

    def get_psbl_qty(self, price): # 구매가능 개수
        return self.current_amt // price
    
    def get_qty(self):
        return self.qty
    
    def get_total_evlu_rate(self, price): # 총자산 증감 비율
        total_amt = self.current_amt + (price * self.qty) # 총 자산 계산
        amt_diff = total_amt - self.start_amt # 초기 자산 차이 계산
        evlu_rate = amt_diff / self.start_amt # 비율 계산

        return evlu_rate * 100

    def get_daily_evlu_rate(self): # 거래당 증감 비율
        amt_diff = self.total_amt_list[-1] - self.total_amt_list[-2]  # 현재 자산 - 이전 자산
        rate = amt_diff / self.total_amt_list[-2] # (현재 자산 - 이전 자산) / 이전 자산
        return rate * 100
    
    def get_next_day_evlu_rate(self, next_price): # 다음날 증감 비율
        next_day_total_amt = self.get_total_amt(next_price)
        amt_diff = next_day_total_amt - self.total_amt_list[-1]  # 예측 다음날 자산 - 현재 자산
        rate = amt_diff / self.total_amt_list[-1] # (예측 다음날 자산 - 현재 자산) / 현재 자산
        return rate * 100
    
    def get_wait_see_next_day_evlu_rate(self, next_price, total_amt):
        next_day_total_amt = self.get_total_amt(next_price)
        amt_diff = next_day_total_amt - total_amt  # 예측 다음날 자산 - 현재 자산
        rate = amt_diff / total_amt # (예측 다음날 자산 - 현재 자산) / 현재 자산
        return rate * 100
    
    def order(self, item_no="005930", order_percent=0.0, price=0):

        if order_percent == 0.0: # 관망
            order_qty = 0
            is_done = True
        elif order_percent < 0 and order_percent >= -1: # 매도
            order_qty, is_done = self.sell(order_percent, price)
        elif order_percent > 0 and order_percent <= 1: # 매수
            order_qty, is_done = self.buy(order_percent, price)
        else:
            order_qty = 0
            is_done = False

        self.total_amt_list.append(self._np_to_float(self.get_total_amt(price))) # 현재 자산 기록
        daily_rate = self._np_to_float(self.get_daily_evlu_rate())
        self.rate_list.append(daily_rate) # 수익률 기록

        assert self.current_amt > 0 # 만약 자산이 음수가 되면 에러 발생
        # 총 자산, 전날 대비 총 자산 증감률 , 주문 수량, 보유 주식 수량, 구매 성공 여부
        return (self.get_total_amt(price), daily_rate, order_qty, self.qty, is_done)  
    
    def sell(self, order_percent=0.0, price=0): # 매도
        if self.qty == 0: # 보유 수량이 없을 때
            return 0, False
        
        order_qty = np.floor(np.abs(self.qty * order_percent)) # 매도 주문 개수 계산

        if order_qty != 0:
            order_price = price * order_qty # 주문 가격 계산
            
            order_fee = self.calculate_fee(order_price) # 수수료 계산
            order_tax = self.calculate_tax(order_price) # 매도 수수료 계산

            total_price = order_price - order_fee - order_tax
            self.current_amt += total_price
            self.qty -= order_qty

        return order_qty, True

    def buy(self, order_percent=0.0, price=0): # 매수
        total_qty = self.current_amt // price # 매수 가능 총 개수
            
        if total_qty == 0: # 구매 가능 수량이 없을 때
            return 0, False
        
        order_qty = np.floor(np.abs(total_qty * order_percent)) # 매수 주문 개수 계산
        
        if order_qty != 0:
            order_price = price * order_qty # 주문 가격 계산

            order_fee = self.calculate_fee(order_price) # 수수료 계산

            total_price = order_price + order_fee
            self.current_amt -= total_price
            self.qty += order_qty

        return order_qty, True
    
    def calculate_fee(self, order_price):
        return int(order_price * (self.fee / 100))
    
    def calculate_tax(self, order_price):
        return int(order_price * (self.tax / 100))
    
    def _np_to_float(self, x):
        if isinstance(x, np.ndarray): # numpy 자료형 바꾸기
            return float(x[0])
        return x
    

def get_price_rate(price, next_price):
    diff = next_price - price
    return (diff / price) * 100

def get_reward(wallet, is_order,order_percent, price, next_price, next_day_rate, wait_see_rate):
    if not is_order:
        return -1
    
    price_rate = get_price_rate(price, next_price) # 다음날 주식 증감률 계산
    
    if price_rate > 0:
        reward = next_day_up_reward(wallet, order_percent, price, next_day_rate, price_rate, wait_see_rate)
    elif price_rate < 0:
        reward = next_day_down_reward(wallet, order_percent, next_day_rate, wait_see_rate)
    else:
        reward = 0

    return reward

def next_day_down_reward(wallet, order_percent, next_day_rate, wait_see_rate): # 다음날 하락할 때
    #rate = self.wallet.get_next_day_evlu_rate(next_price) # 다음날 수익률
    
    if order_percent == 0.0: # 관망
        if wallet.get_qty() == 0: # 매도할 물량이 없는 경우 관망 / 0
            print("매도할 물량이 없는 경우 관망 / 0")
            reward = 0
        elif next_day_rate < 0: # 관망 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
            print("관망 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)")
            reward = next_day_rate
        elif next_day_rate >= 0: # 관망 했는데 +인 경우(물량이 0인 경우) / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0
            print("관망 했는데 +인 경우(물량이 0인 경우) / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0")
            reward = (next_day_rate - wait_see_rate)
        else:
            raise
    elif order_percent < 0 and order_percent >= -1: # 매도
        if next_day_rate <= 0: # 매도 했는데 -인 경우 / 손해를 막은 만큼 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
            print("매도 했는데 -인 경우 / 손해를 막은 만큼 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상")
            reward = (next_day_rate - wait_see_rate)
        else:
            raise

    elif order_percent > 0 and order_percent <= 1: # 매수
        if next_day_rate <= 0: # 매수 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
            print("매수 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)")
            reward = next_day_rate
        else:
            raise
    else:
        reward = -1

    return reward

def next_day_up_reward(wallet, order_percent, price, next_day_rate, price_rate, wait_see_rate): # 다음날 상승할 때
    #rate = self.wallet.get_next_day_evlu_rate(next_price) # 다음날 수익률
    
    if order_percent == 0.0: # 관망
        if wallet.get_psbl_qty(price) == 0: # 매수할 물량이 없는 경우 관망 / 0
            print("매수할 물량이 없는 경우 관망 / 0")
            reward = 0
        elif next_day_rate <= 0: # 관망 했는데 -인 경우(물량이 0 경우) / 가격이 상승한 만큼 페널티 (-price_rate)
            print("관망 했는데 -인 경우(물량이 0 경우) / 가격이 상승한 만큼 페널티 (-price_rate)")
            reward = -price_rate
        elif next_day_rate > 0: # 관망 했는데 +인 경우 / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0
            print("관망 했는데 +인 경우 / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0")
            reward = (next_day_rate - wait_see_rate) 
        else:
            raise
    elif order_percent < 0 and order_percent >= -1: # 매도
        if next_day_rate >= 0: # 매도 했는데 +인 경우 / 다음날 수익 낼 수 있었던 만큼 페널티 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 
            print("매도 했는데 +인 경우 / 다음날 수익 낼 수 있었던 만큼 페널티 (다음날 예상 수익률 - 만약 관망했을 때 수익률) ")
            reward = (next_day_rate - wait_see_rate)
        else:
            raise

    elif order_percent > 0 and order_percent <= 1: # 매수
        if next_day_rate >= 0: # 매수 했는데 +인 경우 / 다음날 수익낸 만큼 보상 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
            print("매수 했는데 +인 경우 / 다음날 수익낸 만큼 보상 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상")
            reward = (next_day_rate - wait_see_rate)
        else:
            raise
    else:
        reward = -1

    return reward 
   
if __name__ == '__main__':
    t = TrainStockWallet(init_price=176000)

    price = 176000
    next_price = 193500
    init_price = price
    order_percent = 1

    wait_see_rate = t.get_wait_see_next_day_evlu_rate(next_price,t.get_total_amt(price))
    print(f"wait_see_rate: {wait_see_rate}")

    info = t.order(order_percent=order_percent, price=price)
    print(info)
    
    next_day_rate = t.get_next_day_evlu_rate(next_price)
    print(f"next_day_rate: {next_day_rate}")
    
    reward = get_reward(t, info[-1],order_percent, price, next_price, next_day_rate, wait_see_rate)
    print(reward)
    print("==========================================") 

    price = 193500
    next_price = 193500
    order_percent = 1.

    wait_see_rate = t.get_wait_see_next_day_evlu_rate(next_price,t.get_total_amt(price))
    print(f"wait_see_rate: {wait_see_rate}")

    info = t.order(order_percent=order_percent, price=price)
    print(info)
    
    next_day_rate = t.get_next_day_evlu_rate(next_price)
    print(f"next_day_rate: {next_day_rate}")
    
    reward = get_reward(t, info[-1],order_percent, price, next_price, next_day_rate, wait_see_rate)
    print(reward)
    print("==========================================")
    
    # 다음날 가격이 떨어질 때
    # 매도할 물량이 없는 경우 관망 / 0
    # 매도 했는데 -인 경우 / 손해를 막은 만큼 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
    # 매도 했는데 +인 경우 / x
    # 매수 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
    # 매수 했는데 +인 경우 / x
    # 관망 했는데 -인 경우 / 손실한 만큼 페널티 (다음날 예상 수익률)
    # 관망 했는데 +인 경우(물량이 0인 경우) / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0

    price = 188500
    next_price = 192000
    order_percent = -0.1

    wait_see_rate = t.get_wait_see_next_day_evlu_rate(next_price,t.get_total_amt(price))
    print(f"wait_see_rate: {wait_see_rate}")

    info = t.order(order_percent=order_percent, price=price)
    print(info)
    
    next_day_rate = t.get_next_day_evlu_rate(next_price)
    print(f"next_day_rate: {next_day_rate}")
    
    reward = get_reward(t, info[-1],order_percent, price, next_price, next_day_rate, wait_see_rate)
    print(reward)
    print("==========================================")
    
    # 다음날 가격이 상승할 때
    # 매수할 물량이 없는 경우 관망 / 0
    # 매도 했는데 -인 경우 / x
    # 매도 했는데 +인 경우 / 다음날 수익 낼 수 있었던 만큼 페널티 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 
    # 매수 했는데 -인 경우 / x
    # 매수 했는데 +인 경우 / 다음날 수익낸 만큼 보상 (다음날 예상 수익률 - 만약 관망했을 때 수익률) 보상
    # 관망 했는데 -인 경우(물량이 0 경우) / 가격이 상승한 만큼 페널티 (-price_rate(price, next_price))
    # 관망 했는데 +인 경우 / (다음날 예상 수익률 - 만약 관망했을 때 수익률) = 0

    price = 192000
    next_price = 289500
    order_percent = 1. 

    wait_see_rate = t.get_wait_see_next_day_evlu_rate(next_price,t.get_total_amt(price))
    print(f"wait_see_rate: {wait_see_rate}")

    info = t.order(order_percent=order_percent, price=price)
    print(info)
    
    next_day_rate = t.get_next_day_evlu_rate(next_price)
    print(f"next_day_rate: {next_day_rate}")
    
    reward = get_reward(t, info[-1],order_percent, price, next_price, next_day_rate, wait_see_rate)
    print(reward)
    print("==========================================")
    
    price = 189500
    next_price = 188000
    order_percent = 1. 

    wait_see_rate = t.get_wait_see_next_day_evlu_rate(next_price,t.get_total_amt(price))
    print(f"wait_see_rate: {wait_see_rate}")

    info = t.order(order_percent=order_percent, price=price)
    print(info)
    
    next_day_rate = t.get_next_day_evlu_rate(next_price)
    print(f"next_day_rate: {next_day_rate}")
    
    reward = get_reward(t, info[-1],order_percent, price, next_price, next_day_rate, wait_see_rate)
    print(reward)
    print("==========================================")
    #print(t.order(order_percent=0.0065, price=60100))
    #print(t.order(order_percent=0.0035, price=61000))
    #print(t.order(order_percent=-0.1, price=61300))
    #print(t.get_total_evlu_rate(61400))