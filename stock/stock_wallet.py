import numpy as np
import abc

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
    def __init__(self, start_amt=100000000, fee=0.0142, tax=0.15):
        super().__init__()
        self.init_balance(start_amt)
        
        self.fee = fee
        self.tax = tax

    def init_balance(self, start_amt):
        self.start_amt = start_amt # 처음 자산
        self.current_amt = start_amt # 현재 자산
        self.qty = 0.0 # 보유한 주식 개수

    def get_total_amt(self, price):
        return self.current_amt + (price * self.qty) # 총 자산 계산

    def get_balance(self): # 현재 자산
        return self.current_amt

    def get_psbl_qty(self, price): # 구매가능 개수
        return self.current_amt // price
    
    def get_total_evlu_rate(self, price): # 총자산 증감 비율
        total_amt = self.current_amt + (price * self.qty) # 총 자산 계산
        amt_diff = total_amt - self.start_amt # 초기 자산 차이 계산
        evlu_rate = amt_diff / self.start_amt # 비율 계산

        return evlu_rate

    def order(self, item_no="005930", order_percent=0.0, price=0):
        if order_percent > 1 or order_percent < -1: # 매매 퍼센트가 -1 <= x <= 1 사이가 아니면 
            return self.get_total_amt(price), 0.0, self.qty, False 

        is_sell = order_percent < 0 # 매도 확인
        total_qty = self.current_amt // price # 매수 가능 총 개수

        if (total_qty == 0 and order_percent > 0) or (self.qty == 0 and order_percent < 0): # 매매 수량 없는데 시도 시
            return self.get_total_amt(price), 0.0, self.qty, False 

        if is_sell: # 매도
            order_qty = np.floor(np.abs(self.qty * order_percent)) # 매도 주문 개수 계산
            order_price = price * order_qty # 주문 가격 계산
            
        else: # 매수
            order_qty = np.floor(np.abs(total_qty * order_percent)) # 매수 주문 개수 계산
            order_price = price * order_qty # 주문 가격 계산

        order_fee = self.calculate_fee(order_price) # 수수료 계산

        if order_qty == 0: # 관망
            pass
        elif is_sell: # 매도
            order_tax = self.calculate_tax(order_price)
            total_price = order_price - order_fee - order_tax
            self.current_amt += total_price
            self.qty -= order_qty 
        else: # 매수
            total_price = order_price + order_fee
            self.current_amt -= total_price
            self.qty += order_qty

        #print(self.current_amt)
        return self.get_total_amt(price), order_qty, self.qty, True # 현재 가지고 있는 현금, 가지고있는 주식 수량, 구매 선공 여부 
    
    def calculate_fee(self, order_price):
        return int(order_price * (self.fee / 100))
    
    def calculate_tax(self, order_price):
        return int(order_price * (self.tax / 100))


if __name__ == '__main__':
    t = TrainStockWallet()
    print(t.order(order_percent=0.0065, price=60100))
    print(t.order(order_percent=0.0035, price=61000))
    print(t.order(order_percent=-0.0001, price=61300))
    #print(t.get_total_evlu_rate(61400))