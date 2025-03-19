import API.kis_auth as ka
import API.kis_domstk as kb
import datetime
import pandas as pd
import numpy as np

class Stock:
    def __init__(self, svr="vps"):
        ka.auth(svr) # 토큰발급

    def datetime_add(self, inqr_strt_dt, days):# 날짜 더하기
        return (datetime.datetime.strptime(inqr_strt_dt,"%Y%m%d") + datetime.timedelta(days=days)).strftime("%Y%m%d")
    
    def datetime_sub(self, inqr_strt_dt, days):# 날짜 빼기
        return (datetime.datetime.strptime(inqr_strt_dt,"%Y%m%d") - datetime.timedelta(days=days)).strftime("%Y%m%d")
    
    def get_daily_stock_info(self, itm_no="005930", inqr_strt_dt=None,count=30):
        '''
        [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년)
        실전계좌/모의계좌의 경우, 한 번의 호출에 최대 100건까지 확인 가능합니다.
        '''
        rt_data = pd.DataFrame()
        
        inqr_end_dt = self.datetime_sub(inqr_strt_dt, count) # 마지막 일자 계산
        
        while True:
            try:# 데이터 가져오기
                rt_data = kb.get_inquire_daily_itemchartprice(output_dv="2", itm_no=itm_no, inqr_strt_dt=inqr_end_dt, inqr_end_dt=inqr_strt_dt).loc[::-1]
                break
            except AttributeError:
                print("wait..")
                continue

        inqr_strt_dt = rt_data.iat[0,0]
        while rt_data.shape[0] < count:# 만약 가져온 데이터가 부족하면
            try:
                # 마지막 일자 다시 계산
                temp_strt_dt = self.datetime_sub(inqr_strt_dt,1) 
                inqr_end_dt = self.datetime_sub(temp_strt_dt, count // 2) 
                
                # 추가로 가져온데이터 추가하기
                extra_data = kb.get_inquire_daily_itemchartprice(output_dv="2", itm_no=itm_no, inqr_strt_dt=inqr_end_dt, inqr_end_dt=temp_strt_dt).loc[::-1]

                rt_data = pd.concat([extra_data, rt_data], ignore_index=True)

                inqr_strt_dt = rt_data.iat[0,0]
                
            except AttributeError:
                print("wait..")
                continue
        
        if rt_data.shape[0] > count:
            rt_data = rt_data.loc[rt_data.shape[0] - count:] # 필요한 데이터 개수만큼 슬라이싱
            rt_data.index = np.arange(count)
        return rt_data
    
    def get_daily_investor(self, inqr_strt_dt=None,count=30):
        """
        [국내주식] 시세분석 > 시장별 투자자매매동향 (일별)
        """
        rt_data = pd.DataFrame()

        while True:
            try:# 데이터 가져오기
                rt_data = kb.get_daily_inquire_investor(inqr_strt_dt=inqr_strt_dt).loc[::-1] # 가져온 데이터 반대로 뒤집기
                rt_data.index = rt_data.index[::-1] # 인덱스 재정렬
                #print(rt_data)
                break
            except AttributeError:
                print("wait..")
                continue
            
        inqr_end_dt = rt_data.iat[0,0]
        while rt_data.shape[0] < count:# 만약 가져온 데이터가 부족하면
            try:

                # 마지막 일자 다시 계산
                temp_end_dt = self.datetime_sub(inqr_end_dt,1) 
            
                # 추가로 가져온데이터 추가하기
                extra_data = kb.get_daily_inquire_investor(inqr_strt_dt=temp_end_dt).loc[::-1]
                rt_data = pd.concat([extra_data, rt_data], ignore_index=True)

                inqr_end_dt = rt_data.iat[0,0]
                
            except AttributeError:
                print("wait..")
                continue
        
        if rt_data.shape[0] > count:
            rt_data = rt_data.loc[:count-1] # 필요한 데이터 개수만큼 슬라이싱
        return rt_data
    
    def get_daily_index_price(self, itm_no="0001", inqr_strt_dt=None,count=30):
        """
        [국내주식] 업종/기타 > 국내업종 일자별지수
        한 번의 조회에 100건까지 확인 가능합니다.
        코스피(0001), 코스닥(1001), 코스피200(2001)
        """
        rt_data = pd.DataFrame()

        while True:
            try:# 데이터 가져오기
                rt_data = kb.get_inquire_index_daily_price(itm_no=itm_no, inqr_strt_dt=inqr_strt_dt).loc[::-1] # 가져온 데이터 반대로 뒤집기
                rt_data.index = rt_data.index[::-1] # 인덱스 재정렬
                #print(rt_data)
                break
            except AttributeError:
                print("wait..")
                continue

        inqr_end_dt = rt_data.iat[0,0]
        while rt_data.shape[0] < count:# 만약 가져온 데이터가 부족하면
            try:
                # 마지막 일자 다시 계산
                temp_end_dt = self.datetime_sub(inqr_end_dt,1) 
                
                # 추가로 가져온데이터 추가하기
                extra_data = kb.get_inquire_index_daily_price(itm_no=itm_no, inqr_strt_dt=temp_end_dt).loc[::-1]
                rt_data = pd.concat([extra_data, rt_data], ignore_index=True)

                inqr_end_dt = rt_data.iat[0,0]
                
            except AttributeError:
                print("wait..")
                continue
        
        if rt_data.shape[0] > count:
            rt_data = rt_data.loc[rt_data.shape[0] - count:] # 필요한 데이터 개수만큼 슬라이싱
            rt_data.index = np.arange(count)
        return rt_data

    def get_daily_chartprice(self, itm_no="COMP", inqr_strt_dt=None, count=30):
        '''
        [해외주식] 기본시세 > 해외주식 종목/지수/환율기간별시세(일/주/월/년)
        해당 API로 미국주식 조회 시, 다우30, 나스닥100, S&P500 종목만 조회 가능합니다.
        종목코드 다우30(.DJI), 나스닥100(COMP), S&P500(SPX)
        '''
        rt_data = pd.DataFrame()
        
        inqr_end_dt = self.datetime_sub(inqr_strt_dt, count) # 마지막 일자 계산
        
        while True:
            try:# 데이터 가져오기
                rt_data = kb.get_inquire_daily_chartprice(itm_no=itm_no, inqr_strt_dt=inqr_end_dt, inqr_end_dt=inqr_strt_dt).loc[::-1]
                break
            except AttributeError:
                print("wait..")
                continue

        inqr_strt_dt = rt_data.iat[0,0]
        while rt_data.shape[0] < count:# 만약 가져온 데이터가 부족하면
            try:
                # 마지막 일자 다시 계산
                temp_strt_dt = self.datetime_sub(inqr_strt_dt,1) 
                inqr_end_dt = self.datetime_sub(temp_strt_dt, count // 2) 
                
                # 추가로 가져온데이터 추가하기
                extra_data = kb.get_inquire_daily_chartprice(itm_no=itm_no, inqr_strt_dt=inqr_end_dt, inqr_end_dt=temp_strt_dt).loc[::-1]

                rt_data = pd.concat([extra_data, rt_data], ignore_index=True)

                inqr_strt_dt = rt_data.iat[0,0]
                
            except AttributeError:
                print("wait..")
                continue
        
        if rt_data.shape[0] > count:
            rt_data = rt_data.loc[rt_data.shape[0] - count:] # 필요한 데이터 개수만큼 슬라이싱
            rt_data.index = np.arange(count)
        return rt_data

    
if __name__ == '__main__':
    s = Stock()
    stock_data = s.get_daily_stock_info(count=30, inqr_strt_dt="20190226")

    investor_data = s.get_daily_investor(count=30, inqr_strt_dt="20190226")
    kospi_data = s.get_daily_index_price(itm_no="0001", count=30, inqr_strt_dt="20190226")
    kosdaq_data = s.get_daily_index_price(itm_no="1001", count=30, inqr_strt_dt="20190226")
    nasdaq_data = s.get_daily_chartprice(itm_no="COMP", count=30, inqr_strt_dt="20190226")
    spx_data = s.get_daily_chartprice(itm_no="SPX", count=30, inqr_strt_dt="20190226")
    

    print("======================주식정보===============================")
    print(stock_data)
    print("======================투자자정보==============================")
    print(investor_data)
    print("======================코스피================================")
    print(kospi_data)
    print("======================코스닥================================")
    print(kosdaq_data)
    print("======================나스닥================================")
    print(nasdaq_data)
    print("======================S&P500================================")
    print(spx_data)