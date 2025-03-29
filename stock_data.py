import API.kis_auth as ka
import API.kis_domstk as kb
import datetime
import time
import pandas as pd
import numpy as np
    
class Stock:
    def __init__(self, svr="vps"):
        ka.auth(svr)  # 토큰발급
        self.delay_time = 1

    def datetime_add(self, inqr_strt_dt, days):  # 날짜 더하기
        return (datetime.datetime.strptime(inqr_strt_dt, "%Y%m%d") + datetime.timedelta(days=days)).strftime("%Y%m%d")
    
    def datetime_sub(self, inqr_strt_dt, days):  # 날짜 빼기
        return (datetime.datetime.strptime(inqr_strt_dt, "%Y%m%d") - datetime.timedelta(days=days)).strftime("%Y%m%d")

    def _fetch_data(self, fetch_function, **kwargs):
        """
        데이터 가져오기
        """
        rt_data = pd.DataFrame()
        while True:
            try:
                rt_data = fetch_function(**kwargs).loc[::-1]
                rt_data.index = np.arange(len(rt_data.index)) # 인덱스 재정렬
                break
            except AttributeError:
                print("wait..")
                time.sleep(self.delay_time)
                continue
        return rt_data
    

    def _fetch_complete_data(self, fetch_function, inqr_strt_dt, count, **kwargs):
        """
        데이터 부족 시 데이터 채우기
        """
        rt_data = self._fetch_data(fetch_function, **kwargs, inqr_strt_dt=inqr_strt_dt)
        inqr_end_dt = rt_data.iat[0, 0]
        
        while rt_data.shape[0] < count:
            try:
                if kwargs.get("inqr_end_dt"):
                    kwargs["inqr_end_dt"] = self.datetime_sub(inqr_end_dt,1) 
                    temp_end_dt = self.datetime_sub(kwargs["inqr_end_dt"], count // 2) 
                else:
                    temp_end_dt = self.datetime_sub(inqr_end_dt, 1)


                extra_data = fetch_function(**kwargs, inqr_strt_dt=temp_end_dt).loc[::-1]
                
                rt_data = pd.concat([extra_data, rt_data], ignore_index=True)
                
                inqr_end_dt = rt_data.iat[0, 0]
            except AttributeError:
                print("wait..")
                time.sleep(self.delay_time)
                continue

        if rt_data.shape[0] > count:
            rt_data = rt_data.loc[rt_data.shape[0] - count:]
            rt_data.index = np.arange(len(rt_data.index))
        return rt_data

    def get_daily_stock_info(self, itm_no="005930", inqr_strt_dt=None, count=30):
        '''
        주식 정보
        [국내주식] 기본시세 > 국내주식기간별시세(일/주/월/년)
        실전계좌/모의계좌의 경우, 한 번의 호출에 최대 100건까지 확인 가능합니다.
        '''

        inqr_end_dt = inqr_strt_dt
        inqr_strt_dt = self.datetime_sub(inqr_strt_dt, count) # 마지막 일자 계산

        return self._fetch_complete_data(
            fetch_function=kb.get_inquire_daily_itemchartprice,
            itm_no=itm_no,
            output_dv="2",
            count=count,
            inqr_strt_dt=inqr_strt_dt,
            inqr_end_dt=inqr_end_dt
        )[["stck_bsop_date", "stck_clpr", "stck_hgpr", "stck_lwpr", "acml_vol", "prdy_vrss"]] # 필요한 정보 필터링 per, eps, pbr


    def get_daily_investor(self, inqr_strt_dt=None, count=30):
        """
        투자자 동향
        [국내주식] 시세분석 > 시장별 투자자매매동향 (일별)
        """

        return self._fetch_complete_data(
            fetch_function=kb.get_daily_inquire_investor,
            count=count,
            inqr_strt_dt=inqr_strt_dt
        )[["stck_bsop_date", "frgn_ntby_qty", "prsn_ntby_qty", "orgn_ntby_qty"]] # 필요한 정보 필터링

    def get_daily_index_price(self, itm_no="0001", inqr_strt_dt=None, count=30):
        """
        코스피, 코스닥
        [국내주식] 업종/기타 > 국내업종 일자별지수
        한 번의 조회에 100건까지 확인 가능합니다.
        코스피(0001), 코스닥(1001), 코스피200(2001)
        """
        return self._fetch_complete_data(
            fetch_function=kb.get_inquire_index_daily_price,
            itm_no=itm_no,
            count=count,
            inqr_strt_dt=inqr_strt_dt
        )[["stck_bsop_date", "bstp_nmix_prpr", "bstp_nmix_hgpr", "bstp_nmix_lwpr", "bstp_nmix_prdy_ctrt", "acml_vol", "acml_tr_pbmn"]] # 필요한 정보 필터링

    def get_daily_chartprice(self, itm_no="COMP", inqr_strt_dt=None, count=30):
        '''
        나스닥, S&P
        [해외주식] 기본시세 > 해외주식 종목/지수/환율기간별시세(일/주/월/년)
        해당 API로 미국주식 조회 시, 다우30, 나스닥100, S&P500 종목만 조회 가능합니다.
        종목코드 다우30(.DJI), 나스닥100(COMP), S&P500(SPX)
        '''
        inqr_end_dt = inqr_strt_dt
        inqr_strt_dt = self.datetime_sub(inqr_strt_dt, count) # 마지막 일자 계산

        return self._fetch_complete_data(
            fetch_function=kb.get_inquire_daily_chartprice,
            itm_no=itm_no,
            count=count,
            inqr_strt_dt=inqr_strt_dt,
            inqr_end_dt=inqr_end_dt
        )[["stck_bsop_date", "ovrs_nmix_prpr", "ovrs_nmix_hgpr", "ovrs_nmix_lwpr"]]# 필요한 정보 필터링, "acml_vol"

    def get_moving_average_line(self, stock_data : pd.DataFrame = None, count=0, moving_days = [5,20,60]):
        '''
        이동 평균선 데이터 구하기
        '''
        rt_data = stock_data[["stck_clpr"]]

        moving_average_data = pd.DataFrame(columns=moving_days) # 이동평균선 저장할 데이터프레임 생성
        
        strt_index = stock_data.shape[0] - count # 시작 인덱스

        for i in range(count): # 이동평균선 구하기 작업
            index = strt_index + i # 현재 인덱스
            
            temp_list = []
            for days in moving_days: # 각 일별로 평균구하기 5일, 20일 60일
                datas = rt_data.loc[index - days + 1 : index].astype(float) # 일별로 데이터 슬라이싱
                temp_list.append(datas.mean()[0]) # 평균 구하기
            
            moving_average_data.loc[i] = temp_list # 데이터 넣기
        
        # 데이터프레임 앞에 날짜 넣어주기
        moving_average_data.insert(loc=0, column='stck_bsop_date', value=stock_data["stck_bsop_date"].values[stock_data.shape[0] - count:])

        return moving_average_data
        
    def get_rsi(self, stock_data : pd.DataFrame = None, count=0, days=14):
        '''
        RSI 구하기
        '''
        rt_data = stock_data[["stck_clpr"]]

        rsi_data = pd.DataFrame(columns=["rsi"]) # RSI 저장할 데이터프레임 생성

        strt_index = stock_data.shape[0] - count # 시작 인덱스

        for i in range(count): # RSI 구하기 작업
            index = strt_index + i # 현재 인덱스
            
            u = []
            d = []

            for j in range(days):
                today_clpr = rt_data.loc[index - j].astype(float)[0] # 당일 종가
                pre_clpr = rt_data.loc[(index - 1) - j].astype(float)[0] # 전일 종가

                difference = today_clpr - pre_clpr # 당일과 전일의 차이 계산
                u.append(difference if difference > 0 else 0) # 전일대비 상승했으면 
                d.append(-difference if difference < 0 else 0) # 전일대비 하강했으면
            
            # 평균구하기
            a_u = np.mean(u)
            a_d = np.mean(d)
            
            if a_d != 0: 
                rs = a_u / a_d # RS 구하기
            else:# 0으로 나누는 문제 방지
                rs = 0

            rsi = rs / (1 + rs) * 100 # RSI 구하기
            rsi_data.loc[i] = rsi # 구한 RSI 넣기

        # 데이터프레임 앞에 날짜 넣어주기
        rsi_data.insert(loc=0, column='stck_bsop_date', value=stock_data["stck_bsop_date"].values[stock_data.shape[0] - count:])
        
        return rsi_data
    
    def get_bollinger_band(self, stock_data : pd.DataFrame = None, moving_average_line : pd.DataFrame = None ,count=0, days=20, k=2 ):
        '''
        볼린저 밴드 구하기
        '''
        
        bb_middle = moving_average_line[[days]].astype(float)
        rt_data = stock_data[["stck_clpr"]]
        bb_data = pd.DataFrame(columns=["bb_upper", "bb_middle", "bb_lower"]) # bollinger_band 저장할 데이터프레임 생성
        
        strt_index = stock_data.shape[0] - count # 시작 인덱스

        for i in range(count):
            index = strt_index + i # 현재 인덱스
                
            datas = rt_data.loc[index - days + 1 : index].astype(float) # 일별로 데이터 슬라이싱
            std_data = np.std(datas)[0] # 표준편차 구하기
            
            middle = bb_middle.loc[i].values[0] # 중단 밴드
            bb_upper = middle + std_data * k # 상단 밴드
            bb_lower = middle - std_data * k # 하단 밴드

            bb_data.loc[i] = [bb_upper, middle, bb_lower] # 밴드 추가
        
        # 데이터프레임 앞에 날짜 넣어주기
        bb_data.insert(loc=0, column='stck_bsop_date', value=stock_data["stck_bsop_date"].values[stock_data.shape[0] - count:])
        return bb_data
    
if __name__ == '__main__':
    s = Stock()
    count = 30
    moving_days = [5,20,60]
    bb_days = 20
    inqr_strt_dt = "20190226"

    start = time.time()
    stock_data = s.get_daily_stock_info(itm_no="005930", count=count+ max(moving_days), inqr_strt_dt=inqr_strt_dt)
    print("======================주식정보===============================")
    print(stock_data)
    #"""
    move_line_data = s.get_moving_average_line(stock_data=stock_data, count=count, moving_days=moving_days)
    print("======================이동평균선===============================")
    print(move_line_data)

    rsi_data = s.get_rsi(stock_data=stock_data, count=count,days=14)
    print("======================RSI===============================")
    print(rsi_data)
    
    print("======================BollingerBand===============================")
    bb_data = s.get_bollinger_band(stock_data=stock_data, moving_average_line=move_line_data, count=count, days=bb_days, k=2)
    print(bb_data)
    
    #investor_data = s.get_daily_investor(count=count, inqr_strt_dt="20190226")
    #print("======================투자자정보==============================")
    #print(investor_data)
    
    kospi_data = s.get_daily_index_price(itm_no="0001", count=count, inqr_strt_dt=inqr_strt_dt)
    print("======================코스피================================")
    print(kospi_data)

    kosdaq_data = s.get_daily_index_price(itm_no="1001", count=count, inqr_strt_dt=inqr_strt_dt)
    print("======================코스닥================================")
    print(kosdaq_data)

    nasdaq_data = s.get_daily_chartprice(itm_no="COMP", count=count+5, inqr_strt_dt=inqr_strt_dt)
    print("======================나스닥================================")
    print(nasdaq_data)

    spx_data = s.get_daily_chartprice(itm_no="SPX", count=count+5, inqr_strt_dt=inqr_strt_dt)
    print("======================S&P500================================")
    print(spx_data)
    #"""
    end = time.time()
    print(f"{end - start:.5f} sec")
