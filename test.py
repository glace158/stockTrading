from stock.stock_adaptor import DailyStockAdaptor
import datetime

min_dt="20190101" 
max_dt="20250131"
days = (datetime.datetime.strptime(max_dt, "%Y%m%d")) - (datetime.datetime.strptime(min_dt, "%Y%m%d"))
print(days)
stock_codes = [ "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030" ] # "005930","000660", "083650", "010120", "035720", "012450", "098460", "064350", "056080", "068270", "039030"
filter = ["stck_clpr","stck_hgpr","stck_lwpr","acml_vol","prdy_vrss",'5','20','60',"rsi","bb_upper","bb_lower"]
a = DailyStockAdaptor(filter)
print(a.load_datas("API\datas\ " + "098460" + ".csv", "20190208", 30))

for i in range(30):
    print("================================")
    #print(a.sortino_ratio())
    print(a.get_info('',0.0))