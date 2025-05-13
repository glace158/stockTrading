from pyts.image import GramianAngularField
import numpy as np
import matplotlib.pyplot as plt # 시각화용
from stock.stock_adaptor import DailyStockAdaptor
np.set_printoptions(suppress=True)
filter = ["stck_clpr","stck_hgpr","stck_lwpr","acml_vol","prdy_vrss",'5','20','60',"rsi","bb_upper","bb_lower"]
a = DailyStockAdaptor(filter)
df = a.load_datas("API\datas\ " + "098460" + ".csv", "20181208", 30, 1000000)

# 예시: 20일간의 종가 데이터 (정규화 필요할 수 있음)\
for d in df.values:
    print(d)

time_series = df.values
# 데이터를 0~1 또는 -1~1 사이로 스케일링하는 것이 좋음 (Min-Max Scaling)
time_series_scaled = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series) + 1e-5) # 0~1 스케일링

for d in time_series_scaled:
    print(d)

image_size = len(df.values) # 시계열 길이와 동일하게 (또는 더 작게) 설정 가능
gasf = GramianAngularField(image_size=image_size, method='summation')
gasf_image = gasf.fit_transform(time_series_scaled.reshape(1, -1)) # (1, image_size, image_size)


# (1, H, W) -> (H, W) 또는 (1, H, W) 그대로 사용 (채널 1개)
single_channel_image = gasf_image[0]
print(f"GASF 이미지 shape: {single_channel_image.shape}, dtype: {single_channel_image.dtype}")

# 시각화 (옵션)
plt.imshow(single_channel_image, cmap='rainbow', origin='lower')
plt.title("GASF Image")
plt.show()

# CNN 입력으로 사용 시 채널 차원 추가: (H, W) -> (1, H, W)
cnn_input_image = single_channel_image[np.newaxis, :, :]
print(f"CNN 입력용 GASF 이미지 shape: {cnn_input_image.shape}")