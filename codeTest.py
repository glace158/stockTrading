from fileManager import Config, File
from datetime import datetime
import os

start_time = datetime.now()

print("현재 시간 (문자열):", start_time.strftime("%Y%m%d_%H%M%S"))
path = str(os.path.dirname(__file__)) + "/" 
#file = File(path + "PPO_logs/", start_time.strftime("%Y%m%d_%H%M%S"))
print(next(os.walk(path + "PPO_logs/" + "MountainCarContinuous-v0"))[2])