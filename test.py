import numpy as np

def distance_based_reward_bounded(reward, alpha=5.0):
    exp_reward = 1 - np.exp(-alpha * (reward ** 2))
    if reward < 0: # 오차가 커질수록 -1에 수렴
        return -exp_reward
    elif reward > 0: # 오차가 커질수록 1에 수렴
        return exp_reward
    else: # 보상은 0
        return 0

# 예시: 오차가 0일 때와 1일 때의 보상 계산
reward_0 = distance_based_reward_bounded(0)
reward_1 = distance_based_reward_bounded(1)
reward_2 = distance_based_reward_bounded(-1)

print(f"Reward when distance = 0: {reward_0}")  # 0
print(f"Reward when distance = 1: {reward_1}")  
print(f"Reward when distance = -1: {reward_2}")  
