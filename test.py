# 초기 표준편차와 최종 표준편차
initial_std = 0.6
final_std = 0.1

# 전체 에포크 및 감소 주기
total_epochs = 40000
step_interval = 5000  # 50만 에포크마다 표준편차 감소

# 주기에 따른 표준편차 감소량 계산
num_steps = total_epochs // step_interval  # 감소 단계 수
std_decrement = (initial_std - final_std) / num_steps  # 단계별 감소량

# 현재 에포크에서 표준편차 계산 함수
def get_stepwise_std(current_epoch, action_std_decay_freq, initial_std, action_std_decay_rate, min_action_std):
    current_step = current_epoch // action_std_decay_freq
    action_std = initial_std - current_step * action_std_decay_rate

    # action_std가 최소값보다 작아지면 최소값으로 설정
    if (action_std <= min_action_std):
        action_std = min_action_std
        print("setting actor output action_std to min_action_std : ", action_std)
    else:
        print("setting actor output action_std to : ", action_std)
    
    return action_std



# 에포크 별 표준편차 출력 예시
for epoch in range(0, total_epochs + 1):  # 샘플로 일부 에포크만 출력
    if epoch % step_interval == 0:
        current_std = get_stepwise_std(epoch, step_interval, initial_std, std_decrement, final_std)
        print(f"Epoch {epoch}: Std = {current_std:.4f}")