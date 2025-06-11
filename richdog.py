import os
import sys
import time
from datetime import datetime
import abc

import torch
import numpy as np
import random

from PPO.PPO2 import PPO
from PPO.environment import GymEnvironment, StockEnvironment
from common.fileManager import Config, File
from common.logger import Logger, DataRecorder

class RichDog:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, config_path=None):
        print("============================================================================================")
        if not config_path:
            config_path = "config/Hyperparameters.yaml"

        self.config = Config.load_config(config_path)
        self.stock_config = Config.load_config("config/StockConfig.yaml")
        self.cur_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.data_recorder: DataRecorder = None
        self.random_search = int(self.config.random_search.value) # 랜덤 파라미터 학습


    def init_parameters(self):
        print("============================================================================================")
        ####### initialize environment hyperparameters ######
        self.env_name = self.config.env_name.value

        self.has_continuous_action_space = bool(self.config.has_continuous_action_space.value)  # continuous action space; else discrete

        self.max_ep_len = int(self.config.max_ep_len.value)      # max timesteps in one episode (에피소드 당 최대 타임 스텝)
        self.max_training_timesteps = int(self.config.max_training_timesteps.value)   # break training loop if timeteps > max_training_timesteps (총 학습 타임스텝)
    
        self.print_freq = int(self.config.print_freq.value)        # print avg reward in the interval (in num timesteps) (출력 주기)
        self.log_freq = int(self.config.log_freq.value)            # log avg reward in the interval (in num timesteps) (로그 파일 생성 주기)
        self.save_model_freq = int(self.config.save_model_freq.value)          # save model frequency (in num timesteps) (모델 저장 주기)

        self.action_std = self._get_random_or_fixed_value(self.config.action_std, float)                # starting std for action distribution (Multivariate Normal) (행동 표준 편차)
        self.min_action_std = self._get_random_or_fixed_value(self.config.min_action_std, float)                # minimum action_std (stop decay after action_std <= min_action_std) (0.05 ~ 0.1) (최소 행동 표준 편차 값)
        
        # min_action_std가 action_std보다 크지 않도록 보정
        if self.action_std < self.min_action_std:
            self.min_action_std = self.action_std

        self.action_std_decay_freq = self._get_random_or_fixed_value(self.config.action_std_decay_freq, int)  # action_std decay frequency (in num timesteps) (표준 편차 감소 주기)
        self.action_std_decay_rate = (self.action_std - self.min_action_std) / ((self.max_training_timesteps - self.action_std_decay_freq)  // self.action_std_decay_freq)     # linearly decay action_std (action_std = action_std - action_std_decay_rate) (행동 표준 편차 감소 값)
        self.action_std_method = self.config.action_std_method.value
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        self.update_timestep = self._get_random_or_fixed_value(self.config.update_timestep, int)      # update policy every n timesteps (정책 업데이트 주기)
        self.K_epochs = self._get_random_or_fixed_value(self.config.K_epochs, int)               # update policy for K epochs in one PPO update (최적화 횟수)

        self.eps_clip = self._get_random_or_fixed_value(self.config.eps_clip, float)          # clip parameter for PPO (클리핑)
        self.gamma = self._get_random_or_fixed_value(self.config.gamma, float)            # discount factor (감가율)
        self.lamda = self._get_random_or_fixed_value(self.config.lamda, float)              # 어드벤티지 감가율
        self.minibatchsize = self._get_random_or_fixed_value(self.config.minibatchsize, int)

        self.lr_actor = self._get_random_or_fixed_value(self.config.lr_actor, float)       # learning rate for actor network (액터의 학습률)
        self.lr_critic = self._get_random_or_fixed_value(self.config.lr_critic, float)      # learning rate for critic network (크리틱 학습률)

        self.value_loss_coef = self._get_random_or_fixed_value(self.config.value_loss_coef, float)     # 가치 손실 계수
        self.entropy_coef = self._get_random_or_fixed_value(self.config.entropy_coef, float)       # 엔트로피 계수

        self.random_seed = int(self.config.random_seed.value)         # set random seed if required (0 = no random seed) (랜덤 시드)
        
        self.cnn_features_dim = int(self.config.cnn_features_dim.value)
        self.mlp_features_dim = int(self.config.mlp_features_dim.value)
        #####################################################

        if self.env_name == "RichDog":
            self.env = StockEnvironment(self.stock_config)
        else:
            self.env = GymEnvironment(env_name=self.env_name)

        # state space dimension
        self.observation_space = self.env.getObservation()
        # action space dimension
        self.action_space = self.env.getActon()


    def print_parameters(self):
        ############# print all hyperparameters #############
        print("training environment name : " + self.env_name)
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_ep_len)
        print("model saving frequency : " + str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", self.observation_space)
        print("action space dimension : ", self.action_space)
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", self.action_std)
            print("decay rate of std of action distribution : ", self.action_std_decay_rate)
            print("minimum std of action distribution : ", self.min_action_std)
            print("decay frequency of std of action distribution : " + str(self.action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(self.update_timestep) + " timesteps")
        print("PPO K epochs : ", self.K_epochs)
        print("PPO epsilon clip : ", self.eps_clip)
        print("discount factor (gamma) : ", self.gamma)
        print("GAE discount factor (lamda) : ", self.lamda)
        print("mini batch size : ", self.minibatchsize)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
        if self.random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", self.random_seed)
            torch.manual_seed(self.random_seed)
            self.env.seed(self.random_seed)
            np.random.seed(self.random_seed)
        #####################################################

        print("============================================================================================")

    def capture_print_parameters(self) -> str:
        """print_parameters 메서드의 출력을 문자열로 캡처하여 로깅에 사용합니다."""
        import io
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        self.print_parameters() # 이 메서드의 출력이 captured_output으로 들어감
        sys.stdout = old_stdout # 표준 출력 복원
        return captured_output.getvalue()
    
    def _get_random_or_fixed_value(self, param_obj, target_type):
        """
        Config에서 읽어온 파라미터 객체(min, max, value)로 랜덤 값 또는 고정 값을 반환
        """

        min_val_str = getattr(param_obj, 'min', None) # 최대 값
        max_val_str = getattr(param_obj, 'max', None) # 최소 값
        default_val_str = getattr(param_obj, 'value', None) # 기본 값

        # min 또는 max 값이 None이 아니고 유효한 문자열일 때 랜덤 값 생성 시도
        if min_val_str is not None and max_val_str is not None and self.random_search > 0:
            try:
                if target_type == int:
                    min_v = int(min_val_str)
                    max_v = int(max_val_str)

                    if min_v > max_v:
                        print(f"경고: 파라미터 '{param_obj.name}'의 min ({min_v})이 max ({max_v})보다 큽니다. 값을 교체합니다.")
                        min_v, max_v = max_v, min_v

                    return random.randint(min_v, max_v)
                
                elif target_type == float:
                    min_v = float(min_val_str)
                    max_v = float(max_val_str)

                    if min_v > max_v:
                        print(f"경고: 파라미터 '{param_obj.name}'의 min ({min_v})이 max ({max_v})보다 큽니다. 값을 교체합니다.")
                        min_v, max_v = max_v, min_v

                    return random.uniform(min_v, max_v)
                
            except ValueError:
                print(f"경고: 파라미터 '{param_obj.name}'의 min/max 값을 {target_type}으로 변환 중 오류 발생. 기본값을 사용합니다.")
                # 오류 발생 시 기본값 사용 로직으로 넘어감
        
        # 랜덤 값 생성이 안됐거나, min/max가 설정되지 않은 경우 기본값 사용
        if default_val_str is None:
            raise ValueError(f"파라미터 '{param_obj.name}' 기본값(value)이 없습니다.")

        if target_type == int:
            return int(default_val_str)
        elif target_type == float:
            return float(default_val_str)
        elif target_type == bool:
            return bool(default_val_str)
        else: 
            return str(default_val_str)
    
class RichDogTrain(RichDog):
    def __init__(self, config_path=None, checkpoint_path=""):
        super().__init__(config_path)
        self.checkpoint_path = checkpoint_path
        
        # DataRecorder의 random_seed는 경로 일관성을 위해 설정 파일의 것을 사용하는 것이 좋음
        self.data_recorder = DataRecorder(
            env_name=self.config.env_name.value, 
            random_seed=int(self.config.random_seed.value), # 설정된 시드를 경로 일관성을 위해 사용
            base_log_dir_prefix="PPO_logs" # 훈련 로그를 위한 최상위 디렉토리 (선호에 따라 변경 가능)
        )
        
        # DataRecorder를 사용하여 설정 파일 저장
        self.data_recorder.setup_config_logging(self.config, self.stock_config)
    
    def init_files(self):
        # 메인 훈련 로그 설정
        self.data_recorder.setup_training_log()

        # 체크포인트 경로 설정
        # 체크포인트 이름 일관성을 위해 설정 파일의 random_seed 전달
        self.data_recorder.setup_checkpointing(trained_random_seed=int(self.config.random_seed.value))
        self.data_recorder.setup_console_log(file_name="PPO_console_log.txt")

    ################################### Training ###################################
    def random_train(self):
        if self.random_search > 0:
            for i in range(self.random_search):
                self.data_recorder.log_to_console(f"Random Traing Count : {i}")
                self.data_recorder.log_to_console("============================================================================================\n")
                self.train()
        else:
            self.train()

    def train(self):
        self.init_parameters()
        self.print_parameters()
        self.init_files()
        # print_parameters 출력을 문자열로 캡처하여 콘솔 로그에 기록
        param_str = self.capture_print_parameters()
        self.data_recorder.log_to_console(param_str)
        
        ################# training procedure ################
        # initialize a PPO agent
        ppo_agent = PPO(self.observation_space, self.action_space, self.lr_actor, self.lr_critic, 
                        self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                        self.action_std, self.value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize,
                        self.cnn_features_dim, self.mlp_features_dim
                        )


        # 만약 모델 데이터 경로가 있으면
        if self.checkpoint_path != "":
            self.data_recorder.log_to_console(f"loading network from : {self.checkpoint_path}\n")
            ppo_agent.load(self.checkpoint_path)
            self.data_recorder.log_to_console("Network load complete.\n--------------------------------------------------------------------------------------------\n")


        # track total training time
        start_time = datetime.now().replace(microsecond=0)

        self.data_recorder.log_to_console("Started training at (GMT) : " + str(start_time) + "\n")
        self.data_recorder.log_to_console("============================================================================================\n")
        
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        loss = 0
        dist_entropy = 0
        policy_loss = 0
        value_loss = 0

        is_save_model = False
        # training loop
        while time_step <= self.max_training_timesteps:

            state, info = self.env.reset()

            current_ep_reward = 0

            is_action_log = is_save_model

            if is_action_log:
                self.data_recorder.log_to_console("Start episode logging\n")
                action_labels = ['timestep', "action", "reward"] + (list(info.keys()) if info else [])
                state_labels = [f"state_{i}" for i in range(ppo_agent.policy.input_dim)]
                
                self.data_recorder.setup_run_data_logs(
                    mode_dir_suffix="train_episode_data", # 훈련 중 에피소드 상세 데이터 저장용 하위폴더
                    run_id=f"{time_step}", # 이 특정 실행 데이터 로그를 위한 ID (예: "ts10000")
                    action_labels=action_labels,
                    state_labels=state_labels
                )

            for t in range(1, self.max_ep_len+1):
                # select action with policy
                action, action_logprob, state_val = ppo_agent.select_action(state)
                
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated # truncated도 에피소드 종료로 간주

                # saving buffer
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                
                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    loss,policy_loss,value_loss,dist_entropy = ppo_agent.update()
                    # 서서히 액션 분포의 표준편차 감소
                    if self.action_std_method == "schedule":
                        ppo_agent.schedule_action_std(self.min_action_std, self.action_std, time_step, self.max_training_timesteps)

                # if continuous action space; then decay action std of ouput action distribution (액션 분포의 표준편차 감소)
                
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0 and self.action_std_method == "freq":
                    ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0 and time_step > 0:
                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = np.round(log_avg_reward, 4)

                    self.data_recorder.log_to_training_file([i_episode, time_step, log_avg_reward, loss, policy_loss, value_loss, dist_entropy])

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = np.round(print_avg_reward, 2)

                    self.data_recorder.log_to_console("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}\n".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 0:
                    chk_path = self.data_recorder.get_checkpoint_path()
                    self.data_recorder.log_to_console("--------------------------------------------------------------------------------------------\n")
                    self.data_recorder.log_to_console("saving model at : " + chk_path + "\n")
                    ppo_agent.save(chk_path)
                    self.data_recorder.log_to_console("model saved")
                    self.data_recorder.log_to_console("Elapsed Time  : " + str(datetime.now().replace(microsecond=0) - start_time) + "\n")
                    self.data_recorder.log_to_console("--------------------------------------------------------------------------------------------\n")
                    self.data_recorder.log_to_console("std : {}\n".format(ppo_agent.action_std))
                    self.data_recorder.log_to_console("--------------------------------------------------------------------------------------------\n")
                    
                    is_save_model = True

                if is_action_log:
                    # 에피소드의 스텝당 행동값 저장
                    action_value = action.item() if not self.has_continuous_action_space and hasattr(action, 'item') else (action[0] if self.has_continuous_action_space else action)
                    action_data_to_log = [t, action_value, reward] 
                    if info: 
                        action_data_to_log.extend(list(info.values()))
                    self.data_recorder.log_to_action_file(action_data_to_log)
                    
                    # 에피소드의 스텝당 상태값 저장
                    state_data_to_log = state.tolist() if isinstance(state, np.ndarray) else [state] # 리스트 형태 보장
                    self.data_recorder.log_to_state_file(state_data_to_log)
                    
                    is_save_model = False

                state = next_state

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1
            

        self.env.close()

        # print total training time
        end_time = datetime.now().replace(microsecond=0)
        final_console_messages = [
            "============================================================================================\n",
            f"Started training at (GMT) : {start_time}\n",
            f"Finished training at (GMT) : {end_time}\n",
            f"Total training time  : {end_time - start_time}\n",
            "============================================================================================\n"
        ]
        
        for msg in final_console_messages:
            self.data_recorder.log_to_console(msg) # 화면에도 출력

        self.data_recorder.close_all() # 모든 로그 파일 닫기

class RichDogTest(RichDog):
    def __init__(self, config_path=None, checkpoint_path=""):
        super().__init__(config_path)
        self.checkpoint_path = checkpoint_path

        try:
            filename = os.path.basename(self.checkpoint_path)
            # 파일명 형식: PPO_{env_name}_{random_seed}_{cur_time}.pth
            parts = filename.split('_') 
            model_env_name = parts[1]
            # model_random_seed = parts[2] # 필요하다면 사용 가능
            self.model_creation_time = parts[3].split('.')[0] # .pth 확장자 제거
        except IndexError:
            print(f"오류: 체크포인트 파일명을 파싱할 수 없습니다: {self.checkpoint_path}")
            print("예상 형식: PPO_환경이름_랜덤시드_타임스탬프.pth")
            sys.exit(1)
            
        model_config_dir_base = os.path.join("PPO_logs", model_env_name, self.model_creation_time, "config")
        
        print(f"Model config load path: {model_config_dir_base}")
        
        # 테스트용 DataRecorder 초기화, 로그는 테스트별 디렉토리에 저장됨
        self.data_recorder = DataRecorder(
            env_name=model_env_name, # 로드된 설정의 env_name 사용
            random_seed=int(datetime.now().timestamp()), # 테스트 로그에는 새로운 시드/ID 사용
            base_log_dir_prefix="PPO_logs",
            cur_time= self.model_creation_time
        )
        self.cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_recorder.setup_console_log(file_name="PPO_console_log.txt")
        
    #################################### Testing ###################################
    def test(self):
        print("============================================================================================")

        ################## hyperparameters ##################
        self.init_parameters()
        render = bool(self.config.render.value)              # render environment on screen
        frame_delay = float(self.config.frame_delay.value)   # if required; add delay b/w frames

        self.action_std = float(self.config.min_action_std.value)
        total_test_episodes = int(self.config.total_test_episodes.value)    # total num of testing episodes
        self.print_parameters()
        # print_parameters 출력을 문자열로 캡처하여 콘솔 로그에 기록
        param_str = self.capture_print_parameters()
        self.data_recorder.log_to_console(param_str)
        #####################################################
        
        # initialize a PPO agent
        ppo_agent = PPO(self.observation_space, self.action_space, self.lr_actor, self.lr_critic, 
                    self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                    self.action_std,self. value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize,
                    self.cnn_features_dim, self.mlp_features_dim
                    )
        
        # preTrained weights directory

        if self.checkpoint_path == "":
            self.data_recorder.log_to_console("Error : Not Setting Model path\n" )
            return


        self.data_recorder.log_to_console(f"loading network from : {self.checkpoint_path}\n")
        ppo_agent.load(self.checkpoint_path)
        self.data_recorder.log_to_console("Network load complete.\n--------------------------------------------------------------------------------------------\n")

        test_running_reward = 0

        for ep in range(1, total_test_episodes+1):
            
            ep_reward = 0
            state, info = self.env.reset()
 
            # 각 테스트 에피소드별로 상세 행동/상태 로그 설정
            action_labels = ['timestep', "action", "reward"] + (list(info.keys()) if info else [])
            state_labels = self.env.get_data_label() if hasattr(self.env, 'get_data_label') else [f"state_{i}" for i in range(ppo_agent.policy.input_dim)]
            
            self.data_recorder.setup_run_data_logs(
                mode_dir_suffix="test_episode_data" + "/" + self.cur_time, # 테스트 중 에피소드 상세 데이터 저장용 하위폴더
                run_id=f"ep{ep}", # 이 특정 테스트 에피소드 로그를 위한 ID (예: "ep1")
                action_labels=action_labels,
                state_labels=state_labels
            )
            
            for t in range(1, self.max_ep_len+1):
                action, action_logprob, state_val = ppo_agent.select_action(state, deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated

                ep_reward += reward
                
                if render:
                    self.env.render()
                    time.sleep(frame_delay)

                # 행동 로깅
                action_value = action.item() if not self.has_continuous_action_space and hasattr(action, 'item') else (action[0] if self.has_continuous_action_space else action)
                action_data_to_log = [t, action_value, reward]
                if info: 
                    action_data_to_log.extend(list(info.values()))

                self.data_recorder.log_to_action_file(action_data_to_log)

                # 상태 로깅
                state_data_to_log = state.tolist() if isinstance(state, np.ndarray) else [state]
                self.data_recorder.log_to_state_file(state_data_to_log)

                state = next_state

                if done:
                    break

            # clear buffer
            ppo_agent.buffer.clear()

            test_running_reward +=  ep_reward
            self.data_recorder.log_to_console(f'Episode: {ep} \t\t Reward: {np.round(ep_reward, 2)}\n')

        self.env.close()

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)

        final_console_messages = [
            "============================================================================================\n",
            f"Average test reward : {avg_test_reward}\n",
            "============================================================================================\n"
        ]
        for msg in final_console_messages:
            self.data_recorder.log_to_console(msg) # 화면에도 출력
        
        self.data_recorder.close_all() # 모든 로그 파일 닫기
        