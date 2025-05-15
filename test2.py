import os
import sys
import time
from datetime import datetime
import abc
import random # 랜덤 모듈 임포트
import io     # 콘솔 출력 캡처용

import torch
import numpy as np

# --- 사용자 정의 모듈 (존재한다고 가정) ---
# from PPO.PPO2 import PPO
# from PPO.environment import GymEnvironment, StockEnvironment
# from common.fileManager import Config, File # Config.Param 객체 사용 가정
# from common.logger import Logger, DataRecorder
# -----------------------------------------

# --- 임시 목업 클래스 (실제 환경에서는 사용자의 모듈 사용) ---
class MockConfigParam: # common.fileManager.Config.Param 모방
    def __init__(self, name, value, min_val=None, max_val=None, note=None):
        self.name = name
        self.value = value # 기본값 또는 고정값
        self.min = min_val
        self.max = max_val
        self.note = note

class MockConfig: # common.fileManager.Config 모방
    def __init__(self, data_dict):
        for key, item_data in data_dict.items():
            setattr(self, key, MockConfigParam(key, item_data.get('value'), item_data.get('min'), item_data.get('max'), item_data.get('note')))

    @staticmethod
    def load_config(config_path):
        # 실제 Config.load_config는 YAML 파일을 로드하여 이런 구조를 만들 것임
        # 여기서는 예시 YAML 데이터를 직접 사용
        print(f"[목업] '{config_path}' 경로에서 설정 로드 시도...")
        if "Hyperparameters.yaml" in config_path:
            # 사용자가 제공한 YAML 데이터 구조 (일부만)
            example_hyperparameters_data = {
                'K_epochs': {'max': '10', 'min': '3', 'note': '최적화 횟수', 'value': '10'},
                'action_std': {'max': '0.7', 'min': '0.4', 'note': '행동 표준 편차', 'value': '0.6'}, # min/max 순서 수정됨
                'action_std_decay_freq': {'max': '1000000', 'min': '100000', 'note': '표준 편차 감소 주기', 'value': '250000'},
                'entropy_coef': {'max': '0.2', 'min': '0.01', 'note': '엔트로피 계수', 'value': '0.01'},
                'env_name': {'max': None, 'min': None, 'note': 'environment name', 'value': 'MountainCarContinuous-v0'},
                'eps_clip': {'max': '0.3', 'min': '0.01', 'note': '클리핑', 'value': '0.2'},
                'gamma': {'max': '0.99', 'min': '0.99', 'note': '감가율', 'value': '0.99'},
                'has_continuous_action_space': {'max': None, 'min': None, 'note': '연속 환경 유무', 'value': 'True'},
                'lamda': {'max': '0.95', 'min': '0.95', 'note': '어드벤티지 감가율', 'value': '0.95'},
                'log_freq': {'max': None, 'min': None, 'note': '로그 파일 생성 주기', 'value': '2000'},
                'lr_actor': {'max': '0.0003', 'min': '0.00005', 'note': '액터 학습률', 'value': '0.0003'},
                'lr_critic': {'max': '0.003', 'min': '0.00005', 'note': '크리틱 학습률', 'value': '0.001'},
                'max_ep_len': {'max': None, 'min': None, 'note': '에피소드 당 최대 타임 스텝', 'value': '1000'},
                'max_training_timesteps': {'max': None, 'min': None, 'note': '총 학습 타임스텝', 'value': '1000000'},
                'min_action_std': {'max': '0.3', 'min': '0.05', 'note': '최소 행동 표준 편차 값', 'value': '0.1'},
                'minibatchsize': {'max': '128', 'min': '32', 'note': '미니 배치 사이즈', 'value': '32'},
                'print_freq': {'max': None, 'min': None, 'note': '출력 주기', 'value': '10000'},
                'random_seed': {'max': None, 'min': None, 'note': '랜덤 시드', 'value': '0'},
                'save_model_freq': {'max': None, 'min': None, 'note': '모델 저장 주기', 'value': '100000'},
                'update_timestep': {'max': '2880', 'min': '720', 'note': '정책 업데이트 주기', 'value': '4000'}, # 쉼표 제거
                'value_loss_coef': {'max': '1.0', 'min': '0.3', 'note': '가치 손실 계수', 'value': '0.5'},
                # 테스트 및 Stock 환경 관련 파라미터 (RichDog 기본 클래스에서 사용)
                'render': {'max': None, 'min': None, 'note': 'environment visual rendering', 'value': 'True'},
                'frame_delay': {'max': None, 'min': None, 'note': 'Time delay per frame', 'value': '0.01'}, # float으로 처리될 것임
                'total_test_episodes': {'max': None, 'min': None, 'note': 'total num of testing episodes', 'value': '3'},
                'stock_code_path': {'max': None, 'min': None, 'note': 'the path of folder where the data is located', 'value': 'API/datas'},
                'min_dt': {'max': None, 'min': None, 'note': 'start of the data', 'value': '20160711'},
                'max_dt': {'max': None, 'min': None, 'note': 'end of the data', 'value': '20250120'},
                'count': {'max': '360', 'min': '360', 'note': 'count of data to use per training', 'value': '360'},
            }
            return MockConfig(example_hyperparameters_data)
        elif "StockConfig.yaml" in config_path:
            # StockConfig.yaml은 다른 구조를 가질 수 있으므로 간단히 목업
            return MockConfig({'some_stock_param': {'value': 'some_value'}})
        return None

    @staticmethod
    def save_config(config_obj, file_path): # RichDogTrain에서 사용
        print(f"[목업] 선택된 파라미터들을 '{file_path}' 경로에 저장 시도...")
        # 실제로는 config_obj.data를 YAML로 저장
        pass

# 실제 사용자 모듈을 사용하도록 아래 주석을 해제하고 목업 클래스들을 제거하세요.
PPO = type('PPO', (), {}) # 임시 목업
GymEnvironment = type('GymEnvironment', (), {'getObservation': lambda: 10, 'getActon': lambda: 2, 'reset': lambda seed=None: (np.zeros(10), {}), 'step': lambda action: (np.zeros(10), 0, False, False, {}), 'close': lambda: None, 'get_data_label': lambda: [f's{i}' for i in range(10)]})
StockEnvironment = type('StockEnvironment', (), {'getObservation': lambda: 20, 'getActon': lambda: 3, 'reset': lambda seed=None: (np.zeros(20), {}), 'step': lambda action: (np.zeros(20), 0, False, False, {}), 'close': lambda: None, 'get_data_label': lambda: [f'stock_feat_{i}' for i in range(20)]})
Config = MockConfig # common.fileManager.Config 대신 목업 사용
# File = type('File', (), {}) # 필요시 목업
DataRecorder = type('DataRecorder', (), { # 임시 목업
    '__init__': lambda self, env_name, random_seed, base_log_dir_prefix: setattr(self, 'log_dir_base', os.path.join(base_log_dir_prefix, env_name, str(random_seed), datetime.now().strftime("%Y%m%d-%H%M%S"))),
    'setup_config_logging': lambda self, cfg1, cfg2: print(f"[목업] 설정 파일 로깅 설정됨."),
    'get_config_log_dir': lambda self: os.path.join(getattr(self, 'log_dir_base', 'PPO_logs/unknown/0/unknown_time'), 'config'), # RichDogTrain에서 사용
    'setup_training_log': lambda self: print(f"[목업] 훈련 로그 설정됨."),
    'setup_checkpointing': lambda self, trained_random_seed: print(f"[목업] 체크포인트 설정됨 (시드: {trained_random_seed})."),
    'setup_console_log': lambda self, file_name: print(f"[목업] 콘솔 로그 '{file_name}' 설정됨."),
    'log_to_console': lambda self, msg: print(f"[콘솔로그] {msg.strip()}"),
    'get_checkpoint_path': lambda self: os.path.join(getattr(self, 'log_dir_base', 'PPO_logs/unknown/0/unknown_time'), "checkpoints", "model.pth"),
    'log_to_training_file': lambda self, data: None,
    'setup_run_data_logs': lambda self, mode_dir_suffix, run_id, action_labels, state_labels: None,
    'log_to_action_file': lambda self, data: None,
    'log_to_state_file': lambda self, data: None,
    'close_all': lambda self: print(f"[목업] 모든 로그 파일 닫힘.")
})
# -----------------------------------------


class RichDog:
    __metaclass__ = abc.ABCMeta

    def __init__(self, config_path=None):
        print("============================================================================================")
        if not config_path:
            config_path = "config/Hyperparameters.yaml" # 실제 경로 사용

        # self.config는 Config.load_config(config_path)의 반환 객체라고 가정
        # 이 객체의 각 속성 (예: self.config.K_epochs)은 min, max, value 등을 가진 Param 객체라고 가정
        self.config = Config.load_config(config_path)
        self.stock_config = Config.load_config("config/StockConfig.yaml") # 실제 경로 사용
        self.cur_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.data_recorder: DataRecorder = None

        # 제공된 YAML에 action_std의 min/max가 반대로 되어있어 수정합니다.
        # action_std: min: '0.4', max: '0.7' (원래: min: '0.7', max: '0.4')
        # update_timestep: max: '2,880' -> '2880' (쉼표 제거)
        # 이러한 수정은 Config.load_config가 처리하거나 YAML 파일 자체에서 수정되어야 합니다.
        # 목업 Config에서는 이미 수정된 값을 사용하도록 했습니다.

    def _get_random_or_fixed_value(self, param_obj, target_type, random_mode = False):
        """
        Config에서 읽어온 파라미터 객체(min, max, value 속성 포함)를 기반으로
        랜덤 값 또는 고정 값을 반환합니다.
        """

        min_val_str = getattr(param_obj, 'min', None) # 최대 값
        max_val_str = getattr(param_obj, 'max', None) # 최소 값
        default_val_str = getattr(param_obj, 'value', None) # 기본 값

        # min 또는 max 값이 None이 아니고 유효한 문자열일 때 랜덤 값 생성 시도
        if min_val_str is not None and max_val_str is not None and random_mode:
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


    def init_parameters(self):
        print("============================================================================================")
        
        # random_seed는 다른 파라미터들의 랜덤화에 영향을 주므로 가장 먼저 확정합니다.
        # self.config.random_seed는 min/max가 없는 고정값으로 가정합니다.
        self.random_seed = self._get_random_or_fixed_value(self.config.random_seed, int)
        if self.random_seed != 0:
            print(f"Python 내장 random 모듈 시드 설정: {self.random_seed}")
            random.seed(self.random_seed)
            # torch, numpy 시드는 이 함수 후반부 또는 필요시 설정

        ####### 환경 하이퍼파라미터 초기화 ######
        self.env_name = self._get_random_or_fixed_value(self.config.env_name, str)
        self.has_continuous_action_space = self._get_random_or_fixed_value(self.config.has_continuous_action_space, bool)

        self.max_ep_len = self._get_random_or_fixed_value(self.config.max_ep_len, int)
        self.max_training_timesteps = self._get_random_or_fixed_value(self.config.max_training_timesteps, int)
    
        self.print_freq = self._get_random_or_fixed_value(self.config.print_freq, int)
        self.log_freq = self._get_random_or_fixed_value(self.config.log_freq, int)
        self.save_model_freq = self._get_random_or_fixed_value(self.config.save_model_freq, int)

        self.action_std = self._get_random_or_fixed_value(self.config.action_std, float)
        self.min_action_std = self._get_random_or_fixed_value(self.config.min_action_std, float)
        
        # min_action_std가 action_std보다 크지 않도록 보정
        if self.action_std < self.min_action_std:
            print(f"경고: 초기 action_std ({self.action_std})가 min_action_std ({self.min_action_std})보다 작습니다. "
                  f"min_action_std를 action_std 값으로 조정합니다.")
            self.min_action_std = self.action_std
            
        self.action_std_decay_freq = self._get_random_or_fixed_value(self.config.action_std_decay_freq, int)
        
        # action_std_decay_rate 계산 (분모가 0이 되지 않도록 주의)
        # (self.max_training_timesteps - self.action_std_decay_freq) 이 부분은 전체 감쇠 스텝 수가 아니라,
        # 감쇠가 시작된 후 남은 총 스텝 수로 해석될 수 있습니다.
        # 감쇠가 일어나는 총 횟수는 (max_training_timesteps / action_std_decay_freq) 입니다.
        # 좀 더 명확한 계산:
        if self.action_std_decay_freq > 0:
            num_decay_steps = self.max_training_timesteps // self.action_std_decay_freq
            if num_decay_steps > 0:
                self.action_std_decay_rate = (self.action_std - self.min_action_std) / num_decay_steps
            else:
                self.action_std_decay_rate = 0 # 감쇠 없음
                print(f"경고: action_std_decay_freq({self.action_std_decay_freq})에 비해 max_training_timesteps({self.max_training_timesteps})가 충분치 않아 감쇠 스텝이 0입니다. 감쇠율 0으로 설정.")
        else:
            self.action_std_decay_rate = 0 # 감쇠 없음
            print(f"경고: action_std_decay_freq가 0 이하({self.action_std_decay_freq})이므로 감쇠율 0으로 설정.")
        #####################################################

        ################ PPO 하이퍼파라미터 ################
        self.update_timestep = self._get_random_or_fixed_value(self.config.update_timestep, int)
        self.K_epochs = self._get_random_or_fixed_value(self.config.K_epochs, int)

        self.eps_clip = self._get_random_or_fixed_value(self.config.eps_clip, float)
        self.gamma = self._get_random_or_fixed_value(self.config.gamma, float)
        self.lamda = self._get_random_or_fixed_value(self.config.lamda, float) # lamda -> lambda (Python 키워드와 충돌 피하기 위해 lamda 사용)
        self.minibatchsize = self._get_random_or_fixed_value(self.config.minibatchsize, int)

        self.lr_actor = self._get_random_or_fixed_value(self.config.lr_actor, float)
        self.lr_critic = self._get_random_or_fixed_value(self.config.lr_critic, float)

        self.value_loss_coef = self._get_random_or_fixed_value(self.config.value_loss_coef, float)
        self.entropy_coef = self._get_random_or_fixed_value(self.config.entropy_coef, float)
        #####################################################

        # StockEnvironment 관련 파라미터 (RichDog 타입일 때 사용)
        self.stock_code_path = self._get_random_or_fixed_value(self.config.stock_code_path, str)
        self.min_dt = self._get_random_or_fixed_value(self.config.min_dt, str) # 날짜는 보통 고정
        self.max_dt = self._get_random_or_fixed_value(self.config.max_dt, str) # 날짜는 보통 고정
        self.count = self._get_random_or_fixed_value(self.config.count, int)

        if self.env_name == "RichDog":
            # self.stock_config는 __init__에서 이미 로드됨
            self.env = StockEnvironment(self.stock_code_path, self.stock_config, self.min_dt, self.max_dt, self.count)
        else:
            self.env = GymEnvironment(env_name=self.env_name)

        self.state_dim = self.env.getObservation()
        self.action_dim = self.env.getActon()

        # torch, numpy 시드는 모든 파라미터 확정 후, 환경 생성 후 설정
        if self.random_seed != 0:
            print(f"torch, numpy 및 환경 시드 설정: {self.random_seed}")
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # 환경 시드 설정 (Gymnasium API는 reset 시 seed 전달)
            # self.env.seed(self.random_seed) # 구 Gym API
            # 최신 Gym/Gymnasium은 env.reset(seed=self.random_seed) 형태로 사용.
            # RichDogTrain.train() 내부의 첫 reset에서 처리하거나, 환경이 내부적으로 numpy.random을 사용하면 np.random.seed로 충분할 수 있음.
        
    def print_parameters(self):
        # 이 메소드는 이제 init_parameters에서 확정된 (랜덤 또는 고정) 값들을 출력합니다.
        print("선택된 하이퍼파라미터:")
        print(f"환경 이름 : {self.env_name}")
        print("--------------------------------------------------------------------------------------------")
        print(f"최대 학습 타임스텝 : {self.max_training_timesteps}")
        print(f"에피소드 당 최대 타임스텝 : {self.max_ep_len}")
        print(f"모델 저장 주기 : {self.save_model_freq} 타임스텝")
        print(f"로그 파일 생성 주기 : {self.log_freq} 타임스텝")
        print(f"평균 보상 출력 주기 (최근 N 타임스텝) : {self.print_freq} 타임스텝")
        print("--------------------------------------------------------------------------------------------")
        print(f"상태 공간 차원 : {self.state_dim}")
        print(f"행동 공간 차원 : {self.action_dim}")
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            print("연속 행동 공간 정책 초기화")
            print("--------------------------------------------------------------------------------------------")
            print(f"행동 분포 초기 표준 편차 : {self.action_std:.4f}")
            print(f"행동 분포 표준 편차 감쇠율 : {self.action_std_decay_rate:.6f}")
            print(f"행동 분포 최소 표준 편차 : {self.min_action_std:.4f}")
            print(f"행동 분포 표준 편차 감쇠 주기 : {self.action_std_decay_freq} 타임스텝")
        else:
            print("이산 행동 공간 정책 초기화")
        print("--------------------------------------------------------------------------------------------")
        print(f"PPO 정책 업데이트 주기 : {self.update_timestep} 타임스텝")
        print(f"PPO K epochs : {self.K_epochs}")
        print(f"PPO epsilon clip : {self.eps_clip:.3f}")
        print(f"할인 계수 (gamma) : {self.gamma:.4f}")
        print(f"GAE 할인 계수 (lambda) : {self.lamda:.4f}")
        print(f"미니 배치 크기 : {self.minibatchsize}")
        print("--------------------------------------------------------------------------------------------")
        print(f"액터 신경망 학습률 : {self.lr_actor:.6f}")
        print(f"크리틱 신경망 학습률 : {self.lr_critic:.6f}")
        print(f"가치 손실 계수 : {self.value_loss_coef:.3f}")
        print(f"엔트로피 계수 : {self.entropy_coef:.4f}")

        if self.random_seed:
            print("--------------------------------------------------------------------------------------------")
            print(f"랜덤 시드 설정됨 : {self.random_seed}")
        else:
            print("--------------------------------------------------------------------------------------------")
            print(f"랜덤 시드 : 설정 안됨 (0), 실행마다 결과가 다를 수 있음")
        #####################################################
        print("============================================================================================")


class RichDogTrain(RichDog):
    def __init__(self, config_path=None):
        # RichDog.__init__이 먼저 호출되어 self.config가 로드되고,
        # 그 다음 self.init_parameters()가 호출되어야 파라미터들이 확정됩니다.
        # 하지만 RichDogTrain의 init_parameters는 super().init_parameters()를 통해 호출되므로,
        # DataRecorder 생성 시점에는 아직 self.env_name, self.random_seed 등이 확정되지 않았을 수 있습니다.
        # 따라서 DataRecorder 생성은 init_parameters 호출 *후* 또는 해당 값들을 사용하는 시점에 이루어져야 합니다.
        # 여기서는 DataRecorder 생성을 train 메서드 시작 부분이나 init_files로 옮기는 것을 고려할 수 있습니다.
        # 또는, __init__에서 init_parameters를 먼저 호출하고 DataRecorder를 설정합니다.
        
        super().__init__(config_path) # self.config 로드
        super().init_parameters()     # self.env_name, self.random_seed 등 파라미터 확정!
                                      # RichDogTest에서는 이 순서가 중요 (훈련된 모델 config 로드 후 init_parameters)

        self.data_recorder = DataRecorder(
            env_name=self.env_name, # 확정된 env_name 사용
            random_seed=self.random_seed, # 확정된 random_seed 사용
            base_log_dir_prefix="PPO_logs"
        )
        
        self.data_recorder.setup_config_logging(self.config, self.stock_config) # 원본 YAML 저장
        self._save_chosen_parameters() # 선택된 파라미터 저장

    def _save_chosen_parameters(self):
        """실제로 선택된 하이퍼파라미터들을 로그 디렉토리에 저장합니다."""
        chosen_params = {
            "env_name": self.env_name,
            "has_continuous_action_space": self.has_continuous_action_space,
            "max_ep_len": self.max_ep_len,
            "max_training_timesteps": self.max_training_timesteps,
            "print_freq": self.print_freq,
            "log_freq": self.log_freq,
            "save_model_freq": self.save_model_freq,
            "action_std": self.action_std,
            "min_action_std": self.min_action_std,
            "action_std_decay_freq": self.action_std_decay_freq,
            "action_std_decay_rate": self.action_std_decay_rate,
            "update_timestep": self.update_timestep,
            "K_epochs": self.K_epochs,
            "eps_clip": self.eps_clip,
            "gamma": self.gamma,
            "lamda": self.lamda,
            "minibatchsize": self.minibatchsize,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "random_seed": self.random_seed,
        }
        if self.env_name == "RichDog":
            chosen_params.update({
                "stock_code_path_used": self.stock_code_path,
                "min_dt_used": self.min_dt,
                "max_dt_used": self.max_dt,
                "count_used": self.count,
            })
        
        try:
            # DataRecorder.get_config_log_dir()는 현재 실행의 config가 저장될 디렉토리를 반환한다고 가정
            config_log_dir = self.data_recorder.get_config_log_dir()
            if config_log_dir:
                # Config 객체를 새로 만들어서 저장 (common.fileManager.Config.save_config 사용 가정)
                # chosen_params_config = Config(None) # 임시 Config 객체
                # chosen_params_config.data = {key: Config.Param(key, value) for key, value in chosen_params.items()} # 실제 Param 객체 구조에 맞게
                
                # 목업 Config 사용 시:
                # chosen_params_config_data = {key: {'value': str(value)} for key, value in chosen_params.items()} # 단순화
                # chosen_params_config = Config(chosen_params_config_data)

                # 더 간단하게는, Config 객체를 통하지 않고 직접 YAML로 저장할 수도 있습니다.
                # 여기서는 Config.save_config를 사용한다고 가정하고, chosen_params를 적절한 Config 객체로 변환 필요
                # 아래는 간단한 YAML 저장을 위한 예시 (실제로는 Config.save_config 사용)
                import yaml
                os.makedirs(config_log_dir, exist_ok=True)
                file_path = os.path.join(config_log_dir, "chosen_hyperparameters.yaml")
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(chosen_params, f, allow_unicode=True, sort_keys=False)
                self.data_recorder.log_to_console(f"선택된 하이퍼파라미터 저장 완료: {file_path}\n")

            else:
                self.data_recorder.log_to_console("경고: 선택된 파라미터를 저장할 config 로그 디렉토리를 결정할 수 없습니다.\n")
        except Exception as e:
            self.data_recorder.log_to_console(f"선택된 하이퍼파라미터 저장 중 오류 발생: {e}\n")

    def init_files(self):
        """DataRecorder 관련 파일 설정을 초기화합니다."""
        # self.data_recorder는 __init__에서 이미 생성됨
        self.data_recorder.setup_training_log()
        # 체크포인트 저장 시 self.random_seed (확정된 값) 사용
        self.data_recorder.setup_checkpointing(trained_random_seed=self.random_seed)
        self.data_recorder.setup_console_log(file_name="PPO_console_log.txt")

    ################################### Training ###################################
    def train(self):
        # self.init_parameters()는 __init__에서 이미 호출됨
        self.print_parameters() # 선택된 파라미터 출력
        self.init_files()       # 로그 파일 등 설정

        # PPO 에이전트 초기화 시 self에 저장된 (랜덤화되었을 수 있는) 파라미터 사용
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, 
                        self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                        self.action_std, self.value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize)

        start_time = datetime.now().replace(microsecond=0)
        self.data_recorder.log_to_console(f"훈련 시작 시간 (GMT) : {start_time}\n")
        self.data_recorder.log_to_console("============================================================================================\n")
        
        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0
        time_step = 0
        i_episode = 0
        loss, policy_loss, value_loss, dist_entropy = 0.0, 0.0, 0.0, 0.0 # 초기화

        while time_step <= self.max_training_timesteps:
            current_episode_seed = (self.random_seed + i_episode) if self.random_seed != 0 else None
            state, info = self.env.reset(seed=current_episode_seed) # Gymnasium API 호환
            current_ep_reward = 0

            # 에피소드/타임스텝별 행동/상태 로깅 조건 (예: 매 모델 저장 시점마다 해당 에피소드 로깅)
            # is_action_log = (self.save_model_freq > 0 and time_step > 0 and time_step % self.save_model_freq == 0)
            # 조금 더 나은 로직: 모델 저장 직후의 에피소드를 로깅하거나, 특정 주기마다 로깅
            log_episode_details_this_time = False
            if self.save_model_freq > 0 and i_episode > 0: # 첫 에피소드는 제외
                # 이전 타임스텝에서 save_model_freq를 넘었고, 현재 타임스텝에서 아직 넘지 않은 경우 (즉, save 직후 에피소드)
                prev_timestep_segment = (time_step - self.max_ep_len) // self.save_model_freq if time_step >= self.max_ep_len else -1
                current_timestep_segment = time_step // self.save_model_freq
                if current_timestep_segment > prev_timestep_segment and prev_timestep_segment != -1 : # 모델 저장이 발생한 직후
                     log_episode_details_this_time = True
                # 또는 매 N번째 에피소드마다 로깅
                # if i_episode % 100 == 0: log_episode_details_this_time = True


            if log_episode_details_this_time:
                action_labels = ['timestep', "action"] + (list(info.keys()) if info else [])
                state_labels = self.env.get_data_label() if hasattr(self.env, 'get_data_label') else [f"state_{i}" for i in range(self.state_dim)]
                self.data_recorder.setup_run_data_logs(
                    mode_dir_suffix="train_episode_data",
                    run_id=f"ep{i_episode}_ts{time_step}",
                    action_labels=action_labels,
                    state_labels=state_labels
                )

            for t in range(1, self.max_ep_len + 1):
                action, action_logprob, state_val = ppo_agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                
                time_step += 1
                current_ep_reward += reward

                if time_step % self.update_timestep == 0:
                    update_result = ppo_agent.update()
                    if update_result: # update가 값을 반환하는 경우
                        loss, policy_loss, value_loss, dist_entropy = update_result
                    # ppo_agent.schedule_action_std(...) 등은 PPO 클래스 내부에서 호출될 수 있음
                    # 또는 여기서 명시적으로 호출:
                    if hasattr(ppo_agent, 'schedule_action_std'):
                         ppo_agent.schedule_action_std(self.min_action_std, self.action_std, time_step, self.max_training_timesteps)
                    elif self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0: # 기존 로직
                         if hasattr(ppo_agent, 'decay_action_std'): # PPO 에이전트에 이 메서드가 있다면
                             ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)


                if time_step > 0 and time_step % self.log_freq == 0:
                    if log_running_episodes > 0:
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = np.round(log_avg_reward, 4)
                        self.data_recorder.log_to_training_file([i_episode, time_step, log_avg_reward, loss, policy_loss, value_loss, dist_entropy])
                    log_running_reward = 0
                    log_running_episodes = 0

                if time_step > 0 and time_step % self.print_freq == 0:
                    if print_running_episodes > 0:
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = np.round(print_avg_reward, 2)
                        self.data_recorder.log_to_console(f"에피소드 : {i_episode} \t 타임스텝 : {time_step} \t 평균 보상 : {print_avg_reward}\n")
                    print_running_reward = 0
                    print_running_episodes = 0

                if time_step > 0 and time_step % self.save_model_freq == 0:
                    chk_path = self.data_recorder.get_checkpoint_path()
                    self.data_recorder.log_to_console("--------------------------------------------------------------------------------------------\n")
                    self.data_recorder.log_to_console(f"모델 저장 경로 : {chk_path}\n")
                    ppo_agent.save(chk_path)
                    self.data_recorder.log_to_console("모델 저장 완료\n")
                    self.data_recorder.log_to_console(f"경과 시간  : {str(datetime.now().replace(microsecond=0) - start_time)}\n")
                    current_agent_action_std = getattr(ppo_agent, 'action_std', 'N/A (PPO 에이전트에 action_std 속성 없음)')
                    self.data_recorder.log_to_console(f"현재 action_std : {current_agent_action_std}\n")
                    self.data_recorder.log_to_console("--------------------------------------------------------------------------------------------\n")

                if log_episode_details_this_time:
                    action_value = action.item() if not self.has_continuous_action_space and hasattr(action, 'item') else (action[0] if self.has_continuous_action_space and isinstance(action, (np.ndarray, list)) and len(action)>0 else action)
                    action_data_to_log = [t, action_value] 
                    if info: action_data_to_log.extend(list(info.values()))
                    self.data_recorder.log_to_action_file(action_data_to_log)
                    
                    state_data_to_log = state.tolist() if isinstance(state, np.ndarray) else [state]
                    self.data_recorder.log_to_state_file(state_data_to_log)
                
                state = next_state
                if done:
                    break
            
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1
        
        self.env.close()
        end_time = datetime.now().replace(microsecond=0)
        final_console_messages = [
            "============================================================================================\n",
            f"훈련 시작 시간 (GMT) : {start_time}\n",
            f"훈련 종료 시간 (GMT) : {end_time}\n",
            f"총 훈련 시간  : {end_time - start_time}\n",
            "============================================================================================\n"
        ]
        for msg in final_console_messages:
            self.data_recorder.log_to_console(msg)
        self.data_recorder.close_all()


class RichDogTest(RichDog):
    def __init__(self, config_path=None, checkpoint_path=""):
        super().__init__(config_path) # 테스트용 기본 config 로드 (Hyperparameters.yaml)
        
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path:
            print("오류: 테스트를 위해서는 체크포인트 경로가 반드시 필요합니다.")
            sys.exit(1)

        model_env_name_parsed, model_creation_time_parsed, model_seed_parsed = self._parse_checkpoint_filename()

        # 테스트 결과 저장을 위한 DataRecorder (모델 정보 기반 경로 사용)
        test_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 테스트 로그는 훈련 로그와 다른 최상위 디렉토리에 저장하는 것이 좋음
        test_log_base = os.path.join(
            "PPO_TEST_logs",
            model_env_name_parsed if model_env_name_parsed else "unknown_env",
            model_seed_parsed if model_seed_parsed else "unknown_seed",
            model_creation_time_parsed if model_creation_time_parsed else "unknown_model_time",
            f"test_run_{test_run_id}"
        )
        self.data_recorder = DataRecorder(
            env_name=model_env_name_parsed if model_env_name_parsed else self.env_name, # 중요: 모델의 env_name
            random_seed=int(datetime.now().timestamp()), # 테스트 실행 자체의 고유 ID
            base_log_dir_prefix=test_log_base # DataRecorder가 이 경로를 직접 사용하도록 수정 필요 (또는 다른 방식)
        )
        # DataRecorder 목업이 base_log_dir_prefix를 그대로 사용한다고 가정하고, 내부 경로 생성 로직을 비활성화
        # setattr(self.data_recorder, 'log_dir_base', test_log_base) # 목업 DataRecorder에 직접 경로 설정

        self.data_recorder.setup_console_log(file_name=f"PPO_test_console_log.txt")
        
        # ★ 중요: 훈련된 모델의 'chosen_hyperparameters.yaml'을 로드하여
        # 에이전트 구조와 관련된 파라미터들(env_name, action_space 등)을 현재 self.config에 덮어쓴다.
        self._load_trained_model_config_and_override(model_env_name_parsed, model_creation_time_parsed, model_seed_parsed)

        # 그 다음, (훈련된 모델의 설정으로 일부 덮어쓰여진) self.config를 사용하여 파라미터 초기화
        super().init_parameters() # self.env_name 등이 훈련 시 값으로 설정됨

    def _parse_checkpoint_filename(self):
        """체크포인트 파일명에서 env_name, creation_time, seed를 파싱합니다."""
        try:
            filename = os.path.basename(self.checkpoint_path)
            # 예: PPO_MyEnv_Seed123_Time20230101-100000.pth
            parts = filename.replace(".pth", "").split('_')
            if len(parts) >= 4 and parts[0] == "PPO":
                model_env_name = parts[1]
                model_seed = parts[2] # 파일명에 시드가 포함되어 있다면
                model_creation_time = parts[3] # 파일명에 생성 시간이 포함되어 있다면
                self.data_recorder.log_to_console(f"파싱된 모델 정보: Env={model_env_name}, Seed={model_seed}, Time={model_creation_time}\n")
                return model_env_name, model_creation_time, model_seed
            else:
                self.data_recorder.log_to_console(f"경고: 표준 체크포인트 파일명 형식이 아닙니다: {filename}. 일부 경로에 기본값을 사용합니다.\n")
                return None, None, None
        except Exception as e:
            self.data_recorder.log_to_console(f"체크포인트 파일명 파싱 오류 {self.checkpoint_path}: {e}\n")
            return None, None, None

    def _load_trained_model_config_and_override(self, model_env_name, model_creation_time, model_seed):
        """훈련된 모델의 'chosen_hyperparameters.yaml'을 로드하고,
        현재 self.config의 주요 파라미터들을 덮어씁니다."""
        if not all([model_env_name, model_creation_time, model_seed]):
            self.data_recorder.log_to_console("경고: 모델 정보를 파싱할 수 없어, 훈련된 모델의 특정 설정을 로드할 수 없습니다. 현재 Hyperparameters.yaml을 사용합니다.\n")
            return

        # 훈련 로그 경로 구조: PPO_logs / {env_name} / {random_seed} / {cur_time} / config / chosen_hyperparameters.yaml
        # DataRecorder의 경로 생성 방식과 일치해야 함.
        # RichDogTrain의 DataRecorder는 base_log_dir_prefix / env_name / random_seed / cur_time 구조를 가짐
        # 여기서 random_seed는 훈련 시 사용된 self.random_seed, cur_time은 훈련 시작 시 self.cur_time
        # model_seed는 파일명에서 파싱된 random_seed, model_creation_time은 파일명에서 파싱된 cur_time
        trained_config_path = os.path.join(
            "PPO_logs", model_env_name, model_seed, model_creation_time, 
            "config", "chosen_hyperparameters.yaml"
        )
        self.data_recorder.log_to_console(f"훈련된 모델의 선택된 설정 로드 시도: {trained_config_path}\n")

        if os.path.exists(trained_config_path):
            try:
                # 실제 Config.load_config 사용
                # trained_chosen_config = Config.load_config(trained_config_path)
                
                # 목업 사용 시: YAML 직접 로드
                import yaml
                with open(trained_config_path, 'r', encoding='utf-8') as f:
                    trained_chosen_params_dict = yaml.safe_load(f)
                # 이 dict를 self.config 객체에 반영할 수 있도록 MockConfig 또는 실제 Config 객체로 변환
                # 여기서는 간단히 필요한 값만 self.config의 Param 객체에 덮어쓰는 형태로 시뮬레이션
                
                self.data_recorder.log_to_console("훈련된 모델의 설정을 로드했습니다. 현재 설정에 적용합니다...\n")

                # 에이전트 구조 및 환경 호환성에 필수적인 파라미터들을 덮어씀
                essential_params_to_override = [
                    "env_name", "has_continuous_action_space",
                    # "action_std", # PPO.load()에서 로드될 수 있음
                    # "state_dim", "action_dim" 등은 env_name에 따라 결정되므로 env_name이 중요
                ]
                # Stock 환경 관련 파라미터도 필요시 덮어쓸 수 있으나, 테스트 시에는
                # 테스트용 데이터셋 경로 등을 사용해야 하므로 주의.
                # 여기서는 env_name과 action_space 관련만 덮어쓰고, 나머지는 PPO.load()에 의존.

                for param_name in essential_params_to_override:
                    if param_name in trained_chosen_params_dict:
                        trained_value = trained_chosen_params_dict[param_name]
                        if hasattr(self.config, param_name):
                            # self.config.param_name이 Param 객체라고 가정하고 .value를 수정
                            # 실제 Config 클래스에 따라 이 부분 수정 필요
                            current_param_obj = getattr(self.config, param_name)
                            setattr(current_param_obj, 'value', str(trained_value)) # 문자열로 저장된 것으로 가정
                            self.data_recorder.log_to_console(f"  '{param_name}' 값을 훈련된 값으로 덮어썼습니다: {trained_value}\n")
                        else: # self.config에 해당 파라미터가 없는 경우 (드묾)
                             # 새로운 Param 객체를 만들어 추가 (실제 Config 클래스에 맞게)
                             setattr(self.config, param_name, MockConfigParam(param_name, str(trained_value))) # 목업
                             self.data_recorder.log_to_console(f"  '{param_name}' 값을 훈련된 값으로 새로 설정했습니다: {trained_value}\n")


            except Exception as e:
                self.data_recorder.log_to_console(f"훈련된 모델의 설정 로드 또는 적용 중 오류: {e}. 현재 Hyperparameters.yaml 설정을 유지합니다.\n")
        else:
            self.data_recorder.log_to_console(f"훈련된 모델의 선택된 설정 파일을 찾을 수 없습니다: {trained_config_path}. 현재 Hyperparameters.yaml 설정을 사용합니다.\n")


    #################################### Testing ###################################
    def test(self):
        # self.init_parameters()는 __init__에서 이미 호출됨 (훈련된 모델 config 반영 후)
        self.data_recorder.log_to_console("=========================== 테스트 시작 =======================================\n")
        
        # 테스트 관련 파라미터는 현재 config (Hyperparameters.yaml)에서 가져옴
        # _get_random_or_fixed_value를 사용하나, 테스트 설정은 보통 고정값이므로 min/max가 없을 것임
        render_config_param = getattr(self.config, 'render', MockConfigParam('render', 'False')) # config에 render가 없을 경우 대비
        render = self._get_random_or_fixed_value(render_config_param, bool)
        
        frame_delay_config_param = getattr(self.config, 'frame_delay', MockConfigParam('frame_delay', '0.01'))
        frame_delay = self._get_random_or_fixed_value(frame_delay_config_param, float)
        
        total_test_episodes_config_param = getattr(self.config, 'total_test_episodes', MockConfigParam('total_test_episodes', '3'))
        total_test_episodes = self._get_random_or_fixed_value(total_test_episodes_config_param, int)

        self.data_recorder.log_to_console("적용된 테스트 파라미터 (일부는 훈련된 모델의 설정일 수 있음):\n")
        param_str = self.capture_print_parameters() # self.print_parameters()의 출력을 문자열로
        self.data_recorder.log_to_console(param_str + "\n")
        self.data_recorder.log_to_console(f"테스트 설정: Render={render}, Frame Delay={frame_delay}, 총 테스트 에피소드={total_test_episodes}\n")
        
        # 테스트 환경 설정
        # self.env_name 등은 _load_trained_model_config_and_override 및 init_parameters를 통해 훈련 시 값으로 설정됨
        if self.env_name == "RichDog":
            # RichDog 테스트 시에는 테스트용 데이터 경로 등을 사용해야 함
            # 이 정보는 현재 config (테스트용 Hyperparameters.yaml)에서 가져옴
            test_stock_code_path = self._get_random_or_fixed_value(self.config.stock_code_path, str)
            test_min_dt = self._get_random_or_fixed_value(self.config.min_dt, str)
            test_max_dt = self._get_random_or_fixed_value(self.config.max_dt, str)
            test_count = self._get_random_or_fixed_value(self.config.count, int)
            
            self.data_recorder.log_to_console(f"테스트용 StockEnvironment 사용. 데이터 경로: {test_stock_code_path}\n")
            self.env = StockEnvironment(
                stock_code_path=test_stock_code_path, # 테스트 데이터 경로
                stock_config=self.stock_config,       # StockConfig는 동일하게 사용
                min_dt=test_min_dt, max_dt=test_max_dt, count=test_count # 테스트용 기간/개수
            )
        else: # 일반 Gym 환경
            self.env = GymEnvironment(env_name=self.env_name, render_mode="human" if render else None)

        # 환경 재생성 후 state/action_dim 재확인 (보통은 동일하겠지만, 안전을 위해)
        self.state_dim = self.env.getObservation()
        self.action_dim = self.env.getActon()
        
        # PPO 에이전트 초기화. lr, K_epochs 등은 init_parameters에서 설정된 값 사용.
        # 이 값들은 훈련된 모델의 것일 수도, 현재 config의 것일 수도 있음 (덮어쓰기 정책에 따라).
        # 핵심은 state_dim, action_dim, has_continuous_action_space가 로드할 모델과 호환되어야 함.
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, 
                    self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                    self.action_std, self.value_loss_coef, self.entropy_coef, self.lamda, self.minibatchsize)
        
        self.data_recorder.log_to_console(f"신경망 로드 경로 : {self.checkpoint_path}\n")
        ppo_agent.load(self.checkpoint_path) # 가중치 및 저장된 action_std 등 로드
        self.data_recorder.log_to_console("신경망 로드 완료.\n--------------------------------------------------------------------------------------------\n")

        test_running_reward = 0
        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            # 테스트 시에도 에피소드별 시드 설정 가능
            current_test_episode_seed = (self.random_seed + ep + 10000) if self.random_seed != 0 else None # 훈련 시드와 겹치지 않게
            state, info = self.env.reset(seed=current_test_episode_seed)
 
            action_labels = ['timestep', "action", "reward"] + (list(info.keys()) if info else [])
            state_labels = self.env.get_data_label() if hasattr(self.env, 'get_data_label') else [f"state_{i}" for i in range(self.state_dim)]
            self.data_recorder.setup_run_data_logs(
                mode_dir_suffix="test_episode_data", # 테스트 결과 하위 폴더
                run_id=f"ep{ep}", 
                action_labels=action_labels,
                state_labels=state_labels
            )
            
            for t in range(1, self.max_ep_len + 1): # self.max_ep_len은 init_parameters에서 설정됨
                action, _, _ = ppo_agent.select_action(state, deterministic=True) # 테스트 시에는 결정론적 행동
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                
                if render:
                    self.env.render() # GymEnvironment에 render 메소드가 있어야 함
                    time.sleep(frame_delay)

                action_value = action.item() if not self.has_continuous_action_space and hasattr(action, 'item') else (action[0] if self.has_continuous_action_space and isinstance(action, (np.ndarray, list)) and len(action)>0 else action)
                action_data_to_log = [t, action_value, reward]
                if info: action_data_to_log.extend(list(info.values()))
                self.data_recorder.log_to_action_file(action_data_to_log)

                state_data_to_log = state.tolist() if isinstance(state, np.ndarray) else [state]
                self.data_recorder.log_to_state_file(state_data_to_log)
                state = next_state
                if done: break
            
            if hasattr(ppo_agent, 'buffer') and hasattr(ppo_agent.buffer, 'clear'):
                ppo_agent.buffer.clear() # 테스트 후 버퍼 클리어

            test_running_reward += ep_reward
            self.data_recorder.log_to_console(f'에피소드: {ep}/{total_test_episodes} \t 보상: {np.round(ep_reward, 2)}\n')

        self.env.close()
        avg_test_reward = test_running_reward / total_test_episodes if total_test_episodes > 0 else 0
        avg_test_reward = round(avg_test_reward, 2)
        final_console_messages = [
            "============================================================================================\n",
            f"총 {total_test_episodes} 에피소드 평균 테스트 보상 : {avg_test_reward}\n",
            "============================================================================================\n"
        ]
        for msg in final_console_messages:
            self.data_recorder.log_to_console(msg)
        self.data_recorder.close_all()
        
    def capture_print_parameters(self) -> str:
        """print_parameters 메서드의 출력을 문자열로 캡처합니다."""
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        self.print_parameters() # 이 메서드의 출력이 captured_output으로 리디렉션됨
        sys.stdout = old_stdout # 표준 출력 복원
        return captured_output.getvalue()