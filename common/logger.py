from common.fileManager import File, Config
import os
from datetime import datetime

class Logger:
    def __init__(self, root_dir=""):
        self.root_dir = root_dir
        self.file_dict = {}

    def add_file(self, key, dir_path, file_name):
        self.file_dict[key] = File(self.root_dir + dir_path, file_name)
    
    def get_file_path(self, key):
        return self.file_dict[key].get_file_path()

    def write_file(self, key, data:str):
        self.file_dict[key].write_flush(data)

    def list_write_file(self, key, data_list:list):
        str_data = [str(item) for item in data_list]
        data = ','.join(str_data) + '\n'
        self.file_dict[key].write_flush(data)

    def print_wirte_file(self, key, data:str):
        print(data)
        self.file_dict[key].write_append(data)

    def close_all(self):
        for key in self.file_dict.keys():
            self.file_dict[key].close()
    
    def is_exists(self, key):
        return key in self.file_dict

class DataRecorder:
    def __init__(self, env_name: str, random_seed: int, 
                 base_log_dir_prefix: str = "PPO_logs",
                 cur_time=""):
        self.env_name = env_name
        self.random_seed = random_seed # 경로 일관성을 위해 설정 파일의 random_seed를 사용할 수 있음

        if cur_time=="":
            self.cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.cur_time = cur_time
        # base_log_dir_prefix는 "PPO_training_logs", "PPO_test_logs" 등 최상위 로그 디렉토리 지정
        # self.run_log_root_path는 이번 실행의 로그가 저장될 기본 경로(예: PPO_training_logs/CartPole-v1/20231027-100000/train/)
        self.run_log_root_path = os.path.join(base_log_dir_prefix, self.env_name, self.cur_time)
        os.makedirs(self.run_log_root_path, exist_ok=True)

        # 범용 로거 인스턴스, 특정 파일들은 필요에 따라 추가될 것임
        self._training_logger = Logger() # 훈련 진행 상황 로그용
        self._console_logger = Logger() # 콘솔 텍스트 파일 로그용
        self._action_logger = Logger() # 에피소드별 또는 타임스텝별 행동 로그용
        self._state_logger = Logger()  # 에피소드별 또는 타임스텝별 상태 로그용
        
        self._checkpoint_path = ""
        self._checkpoint_dir = ""

        print(f"DataRecorder initialized. Logging to: {self.run_log_root_path}")

    def setup_config_logging(self, config_to_save: Config, stock_config_to_save: Config):
        """훈련/테스트 실행에 사용된 설정 파일들을 로그 디렉토리에 저장합니다."""
        config_dir = os.path.join(self.run_log_root_path, "config")
        Config.save_config(config_to_save, os.path.join(config_dir, "Hyperparameters.yaml"))
        Config.save_config(stock_config_to_save, os.path.join(config_dir, "StockConfig.yaml"))
        print(f"Configurations saved to : {config_dir}")

    def setup_training_log(self):
        """훈련 진행 상황(에피소드, 보상 등)을 기록할 메인 로그 파일을 설정합니다."""
        log_file_name = f'PPO_{self.env_name}_log_{self.cur_time}.csv'
        # log_file_dir는 self.run_log_root_path 입니다.
        self._training_logger.add_file("training_log", self.run_log_root_path + "/", log_file_name)
        # CSV 헤더 작성
        self._training_logger.write_file("training_log", 'episode,timestep,reward,loss,policy_loss,value_loss,dist_entropy\n')
        print(f"Training log setup at : {os.path.join(self.run_log_root_path, log_file_name)}")

    def setup_checkpointing(self, trained_random_seed: int):
        """모델 체크포인트 저장 경로를 설정합니다. 파일명 일관성을 위해 설정 파일의 random_seed를 사용합니다."""
        self._checkpoint_dir = os.path.join("PPO_preTrained", self.env_name) # 예: PPO_preTrained/CartPole-v1/
        
        checkpoint_file_name = f"PPO_{self.env_name}_{trained_random_seed}_{self.cur_time}.pth"
        
        self._checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_file_name)
        # 실제 저장은 PPO 에이전트가 담당하므로, 여기서는 경로와 디렉토리만 준비합니다.
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        print(f"Checkpoint path set to : {self._checkpoint_path}")
        
    def get_checkpoint_path(self) -> str:
        """설정된 체크포인트 파일 경로를 반환합니다."""
        if not self._checkpoint_path:
            raise ValueError("체크포인트 경로가 설정되지 않았습니다.")
        return self._checkpoint_path

    def setup_console_log(self, console_log_base_dir="PPO_console", file_name="PPO_console_log.txt"):
        """콘솔 출력을 기록할 로그 파일을 설정합니다."""
        os.makedirs(console_log_base_dir, exist_ok=True)
        # 콘솔 로그는 여러 실행에 걸쳐 추가될 수 있으므로 'a'(append) 모드를 사용
        self._console_logger.add_file("console", console_log_base_dir + "/", file_name) 
        print(f"Console log setup at : {os.path.join(console_log_base_dir, file_name)}")

    def setup_run_data_logs(self, mode_dir_suffix: str, run_id: str, action_labels: list, state_labels: list):
        """
        특정 실행(예: 에피소드 또는 특정 타임스텝)에 대한 행동 및 상태 로그를 설정합니다.
        mode_dir_suffix: 예: "train_run_data" 또는 "test_run_data" (실행별 데이터 로그의 상위 폴더명)
        run_id: 이 특정 실행 로그의 고유 식별자 (예: 에피소드 번호, 타임스텝)
        """
        # 예: PPO_training_logs/CartPole-v1/20231027-100000/train_episode_data/
        base_data_log_path = os.path.join(self.run_log_root_path, mode_dir_suffix)

        # 예: .../train_episode_data/PPO_action_logs/20231027-100000/
        action_log_dir = os.path.join(base_data_log_path, "PPO_action_logs")
        action_file_name = f"PPO_{self.env_name}_action_{self.random_seed}_{self.cur_time}_{run_id}.csv"
        self._action_logger.add_file("action_log", action_log_dir + "/", action_file_name)
        self._action_logger.list_write_file("action_log", action_labels) # CSV 헤더

        # 예: .../train_episode_data/PPO_state_logs/20231027-100000/
        state_log_dir = os.path.join(base_data_log_path, "PPO_state_logs")
        state_file_name = f"PPO_{self.env_name}_state_{self.random_seed}_{self.cur_time}_{run_id}.csv"
        self._state_logger.add_file("state_log", state_log_dir + "/", state_file_name)
        self._state_logger.list_write_file("state_log", state_labels) # CSV 헤더

        print(f"Action logs for run {run_id} setup at: {action_log_dir}")
        print(f"State logs for run {run_id} setup at: {state_log_dir}")
        
    def log_to_training_file(self, data_list: list):
        """메인 훈련 로그 파일에 데이터를 기록합니다."""
        self._training_logger.list_write_file("training_log", data_list)

    def log_to_console(self, message: str, print_to_stdout: bool = True):
        """콘솔 로그 파일에 메시지를 기록하고, 선택적으로 표준 출력(화면)에도 인쇄합니다."""
        if print_to_stdout:
            print(message.strip()) # 메시지에 이미 개행 문자가 있을 수 있으므로 strip() 사용
        # 메시지가 필요에 따라 개행 문자를 포함한다고 가정
        self._console_logger.write_file("console", message)  

    def log_to_action_file(self, data_list: list):
        """행동 로그 파일에 데이터를 기록합니다."""
        self._action_logger.list_write_file("action_log", data_list)

    def log_to_state_file(self, data_list: list):
        """상태 로그 파일에 데이터를 기록합니다."""
        self._state_logger.list_write_file("state_log", data_list)

    def close_all(self):
        """모든 로거의 파일 핸들을 닫습니다."""
        self._training_logger.close_all()
        self._console_logger.close_all()
        self._action_logger.close_all()
        self._state_logger.close_all()
        print("All data recorder files closed.")