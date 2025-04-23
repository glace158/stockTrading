import os
import sys
import time
from datetime import datetime
import abc

import torch
import numpy as np

from PPO.PPO import PPO
from PPO.environment import GymEnvironment, StockEnvironment
from common.fileManager import Config, File
from common.logger import Logger

class RichDog:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, config_path=None):
        print("============================================================================================")
        if not config_path:
            config_path = "config/Hyperparameters.yaml"

        self.config = Config.load_config(config_path)
        self.stock_config = Config.load_config("config/StockConfig.yaml")
        self.log_root_path = ""
        self.cur_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.logger = Logger()
        self.console_logger = Logger()

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

        self.action_std = float(self.config.action_std.value)                  # starting std for action distribution (Multivariate Normal) (행동 표준 편차)
        self.min_action_std = float(self.config.min_action_std.value)                # minimum action_std (stop decay after action_std <= min_action_std) (0.05 ~ 0.1) (최소 행동 표준 편차 값)
        self.action_std_decay_freq = int(self.config.action_std_decay_freq.value)  # action_std decay frequency (in num timesteps) (표준 편차 감소 주기)
        self.action_std_decay_rate = (self.action_std - self.min_action_std) / ((self.max_training_timesteps - self.action_std_decay_freq)  // self.action_std_decay_freq)     # linearly decay action_std (action_std = action_std - action_std_decay_rate) (행동 표준 편차 감소 값)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        self.update_timestep = int(self.config.update_timestep.value)      # update policy every n timesteps (정책 업데이트 주기)
        self.K_epochs = int(self.config.K_epochs.value)               # update policy for K epochs in one PPO update (최적화 횟수)

        self.eps_clip = float(self.config.eps_clip.value)          # clip parameter for PPO (클리핑)
        self.gamma = float(self.config.gamma.value)            # discount factor (감가율)
        self.lamda = float(self.config.lamda.value)              # 어드벤티지 감가율
        self.minibatchsize = int(self.config.minibatchsize.value)

        self.lr_actor = float(self.config.lr_actor.value)       # learning rate for actor network (액터의 학습률)
        self.lr_critic = float(self.config.lr_critic.value)       # learning rate for critic network (크리틱 학습률)

        self.value_loss_coef = float(self.config.value_loss_coef.value)     # 가치 손실 계수
        self.entropy_coef = float(self.config.entropy_coef.value)       # 엔트로피 계수

        self.random_seed = int(self.config.random_seed.value)         # set random seed if required (0 = no random seed) (랜덤 시드)
        #####################################################

        #self.env = GymEnvironment(env_name=self.env_name)
        self.env = StockEnvironment(self.config.stock_code_path.value, self.stock_config, self.config.min_dt.value, self.config.max_dt.value, int(self.config.count.value))

        # state space dimension
        self.state_dim = self.env.getObservation().shape[0]
        # action space dimension
        if self.has_continuous_action_space:
            self.action_dim = self.env.getActon().shape[0]
        else:
            self.action_dim = self.env.getActon().n


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
        print("state space dimension : ", self.state_dim)
        print("action space dimension : ", self.action_dim)
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

    def console_file_log(self):
        ################## console logging ##################
        self.console_file_name = "PPO_console_log.txt"
        console_dir = "PPO_console/"
        self.console_logger.add_file("console", console_dir, self.console_file_name)

        print("save console path : " + console_dir)
        #####################################################
        
    def action_file_log(self, num, mode_dir, actionlabels):
        ################## action logging ##################
        
        action_file_name = "PPO_{}_action_{}_{}_{}.csv".format(self.env_name, self.random_seed, self.cur_time, num)
        action_dir = self.log_root_path + mode_dir + "/PPO_action_logs/" + self.env_name + '/' + str(self.cur_time) + '/'

        self.logger.add_file("action", action_dir, action_file_name)
        self.logger.list_write_file("action", actionlabels)
    
        state_file_name = "PPO_{}_state_{}_{}_{}.csv".format(self.env_name, self.random_seed, self.cur_time, num)
        state_dir = self.log_root_path + mode_dir + "/PPO_state_logs/" + self.env_name + '/' + str(self.cur_time) + '/'

        self.logger.add_file("state", state_dir, state_file_name)
        self.logger.list_write_file("state", self.env.get_data_label())

        print("save action logs path : " + action_dir)
        #####################################################

class RichDogTrain(RichDog):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        self.log_root_path = 'PPO_logs/' + self.config.env_name.value + "/" + self.cur_time + '/'
        self.logger = Logger()
        self.console_logger = Logger()
        
        Config.save_config( self.config, self.log_root_path + "config/Hyperparameters.yaml")
        Config.save_config( self.stock_config, self.log_root_path + "config/StockConfig.yaml")
    
    def init_files(self):
        ###################### logging ######################
        #### log files for multiple runs are NOT overwritten
        log_file_name = 'PPO_' + self.env_name + "_log_" + self.cur_time + ".csv"
        log_file_dir = self.log_root_path
        self.logger.add_file("log_file", log_file_dir, log_file_name)
        
        self.logger.write_file("log_file", 'episode,timestep,reward,loss,dist_entropy\n')

        print("current logging run number for " + self.env_name + " : ", self.cur_time)
        print("logging at : " + log_file_dir + log_file_name)
        #####################################################

        ################### checkpointing ###################
        checkpoint_file_name = "PPO_{}_{}_{}.pth".format(self.env_name, self.random_seed, self.cur_time)
        checkpoint_dir = "PPO_preTrained/" + self.env_name + '/'
        self.logger.add_file("checkpoint", checkpoint_dir, checkpoint_file_name)
        
        print("save checkpoint path : " + checkpoint_dir + checkpoint_file_name)
        #####################################################


    ################################### Training ###################################
    def train(self):
        self.init_parameters()
        self.print_parameters()
        
        ################# training procedure ################
        # initialize a PPO agent
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, 
                        self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                        self.action_std,self. value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize)

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        self.init_files()
        self.console_file_log()

        self.console_logger.print_wirte_file("console", "Started training at (GMT) : " + str(start_time) + "\n")
        self.console_logger.print_wirte_file("console", "============================================================================================\n")
        
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        is_save_model = False

        loss = 0
        dist_entropy = 0

        # training loop
        while time_step <= self.max_training_timesteps:

            state, info = self.env.reset()

            current_ep_reward = 0
            next_state = state

            is_action_log = is_save_model

            if is_action_log:
                self.action_file_log(time_step, "PPO_train_logs", ['timestep', "action", "reward"]  + list(info.keys()))

            for t in range(1, self.max_ep_len+1):

                # select action with policy
                action, action_logprob, state_val = ppo_agent.select_action(state)
                
                next_state, reward, done, _, info = self.env.step(action)
                
                # saving buffer
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                
                state = next_state
                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    loss, dist_entropy = ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution (액션 분포의 표준편차 감소)
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(time_step, self.action_std_decay_freq, self.action_std, self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = np.round(log_avg_reward, 4)

                    self.logger.list_write_file("log_file", [i_episode, time_step, log_avg_reward, loss, dist_entropy])

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = np.round(print_avg_reward, 2)

                    self.console_logger.print_wirte_file("console", "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}\n".format(i_episode, time_step, print_avg_reward))
                    
                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 0:
                    self.console_logger.print_wirte_file("console", "--------------------------------------------------------------------------------------------\n")
                    self.console_logger.print_wirte_file("console", "saving model at : " + self.logger.get_file_path("checkpoint") + "\n")
                    ppo_agent.save(self.logger.get_file_path("checkpoint"))
                    self.console_logger.print_wirte_file("console", "model saved")
                    self.console_logger.print_wirte_file("console", "Elapsed Time  : " + str(datetime.now().replace(microsecond=0) - start_time) + "\n")
                    self.console_logger.print_wirte_file("console", "--------------------------------------------------------------------------------------------\n")

                    is_save_model = True

                if is_action_log:
                    self.logger.list_write_file('action', [t, action[0], reward] + list(info.values()))
                    str_state = [str(item) for item in state]
                    self.logger.list_write_file('state', str_state)
                    is_save_model = False

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1
        
        self.logger.close_all()
        self.env.close()

        # print total training time
        end_time = datetime.now().replace(microsecond=0)
        self.console_logger.print_wirte_file("console", "============================================================================================\n")
        self.console_logger.print_wirte_file("console", "Started training at (GMT) : " + str(start_time) + "\n")
        self.console_logger.print_wirte_file("console", "Finished training at (GMT) : " + str(end_time) + "\n")
        self.console_logger.print_wirte_file("console", "Total training time  : " + str(end_time - start_time) + "\n")
        self.console_logger.print_wirte_file("console", "============================================================================================\n")
        self.console_logger.close_all()


class RichDogTest(RichDog):
    def __init__(self, config_path=None, checkpoint_path=""):
        super().__init__(config_path)
        self.checkpoint_path = checkpoint_path
        file_name, file_extension = os.path.splitext(os.path.basename(self.checkpoint_path)) # 모델 파일 이름 추출
        self.cur_time = file_name.split('_')[-1] # 모델파일 시간 추출

        root_path = 'PPO_logs/' + self.config.env_name.value + "/" + self.cur_time + '/'
        
        self.config = Config.load_config(root_path + "config/Hyperparameters.yaml")
        self.stock_config = Config.load_config(root_path + "config/StockConfig.yaml")
        
        self.cur_time = str(datetime.now().strftime("%Y%m%d-%H%M%S")) # 현재 시간 추출
        self.logger = Logger(root_path)
        self.console_logger = Logger()
        
    #################################### Testing ###################################
    def test(self):
        print("============================================================================================")
        self.console_file_log()

        ################## hyperparameters ##################
        self.init_parameters()
        render = bool(self.config.render.value)              # render environment on screen
        frame_delay = float(self.config.frame_delay.value)   # if required; add delay b/w frames

        total_test_episodes = int(self.config.total_test_episodes.value)    # total num of testing episodes
        self.action_std = float(self.config.min_action_std.value)
        self.print_parameters()
        #####################################################

        #self.env = GymEnvironment(env_name=self.env_name, render_mode="human")
        self.env = StockEnvironment(stock_code_path="API/test_datas",stock_config=self.stock_config)

        # state space dimension
        self.state_dim = self.env.getObservation().shape[0]
        # action space dimension
        if self.has_continuous_action_space:
            self.action_dim = self.env.getActon().shape[0]
        else:
            self.action_dim = self.env.getActon().n

        # initialize a PPO agent
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, 
                    self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                    self.action_std,self. value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize)
        # preTrained weights directory

        if self.checkpoint_path == "":
            #self.console_file.write_append("Error : Not Setting Model path\n")
            self.console_logger.print_wirte_file("console", "Error : Not Setting Model path\n" )
            return

        print("loading network from : " + self.checkpoint_path)
        self.console_logger.print_wirte_file("console", "loading network from : " + self.checkpoint_path + "\n")
        
        ppo_agent.load(self.checkpoint_path)

        self.console_logger.print_wirte_file("console", "--------------------------------------------------------------------------------------------\n")

        test_running_reward = 0

        for ep in range(1, total_test_episodes+1):
            
            ep_reward = 0
            state, info = self.env.reset()
            self.action_file_log(ep, "PPO_test_logs", ['timestep', "action", "reward"]  + list(info.keys()))

            for t in range(1, self.max_ep_len+1):
                action, action_logprob, state_val = ppo_agent.select_action(state)
                state, reward, done, _, info = self.env.step(action)
                ep_reward += reward

                if render:
                    self.env.render()
                    time.sleep(frame_delay)

                # logging
                self.logger.list_write_file('action', [t, action[0], reward] + list(info.values()))
                str_state = [str(item) for item in state]
                self.logger.list_write_file('state', str_state)

                if done:
                    break

            # clear buffer
            ppo_agent.buffer.clear()

            test_running_reward +=  ep_reward
            self.console_logger.print_wirte_file("console", 'Episode: {} \t\t Reward: {}\n'.format(ep, np.round(ep_reward, 2)))

            ep_reward = 0

        self.env.close()
        self.logger.close_all()

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)

        self.console_logger.print_wirte_file("console", "============================================================================================\n")
        self.console_logger.print_wirte_file("console", "average test reward : " + str(avg_test_reward) + "\n")
        self.console_logger.print_wirte_file("console", "============================================================================================\n")
        self.console_logger.close_all()