import os
import sys
import time
from datetime import datetime

import torch
import numpy as np

from PPO.PPO import PPO
from PPO.environment import GymEnvironment, StockEnvironment
from PPO.fileManager import Config, File

class RichDog:
    def __init__(self, config_path=None):
        print("============================================================================================")
        self.path = str(os.path.dirname(__file__)) + "/" 
        if not config_path:
            config_path = self.path + "config/Hyperparameters.yaml"

        self.config = Config.load_config(config_path)
        print("Run Directory : " + config_path)

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
        self.action_std_decay_rate = (self.action_std - self.min_action_std) / (self.max_training_timesteps // self.action_std_decay_freq)     # linearly decay action_std (action_std = action_std - action_std_decay_rate) (행동 표준 편차 감소 값)
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
        self.env = StockEnvironment(self.config.stock_code_path.value, self.config.min_dt.value, self.config.max_dt.value, int(self.config.count.value))

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

    def init_files(self):
        ###################### logging ######################
        #### log files for multiple runs are NOT overwritten
        self.cur_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_file_name = 'PPO_' + self.env_name + "_log_" + self.cur_time + ".csv"
        self.log_file = File(self.path + "PPO_logs/" + self.env_name + '/', log_file_name)

        print("current logging run number for " + self.env_name + " : ", self.cur_time)
        print("logging at : " + self.log_file.get_file_path())
        #####################################################

        ################### checkpointing ###################
        checkpoint_file_name = "PPO_{}_{}_{}.pth".format(self.env_name, self.random_seed, self.cur_time)
        self.checkpoint_file = File(self.path + "PPO_preTrained/" + self.env_name + '/', checkpoint_file_name)

        print("save checkpoint path : " + self.checkpoint_file.get_file_path())
        #####################################################

        ################## console logging ##################
        self.console_file_name = "PPO_console_log.txt"
        self.console_file = File(self.path + "PPO_console/" , self.console_file_name)

        print("save console path : " + self.console_file.get_file_path())
        #####################################################

    def action_file_log(self, num):
        ################## action logging ##################
        action_file_name = "PPO_{}_action_{}_{}_{}.csv".format(self.env_name, self.random_seed, self.cur_time, num)
        self.action_file = File(self.path + "PPO_action_logs/" + self.env_name + '/' + str(self.cur_time) + '/', action_file_name)
        self.action_file.write('timestep,action,reward\n')
        
        
        state_file_name = "PPO_{}_state_{}_{}_{}.csv".format(self.env_name, self.random_seed, self.cur_time, num)
        self.state_file = File(self.path + "PPO_state_logs/" + self.env_name + '/' + str(self.cur_time) + '/', state_file_name)
        self.state_file.write( ','.join(self.env.get_data_label()) +'\n')

        print("save action logs path : " + self.action_file.get_file_path())
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

        print("Started training at (GMT) : ", start_time)
        print("============================================================================================")
        
        self.console_file.write_append("Started training at (GMT) : " + str(start_time) + "\n")
        self.console_file.write_append("============================================================================================"+ "\n")


        # logging file
        self.log_file.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        is_save_model = False

        # training loop
        while time_step <= self.max_training_timesteps:

            state, _ = self.env.reset()
            current_ep_reward = 0
            next_state = state

            is_action_log = is_save_model

            if is_action_log:
                self.action_file_log(time_step)

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
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution (액션 분포의 표준편차 감소)
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(time_step, self.action_std_decay_freq, self.action_std, self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = np.round(log_avg_reward, 4)

                    self.log_file.write_flush('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    
                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = np.round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                    
                    self.console_file.write_append("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}\n".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + self.checkpoint_file.get_file_path())
                    ppo_agent.save(self.checkpoint_file.get_file_path())
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                    is_save_model = True

                    self.console_file.write_append("--------------------------------------------------------------------------------------------"+ "\n")
                    self.console_file.write_append("saving model at : " + self.checkpoint_file.get_file_path()+ "\n")
                    self.console_file.write_append("Elapsed Time  : " + str(datetime.now().replace(microsecond=0) - start_time)+ "\n")
                    self.console_file.write_append("--------------------------------------------------------------------------------------------"+ "\n")

                if is_action_log:
                    self.action_file.write_flush('{},{},{}\n'.format(t, action[0], reward))
                    str_state = [str(item) for item in state]
                    self.state_file.write_flush(','.join(str_state)+'\n')
                    is_save_model = False

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1
        
        self.action_file.close()
        self.state_file.close()
        self.log_file.close()
        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
        
        self.console_file.write_append("============================================================================================"+ "\n")
        self.console_file.write_append("Started training at (GMT) : " + str(start_time)+ "\n")
        self.console_file.write_append("Finished training at (GMT) : "+ str(end_time)+ "\n")
        self.console_file.write_append("Total training time  : " + str(end_time - start_time)+ "\n")
        self.console_file.write_append("============================================================================================"+ "\n")

    #################################### Testing ###################################
    def test(self, checkpoint_path):
        print("============================================================================================")
        self.console_file_name = "PPO_console_log.txt"
        self.console_file = File(self.path + "PPO_console/" , self.console_file_name)
        print("save console path : " + self.console_file.get_file_path())

        ################## hyperparameters ##################
        self.init_parameters()
        render = bool(self.config.render.value)              # render environment on screen
        frame_delay = float(self.config.frame_delay.value)   # if required; add delay b/w frames

        total_test_episodes = int(self.config.total_test_episodes.value)    # total num of testing episodes
        self.action_std = float(self.config.min_action_std.value)
        self.print_parameters()
        #####################################################

        #self.env = GymEnvironment(env_name=self.env_name, render_mode="human")
        self.env = StockEnvironment()

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

        """
        directory = self.path + "PPO_preTrained" + '/' + self.env_name + '/'
        checkpoint_file_list = next(os.walk(directory))[2]
        checkpoint_path = directory + checkpoint_file_list[-1]
        """
        
        if checkpoint_path == "":
            self.console_file.write_append("Error : Not Setting Model path\n")
            return

        print("loading network from : " + checkpoint_path)
        self.console_file.write_append("loading network from : " + checkpoint_path + "\n")
        
        
        ppo_agent.load(checkpoint_path)

        print("--------------------------------------------------------------------------------------------")
        self.console_file.write_append("--------------------------------------------------------------------------------------------" + "\n")

        test_running_reward = 0

        for ep in range(1, total_test_episodes+1):
            ep_reward = 0
            state, _ = self.env.reset()

            for t in range(1, self.max_ep_len+1):
                action, action_logprob, state_val = ppo_agent.select_action(state)
                state, reward, done, _, info = self.env.step(action)
                ep_reward += reward

                if render:
                    self.env.render()
                    time.sleep(frame_delay)

                if done:
                    break

            # clear buffer
            ppo_agent.buffer.clear()

            test_running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            self.console_file.write_append('Episode: {} \t\t Reward: {}\n'.format(ep, round(ep_reward, 2)))

            ep_reward = 0

        self.env.close()

        print("============================================================================================")

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))
        print("============================================================================================")
        
        self.console_file.write_append("============================================================================================\n")
        self.console_file.write_append("average test reward : " + str(avg_test_reward) + "\n")
        self.console_file.write_append("============================================================================================\n")

if __name__ == '__main__':
    richdog = RichDog()
    arg = sys.argv

    if len(arg)<= 1:        
        arg = ["main.py", "train", "PPO_preTrained/Richdog/PPO_Richdog_0_20250403-141334.pth"]
        

    if len(arg) > 1:
        if arg[1] == 'train':
            richdog.train()
        elif arg[1] == 'test':
            richdog.test(arg[2])
