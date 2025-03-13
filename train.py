import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
from environment import GymEnvironment
from fileManager import Config, File

print("============================================================================================")
path = str(os.path.dirname(__file__)) + "/" 
config = Config.load_config(path + "/config/" + "Hyperparameters.yaml")
print("Run Directory : " + path)

class RichDog:
    def __init__(self):
        self.init_parameters()
        self.print_parameters()

    def init_parameters(self):
        print("============================================================================================")
        ####### initialize environment hyperparameters ######
        self.env_name = config.env_name

        self.has_continuous_action_space = config.has_continuous_action_space  # continuous action space; else discrete

        self.max_ep_len = config.max_ep_len      # max timesteps in one episode (에피소드 당 최대 타임 스텝)
        self.max_training_timesteps = int(config.max_training_timesteps)   # break training loop if timeteps > max_training_timesteps (총 학습 타임스텝)
    
        self.print_freq = config.print_freq        # print avg reward in the interval (in num timesteps) (출력 주기)
        self.log_freq = config.log_freq            # log avg reward in the interval (in num timesteps) (로그 파일 생성 주기)
        self.save_model_freq = int(config.save_model_freq)          # save model frequency (in num timesteps) (모델 저장 주기)

        self.action_std = config.action_std                  # starting std for action distribution (Multivariate Normal) (행동 표준 편차)
        self.action_std_decay_rate = config.action_std_decay_rate        # linearly decay action_std (action_std = action_std - action_std_decay_rate) (행동 표준 편차 감소 값)
        self.min_action_std = config.min_action_std                # minimum action_std (stop decay after action_std <= min_action_std) (0.05 ~ 0.1) (최소 행동 표준 편차 값)
        self.action_std_decay_freq = int(config.action_std_decay_freq)  # action_std decay frequency (in num timesteps) (표준 편차 감소 주기)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        self.update_timestep = config.update_timestep      # update policy every n timesteps (정책 업데이트 주기)
        self.K_epochs = config.K_epochs               # update policy for K epochs in one PPO update (최적화 횟수)

        self.eps_clip = config.eps_clip          # clip parameter for PPO (클리핑)
        self.gamma = config.gamma            # discount factor (감가율)
        self.lamda = config.lamda              # 어드벤티지 감가율
        self.minibatchsize = config.minibatchsize

        self.lr_actor = config.lr_actor       # learning rate for actor network (액터의 학습률)
        self.lr_critic = config.lr_critic       # learning rate for critic network (크리틱 학습률)

        self.value_loss_coef = config.value_loss_coef     # 가치 손실 계수
        self.entropy_coef = config.entropy_coef       # 엔트로피 계수

        self.random_seed = config.random_seed         # set random seed if required (0 = no random seed) (랜덤 시드)
        #####################################################

        self.env = GymEnvironment(env_name=self.env_name)

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
        cur_time = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        log_file_name = 'PPO_' + self.env_name + "_log_" + cur_time + ".csv"
        self.log_file = File(path + "PPO_logs/" + self.env_name + '/', log_file_name)

        print("current logging run number for " + self.env_name + " : ", cur_time)
        print("logging at : " + self.log_file.get_file_path())
        #####################################################

        ################### checkpointing ###################
        checkpoint_file_name = "PPO_{}_{}_{}.pth".format(self.env_name, self.random_seed, cur_time)
        self.checkpoint_file = File(path + "PPO_preTrained/" + self.env_name + '/', checkpoint_file_name)

        print("save checkpoint path : " + self.checkpoint_file.get_file_path())
        #####################################################

    ################################### Training ###################################
    def train(self):

        ################# training procedure ################

        # initialize a PPO agent
        ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, 
                        self.gamma, self.K_epochs, self.eps_clip, self.has_continuous_action_space, 
                        self.action_std,self. value_loss_coef, self.entropy_coef,self.lamda, self.minibatchsize)

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        self.log_file.write('episode,timestep,reward\n')
        #log_f = open(log_f_name,"w+")
        #log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= self.max_training_timesteps:

            state, _ = self.env.reset()
            current_ep_reward = 0
            next_state = state

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
                    ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)


                    self.log_file.write_flush('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    #log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    #log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

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

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        self.log_file.close()
        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")


if __name__ == '__main__':

    RichDog().train()
    
    
    
    
    
    
    
