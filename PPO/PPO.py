import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        self.action_dim = action_space.shape[0] if has_continuous_action_space else action_space.n
        if has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, self.action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, self.action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state): # 행동
        #print(state)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action): # 평가

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)

            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, observation_space, action_space, lr_actor, lr_critic, 
                 gamma, K_epochs, eps_clip, has_continuous_action_space, 
                 action_std_init=0.6, value_loss_coef =0.1, entropy_coef= 0.5, lambda_gae=0.95, minibatchsize=32):

        state_dim = observation_space.shape[0]
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer()
        self.minibatchsize = minibatchsize # 이니 배치 사이즈 설정

        self.policy = ActorCritic(state_dim, action_space, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_space, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std): # 표준편차 설정
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std): # 행동 분포의 표준편차 감소
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            #current_step = current_epoch // action_std_decay_freq
            #self.action_std = initial_std - current_step * action_std_decay_rate
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4) # 소수점 4자리까지 반올림

            # action_std가 최소값보다 작아지면 최소값으로 설정
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)

            # 새로운 action_std 값을 정책에 반영
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state): # 행동 선택
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten(), action_logprob, state_val
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item(), action_logprob, state_val
    
    def calculate_gae(self, state_values, rewards):
        """ Generalized Advantage Estimation (GAE) """
        advantages = []
        last_gae_lam = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - self.buffer.is_terminals[step]
                next_value = state_values[step]
            else:
                next_non_terminal = 1.0 - self.buffer.is_terminals[step + 1]
                next_value = state_values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - state_values[step]
            last_gae_lam = delta + self.gamma * self.lambda_gae * last_gae_lam
            advantages.insert(0, last_gae_lam)
            
        return torch.squeeze(torch.stack(advantages, dim=0)).detach().to(device)

    def update(self): # 정책 업데이트

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device) # 상태 (환경)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device) # 행동
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device) # 행동 확률
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device) # 상태 가치 값

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device) # 보상
        
        # calculate advantages
        advantages = self.calculate_gae(old_state_values, rewards) # GAE 어드밴티지 계산 

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 어드밴티지 정규화

        # Optimize policy for K epochs
        inds = np.arange(len(old_states)) # 미니 배치 인덱스 
        nbatch = len(old_states) # 배치 사이즈 설정
        
        for _ in range(self.K_epochs):
            np.random.shuffle(inds) # 인덱스 섞기
            for start in range(0, nbatch, self.minibatchsize): # 0부터 배치 사이즈까지 학습 배치 간격으로 반복
                end = start + self.minibatchsize # minibatchsize 간격만큼 설정
                mbinds = inds[start:end] # 학습 배치 사이즈에 맞게 인덱스 슬라이싱

                # 미니 배치 추출
                old_states_mini = old_states[mbinds]
                old_actions_mini = old_actions[mbinds]
                old_logprobs_mini = old_logprobs[mbinds]
                old_state_values_mini = old_state_values[mbinds]
                rewards_mini = rewards[mbinds]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_mini, old_actions_mini)

                # match state_values tensor dimensions with rewards tensor (불필요한 차원 삭제)
                state_values = torch.squeeze(state_values) 
                
                # Finding the ratio (pi_theta / pi_theta__old) (정책 확률 계산)
                ratios = torch.exp(logprobs - old_logprobs_mini.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages[mbinds] 
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[mbinds]
                
                
                returns = advantages[mbinds] + old_state_values_mini
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + self.value_loss_coef * self.MseLoss(state_values, returns) - self.entropy_coef * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return np.mean(loss.detach().cpu().numpy()), np.mean(dist_entropy.detach().cpu().numpy())
    
    def save(self, checkpoint_path): 
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
