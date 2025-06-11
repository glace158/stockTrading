import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

from typing import Union, Dict
from common.buffers import RolloutBuffer, DictRolloutBuffer
from common.extractors import * 

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
class ActorCritic(nn.Module):
    def __init__(self, observation_space: Union[spaces.Dict, spaces.Box],
                 action_space: spaces.Space,
                 has_continuous_action_space: bool, action_std_init: float,
                 cnn_features_dim: int = 64, # CNN용
                 mlp_features_dim: int = 0 # MLP용
                ):
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.has_continuous_action_space = has_continuous_action_space
        
        self.action_dim = action_space.shape[0] if has_continuous_action_space else action_space.n
        
        self.input_dim = self._set_features(observation_space, cnn_features_dim, mlp_features_dim)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(self.input_dim, 128),
                            nn.LeakyReLU(),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, self.action_dim),
                            nn.Tanh()
                        )
            
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device) # 행동 표준 편차 설정
        else:
            self.actor = nn.Sequential(
                            nn.Linear(self.input_dim, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, self.action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(self.input_dim, 64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        """
            행동 표준편차 설정 (연속환경)
        """        
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    # features 설정
    def _set_features(self, observation_space: Union[spaces.Dict, spaces.Box], 
                 cnn_features_dim: int = 64, # CNN용
                 mlp_features_dim: int = 0, # MLP용
                 ) -> int:
        """
            추가 신경망 (CNN, MLP, Identity 등) 설정
        """

        # Feature Extractor
        if isinstance(observation_space, spaces.Dict): # 상태 데이터가 딕셔너리이면 
            # CNN 과 MLP 추가
            self.features_extractor = CombinedFeaturesExtractor(observation_space,
                                                                cnn_features_dim=cnn_features_dim,
                                                                mlp_features_dim=mlp_features_dim)
        elif isinstance(observation_space, spaces.Box): # 단일 데이터이면
            if len(observation_space.shape) >= 2: # 이미지 또는 2D 이상 데이터
                self.features_extractor = CnnExtractor(observation_space, features_dim=cnn_features_dim)

            elif len(observation_space.shape) == 1: # 1D 벡터
                if mlp_features_dim != 0: # mlp 크기가 0이 아니면
                    self.features_extractor = MlpExtractor(observation_space, features_dim=mlp_features_dim)
                else:
                    self.features_extractor = None
                    #self.features_extractor = IdentityNetwork(observation_space)

            else:
                raise ValueError(f"Unsupported Box observation space shape: {observation_space.shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

        if self.features_extractor != None:
            input_dim = self.features_extractor.features_dim
        else:
            input_dim = observation_space.shape[0]

        return input_dim

    def _get_features(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
            추가 신경망 (CNN, MLP, Identity 등) 반환값 가져오기
        """
        if self.features_extractor != None:
            return self.features_extractor(observations)
        else:
            return observations
      
    def forward(self):
        raise NotImplementedError
    
    def act(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], deterministic: bool = False): # 행동
        """
            행동 선택하기
        """
        features = self._get_features(observations) # 특징(features) 추출하기
        #dist = self._get_actor_dist_from_features(features) # 행동 확률 분포 추출
        
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
        

        if deterministic: # 결정론적 행동
            if self.has_continuous_action_space: # 연속환경일 때
                action = dist.mean # 확률 분포 평균으로 행동
            else: # 이산환경일 때
                # 확률 분포가 최대가 되는 값으로 행동
                action = torch.argmax(dist.probs, dim=-1) # dim=1 -> dim=-1 for robustness
        else: 
            action = dist.sample() # 확률 분포에서 샘플링

        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], action: torch.Tensor): # 평가
        """
            행동 평가하기
        """
        features = self._get_features(observations) # 특징(features) 추출하기
        #dist = self._get_actor_dist_from_features(features) # 행동 확률 분포 추출
        
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)

            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        if self.action_dim == 1 and self.has_continuous_action_space:
            action = action.reshape(-1, self.action_dim)

        state_values = self.critic(features)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy


class PPO:
    def __init__(self, observation_space: Union[spaces.Dict, spaces.Box], 
                 action_space: spaces.Box, 
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 has_continuous_action_space, action_std_init=0.6, 
                 value_loss_coef =0.1, entropy_coef= 0.5, lambda_gae=0.95, minibatchsize=32,
                 cnn_features_dim: int = 64,
                 mlp_features_dim: int = 0
                 ):

        self.observation_space = observation_space
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init
            self.action_std_init = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        #if isinstance(observation_space, spaces.Dict): # 상태 데이터가 딕셔너리이면
        #    self.buffer = DictRolloutBuffer(self.observation_space.keys())
        #else:
        self.buffer = RolloutBuffer()

        self.minibatchsize = minibatchsize # 이니 배치 사이즈 설정

        self.policy = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                  cnn_features_dim, mlp_features_dim).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        if self.policy.features_extractor != None:
             self.optimizer.add_param_group({'params': self.policy.features_extractor.parameters(), 'lr': lr_actor})

        self.policy_old = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                      cnn_features_dim, mlp_features_dim).to(device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std): # 표준편차 설정
        """
            행동 분포의 표준편차 적용
            정책 신경망에 새로운 행동 분표 표준편차 적용
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)       
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std): # 행동 분포의 표준편차 감소
        """
            행동 분포의 표준편차 감소 (계단식 감소)
            action_std_decay_freq마다 action_std_decay_rate만큼 표준편차 감소
        """
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            # action_std를 decay_rate만큼 감소
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

    def schedule_action_std(self, min_action_std, action_std_init, current_step, max_steps):
        """
            행동 분포의 표준편차 감소 (점진적 감소)
            스텝에 따라 서서히 감소
        """
        #print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            action_std = action_std_init + (min_action_std - action_std_init) * (current_step / max_steps)
            #action_std = torch.exp(log_std)
            self.action_std = round(action_std, 4)
            
            # action_std가 최소값보다 작아지면 최소값으로 설정
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                #print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                pass
                #print("setting actor output action_std to : ", self.action_std)

            # 새로운 action_std 값을 정책에 반영
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        #print("--------------------------------------------------------------------------------------------")

    def _obs_to_tensor(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
            관찰(딕셔너리 또는 단일 Numpy 배열)을 PyTorch 텐서로 변환
            필요한 경우 정규화(이미지) 및 배치 차원 추가
        """

        if isinstance(obs, dict): # 만약 CNN 과 관찰 데이터 함께 사용하면
            tensor_dict = {}
            for key, value in obs.items():
                # Gymnasium 환경은 보통 float32 또는 uint8 로 관찰을 반환

                # value.ndim >= 2: 최소 2차원 (예: H, W)
                # value.shape[-1] <= 4: 마지막 차원의 크기가 4 이하 (예: 채널 수 R, G, B, Alpha)
                if value.dtype == np.uint8 and np.max(value) > 1 and (key == "image" or "img" in key or (value.ndim >=2 and value.shape[-1] <=4)): # 이미지로 추정
                    # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                    tensor_val = torch.as_tensor(value, device=device).float() / 255.0
                else: # 이미지가 아닌 일반 데이터
                    tensor_val = torch.as_tensor(value, device=device).float() # 텐서로 변환, float

                tensor_dict[key] = tensor_val # 변환된 텐서를 딕셔너리에 저장
            return tensor_dict # 텐서로 채워진 딕셔너리 반환
        
        else: # 단일 Numpy 배열
            if obs.dtype == np.uint8 and obs.ndim >=2 and np.max(obs) > 1: # (H,W) 또는 (H,W,C) 이미지로 추정
                tensor_obs = torch.as_tensor(obs, device=device).float() / 255.0 # 0~1 사이의 값으로 정규화
            else: # 일반 데이터
                tensor_obs = torch.as_tensor(obs, device=device).float() # 텐서로 변환, float

            return tensor_obs
        
    def select_action(self, observation: Union[Dict[str, np.ndarray], np.ndarray], deterministic: bool = False): # 행동 선택
        with torch.no_grad():
            state = self._obs_to_tensor(observation)
            action, action_logprob, state_val = self.policy_old.act(state, deterministic)

        #if isinstance(self.buffer, DictRolloutBuffer):
        #    for key, val in state.item():
        #        self.buffer.states[key].append(val)
        self.buffer.states.append(state)               
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_val
        else:
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

    def _obs_to_tensor_batch(self, states):
        """
            배치 학습 시 사용
            관찰(딕셔너리 또는 단일 Numpy 배열)을 PyTorch 텐서로 변환
            필요한 경우 정규화(이미지) 및 배치 차원 추가
        """
        # Convert list to tensor
        if isinstance(states[0], dict):
            batched_observations = {}
            # 첫 번째 관찰의 키를 기준으로 모든 관찰을 묶음
            dict_keys = states[0].keys()
            for key in dict_keys:
                obs_list_for_key = []
                for obs_dict_item in states:
                    val = obs_dict_item[key]
                    # 이미지 정규화 (uint8 -> float / 255.0)
                    if val.dtype == np.uint8 and np.max(val) > 1 and (key == "image" or "img" in key or (val.ndim >=2 and val.shape[-1] <=4)):
                        # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                        val = val.astype(np.float32) / 255.0
                    obs_list_for_key.append(val.cpu().numpy())

                batched_observations[key] = torch.as_tensor(np.stack(obs_list_for_key), device=device).detach().float()
        else: # 단일 Numpy 배열 리스트
            if states[0].dtype == np.uint8 and states[0].ndim >=2: # 이미지로 추정
                obs_list = []
                for obs_item in states:
                    val = obs_item
                    # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                    if np.max(val) > 1:
                        val = val.astype(np.float32) / 255.0
                    else:
                        val = val.astype(np.float32)
                        
                    obs_list.append(val.cpu().numpy())
                
                batched_observations = torch.as_tensor(np.stack(obs_list), device=device).detach().float()
            
            else:
                batched_observations = torch.squeeze(torch.stack(states, dim=0)).detach().to(device)

        return batched_observations
    
    def update(self): # 정책 업데이트
        # convert list to tensor
        old_states = self._obs_to_tensor_batch(self.buffer.states) # 상태 값 텐서로 변환
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device) # 행동
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device) # 행동 확률
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device) # 상태 가치 값

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device) # 보상

        # calculate advantages
        advantages = self.calculate_gae(old_state_values, rewards) # GAE 어드밴티지 계산 

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 어드밴티지 정규화

        # Optimize policy for K epochs
        inds = np.arange(len(rewards)) # 미니 배치 인덱스 
        
        nbatch = len(rewards) # 배치 사이즈 설정
        
        for _ in range(self.K_epochs):
            np.random.shuffle(inds) # 인덱스 섞기
            for start in range(0, nbatch, self.minibatchsize): # 0부터 배치 사이즈까지 학습 배치 간격으로 반복
                end = start + self.minibatchsize # minibatchsize 간격만큼 설정
                mbinds = inds[start:end] # 학습 배치 사이즈에 맞게 인덱스 슬라이싱

                # 미니 배치 추출
                if isinstance(old_states, dict):
                    old_states_mini = {}
                    for key in old_states.keys():
                        old_states_mini[key] = old_states[key][mbinds] 
                else:
                    old_states_mini = old_states[mbinds]

                old_actions_mini = old_actions[mbinds]
                old_logprobs_mini = old_logprobs[mbinds]
                old_state_values_mini = old_state_values[mbinds]
                advantages_mini = advantages[mbinds]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_mini, old_actions_mini)
                
                # Finding the ratio (pi_theta / pi_theta__old) (정책 확률 계산)
                ratios = torch.exp(logprobs - old_logprobs_mini.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages_mini 
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_mini
                
                policy_loss = -torch.min(surr1, surr2) # 정책 손실
                
                returns = advantages_mini + old_state_values_mini
                value_loss = self.MseLoss(state_values, returns) # 가치 손실

                entropy_loss = -dist_entropy # 엔트로피

                # final loss of clipped objective PPO
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0) # 그래디언트 클래핑 0.5, 1.0, 5.0
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return loss.mean().item(), policy_loss.mean().item(), value_loss.item(), entropy_loss.mean().item()
    
    def save(self, checkpoint_path): 
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))