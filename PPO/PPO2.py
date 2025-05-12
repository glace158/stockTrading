import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from gym import spaces
from typing import Union, Dict, List

from common.buffers import RolloutBuffer
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
                 mlp_features_dim: int = 32, # MLP용
                 use_mlp: bool = False
                ):
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.has_continuous_action_space = has_continuous_action_space
        
        self.action_dim = action_space.shape[0] if has_continuous_action_space else action_space.n
        
        mlp_extractor_input_dim = self._set_features(observation_space, cnn_features_dim, mlp_features_dim, use_mlp)
        # Actor head
        actor_hidden_dim = 64
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim), nn.Tanh()
                        )
            #self.log_std = nn.Parameter(torch.ones(self.action_dim, device=device) * np.log(action_std_init))
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)
        else:
            self.actor = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim),nn.Softmax(dim=-1)
                        )

        # Critic head
        critic_hidden_dim = 64
        self.critic = nn.Sequential(
                        nn.Linear(mlp_extractor_input_dim, critic_hidden_dim), nn.Tanh(),
                        nn.Linear(critic_hidden_dim, critic_hidden_dim), nn.Tanh(),
                        nn.Linear(critic_hidden_dim, 1)
                    )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    # features 설정
    def _set_features(self, observation_space: Union[spaces.Dict, spaces.Box], 
                 cnn_features_dim: int = 64, # CNN용
                 mlp_features_dim: int = 32, # MLP용
                 use_mlp: bool = False
                 ):
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
                if use_mlp:
                    self.features_extractor = MlpExtractor(observation_space, features_dim=mlp_features_dim)
                else:
                    self.features_extractor = IdentityNetwork(observation_space)

            else:
                raise ValueError(f"Unsupported Box observation space shape: {observation_space.shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

        mlp_extractor_input_dim = self.features_extractor.features_dim
        return mlp_extractor_input_dim

    def _get_features(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
            추가 신경망 (CNN, MLP, Identity 등) 반환값 가져오기
        """
        return self.features_extractor(observations)

    def _get_actor_dist_from_features(self, features: torch.Tensor) -> torch.distributions.Distribution:
        """
            features 으로부터 액터(actor) 네트워크를 사용하여 행동의 확률 분포를 반환
        """
        if self.has_continuous_action_space: # 연속 환경일 때
            action_mean = self.actor(features) # 행동 예측 
            action_std_unbatched = torch.exp(self.log_std) # 표준편차의 로그 값(log_std)을 exp()를 취해 실제 표준편차를 구함
            action_std_batched = action_std_unbatched.expand_as(action_mean) #  unbatched 표준 편차를 action_mean의 형태 (batch_size, action_dim)에 맞게 확장
            # MultivariateNormal은 cov_mat 또는 scale_tril을 사용
            # scale_tril=torch.diag_embed(action_std_batched) 가 더 안정적일 수 있음
                
            # 공분산 행렬 생성: 각 행동 차원이 독립적이라고 가정하고, 대각 성분만 있는 공분산 행렬을 만듬
            # action_std_batched.pow(2)는 분산(variance)을 계산
            # torch.diag_embed()는 1D 또는 2D 텐서를 입력받아 대각 행렬 또는 배치된 대각 행렬 생성
            # cov_mat는 각 배치 샘플에 대해 (action_dim, action_dim) 크기의 대각 행렬이 됨
            cov_mat = torch.diag_embed(action_std_batched.pow(2)) # 배치 크기 고려하여 각 항목에 대해 대각행렬 생성
            
            dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat) # 평균과 공분산 행렬을 사용하여 다변수 정규 분포(MultivariateNormal) 생성
        else: # 이산 환경일 때
            action_logits = self.actor(features) # 액터 네트워크(self.actor)는 특징(features)을 입력받아 각 이산 행동에 대한 로직(logits) 예측
            
            # logits은 소프트맥스 함수를 통과하기 전의 값
            # action_logits의 형태는 (batch_size, num_actions)
            dist = Categorical(logits=action_logits) # 카테고리 분포(Categorical distribution) 객체 생성

        return dist # 생성된 행동 확률 분포 반환

    def forward(self):
        raise NotImplementedError
    
    def act(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], deterministic: bool = False):
        """
            행동 예측
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

        action_logprob = dist.log_prob(action) # 확률 분포에서 action이 나타날 로그 확률
        state_val = self.critic(features) # 크리틱에 상태 가치 값 추출

        return action.detach(), action_logprob.detach(), state_val.detach() # 행동, 확률 로그 확률, 가치 함수

    def evaluate(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], actions: torch.Tensor): # 평가
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
                actions = actions.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
        
        
        state_values = self.critic(features) # 크리틱에 상태 가치 값 추출

        action_logprobs = dist.log_prob(actions) # 확률 분포에서 action이 나타날 로그 확률
        dist_entropy = dist.entropy() # 엔트로피 계산

        return action_logprobs, torch.squeeze(state_values), dist_entropy # 확률 로그 확률, 가치 함수, 엔트로피



class PPO:
    def __init__(self, observation_space: Union[spaces.Dict, spaces.Box],
                 action_space: spaces.Space,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 value_loss_coef=0.5, entropy_coef=0.01,
                 lambda_gae=0.95, minibatch_size=64,
                 # ActorCritic 내부 특징 추출기 크기 제어용 파라미터 추가
                 cnn_features_dim: int = 64,
                 mlp_features_dim: int = 32,
                 use_mlp: bool = False
                 ):

        self.observation_space = observation_space
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.minibatchsize = minibatch_size # 미니 배치 사이즈 설정

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                  cnn_features_dim, mlp_features_dim, use_mlp).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        #if self.has_continuous_action_space:
        #     self.optimizer.add_param_group({'params': self.policy.log_std, 'lr': lr_log_std})
        if self.policy.features_extractor != None:
             self.optimizer.add_param_group({'params': self.policy.features_extractor.parameters(), 'lr': lr_actor})

        self.policy_old = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                      cnn_features_dim, mlp_features_dim, use_mlp).to(device)
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
        """
            행동 분포의 표준편차 감소
            표준편차가 감소함에 따라 행동 탐색이 줄어든다.
        """
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
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
                if value.dtype == np.uint8 and (key == "image" or "img" in key or (value.ndim >=2 and value.shape[-1] <=4)): # 이미지로 추정
                    # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                    tensor_val = torch.as_tensor(value, device=device).float() / 255.0
                else: # 이미지가 아닌 일반 데이터
                    tensor_val = torch.as_tensor(value, device=device).float() # 텐서로 변환, float

                # 현재 observation_space의 해당 subspace를 참조하여 원래 차원과 비교
                #if len(tensor_val.shape) == len(self.observation_space[key].shape):
                #    tensor_val = tensor_val.unsqueeze(0) # 0번 축에 배치 차원(크기 1) 추가

                tensor_dict[key] = tensor_val # 변환된 텐서를 딕셔너리에 저장
            return tensor_dict # 텐서로 채워진 딕셔너리 반환
        
        else: # 단일 Numpy 배열
            if obs.dtype == np.uint8 and obs.ndim >=2 : # (H,W) 또는 (H,W,C) 이미지로 추정
                tensor_obs = torch.as_tensor(obs, device=device).float() / 255.0
            else: # 일반 데이터
                tensor_obs = torch.as_tensor(obs, device=device).float() # 텐서로 변환, float

            # gymnasium의 Box.sample()은 배치 차원 없이 반환, env.step()도 마찬가지
            #if len(tensor_obs.shape) == len(self.observation_space.shape):
            #    tensor_obs = tensor_obs.unsqueeze(0) # 0번 축에 배치 차원(크기 1) 추가

            return tensor_obs
        
    def _obs_to_tensor_batch(self):
        """
            배치 학습 시 사용
            관찰(딕셔너리 또는 단일 Numpy 배열)을 PyTorch 텐서로 변환
            필요한 경우 정규화(이미지) 및 배치 차원 추가
        """
        # Convert list to tensor
        if isinstance(self.buffer.states[0], dict):
            batched_observations = {}
            # 첫 번째 관찰의 키를 기준으로 모든 관찰을 묶음
            dict_keys = self.buffer.states[0].keys()
            for key in dict_keys:
                obs_list_for_key = []
                for obs_dict_item in self.buffer.states:
                    val = obs_dict_item[key]
                    # 이미지 정규화 (uint8 -> float / 255.0)
                    if val.dtype == np.uint8 and (key == "image" or "img" in key or (val.ndim >=2 and val.shape[-1] <=4)):
                        # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                        val = val.astype(np.float32) / 255.0

                    obs_list_for_key.append(val.cpu().numpy())

                batched_observations[key] = torch.as_tensor(np.stack(obs_list_for_key), device=device).float()
        else: # 단일 Numpy 배열 리스트
            obs_list = []
            for obs_item in self.buffer.states:
                val = obs_item
                if val.dtype == np.uint8 and val.ndim >=2: # 이미지로 추정
                    # 이미지는 uint8 타입일 경우 0-1 사이로 정규화하기 위해 255.0로 나눔
                    val = val.astype(np.float32) / 255.0

                obs_list.append(val.cpu().numpy())
            batched_observations = torch.as_tensor(np.stack(obs_list), device=device).float()

        return batched_observations

    def select_action(self, observation: Union[Dict[str, np.ndarray], np.ndarray], deterministic: bool = False): # 행동 선택
        """
            행동 선택하기
        """
        with torch.no_grad():
            state = self._obs_to_tensor(observation)
            action, action_logprob, state_val = self.policy_old.act(state, deterministic)

        #print(state.shape) # 2
        #print(action.shape) # 1,1
        #print(action_logprob.shape) # 1
        #print(state_val.shape)# 1
        #print("==========================")
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_val
        else:
            # 배치 차원이 있다면 squeeze, 없다면 item() (단일 환경에서 단일 행동)
            return action.squeeze().item() if action.numel() == 1 else action.detach().cpu().numpy(), action_logprob, state_val
           
    def calculate_gae(self, rewards: torch.Tensor, state_values: torch.Tensor, is_terminals: torch.Tensor):
        """ Generalized Advantage Estimation (GAE) """
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1: # 마지막 스텝
                next_non_terminal = 1.0 - is_terminals[step]
                next_value = state_values[step]
            else:
                next_non_terminal = 1.0 - is_terminals[step + 1]
                next_value = state_values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - state_values[step]
            last_gae_lam = delta + self.gamma * self.lambda_gae * last_gae_lam
            advantages[step] = last_gae_lam
        
        # Returns는 GAE advantage + value estimates
        returns = advantages + state_values
        return advantages, returns
    

    def update(self): # 정책 업데이트

        batched_states = self._obs_to_tensor_batch() # 상태 값 텐서로 변환
        old_actions_list = [a.to(device) for a in self.buffer.actions] 
        old_logprobs_list = [lp.to(device) for lp in self.buffer.logprobs]
        old_state_values_list = [sv.to(device) for sv in self.buffer.state_values]
        
        old_actions = torch.cat(old_actions_list, dim=0).squeeze(-1)
        
        old_logprobs = torch.cat(old_logprobs_list, dim=0).squeeze(-1)
        old_state_values = torch.cat(old_state_values_list, dim=0).squeeze(-1)

        if not self.has_continuous_action_space and old_actions.ndim > 1 and old_actions.shape[-1] == 1:
            old_actions = old_actions.squeeze(-1)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)
        
        #print(batched_states.shape)
        #print(old_actions.shape)
        #print(old_logprobs.shape)
        #print(old_state_values.shape)
        #print("=====================================================")
        
        # calculate advantages
        advantages, returns = self.calculate_gae(rewards, old_state_values, is_terminals) # GAE 어드밴티지 계산 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 어드밴티지 정규화


        # Optimize policy for K epochs
        num_samples = len(self.buffer.states) # 버퍼 샘플링
        inds = np.arange(num_samples)

        for _ in range(self.K_epochs):
            np.random.shuffle(inds) # 인덱스 섞기
            for start in range(0, num_samples, self.minibatchsize): # 0부터 배치 사이즈까지 학습 배치 간격으로 반복
                end = start + self.minibatchsize # minibatchsize 간격만큼 설정
                mbinds = inds[start:end] # 학습 배치 사이즈에 맞게 인덱스 슬라이싱

                if isinstance(batched_states, dict):# 상태가 딕셔너리 형태이면
                    old_states_mini = { 
                        key: batched_states[key][mbinds] for key in batched_states
                    }
                else: # 일반 데이터이면
                    old_states_mini = batched_states[mbinds]

                # 미니 배치 추출
                old_actions_mini = old_actions[mbinds]
                old_logprobs_mini = old_logprobs[mbinds]
                advantages_mini = advantages[mbinds]
                returns_mini = returns[mbinds]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_mini, old_actions_mini)

                # Finding the ratio (pi_theta / pi_theta__old) (정책 확률 계산)
                ratios = torch.exp(logprobs - old_logprobs_mini.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * advantages_mini 
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_mini

                #returns = advantages[mbinds] + old_state_values_mini

                policy_loss = -torch.min(surr1, surr2).mean() # 정책 손실
                
                value_loss = self.MseLoss(state_values, returns_mini) # 가치 손실
                entropy_loss = -dist_entropy.mean() # 엔트로피

                # final loss of clipped objective PPO
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # 그래디언트 클래핑
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        
        return loss.item(), entropy_loss.item()
        #return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save(self, checkpoint_path): 
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print(f"Saved model to {checkpoint_path}")
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        print(f"Loaded model from {checkpoint_path}")
