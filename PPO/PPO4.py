import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import gym as gym # gymnasium 사용
from gym import spaces
from typing import Union, Dict, List

################################## set device ##################################
print("============================================================================================")
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
        self.observations = [] # 딕셔너리 또는 단일 텐서(Numpy) 저장
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# SB3의 CnnExtractor와 유사한 간단한 CNN 모듈
class SimpleCnnExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__()
        n_input_channels = observation_space.shape[0] # (C, H, W) 가정
        # 만약 입력이 (H, W, C)라면, forward에서 transpose 필요 또는 여기서 shape 조정
        if len(observation_space.shape) == 3 and observation_space.shape[0] > observation_space.shape[2] and observation_space.shape[2] <=4 : # H,W,C일 가능성
             print(f"Warning: SimpleCnnExtractor input shape {observation_space.shape} might be HWC. Assuming CHW for Conv2d.")
             # 실제로는 환경 래퍼에서 CHW로 바꾸는 것이 좋음

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            # observation_space.sample()은 (C,H,W) 또는 (H,W,C)를 반환할 수 있음.
            # CNN은 (N,C,H,W)를 기대함.
            dummy_input_shape = (1, *observation_space.shape) # (1, C, H, W)
            dummy_input = torch.as_tensor(np.zeros(dummy_input_shape)).float()
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_out_dim, features_dim), nn.ReLU())
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 입력이 (N, H, W, C)이고 CNN이 (N, C, H, W)를 기대하면 여기서 transpose
        # if observations.ndim == 4 and observations.shape[1] > observations.shape[3]: # Basic HWC check
        #     observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))

    @property
    def features_dim(self):
        return self._features_dim

class MlpExtractor(nn.Module):
    """단일 벡터 입력을 위한 간단한 MLP 특징 추출기"""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        input_dim = np.prod(observation_space.shape)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            # 필요시 레이어 추가
        )
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(observations))

    @property
    def features_dim(self):
        return self._features_dim

class CombinedFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64, mlp_features_dim: int = 32):
        super().__init__()
        extractors = {}
        total_features_dim = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box) and len(subspace.shape) >= 2: # 보통 이미지 (C,H,W) or (H,W) 등
                 # 이미지 subspace의 채널 수가 1이고 shape이 (H,W)이면 (1,H,W)로 간주
                if len(subspace.shape) == 2: # (H,W) -> (1,H,W)로 가정
                    img_obs_space = spaces.Box(low=subspace.low.reshape(1, *subspace.shape),
                                               high=subspace.high.reshape(1, *subspace.shape),
                                               shape=(1, *subspace.shape),
                                               dtype=subspace.dtype)
                    extractors[key] = SimpleCnnExtractor(img_obs_space, features_dim=cnn_features_dim)
                else:
                    extractors[key] = SimpleCnnExtractor(subspace, features_dim=cnn_features_dim)
                total_features_dim += extractors[key].features_dim
            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1: # 수치형 벡터
                extractors[key] = MlpExtractor(subspace, features_dim=mlp_features_dim)
                total_features_dim += extractors[key].features_dim
            else: # 기타 (예: Discrete) - 여기서는 Flatten 후 사용
                extractors[key] = nn.Flatten()
                total_features_dim += np.prod(subspace.shape)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

    @property
    def features_dim(self):
        return self._features_dim


class ActorCritic(nn.Module):
    def __init__(self, observation_space: Union[spaces.Dict, spaces.Box],
                 action_space: spaces.Space,
                 has_continuous_action_space: bool, action_std_init: float,
                 cnn_features_dim_dict: int = 64, # CombinedExtractor 내부 CNN용
                 mlp_features_dim_dict: int = 32, # CombinedExtractor 내부 MLP용
                 simple_cnn_features_dim: int = 64, # 단일 이미지 입력용
                 simple_mlp_features_dim: int = 64): # 단일 벡터 입력용
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_space.shape[0] if has_continuous_action_space else action_space.n

        # Feature Extractor
        if isinstance(observation_space, spaces.Dict):
            self.features_extractor = CombinedFeaturesExtractor(observation_space,
                                                                cnn_features_dim=cnn_features_dim_dict,
                                                                mlp_features_dim=mlp_features_dim_dict)
        elif isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) >= 2: # 이미지 또는 2D 이상 데이터
                 # (H,W) -> (1,H,W) 처리
                if len(observation_space.shape) == 2:
                    img_obs_space = spaces.Box(low=np.expand_dims(observation_space.low, axis=0),
                                               high=np.expand_dims(observation_space.high, axis=0),
                                               shape=(1, *observation_space.shape),
                                               dtype=observation_space.dtype)
                    self.features_extractor = SimpleCnnExtractor(img_obs_space, features_dim=simple_cnn_features_dim)
                else: # (C,H,W)
                    self.features_extractor = SimpleCnnExtractor(observation_space, features_dim=simple_cnn_features_dim)

            elif len(observation_space.shape) == 1: # 1D 벡터
                self.features_extractor = MlpExtractor(observation_space, features_dim=simple_mlp_features_dim)
            else:
                raise ValueError(f"Unsupported Box observation space shape: {observation_space.shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

        mlp_extractor_input_dim = self.features_extractor.features_dim

        # Actor head
        actor_hidden_dim = 64
        if has_continuous_action_space:
            self.actor_net = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim), nn.Tanh()
                        )
            self.log_std = nn.Parameter(torch.ones(self.action_dim, device=device) * np.log(action_std_init))
        else:
            self.actor_net = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim), nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim)
                        )

        # Critic head
        critic_hidden_dim = 64
        self.critic_net = nn.Sequential(
                        nn.Linear(mlp_extractor_input_dim, critic_hidden_dim), nn.Tanh(),
                        nn.Linear(critic_hidden_dim, critic_hidden_dim), nn.Tanh(),
                        nn.Linear(critic_hidden_dim, 1)
                    )

    def _get_features(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return self.features_extractor(observations)

    def _get_actor_dist_from_features(self, features: torch.Tensor) -> torch.distributions.Distribution:
        if self.has_continuous_action_space:
            action_mean = self.actor_net(features)
            action_std_unbatched = torch.exp(self.log_std)
            action_std_batched = action_std_unbatched.expand_as(action_mean)
            # MultivariateNormal은 cov_mat 또는 scale_tril을 사용
            # scale_tril=torch.diag_embed(action_std_batched) 가 더 안정적일 수 있음
            cov_mat = torch.diag_embed(action_std_batched.pow(2)) # 배치 크기 고려하여 각 항목에 대해 대각행렬 생성
            dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat)
        else:
            action_logits = self.actor_net(features)
            dist = Categorical(logits=action_logits)
        return dist

    def act(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], deterministic: bool = False):
        features = self._get_features(observations)
        dist = self._get_actor_dist_from_features(features)

        if deterministic:
            if self.has_continuous_action_space:
                action = dist.mean
            else:
                action = torch.argmax(dist.probs, dim=-1) # dim=1 -> dim=-1 for robustness
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic_net(features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_actions(self, observations: Union[Dict[str, torch.Tensor], torch.Tensor], actions: torch.Tensor):
        features = self._get_features(observations)
        dist = self._get_actor_dist_from_features(features)
        state_values = self.critic_net(features)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_values, -1), dist_entropy

class PPO:
    def __init__(self, observation_space: Union[spaces.Dict, spaces.Box],
                 action_space: spaces.Space,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 value_loss_coef=0.5, entropy_coef=0.01,
                 lambda_gae=0.95, minibatch_size=64,
                 # ActorCritic 내부 특징 추출기 크기 제어용 파라미터 추가
                 cnn_features_dim_dict: int = 64,
                 mlp_features_dim_dict: int = 32,
                 simple_cnn_features_dim: int = 64,
                 simple_mlp_features_dim: int = 64):

        self.observation_space = observation_space
        self.action_space = action_space
        self.has_continuous_action_space = has_continuous_action_space

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.minibatch_size = minibatch_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                  cnn_features_dim_dict, mlp_features_dim_dict,
                                  simple_cnn_features_dim, simple_mlp_features_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.features_extractor.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_net.parameters(), 'lr': lr_critic}
        ])
        if self.has_continuous_action_space:
             self.optimizer.add_param_group({'params': self.policy.log_std, 'lr': lr_actor})


        self.policy_old = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init,
                                      cnn_features_dim_dict, mlp_features_dim_dict,
                                      simple_cnn_features_dim, simple_mlp_features_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def _obs_to_tensor(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """ 관찰(딕셔너리 또는 단일 Numpy 배열)을 텐서로 변환하고 배치 차원 추가 """
        if isinstance(obs, dict):
            tensor_dict = {}
            for key, value in obs.items():
                # Gymnasium 환경은 보통 float32 또는 uint8 로 관찰을 반환
                # 이미지는 uint8일 경우 /255.0 정규화 필요
                if value.dtype == np.uint8 and (key == "image" or "img" in key or (value.ndim >=2 and value.shape[-1] <=4)): # 이미지로 추정
                    tensor_val = torch.as_tensor(value, device=device).float() / 255.0
                else:
                    tensor_val = torch.as_tensor(value, device=device).float()

                # 현재 observation_space의 해당 subspace를 참조하여 원래 차원과 비교
                # 예: obs_space_shape = self.observation_space[key].shape
                # if tensor_val.ndim == len(obs_space_shape): # 배치 차원 없을 시
                # gymnasium의 Box.sample()은 배치 차원 없이 반환, env.step()도 마찬가지
                if len(tensor_val.shape) == len(self.observation_space[key].shape):
                     tensor_val = tensor_val.unsqueeze(0)
                tensor_dict[key] = tensor_val
            return tensor_dict
        else: # 단일 Numpy 배열
            if obs.dtype == np.uint8 and obs.ndim >=2 : # (H,W) 또는 (H,W,C) 이미지로 추정
                 tensor_obs = torch.as_tensor(obs, device=device).float() / 255.0
            else:
                 tensor_obs = torch.as_tensor(obs, device=device).float()

            # gymnasium의 Box.sample()은 배치 차원 없이 반환, env.step()도 마찬가지
            if len(tensor_obs.shape) == len(self.observation_space.shape):
                 tensor_obs = tensor_obs.unsqueeze(0)

            # 단일 관찰이 이미지이고 (H,W) 이며 CNN이 (C,H,W)를 기대한다면 여기서 채널 차원 추가
            # 예: if tensor_obs.ndim == 3 and self.policy.features_extractor is SimpleCnnExtractor and tensor_obs.shape[0] != 1 and tensor_obs.shape[0]!=3: # N, H, W
            #    tensor_obs = tensor_obs.unsqueeze(1) # N, 1, H, W (만약 SimpleCnnExtractor의 n_input_channels=1 가정)

            return tensor_obs

    def select_action(self, observation: Union[Dict[str, np.ndarray], np.ndarray], deterministic: bool = False):
        with torch.no_grad():
            tensor_observation = self._obs_to_tensor(observation)
            action, action_logprob, state_val = self.policy_old.act(tensor_observation, deterministic)

        self.buffer.observations.append(observation) # 원본 Numpy 관찰 저장
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_val
        else:
            # 배치 차원이 있다면 squeeze, 없다면 item() (단일 환경에서 단일 행동)
            return action.squeeze().item() if action.numel() == 1 else action.detach().cpu().numpy(), action_logprob, state_val

    def calculate_gae(self, rewards: torch.Tensor, state_values: torch.Tensor, is_terminals: torch.Tensor, last_value: torch.Tensor, last_done: torch.Tensor):
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - is_terminals[step + 1]
                next_values = state_values[step + 1]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - state_values[step]
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + state_values
        return advantages, returns

    def update(self, last_observation: Union[Dict[str, np.ndarray], np.ndarray], last_done_val: bool):
        with torch.no_grad():
            # 마지막 관찰에 대한 가치 계산
            # last_done_val이 True이면 last_value는 0, 아니면 신경망 통과
            if last_done_val:
                last_value = torch.tensor([0.0], device=device)
            else:
                last_obs_tensor = self._obs_to_tensor(last_observation)
                # ActorCritic.act는 (action, logprob, value)를 반환하므로, value만 직접 가져오거나
                # critic 네트워크를 직접 호출 (더 일반적)
                features = self.policy_old._get_features(last_obs_tensor)
                last_value = self.policy_old.critic_net(features).reshape(-1)

            last_done_tensor = torch.as_tensor([last_done_val], dtype=torch.float32, device=device)

        # Convert list to tensor
        if isinstance(self.buffer.observations[0], dict):
            batched_observations = {}
            # 첫 번째 관찰의 키를 기준으로 모든 관찰을 묶음
            dict_keys = self.buffer.observations[0].keys()
            for key in dict_keys:
                obs_list_for_key = []
                for obs_dict_item in self.buffer.observations:
                    val = obs_dict_item[key]
                    # 이미지 정규화 (uint8 -> float / 255.0)
                    if val.dtype == np.uint8 and (key == "image" or "img" in key or (val.ndim >=2 and val.shape[-1] <=4)):
                        val = val.astype(np.float32) / 255.0
                    # 이미지 채널 순서 (H,W,C) -> (C,H,W) 변경 (필요시)
                    # 예: if key == "image" and val.ndim == 3 and val.shape[0] > val.shape[2]:
                    #    val = np.transpose(val, (2, 0, 1))
                    obs_list_for_key.append(val)
                batched_observations[key] = torch.as_tensor(np.stack(obs_list_for_key), device=device).float()
        else: # 단일 Numpy 배열 리스트
            obs_list = []
            for obs_item in self.buffer.observations:
                val = obs_item
                if val.dtype == np.uint8 and val.ndim >=2: # 이미지로 추정
                    val = val.astype(np.float32) / 255.0
                # 채널 순서 변경 또는 채널 차원 추가 (필요시)
                # 예: if val.ndim == 2 and self.observation_space.shape==(1, *val.shape): # (H,W)고 (1,H,W) 기대
                #    val = np.expand_dims(val, axis=0)
                obs_list.append(val)
            batched_observations = torch.as_tensor(np.stack(obs_list), device=device).float()

        old_actions_list = [a.to(device) for a in self.buffer.actions]
        old_logprobs_list = [lp.to(device) for lp in self.buffer.logprobs]
        old_state_values_list = [sv.to(device) for sv in self.buffer.state_values]

        old_actions = torch.cat(old_actions_list, dim=0)
        old_logprobs = torch.cat(old_logprobs_list, dim=0).squeeze(-1)
        old_state_values = torch.cat(old_state_values_list, dim=0).squeeze(-1)

        if not self.has_continuous_action_space and old_actions.ndim > 1 and old_actions.shape[-1] == 1:
            old_actions = old_actions.squeeze(-1)


        rewards_tensor = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        is_terminals_tensor = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)


        advantages, returns = self.calculate_gae(rewards_tensor, old_state_values, is_terminals_tensor, last_value, last_done_tensor)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(self.buffer.observations)
        indices = np.arange(num_samples)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                if isinstance(batched_observations, dict):
                    mini_batch_obs = {
                        key: batched_observations[key][minibatch_indices] for key in batched_observations
                    }
                else:
                    mini_batch_obs = batched_observations[minibatch_indices]

                mini_batch_actions = old_actions[minibatch_indices]
                mini_batch_old_logprobs = old_logprobs[minibatch_indices]
                mini_batch_advantages = advantages[minibatch_indices]
                mini_batch_returns = returns[minibatch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate_actions(mini_batch_obs, mini_batch_actions)

                ratios = torch.exp(logprobs - mini_batch_old_logprobs.detach())
                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mini_batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values, mini_batch_returns)
                entropy_loss = -dist_entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print(f"Saved model to {checkpoint_path}")

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        print(f"Loaded model from {checkpoint_path}")