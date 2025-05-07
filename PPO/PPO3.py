import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import gym as gym # gymnasium 사용 명시
from gym import spaces

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
        # self.states = [] # 이제 states는 딕셔너리 리스트가 될 것임
        self.dict_observations = [] # SB3 스타일로 명칭 변경 및 딕셔너리 저장용
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.dict_observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# SB3의 CnnExtractor와 유사한 간단한 CNN 모듈
class SimpleCnnExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64): # SB3 CnnExtractor의 기본 features_dim은 512
        super().__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1), # 채널 수, 필터 크기 등은 데이터에 맞게 조절
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_out_dim, features_dim), nn.ReLU())
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    @property
    def features_dim(self):
        return self._features_dim

class CombinedFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        extractors = {}
        total_features_dim = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # 이미지 데이터용 CNN 추출기 (실제 이미지 크기에 맞게 CNN 내부 파라미터 조절 필요)
                # 여기서는 SimpleCnnExtractor의 features_dim을 임의로 64로 설정
                extractors[key] = SimpleCnnExtractor(subspace, features_dim=64)
                total_features_dim += extractors[key].features_dim
            elif key == "numerical":
                # 수치형 데이터는 Flatten 후 바로 사용하거나 간단한 MLP 추가 가능
                extractors[key] = nn.Flatten()
                total_features_dim += np.prod(subspace.shape) # Flatten 후의 크기
            # 필요시 다른 타입의 데이터 처리 로직 추가
        
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_features_dim

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

    @property
    def features_dim(self):
        return self._features_dim


class ActorCritic(nn.Module):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Space,
                 has_continuous_action_space: bool, action_std_init: float):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_space.shape[0] if has_continuous_action_space else action_space.n

        # Feature Extractor
        self.features_extractor = CombinedFeaturesExtractor(observation_space)
        mlp_extractor_input_dim = self.features_extractor.features_dim

        # Actor head
        # SB3는 policy_kwargs로 actor/critic head의 net_arch를 지정할 수 있게 함
        # 여기서는 고정된 MLP 구조 사용
        actor_hidden_dim = 64 # 예시
        if has_continuous_action_space:
            self.actor_net = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim),
                            nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim),
                            nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim),
                            nn.Tanh() # 값의 범위를 -1 ~ 1로 제한 (필요시 조절)
                        )
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * np.log(action_std_init)) # SB3는 종종 log_std를 학습 파라미터로 사용
        else:
            self.actor_net = nn.Sequential(
                            nn.Linear(mlp_extractor_input_dim, actor_hidden_dim),
                            nn.Tanh(),
                            nn.Linear(actor_hidden_dim, actor_hidden_dim),
                            nn.Tanh(),
                            nn.Linear(actor_hidden_dim, self.action_dim) # Softmax는 Categorical 분포에서 처리
                        )

        # Critic head
        critic_hidden_dim = 64 # 예시
        self.critic_net = nn.Sequential(
                        nn.Linear(mlp_extractor_input_dim, critic_hidden_dim),
                        nn.Tanh(),
                        nn.Linear(critic_hidden_dim, critic_hidden_dim),
                        nn.Tanh(),
                        nn.Linear(critic_hidden_dim, 1)
                    )

    def _get_features(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.features_extractor(observations)

    def _get_actor_dist_from_features(self, features: torch.Tensor) -> torch.distributions.Distribution:
        if self.has_continuous_action_space:
            action_mean = self.actor_net(features)
            action_std = torch.exp(self.log_std)
            # SB3는 SquashedNormal 또는 DiagGaussianDistribution 사용. 여기서는 MultivariateNormal 간소화 버전.
            # cov_mat = torch.diag_embed(action_std.pow(2)).to(device) # 더 정확한 표현
            cov_mat = torch.diag(action_std.pow(2)).unsqueeze(0).repeat(features.size(0), 1, 1) # 배치처리 고려
            dist = MultivariateNormal(action_mean, scale_tril=torch.cholesky(cov_mat)) # scale_tril 사용 권장
        else:
            action_logits = self.actor_net(features)
            dist = Categorical(logits=action_logits)
        return dist

    def act(self, observations: dict[str, torch.Tensor], deterministic: bool = False):
        # 관찰값을 device로 옮기고 배치 차원 추가 (단일 관찰인 경우)
        # 이 부분은 PPO.select_action에서 처리하는 것이 더 적합할 수 있음
        # for key in observations:
        #     observations[key] = torch.as_tensor(observations[key], device=device).float().unsqueeze(0)

        features = self._get_features(observations)
        dist = self._get_actor_dist_from_features(features)

        if deterministic:
            if self.has_continuous_action_space:
                action = dist.mean
            else:
                action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic_net(features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate_actions(self, observations: dict[str, torch.Tensor], actions: torch.Tensor):
        features = self._get_features(observations)
        dist = self._get_actor_dist_from_features(features)
        state_values = self.critic_net(features)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_values), dist_entropy

    # set_action_std 관련 로직은 log_std를 직접 조절하는 방식으로 변경하거나,
    # 고정된 std를 사용한다면 기존 action_var 방식을 유지할 수 있습니다.
    # SB3는 보통 학습 가능한 log_std를 사용하거나, 초기값을 주고 점차 줄이지는 않습니다.
    # 여기서는 action_std_init으로 초기화된 log_std를 사용합니다.


class PPO:
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Space,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,
                 value_loss_coef=0.5, entropy_coef=0.01, # SB3 기본값 참조
                 lambda_gae=0.95, minibatch_size=64): # SB3 기본값 참조

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

        self.policy = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init).to(device)
        # SB3는 actor와 critic 파라미터를 분리하지 않고 하나의 optimizer를 사용하는 경우가 많음.
        # 또는 AdamW를 사용. 여기서는 기존 방식 유지.
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.features_extractor.parameters(), 'lr': lr_actor}, # 특징 추출기도 학습
            {'params': self.policy.actor_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_net.parameters(), 'lr': lr_critic}
        ])
        if self.has_continuous_action_space: # log_std도 학습 대상에 포함
             self.optimizer.add_param_group({'params': self.policy.log_std, 'lr': lr_actor})


        self.policy_old = ActorCritic(observation_space, action_space, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    # action_std decay 관련 로직은 SB3에서는 일반적이지 않음. 필요하다면 유지 가능.
    # def decay_action_std(...)

    def _obs_to_tensor(self, obs_dict: dict) -> dict[str, torch.Tensor]:
        """ 딕셔너리 형태의 관찰을 텐서 딕셔너리로 변환하고 배치 차원 추가 """
        tensor_dict = {}
        for key, value in obs_dict.items():
            # GPU로 옮기고 float 타입으로 변환, 배치 차원(0) 추가
            # 주의: 이미지가 uint8이면 0-255 범위이므로 /255.0 처리가 필요할 수 있음
            tensor_obs = torch.as_tensor(value, device=device).float()
            if tensor_obs.ndim == self.observation_space[key].shape.__len__(): # 배치 차원 없을 시
                 tensor_obs = tensor_obs.unsqueeze(0)
            tensor_dict[key] = tensor_obs
        return tensor_dict

    def select_action(self, observation_dict: dict): # state -> observation_dict
        # 관찰값을 텐서로 변환 (GPU로 이동, 배치 차원 추가 등)
        # SB3에서는 이 처리를 내부 유틸 (obs_as_tensor)이나 VecEnv에서 담당
        with torch.no_grad():
            tensor_observation_dict = self._obs_to_tensor(observation_dict)
            action, action_logprob, state_val = self.policy_old.act(tensor_observation_dict)

        # 버퍼에 원본 numpy observation_dict 저장 (나중에 텐서로 변환)
        self.buffer.dict_observations.append(observation_dict)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_val
        else:
            return action.item(), action_logprob, state_val # action.item()은 단일 환경 가정

    def calculate_gae(self, rewards: torch.Tensor, state_values: torch.Tensor, is_terminals: torch.Tensor, last_value: torch.Tensor, last_done: torch.Tensor):
        """ GAE 계산 (SB3의 OnPolicyAlgorithm._get_advantages와 유사하게 수정) """
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done # 마지막 스텝의 다음 상태는 제공된 last_value
                next_values = last_value
            else:
                next_non_terminal = 1.0 - is_terminals[step + 1]
                next_values = state_values[step + 1]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - state_values[step]
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        # Returns는 GAE advantage + value estimates
        returns = advantages + state_values
        return advantages, returns


    def update(self):
        # 마지막 상태의 가치 추정 (학습 안정성을 위해 필요)
        # 이 부분은 rollout 수집 시 마지막 상태에 대해 policy_old.critic_net(features)를 한번 더 호출해야 함.
        # 여기서는 간단히 0으로 가정하거나, 버퍼의 마지막 state_value를 사용.
        # 정확하려면 rollout 수집 루프에서 마지막 next_observation에 대한 value를 계산해야 함.
        with torch.no_grad():
            last_obs_tensor = self._obs_to_tensor(self.buffer.dict_observations[-1]) # 예시: 단순 마지막 관찰 사용
            features = self.policy_old._get_features(last_obs_tensor)
            last_value = self.policy_old.critic_net(features).reshape(-1)
            # 실제로는 마지막 에피소드가 done이 아니었다면 이 값을 사용, done이면 0
            # 이 로직은 rollout 수집 시점에 더 명확하게 처리되어야 합니다.
            # 여기서는 is_terminals를 통해 done 여부를 알 수 있으므로, last_done은 is_terminals[-1]로 간주
            last_done = torch.as_tensor([self.buffer.is_terminals[-1]], dtype=torch.float32, device=device)


        # Convert list to tensor
        # 1. Observations 딕셔너리 리스트 -> 딕셔너리 of 텐서 (각 키별로 텐서화)
        batched_observations = {}
        for key in self.buffer.dict_observations[0].keys():
            # np.stack을 사용하여 리스트 내 numpy 배열들을 하나의 큰 배열로 합침
            # 이후 torch.as_tensor로 변환
            # 주의: 이미지 데이터가 (H, W, C)라면 (C, H, W)로 transpose 필요할 수 있음 (CNN 입력 형식에 따라)
            # self.observation_space[key].shape를 참조하여 올바른 형태로 stack
            if key == "image" and self.observation_space[key].shape.__len__() == 3: # (C, H, W)
                 stacked_obs = np.stack([obs[key] for obs in self.buffer.dict_observations])
            elif key == "numerical" and self.observation_space[key].shape.__len__() == 1: # (Dim,)
                 stacked_obs = np.array([obs[key] for obs in self.buffer.dict_observations])
            else: # 일반적인 경우
                 stacked_obs = np.array([obs[key] for obs in self.buffer.dict_observations])

            batched_observations[key] = torch.as_tensor(stacked_obs, device=device).float()


        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        rewards_tensor = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        is_terminals_tensor = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)


        # Calculate GAE and returns
        advantages, returns = self.calculate_gae(rewards_tensor, old_state_values, is_terminals_tensor, last_value, last_done)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 어드밴티지 정규화

        num_samples = len(self.buffer.dict_observations)
        indices = np.arange(num_samples)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                # 미니 배치 추출
                # 각 관찰 딕셔너리에서 미니배치 인덱싱
                mini_batch_obs = {
                    key: batched_observations[key][minibatch_indices] for key in batched_observations
                }
                mini_batch_actions = old_actions[minibatch_indices]
                mini_batch_old_logprobs = old_logprobs[minibatch_indices]
                # mini_batch_old_state_values = old_state_values[minibatch_indices] # GAE 사용시에는 직접 사용 X
                mini_batch_advantages = advantages[minibatch_indices]
                mini_batch_returns = returns[minibatch_indices]


                logprobs, state_values, dist_entropy = self.policy.evaluate_actions(mini_batch_obs, mini_batch_actions)

                ratios = torch.exp(logprobs - mini_batch_old_logprobs.detach())
                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mini_batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values, mini_batch_returns) # SB3는 주로 clipped value loss 사용 안함
                entropy_loss = -dist_entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # SB3 기본 max_grad_norm
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        # loss 값들 리턴 (모니터링용)
        return policy_loss.item(), value_loss.item(), entropy_loss.item()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))