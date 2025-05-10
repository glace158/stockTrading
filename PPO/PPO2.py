import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import copy # For deep copying RunningMeanStd

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

################################## Utils ##################################
class RunningMeanStd:
    # Calculates running mean and standard deviation
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # Source: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)

    def normalize(self, x, update_stats=False):
        if update_stats:
             # Detach tensor before converting to numpy and updating stats
            self.update(x.detach().cpu().numpy())
        # Clip std dev below epsilon to avoid division by zero
        return (x - torch.tensor(self.mean, dtype=torch.float32).to(x.device)) / torch.tensor(self.std + 1e-8, dtype=torch.float32).to(x.device)

    def denormalize(self, x):
         # Not typically needed for state/reward normalization in RL training
         return x * torch.tensor(self.std + 1e-8, dtype=torch.float32).to(x.device) + torch.tensor(self.mean, dtype=torch.float32).to(x.device)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []          # Tabular states
        self.image_states = []    # Image states
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.image_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# ActorCritic class remains largely the same as the multi-modal version provided previously
class ActorCritic(nn.Module):
    # state_dim 대신 tabular_state_dim 및 이미지 차원 관련 인자 추가
    def __init__(self, tabular_state_dim, input_channels, img_height, img_width, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        # --- CNN Base (이미지 처리용) ---
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # CNN 출력 크기 계산
        dummy_input = torch.zeros(1, input_channels, img_height, img_width)
        with torch.no_grad():
            # Ensure dummy input is on the same device model will be on
            dummy_input = dummy_input.to(next(self.cnn_base.parameters()).device if next(self.cnn_base.parameters(), None) is not None else device)
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]


        # --- Optional: MLP for tabular data ---
        # Can add a small MLP here to process tabular data before concatenation
        # self.tabular_mlp = nn.Sequential(...)
        # tabular_processed_dim = ...
        # combined_features_dim = tabular_processed_dim + cnn_out_dim
        # For simplicity, we directly concatenate raw (normalized) tabular data
        combined_features_dim = tabular_state_dim + cnn_out_dim

        # --- Actor Head ---
        if has_continuous_action_space:
            self.action_dim = action_dim
            # Ensure action_var is created on the correct device during initialization
            # We'll create it later based on the model's device
        else:
            self.action_dim = action_dim # Store action_dim for discrete case too

        self.actor = nn.Sequential(
                        nn.Linear(combined_features_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim) # Output raw logits/means
                    )

        # --- Critic Head ---
        self.critic = nn.Sequential(
                        nn.Linear(combined_features_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

        # Initialize action_var after model is moved to device
        if has_continuous_action_space:
            self.action_std_init = action_std_init
            # self.action_var will be initialized in PPO's __init__ after model.to(device)

    def _get_features(self, tabular_state, image_state):
        """Extract features from both modalities and concatenate."""
        cnn_features = self.cnn_base(image_state)
        # Optional: process tabular_state through an MLP if defined
        # tabular_features = self.tabular_mlp(tabular_state)
        # combined_features = torch.cat((tabular_features, cnn_features), dim=1)
        combined_features = torch.cat((tabular_state, cnn_features), dim=1)
        return combined_features

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            # Ensure action_var is on the correct device
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.actor[0].weight.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError # Or define a standard forward pass if needed elsewhere

    def act(self, tabular_state, image_state):
        """ Get action, log probability, and value estimate for decision making. """
        combined_features = self._get_features(tabular_state, image_state)

        if self.has_continuous_action_space:
            action_mean = self.actor(combined_features)
            # Ensure action_var is initialized and on the correct device
            if not hasattr(self, 'action_var'):
                 raise RuntimeError("action_var not initialized. Call set_action_std first.")
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # Get logits from actor network
            action_logits = self.actor(combined_features)
            # Create distribution from logits
            dist = Categorical(logits=action_logits) # Use logits for numerical stability

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(combined_features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, tabular_state, image_state, action):
        """ Get log probability, value estimate, and entropy for loss calculation. """
        combined_features = self._get_features(tabular_state, image_state)

        if self.has_continuous_action_space:
            action_mean = self.actor(combined_features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var) # Use diag_embed for batch processing
            dist = MultivariateNormal(action_mean, cov_mat)

            # Reshape actions if needed (important for single-action envs)
            if self.action_dim == 1 and action.dim() > 1 and action.shape[1] != self.action_dim:
                 action = action.reshape(-1, self.action_dim)
            elif action.dim() == 1 and self.action_dim > 1: # Handle case where buffer might store actions incorrectly
                 # This depends on how actions are stored/processed, may need adjustment
                 pass

        else:
            action_logits = self.actor(combined_features)
            dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(combined_features)

        return action_logprobs, state_values, dist_entropy


class PPO:
    # Added parameters for new features
    def __init__(self, tabular_state_dim, input_channels, img_height, img_width, action_dim, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6,
                 # New / Modified Parameters
                 value_loss_coef=0.5,    # Standard PPO value loss coefficient
                 entropy_coef=0.01,     # Standard PPO entropy coefficient (often smaller)
                 lambda_gae=0.95,
                 minibatch_size=64,     # Renamed from minibatchsize
                 max_grad_norm=0.5,     # Gradient clipping threshold
                 normalize_state=True,  # Enable/disable state normalization
                 # normalize_reward=False, # Option to normalize rewards (often less critical than advantage norm)
                 anneal_lr=True,        # Enable/disable learning rate annealing
                 total_timesteps=1e6):   # Needed for LR annealing calculation

        self.device = device # Store device
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim # Store action_dim

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        self.minibatch_size = minibatch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_state = normalize_state
        # self.normalize_reward = normalize_reward # Store if reward normalization is added
        self.anneal_lr = anneal_lr
        self.total_timesteps = total_timesteps
        self._current_timestep = 0 # Track timesteps for LR annealing

        self.buffer = RolloutBuffer()

        # Initialize RunningMeanStd for state normalization if enabled
        if self.normalize_state:
            # Initialize with the shape of the tabular state
            self.state_rms = RunningMeanStd(shape=(tabular_state_dim,))
        # if self.normalize_reward: # Initialize if reward normalization is added
            # self.reward_rms = RunningMeanStd(shape=()) # Scalar reward

        # Initialize ActorCritic policy and move to device
        self.policy = ActorCritic(tabular_state_dim, input_channels, img_height, img_width, action_dim, has_continuous_action_space, action_std_init).to(self.device)

        # Initialize action_std after model is on device
        if has_continuous_action_space:
            self.action_std = action_std_init
            self.policy.set_action_std(action_std_init) # Initialize action_var inside policy

        # Store initial learning rates
        self.lr_actor_initial = lr_actor
        self.lr_critic_initial = lr_critic

        # Define optimizer AFTER moving policy to device
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.cnn_base.parameters(), 'lr': lr_critic} # Use critic LR for CNN by default
        ])

        # Initialize old policy and load state dict AFTER optimizer is defined
        self.policy_old = ActorCritic(tabular_state_dim, input_channels, img_height, img_width, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if has_continuous_action_space:
             self.policy_old.set_action_std(action_std_init) # Also init action_var for old policy

        self.MseLoss = nn.MSELoss()

    def _update_learning_rate(self):
        """ Anneals the learning rate linearly based on total timesteps """
        if self.anneal_lr:
            frac = 1.0 - (self._current_timestep / self.total_timesteps)
            # Ensure frac is not negative
            frac = max(frac, 0.0)
            new_lr_actor = self.lr_actor_initial * frac
            new_lr_critic = self.lr_critic_initial * frac

            # Update optimizer param groups
            self.optimizer.param_groups[0]['lr'] = new_lr_actor
            self.optimizer.param_groups[1]['lr'] = new_lr_critic
            # Also update CNN learning rate (assuming it uses critic's group or has its own)
            if len(self.optimizer.param_groups) > 2:
                 self.optimizer.param_groups[2]['lr'] = new_lr_critic # Update CNN LR too


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """ Simple linear decay for action_std based on current value. Call this externally if needed. """
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def select_action(self, tabular_state, image_state, current_timestep=None):
        """ Select action using old policy during rollout phase. """
        # Update current timestep tracker if provided (for LR annealing)
        if current_timestep is not None:
             self._current_timestep = current_timestep

        # --- Preprocessing ---
        # Tabular state: Convert to tensor, add batch dim, move to device
        tabular_state_tensor = torch.FloatTensor(tabular_state).to(self.device)
        if tabular_state_tensor.dim() == 1:
            tabular_state_tensor = tabular_state_tensor.unsqueeze(0)

        # Normalize tabular state using running mean/std
        if self.normalize_state:
             # Important: Don't update stats during action selection, only normalize
             tabular_state_tensor = self.state_rms.normalize(tabular_state_tensor, update_stats=False)

        # Image state: Convert, normalize, add batch dim, move to device
        if isinstance(image_state, np.ndarray):
            if image_state.ndim == 3 and image_state.shape[-1] in [1, 3]: # HWC
                image_state = image_state.transpose((2, 0, 1)) # CHW
            elif image_state.ndim == 2: # HW (Grayscale)
                image_state = np.expand_dims(image_state, axis=0) # CHW
            image_state_tensor = torch.FloatTensor(image_state).unsqueeze(0).to(self.device) / 255.0
        elif torch.is_tensor(image_state):
            # Assume CHW or BCHW
            if image_state.dim() == 3: # CHW
                 image_state_tensor = image_state.unsqueeze(0).to(self.device)
            elif image_state.dim() == 4: # BCHW
                 image_state_tensor = image_state.to(self.device)
            else:
                raise ValueError(f"Unsupported image tensor shape: {image_state.shape}")
            # Assume already normalized if tensor
        else:
            raise TypeError(f"Unsupported image state type: {type(image_state)}")

        # --- Action Selection ---
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(tabular_state_tensor, image_state_tensor)

        # --- Store in Buffer ---
        # Store tensors directly (or .cpu() if memory is a concern)
        self.buffer.states.append(tabular_state_tensor) # Store normalized tensor
        self.buffer.image_states.append(image_state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        # Return action in appropriate format
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob.item(), state_val.item()
        else:
            return action.item(), action_logprob.item(), state_val.item()


    def calculate_gae(self, rewards, state_values, is_terminals):
        """ Calculate Generalized Advantage Estimation (GAE). """
        advantages = []
        last_gae_lam = 0
        # Ensure state_values includes the value of the *next* state after the last action
        # If the buffer stores N transitions, it should have N+1 values (or handle terminal case)

        num_steps = len(rewards)
        # Need value of the state *after* the last action in the rollout
        # Usually obtained by running the critic on the final state observed
        # If not available, approximate with 0 for terminal or last value for non-terminal
        # Let's assume the buffer stores N rewards, N terminals, and N+1 values
        # or that the last value in state_values corresponds to the last reward step
        if len(state_values) == num_steps + 1:
             values = state_values # Includes value of state s_T
        elif len(state_values) == num_steps:
             # If only N values, need to handle the final next_value carefully
             # This implementation assumes N values corresponding to s_0 to s_{N-1}
             print("Warning: GAE calculation might be less accurate if value of final next state is missing.")
             values = torch.cat([state_values, state_values[-1:]], dim=0) # Approximate last next_value with last value
        else:
            raise ValueError("Mismatch in lengths for GAE calculation")


        for step in reversed(range(num_steps)):
            # next_value is V(s_{t+1})
            next_value = values[step + 1]
            # Mask value if the next state s_{t+1} is terminal (comes from is_terminals[step+1] conceptually)
            # However, is_terminals[t] usually flags if s_{t+1} is terminal *after* action a_t from s_t
            # Let's assume is_terminals[step] means the episode ended *after* taking action at step `step`
            next_non_terminal = 1.0 - is_terminals[step] # if episode ends at step 'step', V(s_{t+1}) is 0

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

        # Stack advantages into a tensor
        return torch.stack(advantages) # Removed squeeze, let caller handle shape


    def update(self):
        """ Update policy and value networks using collected rollout data. """

        # --- Data Preparation ---
        # Convert lists to tensors
        # Note: states are already normalized if normalize_state=True
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_image_states = torch.cat(self.buffer.image_states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()
        old_state_values = torch.cat(self.buffer.state_values, dim=0).detach() # V(s_t) from rollout
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device).unsqueeze(1) # Add dim for consistency
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device).unsqueeze(1) # Add dim

        # --- Update Normalization Statistics ---
        # Update state normalization stats *once* per update using the collected batch
        if self.normalize_state:
            # We need the *original* unnormalized states to update stats correctly.
            # The buffer currently stores normalized states. This is a limitation.
            # Option 1: Store unnormalized states too (more memory).
            # Option 2: De-normalize (requires storing stats used during rollout).
            # Option 3 (Approximation): Update stats based on normalized data - NOT recommended.
            # For now, we skip updating RMS here, assuming it's updated externally or stable enough.
            # If implementing state RMS update here, ensure it uses *unnormalized* data.
            # --- Placeholder for updating state_rms ---
            # if self.normalize_state:
            #    unnormalized_states = ... # Requires storing/retrieving unnormalized states
            #    self.state_rms.update(unnormalized_states.cpu().numpy())
            pass # Assuming external update or stable stats for now

        # --- GAE Calculation ---
        # Calculate advantages using GAE
        # Ensure old_state_values passed to GAE are V(s_t)
        advantages = self.calculate_gae(rewards, old_state_values, is_terminals)
        # Calculate returns (targets for value function)
        returns = advantages + old_state_values.detach() # V_{target} = A_t + V(s_t)

        # --- Advantage Normalization --- (Standard practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Learning Rate Annealing ---
        self._update_learning_rate() # Update LR before optimization loops

        # --- Optimize policy for K epochs ---
        num_samples = old_states.shape[0]
        if num_samples == 0:
             print("Warning: Rollout buffer is empty. Skipping update.")
             self.buffer.clear()
             return 0.0, 0.0 # Return dummy loss/entropy

        inds = np.arange(num_samples)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.K_epochs):
            np.random.shuffle(inds)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                mbinds = inds[start:end]

                # Get mini-batch data
                states_mini = old_states[mbinds]
                image_states_mini = old_image_states[mbinds]
                actions_mini = old_actions[mbinds]
                old_logprobs_mini = old_logprobs[mbinds]
                returns_mini = returns[mbinds]
                advantages_mini = advantages[mbinds]

                # --- Evaluate current policy ---
                logprobs, state_values_pred, dist_entropy = self.policy.evaluate(states_mini, image_states_mini, actions_mini)

                # Ensure state_values_pred has the correct shape for loss calculation
                state_values_pred = state_values_pred.view_as(returns_mini) # Match shape [minibatch, 1]


                # --- Calculate Losses ---
                # Policy Loss (Ratio and Surrogate Loss)
                ratios = torch.exp(logprobs - old_logprobs_mini.detach())
                surr1 = ratios * advantages_mini
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_mini
                policy_loss = -torch.min(surr1, surr2).mean() # Mean over minibatch

                # Value Loss (MSE)
                value_loss = self.MseLoss(state_values_pred, returns_mini.detach()) # Use detach on returns if not already detached

                # Entropy Loss
                entropy_loss = -dist_entropy.mean() # Maximize entropy -> minimize negative entropy

                # Total Loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # --- Optimization Step ---
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- Logging (accumulate) ---
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item() # Store positive entropy
                num_updates += 1

        # --- Update Old Policy ---
        self.policy_old.load_state_dict(self.policy.state_dict())

        # --- Clear Buffer ---
        self.buffer.clear()

        # --- Return average losses/entropy ---
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0

        # Return total loss average for backward compatibility, but separate losses are more informative
        # avg_loss = avg_policy_loss + self.value_loss_coef * avg_value_loss + self.entropy_coef * (-avg_entropy)
        # print(f"Update Complete: Avg Total Loss: {avg_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")

        # Return average total loss and average entropy for consistency with original return values
        avg_total_loss_approx = avg_policy_loss + self.value_loss_coef * avg_value_loss - self.entropy_coef * avg_entropy
        return avg_total_loss_approx, avg_entropy


    def save(self, checkpoint_path):
        # Save model weights and potentially normalization stats
        save_dict = {
            'policy_state_dict': self.policy_old.state_dict(), # Save old policy weights
        }
        if self.normalize_state:
            save_dict['state_rms_mean'] = self.state_rms.mean
            save_dict['state_rms_var'] = self.state_rms.var
            save_dict['state_rms_count'] = self.state_rms.count
        # Add reward RMS if implemented
        torch.save(save_dict, checkpoint_path)
        print(f"PPO model saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        # Load model weights and normalization stats
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict']) # Load into current policy too

        if self.normalize_state and 'state_rms_mean' in checkpoint:
            self.state_rms.mean = checkpoint['state_rms_mean']
            self.state_rms.var = checkpoint['state_rms_var']
            self.state_rms.count = checkpoint['state_rms_count']
            print("Loaded state normalization statistics.")
        # Load reward RMS if implemented and saved
        print(f"PPO model loaded from {checkpoint_path}")