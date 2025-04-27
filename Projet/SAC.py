import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
from tqdm import tqdm
from collections import deque
import os
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for the racetrack environment
config_dict = {
    'action': {
        'lateral': True,
        'longitudinal': False,
        'target_speeds': [0, 20, 40, 60],
        'type': 'ContinuousAction'
    },
    'duration': 120,
    'controlled_vehicles': 1,
    'observation': {
        'align_to_vehicle_axes': True,
        'as_image': False,
        'features': ['presence', 'velocity', 'acceleration'],
        'grid_size': [[-30, 30], [-30, 30]],
        'grid_step': [5, 5],
        'type': 'OccupancyGrid'
    },
    "collision_reward": -10,
    "action_reward": -0.05,
    "show_trajectories": True,
    "lane_centering_cost": 2,
    "progress_reward": 8,
    'other_vehicles': 3,
    'render_agent': True,
    'manual_control': False
    }

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states), 
                np.array(dones, dtype=np.uint8))
    
    def __len__(self):
        return len(self.buffer)

# Actor Network for SAC
class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, hidden_size, action_dim, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Convolutional layers for processing grid data
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 64 * 12 * 12 
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        # Reshape for convolutional processing
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 12, 12)  # Adjust dimensions based on your grid
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Use reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        
        # Squash using tanh to bound actions between -1 and 1
        action = torch.tanh(z)
        
        # Calculate log probability, incorporating the Jacobian adjustment for tanh
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

# Critic Network for SAC (Twin Q-networks)
class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_size):
        super(CriticNetwork, self).__init__()
        
        # Process observation with convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 64 * 12 * 12
        
        # Q1 architecture with Layer Normalization
        self.q1_fc1 = nn.Linear(conv_output_size + action_dim, hidden_size)
        self.q1_ln1 = nn.LayerNorm(hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_ln2 = nn.LayerNorm(hidden_size)
        self.q1 = nn.Linear(hidden_size, 1)
        
        # Q2 architecture with Layer Normalization
        self.q2_fc1 = nn.Linear(conv_output_size + action_dim, hidden_size)
        self.q2_ln1 = nn.LayerNorm(hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_ln2 = nn.LayerNorm(hidden_size)
        self.q2 = nn.Linear(hidden_size, 1)
        
    def forward_conv(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 12, 12)  # Reshape for conv layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(batch_size, -1)  # Flatten
        
    def forward(self, state, action):
        state_features = self.forward_conv(state)
        sa = torch.cat([state_features, action], 1)
        
        # Q1 estimate
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1(q1)
        
        # Q2 estimate
        q2 = F.relu(self.q2_ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2(q2)
        
        return q1, q2

# SAC Agent
class SAC:
    def __init__(self, env, hidden_size=256, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.005, alpha=0.2, auto_entropy_tuning=True, 
                 buffer_size=1000000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.obs_shape = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
        self.action_dim = env.action_space.shape[0]
        self.action_high = env.action_space.high[0]
        self.action_low = env.action_space.low[0]
        
        # Initialize networks
        self.actor = ActorNetwork(self.obs_shape, hidden_size, self.action_dim).to(device)
        self.critic = CriticNetwork(self.obs_shape, self.action_dim, hidden_size).to(device)
        self.critic_target = CriticNetwork(self.obs_shape, self.action_dim, hidden_size).to(device)
        
        # Copy parameters from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        self.total_steps = 0
        self.episodes = 0
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if evaluate:
            # Use mean action for evaluation (no exploration)
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
        else:
            # Sample action for training
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)
        
        # Convert to torch tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
        
        # Compute target Q value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            next_q1, next_q2 = self.critic_target(next_state_batch, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actor loss
        action, log_prob = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, action)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Update alpha if needed
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update of target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return critic_loss.item(), actor_loss.item()
    
    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(directory, "critic_target.pth"))
        
        if self.auto_entropy_tuning:
            torch.save(self.log_alpha, os.path.join(directory, "log_alpha.pth"))
    
    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth")))
        self.critic_target.load_state_dict(torch.load(os.path.join(directory, "critic_target.pth")))
        
        if self.auto_entropy_tuning and os.path.exists(os.path.join(directory, "log_alpha.pth")):
            self.log_alpha = torch.load(os.path.join(directory, "log_alpha.pth"))
            self.alpha = self.log_alpha.exp()

def evaluate_agent(env, agent, num_episodes=5):
    """Evaluate agent performance"""
    rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            on_road = info["rewards"].get("on_road_reward", True)
            done = terminated or (not on_road)
            
            episode_reward += reward
            obs = next_obs
            
            if terminated and reward > 0:  # Success condition
                success_count += 1
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    success_rate = success_count / num_episodes
    
    print(f"Evaluation over {num_episodes} episodes: {avg_reward:.3f} average reward, {success_rate:.2f} success rate")
    return avg_reward, success_rate

def train_agent(env, agent, max_episodes=300, max_steps=100, batch_size=256, 
                updates_per_step=1, eval_interval=10, initial_random_steps=5000,
                save_path="models_sac"):
    """Train the SAC agent"""
    os.makedirs(save_path, exist_ok=True)
    best_eval_reward = -float('inf')
    
    # Tracking metrics
    rewards = []
    eval_rewards = []
    success_rates = []
    
    # Implement curriculum learning
    current_config = config_dict.copy()
    current_config["other_vehicles"] = 0
    env.unwrapped.configure(current_config)
    
    # Difficulty schedule
    difficulty_schedule = {
        50: {"other_vehicles": 1},
        100: {"other_vehicles": 2},
        150: {"other_vehicles": 3},
    }
    
    pbar = tqdm(range(1, max_episodes + 1), desc="Training")
    total_steps = 0
    
    for episode in pbar:
        # Update difficulty based on schedule
        if episode in difficulty_schedule:
            for key, value in difficulty_schedule[episode].items():
                current_config[key] = value
                env.unwrapped.configure(current_config)
                print(f"Increasing difficulty: {key} = {value}")
        
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_steps < max_steps:
            # Select action
            if total_steps < initial_random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check if vehicle is on road
            on_road = info["rewards"].get("on_road_reward", True)
            done = terminated or (not on_road)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, float(done or truncated))
            
            # Update agent
            if total_steps >= initial_random_steps and len(agent.memory) > batch_size:
                for _ in range(updates_per_step):
                    critic_loss, actor_loss = agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done or truncated:
                break
        
        # Record episode reward
        rewards.append(episode_reward)
        agent.episodes += 1
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'steps': total_steps
        })
        
        # Evaluate agent
        if episode % eval_interval == 0:
            avg_reward, success_rate = evaluate_agent(env, agent)
            eval_rewards.append(avg_reward)
            success_rates.append(success_rate)
            
            # Save if best model
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                agent.save(os.path.join(save_path, 'best_model'))
                print(f"New best model saved with reward: {best_eval_reward:.3f}")
            
            # Plot training progress
            plot_training_progress(rewards, eval_rewards, success_rates, eval_interval, save_path)
    
    # Save final model
    agent.save(os.path.join(save_path, 'final_model'))
    
    return agent, rewards, eval_rewards, success_rates

def plot_training_progress(rewards, eval_rewards, success_rates, eval_interval, save_path):
    """Plot training progress metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    if len(rewards) > 100:
        rolling_mean = [np.mean(rewards[max(0, i-100):i]) for i in range(1, len(rewards)+1)]
        plt.plot(rolling_mean, 'r-')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.grid(True)
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    eval_x = np.arange(0, len(rewards), eval_interval)[:len(eval_rewards)]
    plt.plot(eval_x, eval_rewards, 'go-')
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Rewards')
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(2, 2, 3)
    plt.plot(eval_x, success_rates, 'bo-')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate')
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_progress.png'))
    plt.close()

def visualize_episode(env, agent):
    """Visualize one episode with the trained agent"""
    obs, _ = env.reset()
    done = False
    truncated = False
    frames = []
    total_reward = 0
    
    while not (done or truncated):
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Get action
        action = agent.select_action(obs, evaluate=True)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        on_road = info["rewards"].get("on_road_reward", True)
        done = terminated or (not on_road)
        
        total_reward += reward
        obs = next_obs
    
    print(f"Episode reward: {total_reward}")
    
    # Display frames
    plt.figure(figsize=(10, 8))
    for i, frame in enumerate(frames):
        if i % 10 == 0:  # Display every 10th frame
            plt.clf()
            plt.imshow(frame)
            plt.title(f"Frame {i}")
            plt.pause(0.01)
    plt.close()

def main():
    # Create environment
    env = gym.make("racetrack-v0", render_mode='rgb_array')
    env.unwrapped.configure(config_dict)
    env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    """
    # Create agent
    agent = SAC(
        env=env,
        hidden_size=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True
    )
    
    # Train agent
    agent, rewards, eval_rewards, success_rates = train_agent(
        env=env,
        agent=agent,
        max_episodes=800,
        max_steps=100,
        batch_size=256,
        updates_per_step=1,
        eval_interval=50,
        initial_random_steps=1500,
        save_path="models_sac"
    )
    """
    # Switch to human render mode for visualization
    env = gym.make("racetrack-v0", render_mode='human')
    env.unwrapped.configure(config_dict)
    env.reset()
    
    # Load best model
    best_agent = SAC(env)
    best_agent.load("./models_sac/best_model")
    
    # Visualize a few episodes
    for _ in range(5):
        visualize_episode(env, best_agent)
    
    # Close environments
    env.close()

if __name__ == "__main__":
    main()