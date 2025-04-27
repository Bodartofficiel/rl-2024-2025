import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
from tqdm import tqdm
import os
from datetime import datetime

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

class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, hidden_size, action_dim, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Add a convolutional layer for processing grid data
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 32 * 12 * 12 
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        # Reshape for convolutional processing (assuming shape [batch, features, grid_h, grid_w])
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 12, 12)  # Adjust dimensions based on your grid
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, hidden_size):
        super(CriticNetwork, self).__init__()
        obs_size = obs_shape
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        value = self.value(x)
        return value

# PPO Agent
class PPOAgent:
    def __init__(self, env, hidden_size=128, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                 update_epochs=10, minibatch_size=64, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        
        self.obs_shape = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
        self.action_dim = env.action_space.shape[0]
        self.action_high = torch.tensor(env.action_space.high[0])
        self.action_low = torch.tensor(env.action_space.low[0])
        
        self.actor = ActorNetwork(self.obs_shape, hidden_size, self.action_dim)
        self.critic = CriticNetwork(self.obs_shape, hidden_size)

        self.loss_function = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.total_steps = 0
        self.episodes = 0
        
    def get_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs)
            std = torch.exp(log_std) + 1e-8
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            action = torch.clamp(action, min=self.action_low, max=self.action_high)
            log_prob = None if deterministic else dist.log_prob(action).sum(dim=-1)
        return action.numpy()[0], log_prob

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return returns, advantages

    def update(self, trajectories):
        states = torch.FloatTensor(np.array(trajectories['states']))
        actions = torch.FloatTensor(np.array(trajectories['actions']))
        old_log_probs = torch.FloatTensor(np.array(trajectories['log_probs']))
        rewards = np.array(trajectories['rewards'])
        dones = np.array(trajectories['dones'])

        with torch.no_grad():
            values = self.critic(states).squeeze().numpy()
            next_states = torch.FloatTensor(np.array(trajectories['next_state']))
            next_values = self.critic(next_states).squeeze().numpy()

        returns, advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []

        for _ in range(self.update_epochs):
            indices = torch.from_numpy(np.arange(len(states)))
            np.random.shuffle(indices)
            for start_idx in range(0, len(states), self.minibatch_size):
                idx = indices[start_idx:start_idx + self.minibatch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                mean, log_std = self.actor(mb_states)
                std = torch.exp(log_std) + 1e-8
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=1)

                # Entropie explicite comme dans ContinuousPPO
                entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_std.mean()
                entropy_loss = -self.entropy_coef * entropy

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() + entropy_loss
                actor_losses.append(actor_loss.item())

                values = self.critic(mb_states).squeeze()
                value_loss = F.mse_loss(values, mb_returns)
                critic_losses.append(value_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.critic_optimizer.step()

        return np.mean(actor_losses), np.mean(critic_losses)

class Logger:
    def __init__(self, log_dir="logs"):
        # Create log directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"{log_dir}/{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.train_rewards = []
        self.eval_rewards = []
        self.eval_stds = []
        self.success_rates = []
        self.actor_losses = []
        self.critic_losses = []
        self.episodes = []
        
    def log_train_reward(self, episode, reward):
        self.episodes.append(episode)
        self.train_rewards.append(reward)
        
    def log_eval_metrics(self, episode, reward, std, success_rate):
        # Only append if this is a new evaluation point
        if episode not in self.episodes:
            self.episodes.append(episode)
        
        self.eval_rewards.append(reward)
        self.eval_stds.append(std)
        self.success_rates.append(success_rate)
        
    def log_loss(self, actor_loss, critic_loss):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.episodes, self.train_rewards, label='Training Reward')
        
        # Find common evaluation episodes
        eval_episodes = self.episodes[::len(self.episodes)//len(self.eval_rewards)] if len(self.eval_rewards) > 0 else []
        
        if len(eval_episodes) > 0:
            plt.errorbar(eval_episodes[:len(self.eval_rewards)], self.eval_rewards, 
                        yerr=self.eval_stds, fmt='o-', label='Evaluation Reward')
        
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Training and Evaluation Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/rewards.png")
        plt.close()
        
    def plot_success_rate(self):
        if len(self.success_rates) > 0:
            eval_episodes = self.episodes[::len(self.episodes)//len(self.success_rates)]
            
            plt.figure(figsize=(12, 6))
            plt.plot(eval_episodes[:len(self.success_rates)], self.success_rates, 'go-', label='Success Rate')
            plt.xlabel('Episodes')
            plt.ylabel('Success Rate')
            plt.title('Agent Success Rate')
            plt.ylim(0, 1.1)  # Success rate between 0 and 1
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{self.log_dir}/success_rate.png")
            plt.close()
        
    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        episodes = range(1, len(self.actor_losses) + 1)
        plt.plot(episodes, self.actor_losses, label='Actor Loss')
        plt.plot(episodes, self.critic_losses, label='Critic Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/losses.png")
        plt.close()
        
    def save_data(self):
        """Save all metrics to a numpy file for later analysis"""
        np.savez(
            f"{self.log_dir}/training_data.npz",
            episodes=np.array(self.episodes),
            train_rewards=np.array(self.train_rewards),
            eval_rewards=np.array(self.eval_rewards),
            eval_stds=np.array(self.eval_stds),
            success_rates=np.array(self.success_rates),
            actor_losses=np.array(self.actor_losses),
            critic_losses=np.array(self.critic_losses)
        )
        
    def generate_plots(self):
        """Generate all plots at once"""
        self.plot_rewards()
        self.plot_success_rate()
        self.plot_losses()
        self.save_data()
        
    def save_config(self, config):
        """Save configuration parameters"""
        with open(f"{self.log_dir}/config.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

def collect_rollout(env, agent, steps=512):
    """Collect fixed number of steps"""
    obs, _ = env.reset()
    
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'dones': [],
        'next_state': []
    }
    
    episode_reward = 0
    episode_rewards = []
    episode_length = 0
    completed_episodes = 0
    
    for _ in range(steps):
        action, log_prob = agent.get_action(obs)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        on_road = info["rewards"].get("on_road_reward", True)
        env_done = terminated or truncated or (not on_road)
        
        trajectory['states'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['log_probs'].append(log_prob.item())
        trajectory['dones'].append(env_done)
        trajectory['next_state'].append(next_obs)
        
        episode_reward += reward
        episode_length += 1
        
        if env_done:
            obs, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_length = 0
            completed_episodes += 1
        else:
            obs = next_obs
        
        agent.total_steps += 1
    
    # Calculate average reward if any episodes were completed
    avg_episode_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else episode_reward
    
    return trajectory, avg_episode_reward, completed_episodes

def evaluate_agent(env, agent, num_episodes=5):
    """Evaluate agent performance"""
    rewards = []
    successes = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.get_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            on_road = info["rewards"].get("on_road_reward", True)
            done = terminated or truncated or (not on_road)
            
            # Check if episode finished successfully
            if terminated and not truncated and on_road:
                successes += 1
            
            episode_reward += reward
            obs = next_obs
            
        rewards.append(episode_reward)
    
    success_rate = successes / num_episodes
    return np.mean(rewards), np.std(rewards), success_rate

def visualize_episode(env, agent):
    """Visualize one episode"""
    obs, _ = env.reset()
    done = False
    frames = []
    total_reward = 0
    
    while not done:
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Get action
        action, _ = agent.get_action(obs, deterministic=True)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        on_road = info["rewards"].get("on_road_reward", True)
        done = terminated or truncated or (not on_road)
        
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

def train(env, agent, num_episodes=2000, eval_freq=10, save_path="models_ppo_basic", log_dir="logs"):
    """Train the agent"""
    os.makedirs(save_path, exist_ok=True)
    best_eval_reward = -float('inf')
    
    # Initialize logger
    logger = Logger(log_dir)
    
    # Log configuration
    config = {
        "hidden_size": agent.actor.fc1.in_features,
        "lr_actor": agent.actor_optimizer.param_groups[0]['lr'],
        "lr_critic": agent.critic_optimizer.param_groups[0]['lr'],
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "clip_ratio": agent.clip_ratio,
        "update_epochs": agent.update_epochs,
        "minibatch_size": agent.minibatch_size,
        "entropy_coef": agent.entropy_coef,
        "num_episodes": num_episodes,
        "eval_freq": eval_freq
    }
    logger.save_config(config)

    # Start with simpler environment
    current_config = config_dict.copy()
    current_config["other_vehicles"] = 0
    env.unwrapped.configure(current_config)
    
    # Implement curriculum learning
    difficulty_schedule = {
        25: {"other_vehicles": 1},
        30: {"other_vehicles": 2},
        35: {"other_vehicles": 3},
    }

    pbar = tqdm(range(1, num_episodes + 1), desc="Training")
    
    for episode in pbar:
        # Update difficulty based on schedule
        if episode in difficulty_schedule:
            for key, value in difficulty_schedule[episode].items():
                current_config[key] = value
                env.unwrapped.configure(current_config)
                print(f"Increasing difficulty: {key} = {value}")    

        # Collect trajectory
        trajectory, episode_reward, completed_episodes = collect_rollout(env, agent, steps=1024)
        
        # Update agent
        actor_loss, critic_loss = agent.update(trajectory)
        
        # Log metrics
        logger.log_train_reward(episode, episode_reward)
        logger.log_loss(actor_loss, critic_loss)
        
        # Increment episode counter
        agent.episodes += 1
        
        # Update progress bar
        pbar.set_description(f"Episode {episode} | Reward: {episode_reward:.2f}")
        
        # Evaluate agent
        if episode % eval_freq == 0:
            eval_reward, eval_std, success_rate = evaluate_agent(env, agent)
            logger.log_eval_metrics(episode, eval_reward, eval_std, success_rate)
            
            print(f"Episode {episode}/{num_episodes} | Train reward: {episode_reward:.2f} | "
                  f"Eval reward: {eval_reward:.2f} ± {eval_std:.2f} | Success rate: {success_rate:.2f}")
            
            # Generate and save plots
            logger.generate_plots()
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }, os.path.join(save_path, 'best_model.pt'))
                print("Best model saved with eval reward: ", eval_reward)
    
    # Save final model
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
    }, os.path.join(save_path, 'final_model.pt'))
    
    # Generate final plots
    logger.generate_plots()
    
    return agent

def load_agent(env, model_path, hidden_size=512):
    """Load a trained agent"""
    agent = PPOAgent(env, hidden_size=hidden_size)
    
    checkpoint = torch.load(model_path)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    
    return agent

def visualize_training_metrics(log_path):
    """Visualize metrics from a saved log file"""
    data = np.load(log_path)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(data['episodes'], data['train_rewards'], label='Training Reward')
    
    # Find evaluation episodes
    eval_episodes = data['episodes'][::len(data['episodes'])//len(data['eval_rewards'])]
    
    plt.errorbar(eval_episodes[:len(data['eval_rewards'])], data['eval_rewards'], 
                yerr=data['eval_stds'], fmt='o-', label='Evaluation Reward')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot success rate
    plt.figure(figsize=(12, 6))
    plt.plot(eval_episodes[:len(data['success_rates'])], data['success_rates'], 'go-', label='Success Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Agent Success Rate')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    episodes = range(1, len(data['actor_losses']) + 1)
    plt.plot(episodes, data['actor_losses'], label='Actor Loss')
    plt.plot(episodes, data['critic_losses'], label='Critic Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution function
def main():
    # Create environment
    env = gym.make("racetrack-v0", render_mode='rgb_array')

    env.unwrapped.configure(config_dict)
    env.reset()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create agent
    agent = PPOAgent(env, hidden_size=512, lr_actor=3e-4, lr_critic=3e-4, 
                    gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                    update_epochs=4, minibatch_size=32, entropy_coef=0.02)

    # Create log directory
    log_dir = "models_ppo/models_ppo_logs"
    save_path = "models_ppo"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Train or load agent
    train_new = False 
    
    if train_new:
        # Train agent
        agent = train(env, agent, num_episodes=150, eval_freq=10, save_path=save_path, log_dir=log_dir)
    else:
        # Load agent
        agent = load_agent(env, "./models_ppo/best_model.pt")
        
        # Visualize metrics from a saved log if available
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.npz')]
        if log_files:
            newest_log = max([os.path.join(log_dir, f) for f in log_files], key=os.path.getctime)
            visualize_training_metrics(newest_log)
    
    # Visualize trained agent
    for _ in range(5):
        visualize_episode(env, agent)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()