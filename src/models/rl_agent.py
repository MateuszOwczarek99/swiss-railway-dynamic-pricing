"""
proximal policy optimization (ppo) agent for dynamic pricing
this is our main reinforcement learning algorithm for learning optimal prices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
import mlflow
import os

class PricingEnvironment:
    """
    simulated environment for pricing decisions
    we use historical data to create realistic scenarios
    """
    
    def __init__(self, data, demand_model):
        self.data = data.reset_index(drop=True)
        self.demand_model = demand_model
        self.current_step = 0
        
        # price multipliers from 0.5x to 1.5x base price
        self.action_space = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.action_size = len(self.action_space)
        
        # state dimensions: [demand, competitor_ratio, days_left, is_weekend, month]
        self.state_size = 5
    
    def reset(self):
        """start a new episode from the beginning of data"""
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        """extract current state from data"""
        if self.current_step >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_step]
        
        # normalize values to help neural network training
        state = np.array([
            row['demand'],
            row['price'] / (row['competitor_price'] + 0.01),
            min(row['days_to_departure'], 30) / 30.0,
            row['is_weekend'],
            row['month'] / 12.0
        ], dtype=np.float32)
        
        return state
    
    def step(self, action_idx):
        """
        take an action (choose price multiplier)
        returns: next_state, reward, done
        """
        row = self.data.iloc[self.current_step]
        
        # apply the chosen price multiplier
        multiplier = self.action_space[action_idx]
        new_price = row['price'] * multiplier
        
        # predict demand at this new price
        # we use a simple elasticity model for speed
        price_elasticity = -0.45
        demand_change = price_elasticity * (multiplier - 1)
        predicted_demand = max(0.05, min(0.95, row['demand'] + demand_change))
        
        # calculate revenue (our primary reward signal)
        revenue = new_price * predicted_demand
        
        # apply penalty if price is much higher than competitor
        if new_price > row['competitor_price'] * 1.25:
            revenue *= 0.85  # 15% penalty
        
        # apply bonus for high occupancy (selling more tickets)
        occupancy_bonus = predicted_demand * 10
        revenue += occupancy_bonus
        
        # move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        next_state = self._get_state() if not done else None
        
        return next_state, revenue, done


class PPONetwork(nn.Module):
    """neural network architecture for ppo agent"""
    
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        
        # policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        # value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.policy_head(shared_out)
        state_value = self.value_head(shared_out)
        return action_probs, state_value


class PPOAgent:
    """
    proximal policy optimization agent
    this is a state-of-the-art rl algorithm for continuous control tasks
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        
        # main network and old network (for stability)
        self.network = PPONetwork(state_size, action_size)
        self.old_network = PPONetwork(state_size, action_size)
        self.old_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # ppo hyperparameters
        self.clip_epsilon = 0.2  # clipping range for policy updates
        self.value_coef = 0.5    # coefficient for value loss
        self.entropy_coef = 0.01 # coefficient for exploration bonus
        self.gamma = 0.99        # discount factor for future rewards
        self.gae_lambda = 0.95   # generalized advantage estimation parameter
        
        # memory for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # exploration parameter
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
    
    def act(self, state, training=True):
        """choose action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if training and random.random() < self.epsilon:
                # explore: random action
                action = random.randrange(self.action_size)
                log_prob = 0
                value = 0
            else:
                # exploit: use policy network
                action_probs, value = self.network(state_tensor)
                distribution = Categorical(action_probs)
                action = distribution.sample().item()
                log_prob = distribution.log_prob(torch.tensor([action])).item()
        
        return action, log_prob, value.item()
    
    def remember(self, state, action, reward, log_prob, value, done):
        """store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def _compute_advantages(self, next_value):
        """compute generalized advantage estimation (gae)"""
        advantages = []
        gae = 0
        
        # reverse through the trajectory
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            # temporal difference error
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            
            # gae calculation
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        # normalize advantages for stability
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train_step(self, next_value):
        """perform one ppo update using collected trajectories"""
        if len(self.states) == 0:
            return 0
        
        # convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        values = torch.FloatTensor(self.values)
        
        # compute advantages and returns
        advantages = self._compute_advantages(next_value)
        returns = advantages + values
        
        # ppo update for several epochs
        epoch_loss = 0
        
        # multiple update epochs (ppo characteristic)
        for _ in range(4):
            # get current policy predictions
            action_probs, current_values = self.network(states)
            distribution = Categorical(action_probs)
            new_log_probs = distribution.log_prob(actions)
            
            # calculate ratio for importance sampling
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # value function loss
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            
            # entropy bonus for exploration
            entropy = distribution.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # total loss
            loss = policy_loss + self.value_coef * value_loss + entropy_loss
            
            # update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        # decay epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        return epoch_loss / 4
    
    def update_target_network(self):
        """copy network weights to old network"""
        self.old_network.load_state_dict(self.network.state_dict())
    
    def train(self, env, episodes=500, update_frequency=2048):
        """train the ppo agent for specified number of episodes"""
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
        print(f"starting ppo training for {episodes} episodes")
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_step = 0
            
            while True:
                # choose action
                action, log_prob, value = self.act(state)
                
                # execute action
                next_state, reward, done = env.step(action)
                
                # store experience
                self.remember(state, action, reward, log_prob, value, done)
                
                episode_reward += reward
                episode_step += 1
                total_steps += 1
                state = next_state
                
                # update policy periodically
                if total_steps % update_frequency == 0:
                    # get value of next state
                    next_val = 0
                    if next_state is not None:
                        _, next_val = self.network(torch.FloatTensor(next_state).unsqueeze(0))
                        next_val = next_val.item()
                    
                    loss = self.train_step(next_val)
                    self.update_target_network()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            
            # print progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_length = np.mean(episode_lengths[-50:])
                print(f"episode {episode+1}/{episodes} | avg reward: {avg_reward:.2f} | epsilon: {self.epsilon:.3f}")
        
        print("ppo training complete")
        return episode_rewards
    
    def save(self, path='models/ppo_agent.pth'):
        """save trained agent to disk"""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"agent saved to {path}")
    
    def load(self, path='models/ppo_agent.pth'):
        """load trained agent from disk"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_target_network()
        print(f"agent loaded from {path}")
