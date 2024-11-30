from abc import ABC
from typing import Dict

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.optim import Adam


class PPO(BasePolicy):
    """
    Proximal Policy Optimization (PPO) implementation based on BasePolicy.
    """

    def __init__(
        self,
        *,
        action_space: gym.Space,
        observation_space: gym.Space,
        actor_critic: nn.Module,
        train_collector,
        eval_collector,
        optimizer: Adam,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        vf_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        """
        Initializes the PPO policy.

        Args:
            action_space (gym.Space): Action space of the environment.
            observation_space (gym.Space): Observation space of the environment.
            actor_critic (nn.Module): Actor-critic model for policy and value function.
            train_collector: Collector for training data.
            eval_collector: Collector for evaluation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            clip_eps (float, optional): Clipping epsilon for PPO. Defaults to 0.2.
            vf_coeff (float, optional): Value function coefficient in the loss. Defaults to 0.5.
            ent_coeff (float, optional): Entropy coefficient in the loss. Defaults to 0.01.
            epochs (int, optional): Number of training epochs per update. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
        )
        self.actor_critic = actor_critic
        self.train_collector = train_collector
        self.eval_collector = eval_collector
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self) -> None:
        """
        Train the PPO policy using data from the train_collector.
        """
        self.train_collector.collect(n_step=1000, reset_before_collect=True)  # Collect rollout data

        self.train_collector.compute_advantages()

        observations = data["observations"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]

        # Compute returns and advantages
        returns, advantages = self.compute_advantages(rewards, values, dones)

        # Training loop
        for _ in range(self.epochs):
            indices = torch.randperm(len(observations))
            for i in range(0, len(observations), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch = {
                    "observations": observations[batch_indices],
                    "actions": actions[batch_indices],
                    "log_probs": old_log_probs[batch_indices],
                    "advantages": advantages[batch_indices],
                    "returns": returns[batch_indices],
                }
                self.update_policy(batch)

    def evaluate(self, episodes: int = 10) -> float:
        """
        Evaluate the policy using the eval_collector.

        Args:
            episodes (int): Number of episodes to evaluate over. Defaults to 10.

        Returns:
            float: Average reward over the episodes.
        """
        total_rewards = self.eval_collector.collect(self, num_episodes=episodes)
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Evaluation completed: Average Reward = {avg_reward}")
        return avg_reward

    def select_action(self, observation: torch.Tensor, deterministic: bool = False):
        """
        Selects an action using the actor-critic model.

        Args:
            observation (torch.Tensor): Current observation.
            deterministic (bool, optional): If True, selects the action deterministically. Defaults to False.

        Returns:
            Dict: Contains 'action', 'log_prob', and 'value'.
        """
        policy, value = self.actor_critic(observation)
        if deterministic:
            action = policy.mean
        else:
            action = policy.sample()
        log_prob = policy.log_prob(action)
        return {"action": action, "log_prob": log_prob, "value": value}

    def compute_advantages(self, rewards, values, dones):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            rewards (torch.Tensor): Rewards from the environment.
            values (torch.Tensor): Value function predictions.
            dones (torch.Tensor): Done flags from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns and advantages.
        """
        advantages = []
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return torch.tensor(returns), torch.tensor(advantages)

    def update_policy(self, batch: dict) -> None:
        """
        Update the policy using PPO loss.

        Args:
            batch (dict): A batch of data with observations, actions, log_probs, advantages, and returns.
        """
        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Compute new log probabilities, entropy, and values
        policy, value = self.actor_critic(observations)
        new_log_probs = policy.log_prob(actions).sum(dim=-1)
        entropy = policy.entropy().mean()

        # PPO clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        surrogate_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = self.vf_coeff * (returns - value).pow(2).mean()

        # Total loss
        loss = surrogate_loss + value_loss - self.ent_coeff * entropy

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
