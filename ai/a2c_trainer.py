import torch
import numpy as np
from ai.agent import Agent


class A2CTrainer:
	def __init__(
		self,
		agents: list[Agent],
		learning_rate: float = 0.001,
		gamma: float = 0.99,
		value_loss_coef: float = 0.5,
		entropy_coef: float = 0.01
	):
		self.agents = agents
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef
		self._sync_agents(agents)
	
	def _sync_agents(self, agents: list[Agent]) -> None:
		self.agents = agents
		for agent in self.agents:
			agent.gamma = self.gamma
			agent.value_loss_coef = self.value_loss_coef
			agent.entropy_coef = self.entropy_coef
			agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=self.learning_rate)

	def update_agents(self, new_agents: list[Agent]) -> None:
		self._sync_agents(new_agents)

	def train_step(self) -> dict:
		total_losses = []
		actor_losses = []
		critic_losses = []
		for agent in self.agents:
			if len(agent.actions) > 0:
				total_loss, actor_loss, critic_loss = agent.update_policy()
				total_losses.append(total_loss)
				actor_losses.append(actor_loss)
				critic_losses.append(critic_loss)
		return {
			'avg_total_loss': np.mean(total_losses) if total_losses else 0.0,
			'avg_actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
			'avg_critic_loss': np.mean(critic_losses) if critic_losses else 0.0,
			'num_trained_agents': len(total_losses)
		}

	def clear_all_trajectories(self) -> None:
		for agent in self.agents:
			agent.clear_trajectory()

	def get_training_summary(self) -> str:
		return (
			f"A2C Training Configuration:\n"
			f"---------------------------\n"
			f"Number of Agents: {len(self.agents)}\n"
			f"Learning Rate: {self.learning_rate}\n"
			f"Discount Factor (gamma): {self.gamma}\n"
			f"Value Loss Coefficient: {self.value_loss_coef}\n"
			f"Entropy Coefficient: {self.entropy_coef}\n"
			f"Device: {self.agents[0].device if self.agents else 'N/A'}"
		)

	def adjust_learning_rate(self, new_lr: float) -> None:
		self.learning_rate = new_lr
		for agent in self.agents:
			for param_group in agent.optimizer.param_groups:
				param_group['lr'] = new_lr

	def adjust_entropy_coefficient(self, new_entropy_coef: float) -> None:
		self.entropy_coef = new_entropy_coef
		for agent in self.agents:
			agent.entropy_coef = new_entropy_coef
