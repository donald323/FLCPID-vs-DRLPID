#File for storing all Actor-Critic related classes and functions
#Created by Weinan Zhang and his team from Shanghai Jiaotong University, and modified by Donald Cheng
#Please visit https://hrl.boyuai.com/chapter/2/actor-critic%E7%AE%97%E6%B3%95 for more information (it is in Chinese, if you need help in translation, please let me know)

import torch
import torch.nn.functional as F
import collections

#Actor Network
class PolicyNet(torch.nn.Module):
	def __init__(self,state_dim,hidden_dim,action_dim):
		super(PolicyNet,self).__init__()
		self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x),dim=1)

#Critic Network
class ValueNet(torch.nn.Module):
	def __init__(self,state_dim,hidden_dim):
		super(ValueNet,self).__init__()
		self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim,1)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		return self.fc2(x)

#Actor-Critic Class for taking action and updating the actor and critic network
class ActorCritic:
	def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device):
		self.actor = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
		self.critic = ValueNet(state_dim,hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
		self.gamma = gamma #Discount factor
		self.device = device
	def take_action(self,state): #Generate a probability distrubution of all actions to be taken, and select one based on the distribution
		state = torch.tensor([state],dtype=torch.float).to(self.device)
		probs = self.actor(state)
		probs = torch.where(torch.isnan(probs), torch.zeros_like(probs) + 1e-18, probs) #Due to unknown reason, some of the probability will become nan, which cannot be used for calculation, this line acts as a safeguard in case that happens
		action_dist = torch.distributions.Categorical(probs)
		action = action_dist.sample()
		return action.item()
	def update(self,transition_dict):
		states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1,1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
		dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
		td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
		td_delta = td_target - self.critic(states)
		log_probs = torch.log(self.actor(states))
		actor_loss = torch.mean(-log_probs * td_delta.detach())
		critic_loss = torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		actor_loss.backward()
		critic_loss.backward()
		self.actor_optimizer.step()
		self.critic_optimizer.step()

#Class to wrap everything up
class RL_A2C:
	def __init__(self,agent,transition_dict_size):
		self.agent = agent
		self.transition_dict_size = transition_dict_size
		self.transition_dict = {'states':collections.deque(maxlen=transition_dict_size),'actions':collections.deque(maxlen=transition_dict_size),'next_states':collections.deque(maxlen=transition_dict_size),'rewards':collections.deque(maxlen=transition_dict_size),'dones':collections.deque(maxlen=transition_dict_size)}
	def update(self,state,action,reward,next_state,done):
		self.transition_dict['states'].append(state)
		self.transition_dict['actions'].append(action)
		self.transition_dict['rewards'].append(reward)
		self.transition_dict['next_states'].append(next_state)
		self.transition_dict['dones'].append(done)
		self.agent.update(self.transition_dict)
	def transition_dict_reset(self):
		transition_dict =  {'states':collections.deque(maxlen=self.transition_dict_size),'actions':collections.deque(maxlen=self.transition_dict_size),'next_states':collections.deque(maxlen=self.transition_dict_size),'rewards':collections.deque(maxlen=self.transition_dict_size),'dones':collections.deque(maxlen=self.transition_dict_size)}
