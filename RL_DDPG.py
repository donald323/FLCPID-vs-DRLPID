#File for storing all DDPG related classes and functions
#Created by Weinan Zhang and his team from Shanghai Jiaotong University, and modified by Donald Cheng
#Please visit https://hrl.boyuai.com/chapter/2/ddpg%E7%AE%97%E6%B3%95 for more information (it is in Chinese, if you need help in translation, please let me know)

import torch
import torch.nn.functional as F
import numpy as np
import collections
import random

#Class for Replay Buffer
class ReplayBuffer:
	def __init__(self,capacity):
		self.buffer = collections.deque(maxlen=capacity)
	def add(self,state,action,reward,next_state,done):
		self.buffer.append((state,action,reward,next_state,done))
	def sample(self,batch_size):
		transitions = random.sample(self.buffer,batch_size)
		state,action,reward,next_state,done = zip(*transitions)
		return np.array(state),action,reward,np.array(next_state),done
	def size(self):
		return len(self.buffer)

#Actor network
class PolicyNet(torch.nn.Module):
	def __init__(self,state_dim,hidden_dim,action_dim,action_bound,eb_sb):
		super(PolicyNet,self).__init__()
		self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
		self.action_bound = action_bound
		self.eb_sb = eb_sb
	def forward(self,x):
		x = F.relu(self.fc1(x))
		if self.eb_sb: #Specialized for Episode-based DDPGPID, which produce increments to the existed PID gains
			return torch.tanh(self.fc2(x)) * self.action_bound
		else: #Specialized for Step-based DDPGPID, which produce the PID gains directly
			return torch.clamp(self.fc2(x), min=0.0, max=self.action_bound)

#Critic network
class QValueNet(torch.nn.Module):
	def __init__(self,state_dim,hidden_dim,action_dim):
		super(QValueNet,self).__init__()
		self.fc1 = torch.nn.Linear(state_dim + action_dim,hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim,hidden_dim)
		self.fc_out = torch.nn.Linear(hidden_dim,1)
	def forward(self,x,a):
		cat = torch.cat([x,a],dim=1)
		x = F.relu(self.fc1(cat))
		x = F.relu(self.fc2(x))
		return self.fc_out(x)

#DDPG Class for taking action and updating
class DDPG:
	def __init__(self,state_dim,hidden_dim,action_dim,action_bound,sigma,actor_lr,critic_lr,tau,gamma,device,eb_sb):
		self.actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound,eb_sb).to(device)
		self.critic = QValueNet(state_dim,hidden_dim,action_dim).to(device)
		self.target_actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound,eb_sb).to(device)
		self.target_critic = QValueNet(state_dim,hidden_dim,action_dim).to(device)
		self.target_critic.load_state_dict(self.critic.state_dict())
		self.target_actor.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
		self.gamma = gamma
		self.sigma = sigma
		self.tau = tau
		self.action_dim = action_dim
		self.device = device
	def take_action(self,state):
		state = torch.tensor([state],dtype=torch.float).to(self.device)
		action = self.actor(state).cpu().detach().numpy() #Action generated from the actor network
		action = action + self.sigma * np.random.randn(self.action_dim) #Gaussian noise added to the action
		return action
	def soft_update(self,net,target_net):
		for param_target,param in zip(target_net.parameters(),net.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
	def update(self,transition_dict):
		states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
		dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
		next_q_values = self.target_critic(next_states,self.target_actor(next_states))
		q_targets = rewards + self.gamma * next_q_values * (1 - dones)
		critic_loss = torch.mean(F.mse_loss(self.critic(states,actions),q_targets))
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		actor_loss = -torch.mean(self.critic(states,self.actor(states)))
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
		self.soft_update(self.actor,self.target_actor)
		self.soft_update(self.critic,self.target_critic)

#Class for wrapping everything up
class RL_DDPG:
	def __init__(self,ReplayBuffer,agent,replay_buffer_minimum_size,batch_size):
		self.ReplayBuffer = ReplayBuffer
		self.agent = agent
		self.replay_buffer_minimum_size = replay_buffer_minimum_size
		self.batch_size = batch_size
	def update(self,state,action,reward,next_state,done):
		self.ReplayBuffer.add(state,action,reward,next_state,done)
		if self.ReplayBuffer.size() > self.replay_buffer_minimum_size:
			b_s,b_a,b_r,b_ns,b_d = self.ReplayBuffer.sample(self.batch_size)
			transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
			self.agent.update(transition_dict)