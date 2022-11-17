#File for storing all DQN related classes and functions
#Created by Weinan Zhang and his team from Shanghai Jiaotong University, and modified by Donald Cheng
#Please visit https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95 for more information (it is in Chinese, if you need help in translation, please let me know)
#The file uses PyTorch, which other frameworks should also be applicable, but please make sure you use the same architecture
#Strongly recommend to read these papers by DeepMind regarding to the how DQN works:
#- https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
#- https://arxiv.org/pdf/1312.5602.pdf

import torch
import torch.nn.functional as F
import numpy as np
import collections
import random

#Class for replay buffer
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

#Class for the Deep Q Network
class Qnet(torch.nn.Module):
	def __init__(self,state_dim,hidden_dim,action_dim):
		super(Qnet,self).__init__()
		self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		return self.fc2(x)

#All functions related to DQN learning
class DQN:
	def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
		self.action_dim = action_dim
		self.q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device) 
		self.target_q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device)
		self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
		self.gamma = gamma #Discount Factor
		self.epsilon = epsilon #Epsilon-greedy Policy
		self.target_update = target_update #Update frequency of the target network
		self.count = 0
		self.device = device
	def take_action(self,state): #Action based on Epsilon-greed policy
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		else:
			state = torch.tensor([state],dtype=torch.float).to(self.device)
			action = self.q_net(state).argmax().item()
		return action
	def max_q_value(self,state):
		state = torch.tensor([state],dtype=torch.float).to(self.device)
		return self.q_net(state).max().item()
	def update(self,transition_dict): #Update the target and training DQN network
		states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
		dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

		q_values = self.q_net(states).gather(1,actions)
		max_action = self.q_net(next_states).max(1)[1].view(-1,1)
		max_next_q_values = self.target_q_net(next_states).gather(1,max_action)
		q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
		dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))
		self.optimizer.zero_grad()
		dqn_loss.backward()
		self.optimizer.step()
		if self.count % self.target_update == 0:
			self.target_q_net.load_state_dict(self.q_net.state_dict()) #Update target network with same parameters as training network
		self.count += 1

#A class to wrap everything up
class RL_DQN:
	def __init__(self,ReplayBuffer,agent,replay_buffer_minimum_size,batch_size):
		self.ReplayBuffer = ReplayBuffer
		self.agent = agent
		self.replay_buffer_minimum_size = replay_buffer_minimum_size
		self.batch_size = batch_size
	def update(self,state,action,reward,next_state,done): #Add {state,action,reward,next state,done} to the replay buffer, and update when minimum size of the buffer is met
		self.ReplayBuffer.add(state,action,reward,next_state,done)
		if self.ReplayBuffer.size() > self.replay_buffer_minimum_size:
			b_s,b_a,b_r,b_ns,b_d = self.ReplayBuffer.sample(self.batch_size)
			transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
			self.agent.update(transition_dict)