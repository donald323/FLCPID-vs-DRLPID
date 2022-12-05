import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import itertools
import pandas as pd
from statsmodels import api as sm
import utility as uti
import ship
import environment as env
import PID
import APF
import ship_sim_system as sss
import RL_DQN
import os


#SBDQN Demo
#Test ship and environment class
mass = 50
L = 1.5
d = 0.183
t = 0.27
rho = 0.9
j = -293 / 335
j2 = j**2
kt = 324369 / 670000
dp = 0.135
tr = 0.291
ah = 0.28
xh = -0.377 * L
xr = -L / 2
ar = L * d / 37.5
A = 1.801
e = 1.137
w = 0.39
eta = 0.914
k = 0.551
yr = 0.467
lr = -0.67
bs = 0.05
by = 0.05
n = 3000*2*np.pi/60
max_rudder_angle = 35*np.pi/180
coor_range = [0,2000]
rudder_angle = 0.0

default_setting_file = "default_env.txt"

#Simulation Parameters
num_steps = 10000
ref_angle_tracker_capacity = 1000
ac_check = 50
ac_threshold = 0.6
v_limit = 200
v_length = 100
halt_limit = 30
halt_radius = 10
obstacle_radius = 5
update_freq = 100
algorithm = 'DQN'

test_APF_ob = APF.Artificial_Potential_Field(50,-500,50,-500000)
test_APF_t = APF.Artificial_Potential_Field(5000,500,50,500000)

#DQN Parameters
avaliable_actions = [0.01 * x for x in range(-5,6)]
all_actions = []
for i in avaliable_actions:
	for j in avaliable_actions:
		for k in avaliable_actions:
				all_actions.append([i,j,k])
state_dim = 7
action_dim = len(all_actions)
hidden_dim = 128
learning_rate = 1e-2
gamma = 0.9
num_episodes = 2000
epsilon = 1.0
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
reward_1 = 1.0
reward_2 = 0.5

#TOT Test
TOT_SBDQN100T_AAE = []
TOT_SBDQNDT_AAE = []

for test_ID in range(1,6):

	#Inialization
	print('Initial TOT Step-based DQN Test', test_ID)
	folder_path = 'SBDQN\\TOTSBDQN %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	replay_buffer = RL_DQN.ReplayBuffer(buffer_size)
	agent = RL_DQN.DQN(state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device)
	rl_dqn = RL_DQN.RL_DQN(replay_buffer,agent,minimal_size,batch_size)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	for i in range(num_episodes):
		rl_dqn.agent.epsilon = (num_episodes - i) / num_episodes
		print('Episode %i' % i)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)		
		motion_title = 'TOTSBDQN %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'TOTSBDQN %i, PID Tracker (Training), Episode %i' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=100,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('AAE: ', AAE)
		print('Average Reward: ', ship_sim_sys.reward_average)
		reward_list.append(ship_sim_sys.reward_average)

	ship_sim_sys.learn = False
	#100T Test
	aaes = 0
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)
		motion_title = 'TOTSBDQN %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'TOTSBDQN %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('TOTSBDQN %i 100T Result: ' % test_ID, aaes / 100)
	TOT_SBDQN100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)
	motion_title = 'TOTSBDQN %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'TOTSBDQN %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('TOTSBDQN %i DT Result: ' % test_ID, AAE)
	TOT_SBDQNDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('TOTSBDQN %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'TOTSBDQN %i, Episode Reward.png' % test_ID)

#FT Test
FT_SBDQN100T_AAE = []
FT_SBDQNDT_AAE = []

for test_ID in range(1,6):

	#Inialization
	print('Initial FT Step-based DQN Test', test_ID)
	folder_path = 'SBDQN\\FTSBDQN %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	replay_buffer = RL_DQN.ReplayBuffer(buffer_size)
	agent = RL_DQN.DQN(state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device)
	rl_dqn = RL_DQN.RL_DQN(replay_buffer,agent,minimal_size,batch_size)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	for i in range(num_episodes):
		rl_dqn.agent.epsilon = (num_episodes - i) / num_episodes
		print('Episode %i' % i)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)		
		motion_title = 'FTSBDQN %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'FTSBDQN %i, PID Tracker (Training), Episode %i' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=100,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('AAE: ', AAE)
		print('Average Reward: ', ship_sim_sys.reward_average)
		reward_list.append(ship_sim_sys.reward_average)

	ship_sim_sys.learn = False
	#100T Test
	aaes = 0
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)
		motion_title = 'FTSBDQN %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'FTSBDQN %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('FTSBDQN %i 100T Result: ' % test_ID, aaes / 100)
	FT_SBDQN100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	ship_sim_sys.add_DRL_agent(rl_dqn,update_freq,all_actions,reward_1,reward_2)
	motion_title = 'FTSBDQN %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'FTSBDQN %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('FTSBDQN %i DT Result: ' % test_ID, AAE)
	FT_SBDQNDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('FTSBDQN %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'FTSBDQN %i, Episode Reward.png' % test_ID)

print('TOTSBDQN 100T Results')
print(TOT_SBDQN100T_AAE)
print('TOTSBDQN DT Result')
print(TOT_SBDQNDT_AAE)

print('FTSBDQN 100T Results')
print(FT_SBDQN100T_AAE)
print('FTSBDQN DT Result')
print(FT_SBDQNDT_AAE)