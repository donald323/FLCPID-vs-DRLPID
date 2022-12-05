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
import RL_DDPG
import os


#SBDDPG Demo
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
algorithm = 'DDPG'

test_APF_ob = APF.Artificial_Potential_Field(50,-500,50,-500000)
test_APF_t = APF.Artificial_Potential_Field(5000,500,50,500000)

#DDPG Parameters
state_dim = 4
action_dim = 3
hidden_dim = 128
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.9
tau = 0.1
sigma = 0.01
action_bound = 0.9
num_episodes = 2000
buffer_size = 10000
minimal_size = 500
batch_size = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
reward_1 = 1.0
reward_2 = 0.5
eb_sb = False #Episode-based or Step-based

#TOT Test
TOT_SBDDPG100T_AAE = []
TOT_SBDDPGDT_AAE = []

for test_ID in range(1,6):

	#Inialization
	print('Initial TOT Step-based DDPG Test', test_ID)
	folder_path = 'SBDDPG\\TOTSBDDPG %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	replay_buffer = RL_DDPG.ReplayBuffer(buffer_size)
	agent = RL_DDPG.DDPG(state_dim,hidden_dim,action_dim,action_bound,sigma,actor_lr,critic_lr,tau,gamma,device,eb_sb)
	rl_ddpg = RL_DDPG.RL_DDPG(replay_buffer,agent,minimal_size,batch_size)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	for i in range(num_episodes):
		print('Episode %i' % i)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)		
		motion_title = 'TOTSBDDPG %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'TOTSBDDPG %i, PID Tracker (Training), Episode %i' % (test_ID,i)
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
		ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)
		motion_title = 'TOTSBDDPG %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'TOTSBDDPG %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('TOTSBDDPG %i 100T Result: ' % test_ID, aaes / 100)
	TOT_SBDDPG100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)
	motion_title = 'TOTSBDDPG %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'TOTSBDDPG %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('TOTSBDDPG %i DT Result: ' % test_ID, AAE)
	TOT_SBDDPGDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('TOTSBDDPG %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'TOTSBDDPG %i, Episode Reward.png' % test_ID)

#FT Test
FT_SBDDPG100T_AAE = []
FT_SBDDPGDT_AAE = []

for test_ID in range(1,6):

	#Inialization
	print('Initial FT Step-based DDPG Test', test_ID)
	folder_path = 'SBDDPG\\FTSBDDPG %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	replay_buffer = RL_DDPG.ReplayBuffer(buffer_size)
	agent = RL_DDPG.DDPG(state_dim,hidden_dim,action_dim,action_bound,sigma,actor_lr,critic_lr,tau,gamma,device,eb_sb)
	rl_ddpg = RL_DDPG.RL_DDPG(replay_buffer,agent,minimal_size,batch_size)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	state = test_PID.return_PID()
	for i in range(num_episodes):
		print('Episode %i' % i)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)		
		motion_title = 'FTSBDDPG %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'FTSBDDPG %i, PID Tracker (Training), Episode %i' % (test_ID,i)
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
		ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)
		motion_title = 'FTSBDDPG %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'FTSBDDPG %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('FTSBDDPG %i 100T Result: ' % test_ID, aaes / 100)
	FT_SBDDPG100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	ship_sim_sys.add_DRL_agent(rl_ddpg,update_freq,False,reward_1,reward_2)
	motion_title = 'FTSBDDPG %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'FTSBDDPG %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('FTSBDDPG %i DT Result: ' % test_ID, AAE)
	FT_SBDDPGDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('FTSBDDPG %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'FTSBDDPG %i, Episode Reward.png' % test_ID)

print('TOTSBDDPG 100T Results')
print(TOT_SBDDPG100T_AAE)
print('TOTSBDDPG DT Result')
print(TOT_SBDDPGDT_AAE)

print('FTSBDDPG 100T Results')
print(FT_SBDDPG100T_AAE)
print('FTSBDDPG DT Result')
print(FT_SBDDPGDT_AAE)