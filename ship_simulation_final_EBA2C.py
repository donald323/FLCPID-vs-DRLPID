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
import RL_A2C
import os


#EBA2C Demo
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
algorithm = 'A2C'

test_APF_ob = APF.Artificial_Potential_Field(50,-500,50,-500000)
test_APF_t = APF.Artificial_Potential_Field(5000,500,50,500000)

#A2C Parameters
avaliable_actions = [0.01 * x for x in range(-5,6)]
all_actions = []
for i in avaliable_actions:
	for j in avaliable_actions:
		for k in avaliable_actions:
				all_actions.append([i,j,k])
state_dim = 3
action_dim = len(all_actions)
hidden_dim = 128
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.9
num_episodes = 2000
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
reward_1 = 1.5
reward_2 = 1.0

#TOT Test
TOT_EBA2C100T_AAE = []
TOT_EBA2CDT_AAE = []
TOT_PID_Tracker = []

for test_ID in range(1,6):

	#Inialization
	print('Initial TOT Episode-based A2C Test', test_ID)
	folder_path = 'EBA2C\\TOTEBA2C %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	agent = RL_A2C.ActorCritic(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device)
	rl_a2c = RL_A2C.RL_A2C(agent,500)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	state = test_PID.return_PID()
	for i in range(num_episodes):
		action = rl_a2c.agent.take_action(state)
		dPID = all_actions[action][0:3]
		test_PID.PID_gain_update(dPID)
		next_state = test_PID.return_PID()
		pid_tracker.append(next_state)
		print('Episode %i' % i, next_state)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)		
		motion_title = 'TOTEBA2C %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'TOTEBA2C %i, PID Tracker (Training), Episode %i' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=100,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		reward = uti.episode_reward(AAE,old_AAE,reward_1,reward_2,test_PID.return_PID(),0.1,0.9)
		old_AAE = AAE
		print('AAE: ', AAE)
		print('Reward: ', reward)
		reward_list.append(reward)
		rl_a2c.update(state,action,reward,next_state,done)
		state = next_state

	#100T Test
	aaes = 0
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		motion_title = 'TOTEBA2C %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'TOTEBA2C %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('TOTEBA2C %i 100T Result: ' % test_ID, aaes / 100)
	TOT_EBA2C100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	motion_title = 'TOTEBA2C %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'TOTEBA2C %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('TOTEBA2C %i DT Result: ' % test_ID, AAE)
	TOT_EBA2CDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('TOTEBA2C %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'TOTEBA2C %i, Episode Reward.png' % test_ID)

	plt.figure(4)
	plt.clf()
	plt.plot(pid_tracker)
	plt.title('TOTEBA2C %i, Track of PID Gains' % test_ID)
	plt.legend(['Kp','Ki','Kd'])
	plt.xlabel('Episode Number')
	plt.ylabel('Kp, Ki and Kd Values')
	plt.savefig(folder_path + 'TOTEBA2C %i, Track of PID Gains.png' % test_ID)

	TOT_PID_Tracker.append(test_PID.return_PID())	

#FT Test

FT_EBA2C100T_AAE = []
FT_EBA2CDT_AAE = []
FT_PID_Tracker = []
for test_ID in range(1,6):

	#Inialization
	print('Initial FT Episode-based A2C Test', test_ID)
	folder_path = 'EBA2C\\FTEBA2C %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	old_AAE = 0
	agent = RL_A2C.ActorCritic(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device)
	rl_a2c = RL_A2C.RL_A2C(agent,500)
	done = False
	reward_list = []
	pid_tracker = []

	#Training
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	state = test_PID.return_PID()
	for i in range(num_episodes):
		action = rl_a2c.agent.take_action(state)
		dPID = all_actions[action][0:3]
		test_PID.PID_gain_update(dPID)
		next_state = test_PID.return_PID()
		pid_tracker.append(next_state)
		print('Episode %i' % i, next_state)
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)		
		motion_title = 'FTEBA2C %i, Track of Ship\'s Motion (Training), Episode %i' % (test_ID,i)
		pid_title = 'FTEBA2C %i, PID Tracker (Training), Episode %i' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=100,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		reward = uti.episode_reward(AAE,old_AAE,reward_1,reward_2,test_PID.return_PID(),0.1,0.9)
		old_AAE = AAE
		print('AAE: ', AAE)
		print('Reward: ', reward)
		reward_list.append(reward)
		rl_a2c.update(state,action,reward,next_state,done)
		state = next_state

	#100T Test
	aaes = 0
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		motion_title = 'FTEBA2C %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'FTEBA2C %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('FTEBA2C %i 100T Result: ' % test_ID, aaes / 100)
	FT_EBA2C100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	motion_title = 'FTEBA2C %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'FTEBA2C %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=False,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('FTEBA2C %i DT Result: ' % test_ID, AAE)
	FT_EBA2CDT_AAE.append(AAE)

	plt.figure(3)
	plt.clf()
	plt.plot(reward_list)
	plt.title('FTEBA2C %i, Reward of Each Episode' % test_ID)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.savefig(folder_path + 'FTEBA2C %i, Episode Reward.png' % test_ID)

	plt.figure(4)
	plt.clf()
	plt.plot(pid_tracker)
	plt.title('FTEBA2C %i, Track of PID Gains' % test_ID)
	plt.legend(['Kp','Ki','Kd'])
	plt.xlabel('Episode Number')
	plt.ylabel('Kp, Ki and Kd Values')
	plt.savefig(folder_path + 'FTEBA2C %i, Track of PID Gains.png' % test_ID)

	FT_PID_Tracker.append(test_PID.return_PID())	

print('TOTEBA2C 100T Results')
print(TOT_EBA2C100T_AAE)
print('TOTEBA2C DT Result')
print(TOT_EBA2CDT_AAE)
print('Final PID Gains of all TOT Trials')
print(TOT_PID_Tracker)

print('FTEBA2C 100T Results')
print(FT_EBA2C100T_AAE)
print('FTEBA2C DT Result')
print(FT_EBA2CDT_AAE)
print('Final PID Gains of all FT Trials')
print(FT_PID_Tracker)