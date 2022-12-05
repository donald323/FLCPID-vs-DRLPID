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
import os

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
algorithm = 'Fuzzy Logic'

test_APF_ob = APF.Artificial_Potential_Field(50,-500,50,-500000)
test_APF_t = APF.Artificial_Potential_Field(5000,500,50,500000)

#FLC Demo
#TOT Test
TOTFLC_100T_AAE = []
TOTFLC_DT_AAE = []

for test_ID in range(1,6):
	print('Initial TOT Fuzzy Logic Test', test_ID)
	folder_path = 'FLC\\TOTFLC %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	aaes = 0

	#100T Test
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		motion_title = 'TOTFLC %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'TOTFLC %i, PID Tracker (100T), Episode %i ' % (test_ID,i)
		AAE = ship_sim_sys.run_trial(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE : ' % i, AAE)
		aaes += AAE
	print('TOTFLC %i 100T Result: ' % test_ID, aaes / 100)
	TOTFLC_100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	motion_title = 'TOTFLC %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'TOTFLC %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_trial(obstacle_included=False,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('TOTFLC %i DT Result: ' % test_ID, AAE)
	TOTFLC_DT_AAE.append(AAE)

#FT Test
FTFLC_100T_AAE = []
FTFLC_DT_AAE = []

for test_ID in range(1,6):
	print('Initial FT Fuzzy Logic Test', test_ID)
	folder_path = 'FLC\\FTFLC %i\\' % (test_ID)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	
	#100T Test
	aaes = 0
	for i in range(100):
		test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
		test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
		test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
		ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
		motion_title = 'FTFLC %i, Track of Ship\'s Motion (100T), Episode %i' % (test_ID,i)
		pid_title = 'FTFLC %i, PID Tracker (100T), Episode %i' % (test_ID,i)
		AAE = ship_sim_sys.run_trial(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=i,save_plot_freq=10,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
		print('100T Episode %i AAE: ' % i, AAE)
		aaes += AAE
	print('FTFLC %i 100T Result: ' % test_ID, aaes / 100)
	FTFLC_100T_AAE.append(aaes / 100)

	#Default Test
	test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
	test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
	test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0.5,ikD=0.5,aw_threshold=10)
	ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
	ship_sim_sys.load_default_environment(default_setting_file)
	motion_title = 'FTFLC %i, Track of Ship\'s Motion (DT)' % (test_ID)
	pid_title = 'FTFLC %i, PID Tracker (DT)' % (test_ID)
	AAE = ship_sim_sys.run_trial(obstacle_included=True,animate=False,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=True,motion_title=motion_title,pid_title=pid_title,folder_path=folder_path)
	print('FTFLC %i DT Result: ' % test_ID, AAE)
	FTFLC_DT_AAE.append(AAE)

print('TOTFLC 100T Results')
print(TOTFLC_100T_AAE)
print('TOTFLC DT Result')
print(TOTFLC_DT_AAE)

print('FTFLC 100T Result')
print(FTFLC_100T_AAE)
print('FTFLC DT Result')
print(FTFLC_DT_AAE)