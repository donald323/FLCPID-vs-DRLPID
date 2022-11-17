import numpy as np
import matplotlib.pyplot as plt
import random
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

test_ship = ship.Ship(mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by)
test_env = env.Environment(numpoints=10,coormin=200,coormax=1800,wcmin=-1,wcmax=1,speed_limit=2)
test_PID = PID.PID([0.0,1.0],ikP=0.5,ikI=0,ikD=0,aw_threshold=10)
test_APF_ob = APF.Artificial_Potential_Field(50,-500,50,-500000)
test_APF_t = APF.Artificial_Potential_Field(5000,500,50,500000)

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
algorithm = 'No AI'

folder_path = 'Test\\'
if not os.path.exists(folder_path):
	os.makedirs(folder_path)

ship_sim_sys = sss.ship_sim(test_ship,test_env,test_PID,test_APF_ob,test_APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm)
ship_sim_sys.load_default_environment(default_setting_file)
AAE = ship_sim_sys.run_episode(obstacle_included=True,animate=True,save_plot=True,save_plot_counter=0,save_plot_freq=1,step_update=False,motion_title='Track of Ship\'s Motion (10 Target Points), Episode Test',pid_title='PID Tracker, Episode Test',folder_path=folder_path)
