#File for all functions needed to run an episode of ship navigation in the environment
#This is the major file for Fuzzy PIDs, DRLPIDs, or just a ship simulation with a static PID gains

import collections
from statsmodels import api as sm
import numpy as np
import matplotlib.pyplot as plt
import Fuzzy_Logic as FLC
import utility as uti

class ship_sim:
	def __init__(self,test_ship,test_env,test_PID,APF_ob,APF_t,num_steps,ref_angle_tracker_capacity,ac_check,ac_threshold,v_limit,v_length,halt_limit,halt_radius,obstacle_radius,algorithm):
		#All major objects
		self.test_ship = test_ship
		self.test_env = test_env
		self.test_PID = test_PID
		self.APF_ob = APF_ob
		self.APF_t = APF_t

		#Parameters
		self.num_steps = num_steps
		self.step_PID_update = False

		#V Mode Parameters
		self.ref_angle_tracker_capacity = ref_angle_tracker_capacity
		self.ac_check = ac_check
		self.ac_threshold = ac_threshold
		self.v_limit = v_limit		
		self.v_length = v_length

		#Halt Mode Parameters
		self.halt_limit = halt_limit
		self.halt_radius = halt_radius
		self.halt_mode_r_speed = self.test_ship.n

		#Obstacle Radius
		self.obstacle_radius = obstacle_radius

		#Others
		self.algorithm = algorithm
		self.old_AAE = 0
		self.reward_average = 0
		self.learn = True

	def load_default_environment(self,default_file): #Load default environment generated from the random_parameters_generator.py
		#Read File
		para_read = {}
		with open(default_file,"rt") as f:
			for line in f:
				result = line.strip()
				result = result.split(':')
				if result[1][1] == '[':
					arr = result[1]
					arr = arr.replace('[','')
					arr = arr.replace(']','')
					arr = arr.replace(' ','')
					arr_split = arr.split(',')
					arr_list = []
					for a in arr_split:
						arr_list.append(float(a))
					para_read[result[0]] = arr_list
				else:
					para_read[result[0]] = float(result[1])

		#Load Parameters
		self.test_env.wcx = para_read['wcx']
		self.test_env.wcy = para_read['wcy']
		self.test_ship.xe = para_read['xe']
		self.test_env.obstacle_listx = para_read['obstacle_listx']
		self.test_env.obstacle_listy = para_read['obstacle_listy']
		self.test_env.targets_listx = para_read['targets_listx']
		self.test_env.targets_listy = para_read['targets_listy']
		self.test_env.move_obstacles_x = para_read['move_obstacles_x']
		self.test_env.move_obstacles_y = para_read['move_obstacles_y']
		self.test_env.moving_vectors = para_read['moving_vectors']
	
	def add_DRL_agent(self,agent,update_freq,all_actions,reward_1,reward_2): #Add the DRL algorithm you want for step-based DRLPIDs
		self.agent = agent
		self.update_freq = update_freq #The frequency of updating PID gains can be customized (warning: higher frequency will take more time to process)
		self.all_actions = all_actions #Discrete actions for step-based DQNPID/A2CPID
		self.reward_1 = reward_1 #Threshold 1 for reward
		self.reward_2 = reward_2 #Threshold 2 for reward

	#Function for ploting the track of ship's motion in the environment
	def plot_motion(self,plot_title,ship_trackerx,ship_trackery,ref_angle,olistx,olisty,tlistx,tlisty,molistx,molisty,obstacle_included):
		plt.figure(1)
		plt.clf()
		plt.plot(ship_trackerx,ship_trackery,marker='o',markersize=1,color='orange')
		for j in range(len(self.test_env.targets_listx)):
			plt.plot(self.test_env.targets_listx[j],self.test_env.targets_listy[j],marker='o',markersize=5,color='green')
		if obstacle_included:
			for j in range(len(self.test_env.obstacle_listx)):
				plt.plot(self.test_env.obstacle_listx[j],self.test_env.obstacle_listy[j],marker='o',markersize=5,color='red')
			for j in range(len(self.test_env.move_obstacles_x)):
				plt.plot(self.test_env.move_obstacles_x[j],self.test_env.move_obstacles_y[j],marker='o',markersize=5,color='blue')
		for j in range(len(self.test_env.reached_target_listx)):
			plt.plot(self.test_env.reached_target_listx[j],self.test_env.reached_target_listy[j],marker='o',markersize=5,color='black')	
		plt.xlim([self.test_ship.coor_range[0] - 5 ,self.test_ship.coor_range[1] + 5])
		plt.ylim([self.test_ship.coor_range[0] - 5 ,self.test_ship.coor_range[1] + 5])
		plt.title(plot_title)
	#Function for plotting the track of PID gains
	def plot_PID_tracker(self,kp_tracker,ki_tracker,kd_tracker,title):
		plt.figure(2)
		plt.clf()
		plt.plot(kp_tracker)
		plt.plot(ki_tracker)
		plt.plot(kd_tracker)
		plt.legend(['Kp','Ki','Kd'])
		plt.xlabel('Time (s)')
		plt.ylabel('Kp, Ki and Kd Values')
		plt.title(title)
	#Run an episode of the ship's motion in the environment
	def run_episode(self,obstacle_included,animate,save_plot,save_plot_counter,save_plot_freq,step_update,motion_title,pid_title,folder_path):
		#Initialization
		ship_trackerx = [self.test_ship.xe[0]] #Track of ship's coordinate throughout the episode
		ship_trackery = [self.test_ship.xe[1]]
		ref_angle_tracker = collections.deque(maxlen=self.ref_angle_tracker_capacity)
		v_mode = False
		v_counter = 0
		halt_mode = False
		halt_count = 0
		step_counter = 0
		done = False

		#Step-based PID Tracker
		kp_tracker = [self.test_PID.kp]
		ki_tracker = [self.test_PID.ki]
		kd_tracker = [self.test_PID.kd]

		past_rudder_angle = self.test_ship.rudder_angle #Record the rudder angle from previous rudder
		aes = 0 #Sum of absolute error
		reward_sum = 0 #Sum of reward
		action_counter = 0

		if self.algorithm == 'A2C' and step_update: #Reset transition dictionary for every episode
			self.agent.transition_dict_reset()

		#Run episode until termination			
		while not done:

			#Virtual Mode Check, Activate if the ship starts to orbit and not able to go to other targets
			if len(ref_angle_tracker) == self.ref_angle_tracker_capacity:
				#Uses self-correlation to detect repeated signal and determine whether the ship is repeating itself or not
				angle_Norm = ref_angle_tracker - np.mean(ref_angle_tracker)
				acf = sm.tsa.acf(angle_Norm,nlags=len(angle_Norm))
				acf_50 = acf[self.ac_check:]
				if max(np.abs(acf_50)) > self.ac_threshold:
					if v_mode == False:
						#If activated for the first time, the virtual target/obstacle will be placed certain units from the nearest obstacle/target to force the ship to move to other places from the trapped region
						#If obstacles are not applied for this episode, do not add the virtual obstacle into the calculation
						if obstacle_included:
							vo_centerx,vo_centery = self.test_env.closest_obstacle(self.test_ship.return_coordinates())
							v_obstacle_listx = [vo_centerx+self.v_length]
							v_obstacle_listy = [vo_centery+self.v_length]
						vt_centerx,vt_centery = self.test_env.closest_target(self.test_ship.return_coordinates())
						v_target_listx = [vt_centerx+self.v_length]
						v_target_listy = [vt_centery+self.v_length]
						v_mode = True #Turn on virtual mode, which the virtual obstacle/target will be put in the consideration when calculating the APF forces
						if animate: #Display if the ship motion is animated on screen
							print('GNRON Detected at Step',step_counter)
							print('Entering V Mode')	

			#Obstacle Check
			olistx,olisty = self.APF_ob.generate_list(self.test_env.obstacle_listx,self.test_env.obstacle_listy,self.test_ship.return_coordinates()) #Static obstacle
			molistx,molisty = self.APF_ob.generate_list(self.test_env.move_obstacles_x,self.test_env.move_obstacles_y,self.test_ship.return_coordinates()) #Moving obstacle
			#Only calculate the force if obstacles were added into the environment
			if obstacle_included:
				xf1,yf1 = self.APF_ob.cal_force(olistx,olisty,self.test_ship.return_coordinates(),self.APF_ob.k)
				xf2,yf2 = self.APF_ob.cal_force(molistx,molisty,self.test_ship.return_coordinates(),self.APF_ob.k)
			else:		
				xf1,yf1 = [0,0]
				xf2,yf2 = [0,0]
			
			#Target Check
			tlistx,tlisty = self.APF_t.generate_list(self.test_env.targets_listx,self.test_env.targets_listy,self.test_ship.return_coordinates())
			xf3,yf3 = self.APF_t.cal_force(tlistx,tlisty,self.test_ship.return_coordinates(),self.APF_t.k)
			
			#Boundary Force Calculation
			xf4,yf4 = self.APF_t.bounadry_cal_force(self.test_ship.coor_range,self.test_ship.return_coordinates()) 

			#Virtual Target/Obstacle Calculation
			if v_mode:
				if v_counter == self.v_limit: #Deactivate Virtual Mode if the time limit has been met
					v_mode = False
					v_counter = 0
					if animate:
						print('Exit V Mode')
				else:
					#Calculate the APF forces from virtual obstacle and target
					v_counter += 1
					xf5,yf5 = self.APF_t.cal_force(v_target_listx,v_target_listy,self.test_ship.return_coordinates(),self.APF_t.k)	
					if obstacle_included:
						xf6,yf6 = self.APF_ob.cal_force(v_obstacle_listx,v_obstacle_listy,self.test_ship.return_coordinates(),self.APF_ob.k)
			else: #Do not include into the calculation if Virtual mode was not on
				xf5,yf5 = [0,0]
				xf6,yf6 = [0,0]

			#Sum of APF Forces
			APF_xf = xf1 + xf2 + xf3 + xf4 + xf5
			APF_yf = yf1 + yf2 + yf3 + yf4 + yf5
			if obstacle_included:
				APF_xf += xf6
				APF_yf += yf6

			#Reference Angle from the Sum of APF forces
			ref_angle = np.arctan2(APF_yf,APF_xf)
			ref_angle_tracker.append(ref_angle)

			#Error and PID Output
			error = ref_angle - self.test_ship.return_angle()
			if error < -np.pi:
				error += 2 * np.pi
			elif error > np.pi:
				error -= 2 * np.pi

			ed = error - self.test_PID.error_old
			ei = error + self.test_PID.error_sum

			#Fuzzy PID Gain Update
			if self.algorithm == 'Fuzzy Logic':
				#Run Fuzzy Logic Calculation
				FLC.kp_sim.input['Error'] = error
				FLC.kp_sim.input['Error Derivative'] = error - self.test_PID.error_old
				FLC.kp_sim.compute()
				kp_defuzz = FLC.kp_sim.output['Kp Output']

				FLC.ki_sim.input['Error'] = error
				FLC.ki_sim.input['Error Derivative'] = error - self.test_PID.error_old
				FLC.ki_sim.compute()
				ki_defuzz = FLC.ki_sim.output['Ki Output']

				FLC.kd_sim.input['Error'] = error
				FLC.kd_sim.input['Error Derivative'] = error - self.test_PID.error_old
				FLC.kd_sim.compute()
				kd_defuzz = FLC.kd_sim.output['Kd Output']

				dPID = [kp_defuzz,ki_defuzz,kd_defuzz]
				self.test_PID.PID_gain_update(dPID)

				#Add to the PID gain tracker
				kp_tracker.append(self.test_PID.kp)
				ki_tracker.append(self.test_PID.ki)
				kd_tracker.append(self.test_PID.kd)
			else:
				if step_counter == 0:
					#Define the initial state for DRLPID
					if step_update and self.algorithm != 'Fuzzy Logic':
						if self.algorithm == 'DQN' or self.algorithm == 'A2C':
							state = [self.test_PID.kp,self.test_PID.ki,self.test_PID.kd,error,ed,ei,past_rudder_angle]
						elif self.algorithm == 'DDPG':
							state = [error,ed,ei,past_rudder_angle]
				else:
					if step_update and self.algorithm != 'Fuzzy Logic':
						if self.algorithm == 'DQN' or self.algorithm == 'A2C':
							next_state = [self.test_PID.kp,self.test_PID.ki,self.test_PID.kd,error,ed,ei,past_rudder_angle]
						elif self.algorithm == 'DDPG':
							next_state = [error,ed,ei,past_rudder_angle]					

			#DRLPID Action
			if step_update and self.algorithm != 'Fuzzy Logic':
				if step_counter % self.update_freq == 0:
					action_counter += 1
					if self.algorithm == 'DQN' or self.algorithm == 'A2C':
						action = self.agent.agent.take_action(state)
						dPID = self.all_actions[action][0:3] #Add the increment to the PID gains
						self.test_PID.PID_gain_update(dPID)
					elif self.algorithm == 'DDPG':
						action = self.agent.agent.take_action(state)
						action = list(action[0])
						#Override the PID gains if the algorithm uses step-based DDPGPID
						self.test_PID.kp = action[0]
						self.test_PID.ki = action[1]
						self.test_PID.kd = action[2]

			#Step-based DRLPID Update
			if step_update and self.algorithm != 'Fuzzy Logic':
				if (step_counter + 1) % self.update_freq == 0 and step_counter > 0:
					if self.learn:
						reward = uti.step_reward(error,self.test_PID.error_old,self.reward_1,self.reward_2,self.test_PID.return_PID(),0.1,0.9) #Calculate Reward
						reward_sum += reward
						self.agent.update(state,action,reward,next_state,done)
					state = next_state

				#Add to the PID gain tracker
				kp_tracker.append(self.test_PID.kp)
				ki_tracker.append(self.test_PID.ki)
				kd_tracker.append(self.test_PID.kd)


			self.test_ship.rudder_angle = self.test_PID.PID_output(error,[-self.test_ship.max_rudder_angle,self.test_ship.max_rudder_angle])
			aes += np.abs(error) #Add to the sum of absolute error

			#Update Ship and Moving Obstacle Position
			self.test_ship.update(self.test_env.wcx,self.test_env.wcy)	
			ship_trackerx.append(self.test_ship.xe[0])
			ship_trackery.append(self.test_ship.xe[1])
			self.test_env.update_moving_obstacles(self.test_ship.coor_min,self.test_ship.coor_max)

			#Halt Mode Check, activate when target is met, where the engine will stop running for a certain amount of time (use for mimicking the action of the ship stopping to gather sample from the water)
			halt_check = self.test_env.target_check(self.test_ship.return_coordinates(),self.halt_radius)			
			if halt_mode == False and halt_check == True:
				halt_mode = True
				if animate:
					print('Halt Mode Activated')
			if halt_mode:
				halt_count += 1
				self.test_ship.n = 0
				if halt_count == self.halt_limit:
					halt_count = 0
					halt_mode = False
					self.test_ship.n = self.halt_mode_r_speed
					if animate:
						print('Halt Mode Deactivated')

			#Boundary Check
			if self.test_ship.xe[0] == self.test_ship.coor_range[0] or self.test_ship.xe[0] == self.test_ship.coor_range[1] or self.test_ship.xe[1] == self.test_ship.coor_range[0] or self.test_ship.xe[1] == self.test_ship.coor_range[1]:
				boundary_check = True
			else:
				boundary_check = False

			obstacle_hit = False

			#Termination Check, which the episode can be terminated if there is a collision with the obstacles and boundaries, all targets are met, or maximum number of steps was reached
			if obstacle_included:
				obstacle_hit = self.test_env.obstalce_check(self.test_ship.return_coordinates(),self.obstacle_radius) or self.test_env.moving_obstalce_check(self.test_ship.return_coordinates(),self.obstacle_radius)
				if obstacle_hit:
					done = True
			if len(self.test_env.targets_listx) == 0 or boundary_check or step_counter == self.num_steps:
				done = True

			#Animation if needed for seeing the motion in real time
			if animate:
				self.plot_motion(motion_title,ship_trackerx,ship_trackery,ref_angle,olistx,olisty,tlistx,tlisty,molistx,molisty,obstacle_included)
				if step_update:
					self.plot_PID_tracker(kp_tracker,ki_tracker,kd_tracker,pid_title)
				plt.pause(0.01)

			#Update information for next step
			self.test_PID.old_error_update(error)
			past_rudder_angle = self.test_ship.rudder_angle
			step_counter += 1

		AAE = aes / step_counter #Calculate the Average Absolute Error for the reference of the episode's performance

		if save_plot and (save_plot_counter % save_plot_freq == 0):
			self.plot_motion(motion_title,ship_trackerx,ship_trackery,ref_angle,olistx,olisty,tlistx,tlisty,molistx,molisty,obstacle_included)
			plt.savefig(folder_path+motion_title+'.png')
			if step_update:
				self.plot_PID_tracker(kp_tracker,ki_tracker,kd_tracker,pid_title)
				plt.savefig(folder_path+pid_title+'.png')
		if animate:
			print('Total Steps: ', step_counter)
			print('AAE: ', aes / step_counter)
			print('Terminate? Y/N')
			x = input()
			if x == 'Y':
				quit()
			plt.show()
		if step_update and self.algorithm != 'Fuzzy Logic':
			self.reward_average = reward_sum / action_counter #Calcualte average reward for all actions taken
		return AAE
