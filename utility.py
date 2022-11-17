#Some other functions that would useful for other tasks

import numpy as np

#A customized clamp-limit function
def clamp(x,minx,maxx):
	if x < minx:
		return minx
	elif x > maxx:
		return maxx
	else:
		return x

#Return the Euclidean distance of two points
def distance(cx,cy,tox,toy):
	return np.sqrt((toy - cy) ** 2 + (tox - cx) ** 2)

#Calculating episode-based DRLPID reward
def episode_reward(AAE,old_AAE,r1,r2,PID_gains,PID_1,PID_2):
	AAE_diff = AAE - old_AAE
	if AAE < r2:
		reward = 2
	elif AAE < r1 and AAE_diff < 0:
		reward = 1
	else:
		reward = 0
	for PID in PID_gains:
		if PID < PID_1 or PID > PID_2:
			reward = -2
			break
	return reward

#Calculating step-based DRLPID reward
def step_reward(error,error_old,r1,r2):
	error_diff = np.abs(error) - np.abs(error_old)
	if np.abs(error) < r2:
		reward = 2
	elif np.abs(error) < r1 and error_diff < 0:
		reward = 1
	else:
		reward = 0	
	return reward