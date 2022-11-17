#File for constructing a 2D environment for autonomous ship

import random
import utility as uti
import numpy as np

class Environment:
	def __init__(self,numpoints,coormin,coormax,wcmin,wcmax,speed_limit):
		self.obstacle_listx = [random.uniform(coormin,coormax) for i in range(numpoints)] #List of random static obstacles
		self.obstacle_listy = [random.uniform(coormin,coormax) for i in range(numpoints)]
		self.targets_listx = [random.uniform(coormin,coormax) for i in range(numpoints)] #List of random targets
		self.targets_listy = [random.uniform(coormin,coormax) for i in range(numpoints)]
		self.reached_target_listx = [] #Empty list for storing targets that have already been reached
		self.reached_target_listy = []
		self.move_obstacles_x = [random.uniform(coormin,coormax) for i in range(numpoints)] #List of random moving obstacles
		self.move_obstacles_y = [random.uniform(coormin,coormax) for i in range(numpoints)]
		self.move_obstacles_speed_limit = speed_limit #Speed limit of the moving obstacles
		self.moving_vectors = [random.uniform(-np.pi,np.pi) for i in range(numpoints)] #List of the directions of each moving obstacles to move
		self.wcx = random.uniform(wcmin,wcmax) #Water current speed
		self.wcy = random.uniform(wcmin,wcmax)
		self.boundary = [coormin,coormax] #Coordinate of the boundaries
	def target_check(self,coors,range_threshold): #Check whether a target was met
		cx,cy = coors
		halt_flag = False
		for i in range(len(self.targets_listx)):
			temp_targetx = self.targets_listx[i]
			temp_targety = self.targets_listy[i]
			if uti.distance(cx,cy,temp_targetx,temp_targety) < range_threshold: #If the distance between the ship and a target is shorter than threshold, it is considered reached
				self.reached_target_listx.append(self.targets_listx[i]) #Add to the reached target list
				self.reached_target_listy.append(self.targets_listy[i])
				del self.targets_listx[i] #Delete from the existed list to prvent the ship to revisit this target again
				del self.targets_listy[i]
				halt_flag = True #Activate halt mode
				break
		return halt_flag
	def obstalce_check(self,coors,range_threshold): #Check whether a static obstacle collided with a ship (same as target check)
		cx,cy = coors
		obstacle_hit = False
		for i in range(len(self.obstacle_listx)):
			temp_targetx = self.obstacle_listx[i]
			temp_targety = self.obstacle_listy[i]
			if uti.distance(cx,cy,temp_targetx,temp_targety) < range_threshold:
				obstacle_hit = True
				break
		return obstacle_hit		
	def closest_target(self,coors): #Search for the nearest target (use for Virtual Mode)
		cx,cy = coors
		close_targetx = self.targets_listx[0]
		close_targety = self.targets_listy[0]
		shortest_dist = uti.distance(cx,cy,close_targetx,close_targety)
		for i in range(len(self.targets_listx)):
			if uti.distance(cx,cy,self.targets_listx[i],self.targets_listy[i]) < shortest_dist:
				shortest_dist = uti.distance(cx,cy,self.targets_listx[i],self.targets_listy[i])
				close_targetx = self.targets_listx[i]
				close_targety = self.targets_listy[i]
		return close_targetx,close_targety
	def closest_obstacle(self,coors): #Search for the nearest obstacle (use for Virtual Mode)
		cx,cy = coors
		close_obstaclex = self.obstacle_listx[0]
		close_obstacley = self.obstacle_listy[0]
		shortest_dist = uti.distance(cx,cy,close_obstaclex,close_obstacley)
		for i in range(len(self.obstacle_listx)):
			if uti.distance(cx,cy,self.obstacle_listx[i],self.obstacle_listy[i]) < shortest_dist:
				shortest_dist = uti.distance(cx,cy,self.obstacle_listx[i],self.obstacle_listy[i])
				close_obstaclex = self.obstacle_listx[i]
				close_obstacley = self.obstacle_listy[i]
		return close_obstaclex,close_obstacley
	def update_moving_obstacles(self,coormin,coormax): #Update coordiante of the moving obstacles
		for i in range(len(self.move_obstacles_x)):
			self.move_obstacles_x[i] += self.move_obstacles_speed_limit * np.cos(self.moving_vectors[i])
			self.move_obstacles_y[i] += self.move_obstacles_speed_limit * np.sin(self.moving_vectors[i])
			if self.move_obstacles_x[i] > coormax or self.move_obstacles_x[i] < coormin or self.move_obstacles_y[i] > coormax or self.move_obstacles_y[i] < coormin: #If the moving obstacle hit the boundary, it move in the oppositie direction until the next collision
				self.moving_vectors[i] = self.moving_vectors[i] + np.pi
				if self.moving_vectors[i] < -np.pi:
					self.moving_vectors[i] += 2 * np.pi
				elif self.moving_vectors[i] > np.pi:
					self.moving_vectors[i] -= 2 * np.pi
	def moving_obstalce_check(self,coors,range_threshold): #Check whether a moving obstacle collided with a ship
		cx,cy = coors
		obstacle_hit = False
		for i in range(len(self.move_obstacles_x)):
			temp_targetx = self.move_obstacles_x[i]
			temp_targety = self.move_obstacles_y[i]
			if uti.distance(cx,cy,temp_targetx,temp_targety) < range_threshold:
				obstacle_hit = True
				break
		return obstacle_hit
