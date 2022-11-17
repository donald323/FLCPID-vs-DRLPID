#File for Artificial Potential Field (APF), to generate objects for interaction and calculated the forces from the APF

import utility as uti
import numpy as np
from statsmodels import api as sm

class Artificial_Potential_Field:
	def __init__(self,r,k,kb,kv):
		self.detection_radius = r
		self.k = k #Coefficient for calculating attractive/repulsive forces
		self.kb = kb #Coefficient for calculating boundary repulsive forces
		self.kv = kv #Coefficient for calculating virtual attractive/repulsive forces
	def generate_list(self,full_listx,full_listy,coor): #Generate a list of objects within detection radius
		coorx,coory = coor
		detection_listx = []
		detection_listy = []
		for i in range(len(full_listx)):
			if uti.distance(coorx,coory,full_listx[i],full_listy[i]) < self.detection_radius:
				detection_listx.append(full_listx[i])
				detection_listy.append(full_listy[i])
		return detection_listx,detection_listy
	def cal_force(self,test_listx,test_listy,coor,coeff): #Calculate the sum of all forces from the list of objects in both axes
		coorx,coory = coor
		xforce = 0
		yforce = 0
		for i in range(len(test_listx)):
			dist = uti.distance(coorx,coory,test_listx[i],test_listy[i])
			force = (coeff/ (dist ** 2))
			force_angle = np.arctan2(test_listy[i] - coory, test_listx[i] - coorx)
			xforce += force * np.cos(force_angle)
			yforce += force * np.sin(force_angle)
		return xforce,yforce

	def bounadry_cal_force(self,coor_range,coor): #Calculating repulsive forces from the boundaries
		xforce = 0
		yforce = 0
		coorx,coory = coor
		coormin,coormax = coor_range
		xdistL = coormin - coorx
		xdistR = coormax - coorx
		ydistD = coormin - coory
		ydistU = coormax - coory
		xforce += (self.kb /(xdistL**2))
		xforce -= (self.kb /(xdistR**2))
		yforce -= (self.kb /(ydistU**2))
		yforce += (self.kb /(ydistD**2))
		return xforce,yforce