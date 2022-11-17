#File for storing the class and functions for the autonomous ship
#The ship is based on a modified MMG-SMS ship motion models, the papers of these two models are as follow:
#- Mathematical Model for Manoeuvring Ship Motion, Yasuo Yoshimura
#- A ship motion simulation system, Shyuh-Kuang Ueng, David Lin, Chieh-Hong Liu

import numpy as np
import random
import utility as uti

class Ship:
	def __init__(self,mass,L,d,t,rho,kt,dp,tr,ah,xh,xr,j2,k,ar,A,e,w,eta,yr,lr,n,max_rudder_angle,rudder_angle,coor_range,bs,by):
		#Basic parameters
		self.mass = mass
		self.L = L
		self.d = d
		self.t = t
		self.rho = rho
		self.kt = kt
		self.dp = dp
		self.tr = tr
		self.ah = ah
		self.xh = xh
		self.xr = xr
		self.j2 = j2
		self.ar = ar
		self.A = A
		self.e = e
		self.w = w
		self.eta = eta
		self.k = k
		self.yr = yr
		self.lr = lr
		self.bs = bs
		self.by = by
		self.fa = (6.13 * self.A) / (2.25 + self.A)
		self.n = n
		self.max_rudder_angle = max_rudder_angle
		self.moi = self.mass * (self.L ** 2) / 12
		self.speed = np.array([0.0,0.0,0.0]) #Initial translational and rotational speed
		self.xe = [random.uniform(coor_range[0] + 100,coor_range[1] - 100), random.uniform(coor_range[0] + 100,coor_range[1] - 100), random.uniform(-np.pi,np.pi)] #Initialize the ship's coordinate
		self.rudder_angle = rudder_angle
		self.coor_min, self.coor_max = coor_range #Boundaries
		self.coor_range = coor_range
	def return_coordinates(self):
		return self.xe[0:2]
	def return_angle(self):
		return self.xe[2]
	def R_Transform(self,inputs,angle): #Change of reference frame from the ship to the global coordinate system
		r_matrix = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
		return np.matmul(r_matrix,inputs)
	def update(self,wcx,wcy): #Calculate forces and acclereation, and update the speed and coordainte of the ship
		#Layer 1
		ur = np.sqrt((1 + 8 * self.kt) / (np.pi * self.j2)) - 1
		ur = 1 + self.k * ur
		ur = self.eta * (ur ** 2) + (1 - self.eta)
		ur = self.e * (1 - self.w) * self.speed[0] * ur
		vr = self.yr * (self.speed[1] + self.speed[2] * self.lr)
		#Layer 2
		Ur = np.sqrt(ur ** 2 + vr ** 2)
		alphaR = self.rudder_angle - np.arctan2(-vr,ur)
			
		#Layer 3
		Fn = 0.5 * self.rho * self.ar * self.fa * (Ur ** 2) * np.sin(alphaR)

		#Layer 4: Calculate all forces
		XP = (1 - self.t) * self.rho * self.kt * (self.dp ** 4) * (self.n ** 2)
		XR = -(1 - self.tr) * Fn * np.sin(self.rudder_angle)
		YR = -(1 + self.ah) * Fn * np.cos(self.rudder_angle)
		NR = -(self.xr + self.ah * self.xh) * Fn * np.cos(self.rudder_angle)
		XD = self.bs * self.mass * (self.speed[0]**2)
		ND = self.by * self.moi * self.speed[2]

		#Layer 5: Forces sum
		X = XP + XR - XD
		Y = YR
		N = NR - ND

		#Layer 6: Accelerations
		aX = X / self.mass
		aY = Y / self.mass
		aN = N / self.moi

		#Layer 7: Change in speed
		self.speed += [aX,aY,aN]
		tspeed = self.R_Transform(self.speed,self.return_angle())

		#Layer 8: Update Position
		self.xe += tspeed
		self.xe += np.array([wcx,wcy,0])
		self.xe[0] = uti.clamp(self.xe[0],self.coor_min,self.coor_max)
		self.xe[1] = uti.clamp(self.xe[1],self.coor_min,self.coor_max)
		if self.xe[2] > np.pi:
			self.xe[2] -= 2 * np.pi
		elif self.xe[2] < -np.pi:
			self.xe[2] += 2 * np.pi