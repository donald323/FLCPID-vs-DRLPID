#File for PID class, allows control of the PID gains and PID output, with updating error information (error, error derivative and error integral)

import utility as uti
import numpy as np

class PID:
	def __init__(self,PID_gain_range,ikP,ikI,ikD,aw_threshold):
		self.PID_min, self.PID_max = PID_gain_range
		self.ikP = ikP
		self.ikI = ikI
		self.ikD = ikD
		self.kp = ikP
		self.ki = ikI
		self.kd = ikD
		self.error_old = 0
		self.error_sum = 0
		self.error_diff = 0
		self.aw_threshold = aw_threshold		
	def PID_gain_update(self,dPID):
		dkp,dki,dkd = dPID
		self.kp += dkp
		self.ki += dki
		self.kd += dkd
		self.kp = uti.clamp(self.kp,self.PID_min,self.PID_max)
		self.ki = uti.clamp(self.ki,self.PID_min,self.PID_max)
		self.kd = uti.clamp(self.kd,self.PID_min,self.PID_max)
	def PID_gain_reset(self):
		self.kp = self.ikP
		self.ki = self.ikI
		self.kd = self.ikD
	def old_error_update(self,error):
		self.error_old = error
	def return_PID(self):
		return [self.kp,self.ki,self.kd]
	def PID_output(self,error,output_range):
		output_min, output_max = output_range
		self.error_sum += error
		self.error_diff = error - self.error_old
		#Anti wind-up measure
		if np.abs(self.error_sum) < self.aw_threshold:
			output = self.kp * error + self.ki * self.error_sum + self.kd * self.error_diff
		else:
			output =  self.kp * error + self.kd * self.error_diff
		return uti.clamp(output, output_min, output_max) #Clamp output based on the conditions
	def error_reset(self):
		self.error_old = 0
		self.error_sum = 0
		self.error_diff = 0	