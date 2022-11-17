#File for generating environments for the ship simulator and export as a txt file

import random
import numpy as np
import json

numpoints = 10
mo_speed = 2
wcx = random.uniform(-1,1)
wcy = random.uniform(-1,1)
coormin,coormax = [0,2000]
xe = [random.uniform(coormin + 100,coormax - 100), random.uniform(coormin + 100,coormax - 100), random.uniform(-np.pi,np.pi)]
obstacle_listx = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
obstacle_listy = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
targets_listx = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
targets_listy = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
move_obstacles_x = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
move_obstacles_y = [random.randrange(coormin + 500,coormax - 500) for i in range(numpoints)]
moving_vectors = [random.uniform(-np.pi,np.pi) for i in range(numpoints)]

para = {"wcx": wcx, "wcy": wcy, "xe": xe, "obstacle_listx": obstacle_listx, "obstacle_listy": obstacle_listy, "targets_listx": targets_listx, "targets_listy": targets_listy
	, "move_obstacles_x": move_obstacles_x, "move_obstacles_y": move_obstacles_y, "moving_vectors": moving_vectors}

with open("test.txt","w") as f:
	for key,value in para.items():
		f.write("%s: %s\n" % (key,value))

#Print all content for checking
para_read = {}
with open("test.txt","rt") as f:
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
print(para_read)
