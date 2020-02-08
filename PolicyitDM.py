#!/usr/bin/env python
# coding: utf-8

# In[23]:

import numpy as np
import random
from GridDM import GridWorld
from GridDM import actions

delta= 0.000001
delta_list = []
grid = GridWorld(5)
discount = 0.85
converged = False
policy = [[None]*grid.Size]*grid.Size             #np.zeros((gridSize, gridSize))

for m in range(grid.Size): #random policy 
    for n in range(grid.Size):
        policy[m][n] = np.random.choice(actions)


IterationNo=0 #policy evaluation
while converged is False:
	IterationNo+=1 #Saving number of iterations
	print("Iteration N: {}".format(IterationNo))
	for i in range(grid.Size):
		for j in range(grid.Size):
			action=policy[i][j]
			[NewStates,rewards] = grid.new_state(i, j, action)
			grid.Updated_GridValue[i,j]=(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) 
	d = np.max(np.abs(np.subtract(grid.Updated_GridValue, grid.Initial_GridValue))) #diference beetwen V and V' 
	delta_list.append(d)   #storage for graph 
	if d < delta:
	#	converged=True
		grid.Final_Grid = grid.Updated_GridValue
		for x in range(grid.Size): #policy improvement
			for y in range(grid.Size): 
				old_a = policy[x][y]
				new_a = old_a
				best_value = grid.Final_Grid[x,y]
				#find best action
				for a in actions:
					[NewStates2,rewards2] = grid.new_state(x, y, a)
					grid.Updated_GridValue[x,y]=(rewards2 + discount * grid.Final_Grid[NewStates2[0], NewStates2[1]]) 
					if grid.Updated_GridValue[x,y] > best_value:
						best_value = grid.Updated_GridValue[x,y]
						new_a = a
				if new_a != old_a:
					converged is False 
					policy[x][y] = new_a
		if converged is False:
			grid.Initial_GridValue = grid.Updated_GridValue
		else: 
			break
	else:
		grid.Initial_GridValue = grid.Updated_GridValue

	
    #if conververged
    #break 
            
print("Total Iteration {}".format(IterationNo))
print(grid.Final_Grid)
print(policy)

# In[ ]:




