#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import random
from GridDM import GridWorld
maximum_change= 0.000001
grid = GridWorld(5)
discount = 0.9
converged = False
actions=['N', 'S', 'W', 'E']
policy = [[None]*grid.Size]*grid.Size             #np.zeros((gridSize, gridSize))


for i in range(grid.Size): #random policy 
    for j in range(grid.Size):
        policy[i][j] = np.random.choice(actions)

print('Initial policy:')
print(policy)

grid.Updated_GridValue=np.zeros((grid.Size, grid.Size)) #Initialization V(s) = 0
IterationNo=0 #policy evaluation
while converged==False:
    IterationNo+=1 #Saving number of iterations
    
    for i in range(grid.Size):
        for j in range(grid.Size):
            action=policy[i][j]
            [NewStates,rewards] = grid.new_state(i, j, action)
            grid.Updated_GridValue[i,j]=(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) 

    if np.sum(np.abs(grid.Updated_GridValue - grid.Initial_GridValue)) < maximum_change:
        converged=True
        grid.Final_Grid= grid.Updated_GridValue
    else:
        grid.Initial_Grid= grid.Updated_GridValue

    for i in range(grid.Size): #policy improvement
        for j in range(grid.Size): 
            old_a = policy[i][j]
            new_a = None
            best_value = 0
            #find best action
            for a in actions:
                [newStates,rewards] = grind.new_state(i,j,a)
                grid.Updated_GridValue[i,j]=(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) ##revisar matriz V
                if grid.Updated_GridValue[i,j] > best_value
                    best_value = grid.Updated_GridValue[i,j]
                    new_a = a
            policy[i][j] = new_a
            if new_a != old_a
            converged=False 
    if conververged
    break 
            
print("Total Iteration {}".format(IterationNo))


# In[ ]:




