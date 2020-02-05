#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from GridDM import GridWorld

## action_prob = {'N': 0.25, 'S': 0.25, 'E': 0.25, 'W': 0.25} just for the policy evaluation?
maximum_change= 0.000001
grid = GridWorld(5)
discount = 0.9
converged = False

IterationNo=0
while converged==False:
    IterationNo+=1 #Saving number of iterations
    grid.Updated_GridValue=np.zeros((grid.Size, grid.Size)) #Initialization V(s) = 0
    for i in range(grid.Size):
        for j in range(grid.Size):
            action_value=[] #array to store 4 actions values and maximize
            for action in actions: #picks an action for each state (i,j)
                [NewStates,rewards] = grid.new_state(i,j,action)   #action takes to S+1 with reward r
                action_value.append(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) #store each action value
            grid.Updated_GridValue[i,j]=action_value[np.random.choice(np.flatnonzero(action_value==np.max(action_value)))]

    if np.sum(np.abs(grid.Updated_GridValue - grid.Initial_GridValue)) < maximum_change:
        converged=True
        grid.Final_Grid= grid.Updated_GridValue
    else:
        grid.Initial_Grid= grid.Updated_GridValue

print("Total Iteration {}".format(IterationNo))




# In[ ]:





# In[ ]:




