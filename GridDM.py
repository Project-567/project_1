#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import random


actions=['N', 'S', 'W', 'E']


class GridWorld():
    def __init__(self, GridSize):
        self.Updated_GridValue = np.zeros((GridSize, GridSize))
        self.Initial_GridValue = np.zeros((GridSize, GridSize))
        self.states = [[i, j] for i in range(GridSize) for j in range(GridSize)]
        self.Size = GridSize
        self.Final_Grid=[]

    def initial_state(self):
        i = random.randint(0, len(self.states)-1)
        rand_state = self.states[i]
        return rand_state #return initial random state
        
    def new_state(self,i,j,action):
        if action=='N':
            if i==0:
                NewState=[i,j]
                reward=-1
            else:
                NewState=[i-1,j]
                reward=0
        elif action=='S':
            if i==self.Size-1:
                NewState=[i,j]
                reward=-1
            else:
                NewState=[i+1,j]
                reward=0
        elif action=='W':
            if j==0:
                NewState=[i,j]
                reward=-1
            else:
                NewState=[i,j-1]
                reward=0
        elif action=='E':
            if j==self.Size-1:
                NewState=[i,j]
                reward=-1
            else:
                NewState=[i,j+1]
                reward=0
        if i==0 and j==1:# A to A prime
            NewState=[4,1]
            reward=10
        if i==0 and j==3:# B to B prime
            NewState=[2,3]
            reward=5

        return (NewState,reward)






