import numpy as np
from GridDM import GridWorld
from GridDM import actions
import matplotlib.pyplot as plt

delta = 0.000001
delta_list = []
grid = GridWorld(5)
discount = 0.8
converged = False
state_count = grid.Size*grid.Size

IterationNo=0
while converged==False:
	IterationNo+=1 #Saving number of iterations
	print("Iteration N: {}".format(IterationNo))
	grid.Updated_GridValue=np.zeros((grid.Size, grid.Size)) #Initialization V(s) = 0	
	for i in range(grid.Size):
		for j in range(grid.Size):
			action_value=[] #array to store 4 actions values and maximize
			for action in actions: #picks an action for each state (i,j)
				#print(i, j, action)
				[NewStates,rewards] = grid.new_state(i,j,action)   #action takes to S+1 with reward r
				action_value.append(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) #store each action value
			grid.Updated_GridValue[i,j]=action_value[np.random.choice(np.flatnonzero(action_value==np.max(action_value)))] 

	d = np.max(np.abs(np.subtract(grid.Updated_GridValue, grid.Initial_GridValue))) #diference beetwen V and V' 
	delta_list.append(d)   #storage for graph 
	if d < delta: #convergence calculation
		converged=True
		grid.Final_Grid = grid.Updated_GridValue
		print(d)
	else:
		grid.Initial_GridValue = grid.Updated_GridValue
		print(d)

print("Total Iteration {}".format(IterationNo))
np.set_printoptions(precision=4)
print(grid.Final_Grid)

## Ploting

delta_list2 = delta_list[0::state_count]

plt.plot(range(IterationNo), delta_list)
plt.title('Value Iteration with Discount Factor ' + str(discount))
plt.xlabel('Iterations')
plt.ylabel('Max Delta')
plt.savefig('Desktop\MIE 1 YEAR\MIE567H1 Dyn and Dist Decision Making\Project 1\GridDM\graphs\ValueIteration-'+str(discount)+'.png')









# In[ ]:




