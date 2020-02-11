
import numpy as np
import random
from GridDM import GridWorld
from GridDM import actions
import copy
import matplotlib.pyplot as plt

delta= 0.000001
delta_list = []
grid = GridWorld(5)
discount = 0.99
converged = False
policymatrix = np.zeros((grid.Size, grid.Size))
policy = np.where(policymatrix<0,policymatrix,'A')            

for m in range(grid.Size): #random policy 
    for n in range(grid.Size):
    	policy[m,n] = np.random.choice(actions)


IterationNo=0 #policy evaluation
while converged is False:
	IterationNo+=1 #Saving number of iterations
	print("Iteration N: {}".format(IterationNo))
	for i in range(grid.Size):
		for j in range(grid.Size):
			action=policy[i,j]
			[NewStates,rewards] = grid.new_state(i, j, action)
			grid.Updated_GridValue[i,j]=(rewards + discount * grid.Initial_GridValue[NewStates[0], NewStates[1]]) 
	d = np.max(np.abs(np.subtract(grid.Updated_GridValue, grid.Initial_GridValue))) #diference beetwen V and V'
	print("Iteration N: {}".format(IterationNo), 'Delta:',d) 
	delta_list.append(d)   #storage for graph 
	grid.Final_Grid = copy.copy(grid.Updated_GridValue)
	if d < delta:
		a_changes=0
		npolicy=copy.copy(policy)
		for x in range(grid.Size): #policy improvement
			for y in range(grid.Size): 
				old_a = npolicy[x,y]
				new_a = npolicy[x,y]
				best_value = grid.Final_Grid[x,y]
				#find best action
				for a in actions:
					[NewStates2,rewards2] = grid.new_state(x, y, a)
					grid.Updated_GridValue[x,y]=(rewards2 + discount * grid.Final_Grid[NewStates2[0], NewStates2[1]]) 
					if grid.Updated_GridValue[x,y] > best_value:
						best_value = grid.Updated_GridValue[x,y]
						new_a = a
						
				if new_a != old_a:
					a_changes+=1
					npolicy[x,y]=new_a
		#policy convergence 
		if a_changes == 0:
			break
		else:
			grid.Initial_GridValue = copy.copy(grid.Final_Grid)
			policy = copy.copy(npolicy)
	else:
		grid.Initial_GridValue = copy.copy(grid.Final_Grid)


            
print("Total Iterations: {}".format(IterationNo))
print(policy)

print("Value Function: ")
np.set_printoptions(precision=4)
print(grid.Final_Grid)


    # plot iteration vs delta
plt.plot(range(IterationNo), delta_list)
plt.title('Policy Evaluation with Discount Factor ' + str(discount))
plt.xlabel('Iterations')
plt.ylabel('Max Delta')
plt.savefig('Desktop\MIE 1 YEAR\MIE567H1 Dyn and Dist Decision Making\Project 1\GridDM\graphs/Policy_Iteration-'+str(discount)+'.png')
plt.show()
# In[ ]:




