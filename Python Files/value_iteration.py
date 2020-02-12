# implement value iteration to solve the Bellman equations derived earlier
# run the algorithm with different discount factor

# Value Iteration
from Gridworld import Gridworld
from policy_evaluation import policy_evaluation
import numpy as np
from random import randint

# display output
from random import uniform
import time
from IPython.display import display, clear_output

actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) # total number of actions
gridSize = 5 # create a square grid of gridSize by gridSize
state_count = gridSize*gridSize # total number of states

iterations = 0
theta = 0.000001
discount_factor = 0.8
delta_list = []

# UNCOMMENT THE FOLLOWING FOR EVEN POLICY
# # initialize a policy: create an array of dimension (number of states by number of actions)
# # for equal probability amongst all actions, divide everything by the number of actions
# policy = np.ones([state_count, action_count]) / action_count

# create a random policy
random_policy = np.random.randint(1000, size=(state_count, action_count))
random_policy = random_policy/random_policy.sum(axis=1)[:,None]
policy = random_policy

# create a grid object
grid = Gridworld(5)

def calculate_action_value(state, value):
    A = np.zeros(action_count)
    
    # perform 4 actions per state and add the rewards (value)
    for action_number, action in enumerate(actions):
            
        # get next position and reward
        new_position = grid.p_transition(state, action)
        reward = grid.reward(state, action)
        
        # get next position and reward
        new_position = grid.p_transition(state, action)
        reward = grid.reward(state, action)

        # calculate value of action: transition_prob*[r + gamma * value(s')]
        A[action_number] += grid.transition_prob*(reward+(discount_factor*value[new_position[0], new_position[1]]))
    
    return A


while True:
    delta = 0
    iterations+=1
    valueMap_copy = np.copy(grid.valueMap)

    # FIND OPTIMAL VALUE #######################################################   
    # start with the first state in the state list
    for state_number, state in enumerate(grid.states):
        value = 0

        # calculate best action value given current state and value function
        action_values = calculate_action_value(state, grid.valueMap)

        # choose the best action value
        best_action_value = np.max(action_values)

        # value of current state is equal to the best action value
        value = best_action_value

        # replace the value in valueMap with the value
        valueMap_copy[state[0], state[1]] = value

        # calculate delta
        delta = max(delta, np.abs(value - grid.valueMap[state[0], state[1]]))       
        clear_output(wait=True)
        display('delta: ' + str(delta) + ' iterations: ' + str(iterations))

    # save data for plot
    delta_list.append(delta)

    # overwrite the original value map (after complete iteration of every step)
    grid.valueMap = valueMap_copy

    # stop when change in value function falls below a given threshold
    if delta < theta:
        break

# EXTRACT POLICY FROM OPTIMAL VALUE #####################################################
for state_number, state in enumerate(grid.states):
    # using the current value map (optimal at this point), calculate the action values
    action_values = calculate_action_value(state, grid.valueMap)
    # return the action with the highest action value
    best_action = np.argmax(action_values)
    # update policy accordingly
    policy[state_number] = np.eye(action_count)[best_action]

# PRINT POLICY TABLE ################################################################################
# import pandas library
import pandas as pd
# define column and index
columns=range(grid.size)
index = range(grid.size)
# define dataframe to represent policy table
policy_table = pd.DataFrame(index = index, columns=columns)

# iterate through policy to make a table that represents action number
# as action name (eg. left, right, up, down)
for state in range(len(policy)):
    for action in range(policy.shape[1]):
        if policy[state][action] == 1:

            # calculate the row and column coordinate of the current state number
            row = int(state/grid.size)
            column = round((state/grid.size - int(state/grid.size))*grid.size)

            # get action name
            if action == 0:
                action_name = 'up'
            elif action == 1:
                action_name = 'right'
            elif action == 2:
                action_name = 'down'
            else:
                action_name = 'left'
            
            # assign action name
            policy_table.loc[row][column] = action_name

print("Policy Table: ")
print(policy_table)
print("Value Map: ")
np.set_printoptions(precision=4)
print(grid.valueMap)
# print("Policy: ")
# print(policy)
# np.save("value_iteration_80", policy)


# PRINT DELTA PLOT #####################################################################
import matplotlib.pyplot as plt
# plot iteration vs delta
plt.plot(range(iterations), delta_list)
plt.title('Value Iteration with Discount Factor ' + str(discount_factor))
plt.xlabel('Iterations')
plt.ylabel('Max Delta')
plt.savefig('graphs/value_iteration_'+str(int(discount_factor*100))+'.png')
plt.show()