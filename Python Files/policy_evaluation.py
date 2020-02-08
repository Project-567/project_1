# Find the value function of policy
# Can use either an iterative method or solve the system of equations
# Show the value function you obtained to at least 4 decimals
from Gridworld import Gridworld
import numpy as np

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
delta_list = []
discount_factor = 0.99 # small prefer immediate reward, large prefer future reward

# initialize a policy: create an array of dimension (number of states by number of actions)
# for equal probability amongst all actions, divide everything by the number of actions
policy = np.ones([state_count, action_count]) / action_count

# policy at state 0 = [0, 0]
# returns a probability for each action given state
policy[0]

# create a grid object
grid = Gridworld(5)

def policy_evaluation(value_map, states, ):

    while True:
        delta = 0
        iterations+=1
        valueMap_copy = np.copy(grid.valueMap)

        # start with the first state in the state list
        for state_number, state in enumerate(grid.states):
            value = 0

            # perform 4 actions per state and add the rewards (value)
            for action_number, action in enumerate(actions):

                # get next position and reward
                new_position = grid.p_transition(state, action)
                reward = grid.reward(state, action)

                # calculate value: policy*transition_prob*[r + gamma * value(s')]
                value += policy[state_number][action_number]*grid.transition_prob*(reward+(discount_factor*grid.valueMap[new_position[0], new_position[1]]))          

            # replace the value in valueMap with the value
            valueMap_copy[state[0], state[1]] = value

            # calculate delta
            delta = max(delta, np.abs(value - grid.valueMap[state[0], state[1]]))       
            clear_output(wait=True)
            display('delta: ' + str(delta) + ' iterations: ' + str(iterations))

            # save data for plot
            delta_list.append(delta)

        # overwrite the original value map (update valuemap after one complete iteration of every state)
        grid.valueMap = valueMap_copy

        # stop when change in value function falls below a given threshold
        if delta < theta:
            break
    
    return value_map



# print the final value function
print("Value Function: ")
np.set_printoptions(precision=4)
print(grid.valueMap)