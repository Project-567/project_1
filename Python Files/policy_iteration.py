# adapt the value_iteration.py code to be policy_iteration

# policy iteration
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

# iterations = 0
theta = 0.000001
discount_factor = 0.8
delta_list = []

# initialize a policy: create an array of dimension (number of states by number of actions)
# for equal probability amongst all actions, divide everything by the number of actions
policy = np.ones([state_count, action_count]) / action_count

# # create a random policy
# random_policy = np.random.randint(1000, size=(state_count, action_count))
# random_policy = random_policy/random_policy.sum(axis=1)[:,None]
# policy = random_policy


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

final_max_iter = 0

# POLICY ITERATION #####################################3
while True:
    
    # POLICY EVALUATION ####################################
        # iterate through all 25 states. At each state, iterate through all 4 actions
        # to calculate the value of each action.
        # Replace the value map with the calculated value.

    # run policy evaluation
    final_value_map, max_iter, delta, policy = policy_evaluation(grid.valueMap, grid.states, discount_factor, theta, grid.reward, 
                                                                    grid.p_transition, grid.transition_prob, policy)

    # for plotting purpose
    # print("iter: ", max_iter)
    final_max_iter += max_iter
    # print("sum iter: ", final_max_iter)
    # print(len(delta))
    # print(delta)
    delta_list.extend(delta)
    print("Value Map: ")
    print(final_value_map)

    # POLICY IMPROVEMENT #######################################
        # iterate through every state and choose the best action with the current policy
        # calculate the action values of every state
        # take the best action and compare whether the best action is the same as the chosen one
        # update the policy with the best action
    
    # initate policy_true as stable
    policy_stable = True

    # iterate over every state
    for state_number, state in enumerate(grid.states):

        # choose the best action with the current policy
        choose_action = np.argmax(policy[state_number])

        # calculate the action values for each state using the current value function
        # eg. action_values = [#, #, #, #] = a value for each of the 4 actions
        action_values = calculate_action_value(state, final_value_map)

        # using the calculated action values, find the best action
        best_action = np.argmax(action_values)

        # if the chosen action is different than the calculated best action
        # then the current policy is not stable
        if choose_action != best_action:
            policy_stable = False

        # update the current policy with the new best action
        policy[state_number] = np.eye(action_count)[best_action]

    # if the policy is stable (eg. chosen action is the same as best action)
    # then we can exit
    # however, if it is not, then we need to perform policy evaluation and improvement again
    if policy_stable:
        break

print("Iterations: ")
print(final_max_iter)

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
print(final_value_map)

# PRINT DELTA PLOT #####################################################################
import matplotlib.pyplot as plt
# plot iteration vs delta
plt.plot(range(final_max_iter), delta_list)
plt.title('Policy Iteration with Discount Factor ' + str(discount_factor))
plt.xlabel('Iterations')
plt.ylabel('Max Delta')
plt.savefig('graphs/Policy-'+str(discount_factor)+'.png')
plt.show()