# Find the value function of policy
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

def policy_evaluation(value_map, states, discount_factor, theta, reward, transition, trans_prob, policy):
    iterations = 0
    delta_list = []
    while True:
        delta = 0
        iterations+=1
        valueMap_copy = np.copy(value_map)

        # start with the first state in the state list
        for state_number, state in enumerate(states):
            value = 0

            # perform 4 actions per state and add the rewards (value)
            for action_number, action in enumerate(actions):

                # get next position and reward
                new_position = transition(state, action)
                rewards = reward(state, action)

                # calculate value: policy*transition_prob*[r + gamma * value(s')]
                value += policy[state_number][action_number]*trans_prob*(rewards+(discount_factor*value_map[new_position[0], new_position[1]]))          

            # replace the value in valueMap with the value
            valueMap_copy[state[0], state[1]] = value

            # calculate delta
            delta = max(delta, np.abs(value - value_map[state[0], state[1]]))      
            # clear_output(wait=True)
            display('delta: ' + str(delta) + ' iterations: ' + str(iterations))

        # save data for plot
        delta_list.append(delta)

        # overwrite the original value map (update valuemap after one complete iteration of every state)
        value_map = valueMap_copy

        # stop when change in value function falls below a given threshold
        if delta < theta:
            break
    
    return value_map, iterations, delta_list, policy

def main():

    # define variables
    theta = 0.000001
    discount_factor = 0.99

    # create a grid object
    grid = Gridworld(5)

    # initialize a policy: create an array of dimension (number of states by number of actions)
    # for equal probability amongst all actions, divide everything by the number of actions
    policy = np.ones([state_count, action_count]) / action_count

    # run policy evaluation
    final_value_map, max_iter, delta, policy = policy_evaluation(grid.valueMap, grid.states, discount_factor, theta, grid.reward, 
                                                                    grid.p_transition, grid.transition_prob, policy)

    # print the final value function
    print("Total Iterations: ")
    print(max_iter)
    print("Value Function: ")
    np.set_printoptions(precision=4)
    print(final_value_map)

    # print delta vs iterations
    import matplotlib.pyplot as plt
    # plot iteration vs delta
    plt.plot(range(max_iter), delta)
    plt.title('Policy Evaluation with Discount Factor ' + str(discount_factor))
    plt.xlabel('Iterations')
    plt.ylabel('Max Delta')
    plt.savefig('graphs/Policy_Evaluation-'+str(discount_factor)+'.png')
    plt.show()

if __name__ == "__main__":
    main()