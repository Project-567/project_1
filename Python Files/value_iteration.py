# implement value iteration to solve the Bellman equations derived earlier
# run the algorithm with different discount factor

# Value Iteration

discount_factor = 0.8 # small prefer immediate reward, large prefer future reward
iterations = 0
theta = 0.000001
delta_list = []

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