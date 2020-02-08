# adapt the value_iteration.py code to be policy_iteration

# policy iteration

while True:
    
    # POLICY EVALUATION ####################################
        # iterate through all 25 states. At each state, iterate through all 4 actions
        # to calculate the value of each action.
        # Replace the value map with the calculated value.
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
        action_values = calculate_action_value(state, grid.valueMap)

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

