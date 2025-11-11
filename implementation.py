# ========================================================
# This file implement a MDP problem (like taxi driver) 
# with stochastic/deterministic policy and its trajectory
# ========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import itertools
from mdp_solver import MDP_class, policy_class, read_mdp_file
from policy_evaluation import linear_algebra, bellman_iteration
import matplotlib.pyplot as plt

# =============================================================================
# 1. Implementation function of taxi driver
# =============================================================================
def taxi_driver_impl():
    """
    Compute the value of all deterministic policies for the taxi driver problem (for γ = 0.9).

    Args :
        None
    Returns : 
        all_policies_with_V_values (dict): 
            A dictionary mapping each policy (as a tuple) to its V-function.
    """
    number_of_states = 3
    number_of_actions = 3
    discount_factor   = 0.9

    # Extract P and R from data file
    transition_f, return_f = read_mdp_file(path="data.txt", N=number_of_states, M=number_of_actions)

    # Create a MDP object for taxi driver
    mdp_obj = MDP_class(P=transition_f, R=return_f, gamma=discount_factor, N=number_of_states, M=number_of_actions)

    # Create a Policy object 
    policy_obj = policy_class(N=number_of_states, M=number_of_actions,policy_type='d')

    # Create dictionary to store all policies with their keys
    all_policies_with_V_values = {} 
    action_range = range(number_of_actions) # [0, 1, 2]

    # Systematically generate every possible policy by itertools.product
    total_policies_generator = itertools.product(action_range, repeat=number_of_states)

    # Loop through all 27 policies
    for policy_tuple in total_policies_generator:
        
        # Set the current policy in our policy object
        policy_obj.policy_array = np.array(policy_tuple)

        # Compute the value for this specific policy
        v_value = linear_algebra(mdp_obj, policy_obj)

        # Store it in the dictionary using the HASHABLE tuple as the key
        all_policies_with_V_values[policy_tuple] = v_value
    
    # Plot the optimal value for each three states:
    # This will be a list of arrays: [array([10, 8, 5]), array([9, 12, 6]), ...]
    all_v_arrays = all_policies_with_V_values.values()

    # This converts the list into a 4x3 array and finds the max of each column
    V_optimal = np.max(np.array(list(all_v_arrays)), axis=0)

    number_of_states = len(V_optimal) # This will be 3

    # Create x-axis labels for the bars
    states = [f"State {i}" for i in range(number_of_states)]

    # Create the bar chart
    plt.figure(figsize=(8, 5)) # Set a nice size
    plt.bar(states, V_optimal)

    # Add labels and a title
    plt.title("Optimal Value Function (V*) for Taxi Problem")
    plt.xlabel("State")
    plt.ylabel("Value (V)")

    # Add the value on top of each bar (optional, but helpful)
    for i in range(number_of_states):
        plt.text(i, V_optimal[i] + 0.1, f"{V_optimal[i]:.2f}", ha='center')

    # Display the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# =============================================================================
# 2. Create 21 with a die
# =============================================================================
def create_21_mdp():
    """
    Create the 21-with-one-dice MDP.

    States:
        0-20: current score
        21: terminal/bust state

    Actions:
        0: roll
        1: stop

    Returns:
        P: Transition probabilities, shape (N, M, N)
        R: Rewards, shape (N, M, N)
    """
    N = 22  # states 0-21
    M = 2   # actions: 0=roll, 1=stop

    P = np.zeros((N, M, N))
    R = np.zeros((N, M, N))
    
    for s in range(21):  # terminal state 21 has no outgoing transitions
        # roll 
        for dice in range(1, 7):  # dice outcomes 1-6
            s_next = s + dice
            if s_next > 21:
                s_next = 21  # bust/terminal
                reward = 0
            elif s_next == 21:
                reward = 1  # win
            else:
                reward = 0
            P[s, 0, s_next] += 1/6
            R[s, 0, s_next] = reward
        
        # stop 
        s_next = 21  # terminal state
        P[s, 1, s_next] = 1
        R[s, 1, s_next] = s / 21  # normalized reward for stopping
    
    # Terminal state 21: no transitions
    P[21, :, 21] = 1
    R[21, :, 21] = 0
    
    return P, R

def twenty_one_with_a_die_impl():
    """
    Evaluate a uniformly random policy for 21-with-a-die by both methods.

    Args :
        None
    Returns : 
        all_policies_with_V_values (dict): 
            A dictionary mapping each policy (as a tuple) to its V-function.
    """
    transition_f,return_f = create_21_mdp()
    # Create a MDP object for 21
    mdp_obj = MDP_class(P=transition_f, R=return_f, gamma=0.9, N=22, M=2)
    # Create a Policy object 
    policy_obj = policy_class(N=22, M=2,policy_type='s')
    policy_obj.generate_policy() # creates the policy array

    print(f"Policy for 21 problem : {policy_obj.policy_array}")

    # Compute the value for this specific policy
    v_value = linear_algebra(mdp_obj, policy_obj)
    print(f"Value of linear algebra : {v_value}")

    v_value = bellman_iteration(mdp_obj, policy_obj, tol=1e-6, max_iter=10000)
    print(f"Value of bellman iteration : {v_value}")

# =============================================================================
# 5. Main Function (for testing)
# =============================================================================
def main_implementation():
    """
    Start the process of Implementing.
    
    Args:
        None
    
    Returns:
        None
    """

    taxi_driver_impl()
    twenty_one_with_a_die_impl()

# =============================================================================
# 6. Main Execution (only when the script is executed directly)
# =============================================================================
if __name__ == '__main__':
    main_implementation()

