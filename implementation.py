# ========================================================
# This file implement a MDP problem (like taxi driver) 
# with stochastic/deterministic policy and its trajectory
# ========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import itertools
from mdp_solver import MDP_class, policy_class, trajectory, read_mdp_file
from policy_evaluation import linear_algebra
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
    # Get the inputs
    file_path         = "data.txt" # in the same directory
    number_of_states  = 3
    number_of_actions = 3
    discount_factor   = 0.9

    # Extract P and R from data file
    transition_f, return_f = read_mdp_file(path=file_path, N=number_of_states, M=number_of_actions)

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

taxi_driver_impl()
