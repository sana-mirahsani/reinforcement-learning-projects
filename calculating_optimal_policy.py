# ==========================================================
# This file finds the optimal policy by different methods : 
# 1.Policy Iteration 2.Iteration over the value 3.linear programming
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
from mdp_solver import MDP_class, policy_class, trajectory, read_mdp_file
from policy_evaluation import linear_algebra
from implementation import create_21_mdp
import pulp
import matplotlib.pyplot as plt

# =============================================================================
# 1. Find optimal policy functions (all three)
# =============================================================================

# Method one : Policy Iteration
def policy_iteration(mdp, policy, evaluate_policy_func, max_iterations=1000):
    """
    Policy Iteration algorithm to find the optimal policy.

    Args:
        mdp : obj
            An object from MDP_class containing P, R, gamma, N (states), M (actions).
        policy : obj
            An object from Policy_class (can be initialized randomly).
        evaluate_policy_func : function
            A function to evaluate a given policy (e.g., linear_algebra).
        max_iterations : int
            Maximum number of iterations to prevent infinite loops.

    Returns:
        policy : obj
            The optimal policy.
        V : 1D vector
            The value function of the optimal policy.
    """
    for iteration in range(max_iterations):
        # Policy Evaluation
        V = evaluate_policy_func(mdp, policy)

        # Policy Improvement
        policy_stable = True
        new_policy_array = np.copy(policy.policy_array)

        for s in range(mdp.N):
            old_action = policy.policy_array[s]
            Q_values = np.zeros(mdp.M)

            # Compute Q(s, a) for each possible action
            for a in range(mdp.M):
                Q_values[a] = np.sum(mdp.P[s, a] * (mdp.R[s, a] + mdp.gamma * V))

            # Choose the action with the maximum Q(s, a)
            best_action = np.argmax(Q_values)
            new_policy_array[s] = best_action

            # Check if policy has changed
            if best_action != old_action:
                policy_stable = False

        # Update the policy
        policy.policy_array = new_policy_array

        # Check for convergence
        if policy_stable:
            print(f"Policy iteration converged after {iteration + 1} iterations.")
            break

    return policy, V

# Method two : value_iteration
def value_iteration(mdp, epsilon=1e-6, max_iterations=1000):
    """
    Value Iteration algorithm to find the optimal value function and policy.
    
    Args:
        mdp : obj
            An object from MDP_class containing:
            - P: transition probabilities, shape (N, M, N)
            - R: rewards, shape (N, M, N)
            - gamma: discount factor
            - N: number of states
            - M: number of actions
        epsilon : float
            Convergence threshold
        max_iterations : int
            Maximum number of iterations

    Returns:
        V : 1D numpy array
            Optimal value function
        policy : 1D numpy array
            Optimal deterministic policy (best action per state)
    """

    # Initialization
    V = np.zeros(mdp.N)
    
    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(mdp.N)

        # Bellman Optimality Update
        for s in range(mdp.N):
            Q_values = np.zeros(mdp.M)
            for a in range(mdp.M):
                Q_values[a] = np.sum(mdp.P[s, a] * (mdp.R[s, a] + mdp.gamma * V))
            V_new[s] = np.max(Q_values)

        # Check Convergence
        delta = np.max(np.abs(V_new - V))
        V = V_new.copy()

        if delta < epsilon:
            print(f"Value iteration converged after {iteration + 1} iterations.")
            break

    # Derive Optimal Policy
    policy = np.zeros(mdp.N, dtype=int)
    for s in range(mdp.N):
        Q_values = np.zeros(mdp.M)
        for a in range(mdp.M):
            Q_values[a] = np.sum(mdp.P[s, a] * (mdp.R[s, a] + mdp.gamma * V))
        policy[s] = np.argmax(Q_values)

    return V, policy

# Method three : linear programming
def linear_programming(mdp):
    """
    Solve Bellman's optimality equation using Linear Programming.

    Args:
        mdp : obj
            MDP object with attributes:
            - P: transition probabilities (N x M x N)
            - R: rewards (N x M x N)
            - gamma: discount factor
            - N: number of states
            - M: number of actions

    Returns:
        V_opt : np.array
            Optimal value function.
        policy_opt : np.array
            Optimal policy.
    """
    # Define LP problem ---
    prob = pulp.LpProblem("Bellman_LP", pulp.LpMinimize)

    # Define variables ---
    V_vars = pulp.LpVariable.dicts("V", range(mdp.N), lowBound=None)

    # Objective function
    # Minimize sum of all V(s)
    prob += pulp.lpSum([V_vars[s] for s in range(mdp.N)])

    # Constraints (Bellman inequalities)
    for s in range(mdp.N):
        for a in range(mdp.M):
            rhs = 0
            for s_next in range(mdp.N):
                rhs += mdp.P[s, a, s_next] * (
                    mdp.R[s, a, s_next] + mdp.gamma * V_vars[s_next]
                )
            prob += V_vars[s] >= rhs

    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the optimal value function
    V_opt = np.zeros(mdp.N)
    for s in range(mdp.N):
        V_opt[s] = V_vars[s].varValue

    # Derive the optimal policy
    policy_opt = np.zeros(mdp.N, dtype=int)
    for s in range(mdp.N):
        Q_values = np.zeros(mdp.M)
        for a in range(mdp.M):
            Q_values[a] = np.sum(mdp.P[s, a] * (mdp.R[s, a] + mdp.gamma * V_opt))
        policy_opt[s] = np.argmax(Q_values)

    return V_opt, policy_opt

# =============================================================================
# 2. Influence of γ
# =============================================================================
def gamma_sensitivity(policy_iteration_func, linear_algebra_func, mdp, policy, step=0.01):
    """
    Analyze the influence of gamma on the optimal policy for a given MDP.

    Args:
        policy_iteration_func: function
        linear_algebra_func: function
        mdp: object
            MDP object with attributes P, R, gamma, N, M
        policy: object
            Initial policy object (Policy class)
        step: float
            Step size for gamma (default 0.01)

    Returns:
        policy_changes: list of tuples
            Each tuple: (gamma, optimal_policy_array)
    """
    # Prepare gamma values 
    gamma_values = np.arange(step, 1.0, step)
    previous_policy = None
    policy_changes = []

    # Loop over gamma values
    for gamma in gamma_values:
        mdp.gamma = gamma
        
        # Reinitialize policy for each gamma (keep the same policy type)
        if policy.policy_type == 'd':
            policy.policy_array = np.random.choice(mdp.M, size=mdp.N)
        elif policy.policy_type == 's':
            policy.policy_array = np.ones((mdp.N, mdp.M)) / mdp.M
        
        # Run Policy Iteration
        optimal_policy, _ = policy_iteration_func(mdp, policy, linear_algebra_func)
        
        # Detect policy change
        if previous_policy is None or not np.array_equal(optimal_policy.policy_array, previous_policy):
            policy_changes.append((gamma, optimal_policy.policy_array.copy()))
            previous_policy = optimal_policy.policy_array.copy()

    print("Gamma values where the optimal policy changes:")
    for gamma, pol in policy_changes:
        print(f"Gamma = {gamma:.2f}, Optimal policy = {pol}")

    return policy_changes

# =============================================================================
# 3. plot the chnages
# =============================================================================
def plot_policy_changes(policy_changes, mdp):
    """
    Plot the optimal policy for each state as a function of gamma.

    Args:
        policy_changes: list of tuples
            Output of gamma_sensitivity function: (gamma, optimal_policy_array)
        mdp: MDP object
            Needed to get number of states
    """
    gammas = [g for g, _ in policy_changes]
    num_states = mdp.N
    
    # Create a matrix: rows = states, columns = gamma indices
    policy_matrix = np.zeros((num_states, len(gammas)))
    
    for j, (_, pol) in enumerate(policy_changes):
        policy_matrix[:, j] = pol  # store optimal action for each state
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(policy_matrix, aspect='auto', origin='lower', cmap='tab20')
    plt.colorbar(label='Optimal Action')
    plt.xlabel('Gamma index')
    plt.ylabel('State')
    plt.title('Optimal Policy for Each State vs Gamma')
    plt.xticks(ticks=np.arange(0, len(gammas), max(1, len(gammas)//10)),
               labels=[f"{gammas[i]:.2f}" for i in range(0, len(gammas), max(1, len(gammas)//10))])
    plt.yticks(ticks=np.arange(0, num_states, max(1, num_states//10)))
    plt.show()

# =============================================================================
# 4. Taxi driver Implementation
# =============================================================================
def find_optimal_policy_taxi_driver():
    number_of_states = 3
    number_of_actions = 3
    discount_factor   = 0.9
    # Extract P and R from data file
    transition_f, return_f = read_mdp_file(path="data.txt", N=number_of_states, M=number_of_actions)

    # Create a MDP object for taxi driver
    mdp_obj = MDP_class(P=transition_f, R=return_f, gamma=discount_factor, N=number_of_states, M=number_of_actions)

    # Create a Policy object 
    policy_obj = policy_class(N=number_of_states, M=number_of_actions,policy_type='d')
    policy_obj.generate_policy() # creates the policy array

    print('========= Method 1 : Policy iteration =========')
    policy_iteration(mdp=mdp_obj, policy=policy_obj, evaluate_policy_func=linear_algebra, max_iterations=1000)

    print('========= Method 2 : Value iteration =========')
    value_iteration(mdp=mdp_obj, epsilon=1e-6, max_iterations=1000)

    print('========= Method 3 : Linear programming =========')
    print(linear_programming(mdp=mdp_obj))

    print('========= Influence of y =========')
    changes = gamma_sensitivity(policy_iteration,linear_algebra ,mdp_obj, policy_obj)
    plot_policy_changes(changes, mdp_obj)

# =============================================================================
# 5. 21 Implementation
# =============================================================================
def find_optimal_policy_21():
    transition_f,return_f = create_21_mdp()
    # Create a MDP object for 21
    mdp_obj = MDP_class(P=transition_f, R=return_f, gamma=0.9, N=22, M=2)
    # Create a Policy object 
    policy_obj = policy_class(N=22, M=2,policy_type='d')
    policy_obj.generate_policy() # creates the policy array

    print('========= Method 1 : Policy iteration =========')
    policy_iteration(mdp=mdp_obj, policy=policy_obj, evaluate_policy_func=linear_algebra, max_iterations=1000)

    print('========= Method 2 : Value iteration =========')
    value_iteration(mdp=mdp_obj, epsilon=1e-6, max_iterations=1000)

    print('========= Method 3 : Linear programming =========')
    print(linear_programming(mdp=mdp_obj))

    print('========= Influence of y =========')
    changes = gamma_sensitivity(policy_iteration,linear_algebra ,mdp_obj, policy_obj)
    plot_policy_changes(changes, mdp_obj)

# =============================================================================
# 5. Main Function (for testing)
# =============================================================================
def main_calculation_optimal_policy():
    """
    Start the process of Implementing.
    
    Args:
        None
    
    Returns:
        None
    """
    print('Optimal policy for Taxi driver problem')
    find_optimal_policy_taxi_driver()

    print('Optimal policy for 21 problem')
    find_optimal_policy_21()

# =============================================================================
# 6. Main Execution (only when the script is executed directly)
# =============================================================================
if __name__ == '__main__':
    main_calculation_optimal_policy()

