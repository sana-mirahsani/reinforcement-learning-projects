# ==========================================================
# This file evaluates a policy by three methods : 
# 1.Linear Algebra 2.Bellman Iteration 3.Monte Carlo Value
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
from mdp_solver import MDP_class, policy_class, trajectory, read_mdp_file

# =============================================================================
# 1. Evaluations functions (all three)
# =============================================================================
# Method one : linear algebra
def linear_algebra(mdp,policy):
    """
    Calculate the V value of a policy by linear algebra.
    
    Args:
        mdp : obj
            An object from MDP_class
        policy : obj
            An bject from Policy_class
    
    Returns:
        V : 1D vector of size N (number of states).
    """
    # Build P^π and R^π
    P_pi = np.zeros((mdp.N, mdp.N))
    R_pi = np.zeros(mdp.N)
    
    if policy.policy_type == 'd':  # deterministic
        for s in range(mdp.N):
            a = policy.policy_array[s]
            P_pi[s] = mdp.P[s, a]
            R_pi[s] = np.sum(mdp.P[s, a] * mdp.R[s, a])
       
    else:  # stochastic
        for s in range(mdp.N):
            R_pi[s] = 0
            for a in range(mdp.M):
                P_pi[s] += policy.policy_array[s, a] * mdp.P[s, a]
                R_pi[s] += policy.policy_array[s, a] * np.sum(mdp.P[s, a] * mdp.R[s, a])

    # Solve (I - γPπ)V = Rπ
    I = np.eye(mdp.N)
    V = np.linalg.solve(I - mdp.gamma * P_pi, R_pi)

    return V

# Method two : Bellman iteration
def bellman_iteration(mdp, policy, tol=1e-6, max_iter=10000):
    """
    Calculate the V value of a policy by Bellman iteration.
    
    Args:
        mdp : obj
            An object from MDP_class
        policy : obj
            An bject from Policy_class
    
    Returns:
        V : 1D vector of size N(number of states).
        n_iter : int Number of iterations until convergence
    """
    V = np.zeros(mdp.N)  # initial guess for V(s)

    for iteration in range(max_iter):
        V_new = np.zeros(mdp.N)

        for s in range(mdp.N):
            if policy.policy_type == 'd':  # deterministic policy
                a = policy.policy_array[s]
                if mdp.R.ndim == 3: # if the reward function is based on the outcome; like R(s,a,s')
                    expected_reward = np.sum(mdp.P[s, a] * mdp.R[s, a])
                else:
                    expected_reward = mdp.R[s, a] # if the reward function is immediate; like R(s,a)
                V_new[s] = expected_reward + mdp.gamma * np.sum(mdp.P[s, a] * V)
            else:  # stochastic policy
                for a in range(mdp.M):
                    if mdp.R.ndim == 3: # if the reward function is based on the outcome; like R(s,a,s')
                        expected_reward = np.sum(mdp.P[s, a] * mdp.R[s, a])
                    else:
                        expected_reward = mdp.R[s, a] # if the reward function is immediate; like R(s,a)
                    V_new[s] += policy.policy_array[s, a] * (
                        expected_reward + mdp.gamma * np.sum(mdp.P[s, a] * V)
                    )

        # Check convergence
        diff = np.max(np.abs(V_new - V))
        if diff < tol:
            print(f"Converged after {iteration+1} iterations with diff={diff:.2e}")
            return V_new, iteration + 1

        V = V_new

    print("Warning: did not converge within max_iter")
    return V, max_iter

# Method third : Monte Carlo Value
def monte_carlo_value(mdp, policy, start_state, n_trajectories=1000, threshold=1e-3, max_steps=100):
    """
    Estimate the value function for a given start state using Monte Carlo method.

    Args:
        mdp : obj
            An object from MDP_class.
        policy : obj
            An bject from Policy_class
        start_state : int
            The state from which all trajectories start.
        n_trajectories : int
            Number of trajectories to generate.
        threshold : float
            Stop condition for gamma^k (passed to trajectory()).
        max_steps : int
            Maximum steps per trajectory.

    Returns:
        V_est : float
            Estimated value of the start state (average return).
        all_returns : list
            List of all G (zeta) values from trajectories.
    """
    
    all_return_vlaues = []
    for i in range(n_trajectories):
        traj, G = trajectory(mdp, policy, start_state, threshold=threshold, max_steps=max_steps)
        all_return_vlaues.append(G)

    V_est = np.mean(all_return_vlaues)
    return V_est, all_return_vlaues

# =============================================================================
# 2. Main Function (for testing)
# =============================================================================
def main_policy_evaluation():
    """
    Start the process of evaluating a policy by three methods.
    
    Args:
        None
    
    Returns:
        None
    """
    # Get the inputs
    file_path         = str(input("Enter the path of data file :"))
    number_of_states  = int(input("Enter number of states :"))
    number_of_actions = int(input("Enter number of actions :"))
    discount_factor   = float(input("Enter the discount factor :"))

    # Extract P and R from data file
    transition_f, return_f = read_mdp_file(path=file_path, N=number_of_states, M=number_of_actions)

    # Create a MDP object
    mdp_obj = MDP_class(P=transition_f, R=return_f, gamma=discount_factor , N=number_of_states, M=number_of_actions)
    
    # Create a Policy object
    type_of_policy = str(input('Write d for deterministic or s for stochastic : '))
    policy_obj = policy_class(N=number_of_states, M=number_of_actions,policy_type=type_of_policy)
    policy_obj.generate_policy() # creates the policy array
    
    # Evaluate by first method :  
    print(f"=== Value function by linear algebra ===")
    V_linear_algebra = linear_algebra(mdp_obj, policy_obj)
    print(V_linear_algebra)

    # Evaluate by second method : 
    print(f"=== Value function by bellman iteration ===")
    V_bellman_iteration, num_iterations = bellman_iteration(mdp_obj, policy_obj, tol=1e-6, max_iter=10000)
    print(V_bellman_iteration)

    # Evaluate by third method : 
    print(f"=== Value function by Monte Carlo ===")
    start_state = int(input("Enter an integer for start state: "))
    V_mc, all_G = monte_carlo_value(mdp_obj, policy_obj, start_state, n_trajectories=500)
    print(f"Estimated value of state {start_state}: {V_mc:.4f}")
    print(f"Average of {len(all_G)} trajectories")

# =============================================================================
# 3. Main Execution (only when the script is executed directly)
# =============================================================================
if __name__ == '__main__':
    main_policy_evaluation()