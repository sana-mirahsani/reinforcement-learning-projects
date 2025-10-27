# =============================================================================
# This file creates a MDP problem and stochastic/deterministic policy and a trajectory
# =============================================================================

# S  = number of states, from 0 to N-1
# A  = number of actions, from 0 to M-1 
# P  = transition function, 3_dimensional N * M * N
# R  = return function, 3_dimensional N * M * N
# y  = the discount factor

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np

# =============================================================================
# 1. MDP CLASS
# =============================================================================
class MDP_class:
    def __init__(self, P, R, gamma, N, M):
        """
        Initialize a Markov Decision Problem (MDP)

        Args:
            P : np.ndarray
                Transition function with shape (N, M, N)
            R : np.ndarray
                Reward function with shape (N, M, N)
            gamma : float
                Discount factor in [0, 1]
            N : int
                Number of states
            M : int
                Number of actions
        Returns:
            mdp object
        """
        self.P = P
        self.R = R
        self.gamma = gamma
        self.N = N
        self.M = M

        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")

        # Create sets of states and actions
        self.S = self._create_states()
        self.A = self._create_actions()
        
    # ---------- Create states -------------
    def _create_states(self):
        """Create the set of states S = {e0, e1, ..., e(N-1)}"""
        return [f"e{i}" for i in range(self.N)]

    # ---------- Create actions -------------
    def _create_actions(self):
        """Create the set of actions A = {a0, a1, ..., a(M-1)}"""
        return [f"a{j}" for j in range(self.M)]

    # ---------- Summary of created MDP -------------
    def summary(self):
        """Print a summary of the MDP structure"""
        print("=== Markov Decision Process ===")
        print(f"States (N={self.N}): {self.S}")
        print(f"Actions (M={self.M}): {self.A}")
        print(f"Discount factor γ = {self.gamma}")
        print(f"P shape: {self.P.shape}")
        print(f"R shape: {self.R.shape}")

# =============================================================================
# 2. Policy class
# =============================================================================
class policy_class:
    def __init__(self, N, M, policy_type):
        """
        Initialize a policy (stochastic/deterministic)

        Args:
            N : int
                Number of states
            M : int
                Number of actions
            policy_type : str (deterministic or stochastic)
        Returns:
            Policy object
        """
        self.number_states  = N
        self.number_actions = M
        self.policy_type = policy_type
        self.policy_array = None

    def generate_policy(self):
        """
        Generating a stochastic or deterministic policy depends on policy_type's value.
        
        Returns: 
            A random numpy array of a policy
        """
        if self.policy_type == "d": # deterministic
            
            self.policy_array = np.random.randint(0, self.number_actions, size=self.number_states)
        
        elif self.policy_type == "s": # stochastic
            self.policy_array = np.random.rand(self.number_states, self.number_actions)
            self.policy_array = self.policy_array / self.policy_array.sum(axis=1, keepdims=True) # Normalization
            
        else: # invalide input
            raise ValueError("Policy type must be 'deterministic' or 'stochastic'.")

# =============================================================================
# 3. Trajectory Function
# =============================================================================
def trajectory(mdp, policy, start_state=0, threshold=1e-3, max_steps=100):
    """
    Generate a trajectory from an MDP and compute the return (G) value.
    
    Args:
        mdp : MDP object (must contain P, R, gamma)
        policy : Policy object
        start_state : initial state index
        threshold : stopping condition based on gamma^k
        max_steps : maximum number of transitions
    
    Returns:
        trajectory : list of (state, action, reward, next_state)
        G : total discounted return (zeta)
    """
    s = start_state
    gamma = mdp.gamma
    G = 0
    traj = []
    discount = 1.0
    
    for _ in range(max_steps):
        # choose action according to the policy
        if policy.policy_type == "d":  # deterministic
            a = policy.policy_array[s]
        else:  # stochastic
            a = np.random.choice(range(mdp.M), p=policy.policy_array[s])
        
        # sample next state
        next_s = np.random.choice(range(mdp.N), p=mdp.P[s, a])
        r = mdp.R[s, a, next_s]
        
        # accumulate discounted reward
        G += discount * r
        traj.append((s, a, r, next_s))
        
        # update state and discount
        s = next_s
        discount *= gamma
        
        # stop condition
        if discount < threshold:
            break
    
    return traj, G

# =============================================================================
# 4. File reading Function
# =============================================================================
def read_mdp_file(path="data.txt", N=1, M=1):
    """
    Reads an MDP file with transition and reward data.
    
    Args:
        path (str): path to text file.
        N (int): number of states.
        M (int): number of actions.
    
    Returns:
        P (np.ndarray): transition matrix of shape (N, M, N)
        R (np.ndarray): reward matrix of shape (N, M, N)
    """
    with open(path, 'r') as f:
        content = f.read().strip().split('\n\n')  # split at blank line

    # Transition function
    P_lines = content[0].strip().split('\n')
    P_values = [list(map(float, line.split())) for line in P_lines]
    P = np.array(P_values).reshape(M, N, N)  # shape (M, N, N)
    P = np.transpose(P, (1, 0, 2))  # shape (N, M, N) => (state, action, next_state)

    # Normalize rows so that sum = 1
    for s in range(N):
        for a in range(M):
            P[s, a, :] = normalize_transition_matrix(P[s, a, :].reshape(1, -1))[0]

    # Reward function
    R_lines = content[1].strip().split('\n')
    R_values = [list(map(float, line.split())) for line in R_lines]
    R = np.array(R_values).reshape(M, N, N)
    R = np.transpose(R, (1, 0, 2))  # shape (N, M, N)

    return P, R

# check if all probabilities' sum is equal to 1
def normalize_transition_matrix(P, tol=1e-8):
    """
    Ensure each row of P sums to 1 by adjusting the last element.
    
    Args:
        P: Transition function.
        tol: tolerance.
    
    Returns:
        P : normalized P
    """
    for row in P:
        total = np.sum(row)
        if abs(total - 1.0) > tol:
            row[-1] = max(0.0, min(1.0, 1.0 - np.sum(row[:-1])))
    return P

# =============================================================================
# 5. Main Function (for testing)
# =============================================================================
def main_mdp_solver():

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

    # Create a trajectory
    start_state = int(input("Enter an integer as the start state : "))
    traj, G_value = trajectory(mdp=mdp_obj, policy=policy_obj , start_state=start_state, threshold=1e-3, max_steps=100)
    
    # Print out
    print("\n")
    mdp_obj.summary()
    print(f"=== Your {policy_obj.policy_type} Policy ===")
    print(policy_obj.policy_array)

    print(f"=== Your trajectory [s, a, r, next_s] ===")
    print(traj)

    print(f"=== Your Zheta value ===")
    print(G_value)

# =============================================================================
# 6. Main Execution (only when the script is executed directly)
# =============================================================================
if __name__ == '__main__':
    main_mdp_solver()