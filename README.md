# Dynamic programming for solving a Markov decision problem

## Preliminaries:
### mdp_solver.py

**This project implements a Markov Decision Process (MDP) framework in Python. It allows you to:**

- Define an MDP with transition and reward matrices.
- Generate deterministic or stochastic policies.
- Simulate trajectories using a given policy.
- Evaluate the total discounted return (G value) of a trajectory.
- Read MDP data from a text file for generalization.
- The code is modular, with separate classes and functions for the MDP, policy, and trajectory generation.
<br>

**File Structure:**

- MDP_class : Represents the MDP with states, actions, transition function, reward function, and discount factor.

- policy_class : Generates deterministic or stochastic policies.

- trajectory : Simulates a trajectory and computes the total discounted reward.

- read_mdp_file : Reads transition and reward data from a text file.

- normalize_transition_matrix : Ensures transition probabilities sum to 1.

- main_mdp_solver : Main function to run the workflow with user inputs.

<br>

**Features**

1. MDP Class

    - States (S): {e0, e1, ..., e(N-1)}

    - Actions (A): {a0, a1, ..., a(M-1)}

    - Transition Function (P): Shape (N, M, N)

    - Reward Function (R): Shape (N, M, N)

    - Discount Factor (γ): Must be in [0,1]

2. Policy Class

    - Supports deterministic or stochastic policies.

    - Deterministic: Each state maps to a single action.

    - Stochastic: Each state maps to a probability distribution over actions.

3. Trajectory Generation

    - Simulates a trajectory starting from a given state.

    - Chooses actions according to the policy.

    - Samples next states using the transition function.

    - Accumulates discounted rewards.

4. Reading MDP from File

    - Reads a text file containing:

    - Transition probabilities

    - Reward values

    - File format:

        - First block: transition probabilities (shape N x M x N)

        - Second block: reward values (shape N x M x N)

5. Main Workflow

    - Load transition and reward matrices from a file.

    - Create an MDP_class object.

    - Generate a policy (deterministic or stochastic).

    - Generate a trajectory from a start state.

    - Compute and print the total discounted reward.

### policy_evaluation.py

- Overview

    - This project implements policy evaluation for a given MDP using three different methods:

    - Linear Algebra – exact evaluation using matrix operations.

    - Bellman Iteration – iterative solution of the Bellman equations until convergence.

    - Monte Carlo Value Estimation – approximate evaluation using sampled trajectories.

    - It builds on the MDP and policy classes from mdp_solver.py and supports both deterministic and stochastic policies.


- File Structure

    - linear_algebra – Compute the value function 
    V exactly by solving 
    (
    𝐼
    −
    𝛾
    𝑃
    𝜋
    )
    𝑉
    =
    𝑅
    𝜋
    (I−γP
    π
    )V=R
    π
    .

    - bellman_iteration – Iteratively update the value function using the Bellman equation until convergence.

    - monte_carlo_value – Generate multiple trajectories from a start state and estimate the expected return.

- main_policy_evaluation – Main workflow that takes user input and evaluates the policy using all three methods.

**Features**

1. Linear Algebra

    Uses matrix inversion to solve for the exact value function of a given policy.

    Supports deterministic and stochastic policies.

    Returns a 1D vector of state values.

2. Bellman Iteration

    Iteratively updates 
    V(s) until convergence.

    Uses a tolerance parameter tol to stop iterations.

    Returns:

    V_new: Value function after convergence

    n_iter: Number of iterations

3. Monte Carlo Value

    Generates n_trajectories from a specified start state.

    Computes the average discounted return.

    Useful for stochastic simulations or when P and R are unknown.

**Main Workflow**

    Read MDP data from a text file (read_mdp_file) with transition and reward matrices.

    Create MDP object using MDP_class.

    Generate a policy using policy_class (deterministic or stochastic).

    Evaluate the policy using all three methods:

    Linear Algebra

    Bellman Iteration

    Monte Carlo

    Print results for comparison.

## policy evaluation
### implementation.py

Simply just run the script it shows you the plotting result for taxi driver problem with all deterministic policies
and also the 21 dice problem.

## Calculating an optimal policy
### calculating_optimal_policy.py
Run this script and it shows you the results of all three method calculation the optimal policy for both taxi driver and
21 dice roll problem, all with the influence of y for both problem with plotting the result.
