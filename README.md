# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Design and implement a reinforcement learning agent using the Q-Learning algorithm to solve the FrozenLake-v1 environment provided by Gymnasium. The agent must learn the optimal action-value function through repeated interaction with the environment and use an epsilon-greedy strategy to balance exploration and exploitation.The learned Q-table should be used to determine the best action for each state and obtain a policy that guides the agent from the starting state to the goal while avoiding holes.


## Software Requirements
```
pip install gymnasium numpy
```

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

1.Create the FrozenLake-v1 environment.

2.Obtain the number of states and actions.

3.Initialize the Q-table with zeros.

4.Set the learning rate $\alpha$, discount factor $\gamma$, initial epsilon, minimum epsilon, and epsilon decay.

5.For each training episode:
   1.Reset the environment and obtain the initial state.
   2.Select an action using the epsilon-greedy strategy.
   3.Execute the selected action.
   4.Observe the next state and reward.
   5.Update the Q-value using the Q-Learning update equation.
   6.Continue until the episode terminates.
   7.Reduce epsilon.

6.Calculate the state-value function using:

$$
V(s) = \max_a Q(s,a)
$$

Obtain the learned policy using:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

7.Display the final Q-table, value function, learned policy, and average reward.

8.Plot the learning curve.

## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
# Write your code here

for episode in range(num_episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning update
        Q[state, action] = Q[state, action] + learning_rate * (
            reward
            + discount_factor * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


```
---

## Output

```text
Final Q-table:
[[0.735 0.774 0.698 0.735]
 [0.735 0.    0.565 0.555]
 [0.68  0.    0.    0.   ]
 [0.463 0.    0.    0.   ]
 [0.774 0.815 0.    0.735]
 [0.    0.    0.    0.   ]
 [0.    0.902 0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.815 0.    0.857 0.774]
 [0.815 0.902 0.902 0.   ]
 [0.857 0.95  0.    0.857]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.902 0.95  0.857]
 [0.902 0.95  1.    0.902]
 [0.    0.    0.    0.   ]]




Estimated State-Value Function:
[[0.774 0.735 0.68  0.463]
 [0.815 0.    0.902 0.   ]
 [0.857 0.902 0.95  0.   ]
 [0.    0.95  1.    0.   ]]





Learned Policy:
[['D' 'L' 'L' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'D' 'D' 'L']
 ['L' 'R' 'R' 'L']]



Average reward over last 1000 episodes: 0.988
```


<img width="934" height="542" alt="image" src="https://github.com/user-attachments/assets/7121625b-2f7a-4f0b-b691-236b52df7318" />


---

## Result

```text
The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.Through repeated interaction with the environment, the agent learned an action-value function and obtained a policy for navigating from the starting state toward the goal while avoiding holes.


```

---

## Inference

```text
The experiment demonstrates that Q-Learning can learn an effective policy without being explicitly provided with the correct path.Initially, the agent performs considerable exploration because the value of $\epsilon$ is high. As training progresses, epsilon decreases and the agent increasingly selects actions based on the learned Q-values.The final Q-table represents the learned action values for each state-action pair. The state-value function is obtained by selecting the maximum Q-value for each state, while the learned policy selects the action having the maximum Q-value.A high average reward over the final 1000 episodes indicates that the agent has learned a successful route to the goal.


```

---

