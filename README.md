# Maze_Solver_Q_learning_ML_Project


# Maze Solver using Reinforcement Learning (Q-Learning)

## Project Overview

This project demonstrates the implementation of a Maze Solver using the Q-Learning algorithm, a popular Reinforcement Learning technique. The agent learns to navigate through a maze environment by interacting with it, receiving rewards and penalties, and updating its knowledge through a Q-Table. The objective of the agent is to find the optimal path from the start position to the goal position while avoiding obstacles.

---

## Objective

The primary objective of this project is to understand the fundamentals of Reinforcement Learning by implementing a Q-Learning based agent that learns to solve a maze through trial and error.

---

## Problem Statement

Given a maze environment containing:

* Empty cells
* Obstacles
* Start position
* Goal position

The agent must learn the shortest and most rewarding path to reach the goal using Reinforcement Learning.

---

## Reinforcement Learning Concepts Used

### State

A state represents the current position of the agent inside the maze.

### Action

The agent can perform four actions:

* Up
* Down
* Left
* Right

### Reward

The environment provides feedback to the agent:

| Action Result | Reward |
| ------------- | ------ |
| Normal Move   | -1     |
| Reach Goal    | +100   |

### Policy

The strategy used by the agent to choose actions.

### Q-Table

The Q-Table stores the expected future reward for each state-action pair.

---

## Maze Environment

### Maze Size

4 × 4 Grid

### Maze Representation

```python
maze = [
 [0, 0, 0, 0],
 [0,-1, 0,-1],
 [0, 0, 0,-1],
 [0,-1, 0,100]
]
```

### Legend

| Value | Meaning    |
| ----- | ---------- |
| 0     | Empty Cell |
| -1    | Obstacle   |
| 100   | Goal State |

---

## Algorithm Used

### Q-Learning

Q-Learning is a model-free Reinforcement Learning algorithm that learns the value of taking a particular action in a given state.

### Q-Learning Formula

Q(s,a) = Q(s,a) + α [ r + γ max(Q(s')) - Q(s,a) ]

Where:

* Q(s,a) = Current Q-Value
* α = Learning Rate
* r = Immediate Reward
* γ = Discount Factor
* max(Q(s')) = Best Future Reward

---

## Project Configuration

### Hyperparameters

| Parameter           | Value |
| ------------------- | ----- |
| Learning Rate (α)   | 0.1   |
| Discount Factor (γ) | 0.9   |
| Episodes            | 10    |
| Number of Actions   | 4     |

---

## State Space

Since the maze size is:

4 × 4

Total states:

16

State indexing:

| Position | State |
| -------- | ----- |
| (0,0)    | 0     |
| (1,0)    | 4     |
| (3,3)    | 15    |

---

## Action Space

| Action Index | Action |
| ------------ | ------ |
| 0            | Up     |
| 1            | Down   |
| 2            | Left   |
| 3            | Right  |

---

## Q-Table Structure

The Q-Table dimensions are:

16 × 4

Explanation:

* 16 states
* 4 possible actions

Total Q-Values:

64

---

## Training Process

### Step 1

Initialize the Q-Table with zeros.

### Step 2

Start the agent from the initial state.

### Step 3

Select an action.

### Step 4

Move to the next state.

### Step 5

Receive reward from the environment.

### Step 6

Update the Q-Value using the Q-Learning formula.

### Step 7

Repeat until the goal is reached.

### Step 8

Continue for multiple episodes.

---

## Sample Output

### Initial Q-Table

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 ...
 [0. 0. 0. 0.]]
```

### Reward Testing

```text
Normal Move = -1

Goal State = 100
```

### Training Progress

```text
Episode 1 Completed
Episode 2 Completed
Episode 3 Completed
...
Episode 10 Completed
```

---

## Final Q-Table Observations

Some important learned Q-Values:

| State    | Best Q-Value |
| -------- | ------------ |
| State 6  | 13.70        |
| State 10 | 31.39        |
| State 14 | 65.13        |

### Interpretation

The increasing Q-Values indicate that the agent has learned which states are closer to the goal.

The highest Q-Value was observed near the goal state because reaching the goal provides the highest reward.

---

## Learning Analysis

### Episode 1

Most Q-Values were near zero because the agent had little experience.

### Episode 5

Positive Q-Values started appearing, indicating that the agent discovered rewarding paths.

### Episode 10

The Q-Values significantly increased for states leading toward the goal, demonstrating successful learning.

---

## Results

The agent successfully learned:

* Valid movements
* Obstacle avoidance
* Reward maximization
* Goal-directed behavior
* Optimal path estimation

The Q-Table showed clear evidence that the agent was learning from experience and improving its decision-making over time.

---

## Skills Demonstrated

* Python Programming
* Reinforcement Learning
* Q-Learning Algorithm
* State Space Modeling
* Reward Engineering
* NumPy
* Decision Making Systems
* Artificial Intelligence Fundamentals

---

## Future Improvements

* Increase training episodes to 1000+
* Implement epsilon-greedy exploration
* Visualize agent movement using Pygame
* Create a web-based simulator
* Support dynamic maze generation
* Add larger maze environments
* Implement Deep Q-Networks (DQN)

---

## Conclusion

This project successfully demonstrates the implementation of a Reinforcement Learning agent using the Q-Learning algorithm. Through repeated interaction with the environment, the agent learned to maximize rewards and identify the optimal route to the goal. The project provides a strong foundation for understanding Reinforcement Learning concepts and serves as a stepping stone toward more advanced AI applications.
