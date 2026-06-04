import numpy as np
import random

# =========================
# STEP 1: CREATE MAZE
# =========================

maze = np.array([
    [0, 0, 0, 0],
    [0,-1, 0,-1],
    [0, 0, 0,-1],
    [0,-1, 0,100]
])

# =========================
# STEP 2: DEFINE STATES
# =========================

num_states = 16

# =========================
# STEP 3: DEFINE ACTIONS
# =========================

actions = ['Up', 'Down', 'Left', 'Right']
num_actions = 4

# =========================
# STEP 4: CREATE Q TABLE
# =========================

q_table = np.zeros((num_states, num_actions))

print("Initial Q Table:")
print(q_table)

# =========================
# STEP 5: LEARNING PARAMETERS
# =========================

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0

episodes = 10   # Keep 10 for testing

# =========================
# STEP 6: STATE CONVERSION
# =========================

def state_to_index(row, col):
    return row * 4 + col

print("\nState Index Examples:")
print(state_to_index(0,0))
print(state_to_index(1,0))
print(state_to_index(3,3))

# =========================
# STEP 7: MOVEMENT FUNCTION
# =========================

def get_next_position(row, col, action):

    if action == 0:
        row -= 1

    elif action == 1:
        row += 1

    elif action == 2:
        col -= 1

    elif action == 3:
        col += 1

    return row, col

print("\nMovement Examples:")
print(get_next_position(0,0,3))
print(get_next_position(0,0,1))

# =========================
# STEP 8: VALID POSITION CHECK
# =========================

def is_valid_position(row, col):

    if row < 0 or row >= 4:
        return False

    if col < 0 or col >= 4:
        return False

    if maze[row][col] == -1:
        return False

    return True

print("\nValid Position Tests:")
print(is_valid_position(0,0))
print(is_valid_position(1,1))
print(is_valid_position(-1,0))

# =========================
# STEP 9: REWARD FUNCTION
# =========================

def get_reward(row, col):

    if maze[row][col] == 100:
        return 100

    return -1

print("\nReward Tests:")
print(get_reward(0,0))
print(get_reward(3,3))

# =========================
# STEP 10: TRAINING LOOP
# =========================

for episode in range(episodes):

    row = 0
    col = 0

    while (row, col) != (3,3):
        # Current State
        state = state_to_index(row, col)
        action = random.randint(0,3)

        new_row, new_col = get_next_position(
            row,
            col,
            action
        )

        if not is_valid_position(new_row, new_col):
            continue

        reward = get_reward(
            new_row,
            new_col
        )
        # Next State
        next_state = state_to_index(
            new_row,
            new_col
        )
        # IMPORTANT
        # Update agent position
        row = new_row
        col = new_col
        # Q-Learning Formula
        old_q = q_table[state, action]

        max_future_q = np.max(
            q_table[next_state]
        )

        new_q = old_q + learning_rate * (
            reward +
            discount_factor * max_future_q -
            old_q
        )

        q_table[state, action] = new_q

        # Move Agent
        row = new_row
        col = new_col

    print("\nTraining Completed!")
    print(f"Episode {episode+1} completed")

    print("\nFinal Q Table:\n")
    print(q_table)
print("\nTraining Loop Working Successfully!")



"""
Output:
Initial Q Table:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

State Index Examples:
0
4
15

Movement Examples:
(0, 1)
(1, 0)

Valid Position Tests:
True
False
False

Reward Tests:
-1
100

Training Completed!
Episode 1 completed

Final Q Table:

[[ 0.   0.   0.  -0.1]
 [ 0.   0.   0.  -0.1]
 [ 0.  -0.1  0.  -0.1]
 [ 0.   0.  -0.1  0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.  -0.1  0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.  -0.1  0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.   0. ]
 [ 0.   0.   0.  10. ]
 [ 0.   0.   0.   0. ]]

Training Completed!
Episode 2 completed

Final Q Table:

[[ 0.        -0.468559   0.        -0.3439   ]
 [ 0.         0.        -0.19      -0.3439   ]
 [ 0.        -0.3439    -0.19      -0.271    ]
 [ 0.         0.        -0.271      0.       ]
 [-0.468559  -0.5217031  0.         0.       ]
 [ 0.         0.         0.         0.       ]
 [-0.19      -0.271      0.         0.       ]
 [ 0.         0.         0.         0.       ]
 [-0.5217031 -0.271      0.        -0.40951  ]
 [ 0.         0.        -0.40951   -0.19     ]
 [-0.1        0.71      -0.19       0.       ]
 [ 0.         0.         0.         0.       ]
 [-0.271      0.         0.         0.       ]
 [ 0.         0.         0.         0.       ]
 [ 0.         0.         0.        19.       ]
 [ 0.         0.         0.         0.       ]]

Training Completed!
Episode 3 completed

Final Q Table:

[[ 0.         -0.5217031   0.         -0.5217031 ]
 [ 0.          0.         -0.271      -0.61257951]
 [ 0.         -0.5217031  -0.40951    -0.40951   ]
 [ 0.          0.         -0.40951     0.        ]
 [-0.56953279 -0.5217031   0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.271      -0.3078559   0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.56953279 -0.271       0.         -0.468559  ]
 [ 0.          0.         -0.5217031  -0.2071    ]
 [-0.3439      2.249      -0.3439      0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.271       0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.         27.1       ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 4 completed

Final Q Table:

[[ 0.         -0.74581342  0.         -0.84990536]
 [ 0.          0.         -0.71757046 -0.87842335]
 [ 0.         -0.81469798 -0.77123208 -0.71757046]
 [ 0.          0.         -0.71757046  0.        ]
 [-0.79410887 -0.79410887  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.41461525  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.83322818 -0.56953279  0.         -0.75305666]
 [ 0.          0.         -0.81469798  0.3883391 ]
 [-0.39731638  6.26579    -0.50352768  0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.56953279  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.292679    0.          0.         34.39      ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 5 completed

Final Q Table:

[[ 0.         -0.74581342  0.         -0.89058101]
 [ 0.          0.         -0.77123208 -0.9282102 ]
 [ 0.         -0.79591281 -0.84990536 -0.77123208]
 [ 0.          0.         -0.77123208  0.        ]
 [-0.79410887 -0.79410887  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.83707482  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.83322818 -0.56953279  0.         -0.75305666]
 [ 0.          0.         -0.81469798  0.3883391 ]
 [-0.39731638  8.634311   -0.50352768  0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.56953279  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.292679    0.          0.         40.951     ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 6 completed

Final Q Table:

[[ 0.         -0.77123208  0.         -0.89058101]
 [ 0.          0.         -0.77123208 -0.9282102 ]
 [ 0.         -0.79591281 -0.84990536 -0.77123208]
 [ 0.          0.         -0.77123208  0.        ]
 [-0.79410887 -0.81469798  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.83707482  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.83322818 -0.61257951  0.         -0.74280047]
 [ 0.          0.         -0.81469798  1.02659318]
 [-0.39731638 11.3564699  -0.50352768  0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.61257951  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.292679    0.          0.         46.8559    ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 7 completed

Final Q Table:

[[ 0.         -0.86491483  0.         -0.91137062]
 [ 0.          0.         -0.81469798 -0.9282102 ]
 [ 0.         -0.79591281 -0.84990536 -0.77123208]
 [ 0.          0.         -0.77123208  0.        ]
 [-0.86491483 -0.86491483  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.83707482  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.86491483 -0.79410887  0.         -0.47599962]
 [ 0.          0.         -0.83322818  3.515554  ]
 [-0.39731638 17.02109952 -0.2158154   0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.79410887  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 1.45381795  0.          0.         52.17031   ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 8 completed

Final Q Table:

[[ 0.         -0.89058101  0.         -0.94185026]
 [ 0.          0.         -0.87842335 -0.94766524]
 [ 0.         -0.79591281 -0.89058101 -0.81469798]
 [ 0.          0.         -0.81469798  0.        ]
 [-0.87842335 -0.89058101  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.83707482  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.87842335 -0.84990536  0.         -0.2119998 ]
 [ 0.          0.         -0.83322818  4.59589755]
 [-0.39731638 19.91431747 -0.2158154   0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.84990536  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 1.45381795  0.          0.         56.953279  ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 9 completed

Final Q Table:

[[ 0.         -0.91137062  0.         -0.95289871]
 [ 0.          0.         -0.90152291 -0.95760884]
 [ 0.         -0.79591281 -0.91137062 -0.84990536]
 [ 0.          0.         -0.84990536  0.        ]
 [-0.89058101 -0.90031583  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.65132156  0.83707482  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.89058101 -0.84990536  0.          0.42417865]
 [ 0.          0.         -0.83885058  5.82859637]
 [-0.39731638 22.94868083 -0.2158154   0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.84990536  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 1.45381795  0.          0.         61.2579511 ]
 [ 0.          0.          0.          0.        ]]

Training Completed!
Episode 10 completed

Final Q Table:

[[ 0.         -0.96184796  0.         -0.96891232]
 [ 0.          0.         -0.9774716   0.43152375]
 [ 0.          4.53105175 -0.94281381 -0.90985284]
 [ 0.          0.          0.43463996  0.        ]
 [-0.95760884 -0.17841576  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.07083579 13.70249894  0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.96184796 -0.9282102   0.          3.06388993]
 [ 0.          0.         -0.36608086 12.23275649]
 [ 2.77047104 31.3994026   2.6943971   0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.23631497  0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 5.69764054  0.          0.         65.13215599]
 [ 0.          0.          0.          0.        ]]

Training Loop Working Successfully!
"""
