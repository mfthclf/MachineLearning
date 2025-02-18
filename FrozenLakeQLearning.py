import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q Table: ")
print(qtable)

action = environment.action_space.sample()
# left -> 0, down -> 1, right -> 2, up -> 3

new_state, reward, done, info,  _ = environment.step(action)

# %%
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q Table: ")
print(qtable)


episodes = 1000
alpha = 0.5 # learning rate
gamma = 0.9 # discount factor

outcomes = []

# training
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False
    outcomes.append("Failure")
    
    while not done:
        
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()

        new_state, reward, done, info,  _ = environment.step(action)
        
        # update q table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state
        
        if reward:
            outcomes[-1] = "Success"

print("Q Table after training: ")
print(qtable)

plt.bar(range(episodes), outcomes)


# test

episodes = 100
nb_success = 0

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False
    
    while not done:
        
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        
        new_state, reward, done, info,  _ = environment.step(action)
        
        state = new_state
        
        nb_success += reward

print("Success Rate: ", 100 * nb_success / episodes)        
