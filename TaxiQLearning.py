import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())

"""
Actions:
    0: Move South
    1: Move North
    2: Move East
    3: Move West
    4: Pickup Passanger
    5: Drop off Passanger
"""

action_space = env.action_space.n
state_space = env.observation_space.n

qtable = np.zeros((state_space, action_space))

alpha = 0.1 # learning rate
gamma = 0.6 # discount rate
epsilon = 0.1

for i in tqdm(range(1, 100001)):
    
    state, _ = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon: # explore %10
            action = env.action_space.sample()
        else: # exploit
            action = np.argmax(qtable[state])
        
        next_state, reward, done, info, _ = env.step(action) 
        
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state]) - qtable[state, action])
        
        state = next_state
        
print("Training finished!")   


# test

total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(100)):
    
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
       
        action = np.argmax(qtable[state])
        
        next_state, reward, done, info, _ = env.step(action) 
        
        state = next_state
        
        if reward == -10:
            penalties += 1
        
        epochs += 1
        
    total_epoch += epochs
    total_penalties += penalties
        
print("Result after {} episodes".format(episodes))
print("Average timesteps per episode: ", total_epoch/episodes)
print("Average penalties per episode: ", total_penalties/episodes)