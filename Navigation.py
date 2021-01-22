from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../Value_methods')
from agent import Agent
from paramutils import *
import pickle as pk

params=parameters()
params.network = 'dqn'  # 'dqn, dueling_dqn'
params.double_dqn = 'disable'  # enable, disable
params.scores_filename ='_v0.txt'
params.env_seed=4

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
#env.seed(params.env_seed)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

seed=0
agent = Agent(state_size, action_size ,seed,params)


params.eps_start=1.0
params.eps_end=0.01
params.eps_decay=0.995
eps = params.eps_start  # initialize epsilon
n_episodes=2000
scores=[]
scores_window=deque(maxlen=100)
for i_episode in range(1,n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score

    while True:
        action = agent.act(state,eps)
        env_info = env.step(int(action))[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)

        score += reward
        state = next_state
        if done:
            break

    scores_window.append(score)  # save most recent score
    scores.append(score)  # save most recent score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="\r")

    if i_episode % 50 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window) >= 13.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,np.mean(scores_window)))
        break

    eps = max(params.eps_end, params.eps_decay * eps)  # decrease epsilon

print("Score: {}".format(score))

env.close()
plt.plot(scores)


