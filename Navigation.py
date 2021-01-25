from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, '../Value_methods')
from paramutils import *
from agent import Agent

import pickle as pk

#Set up the parameters
params=parameters()
params.network = 'dqn'  # 'dqn, dueling_dqn'
params.double_dqn = 'disable'  # enable, disable
params.buffer = 'baseline'  # baseline (other not supported yet)
params.op_filename_prefix ='dummy'
params.eps_start=1.0
params.eps_end=0.01
params.eps_decay=0.995
params.n_episodes=2000


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe",no_graphics=True)

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

seed=0 #Seed for epsilon greedy as well as the NN
agent = Agent(state_size, action_size ,seed,params)



eps = params.eps_start  # initialize epsilon
n_episodes=params.n_episodes
scores=[]
scores_window=deque(maxlen=100) #Window for moving average.

for i_episode in range(1,n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score

    while True:
        action = agent.act(state,eps) #Call the Agent to act on the current state
        env_info = env.step(int(action))[brain_name] #Envrionment response to action.
        next_state = env_info.vector_observations[0] #Next state transition
        reward = env_info.rewards[0] #Reward earned when state->next_state
        done = env_info.local_done[0] #Env signals if episode is done (boolean)
        agent.step(state, action, reward, next_state, done) #Agent updates learning rule based on current state,action,rewards, next_state

        score += reward #Accumulate reward for this state
        state = next_state #Move to next state.
        if done: #If env is done with the current episode, then exit the episode.
            break

    scores_window.append(score)  # save most recent score
    scores.append(score)  # save most recent score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="\r")

    if i_episode % 50 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window) >= 13.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), params.op_filename_prefix+'_checkpoint.pth')
        break

    eps = max(params.eps_end, params.eps_decay * eps)  # decrease epsilon

print("Score: {}".format(score))
env.close()
pk.dump([scores, params],open(params.op_filename_prefix+'.pk','wb'))

