# Unity-ML Banana collector using DQN

This repo contains a working code for solving the Unity ML Agent called Banana Collector using Deel Q-Network and its variants.

## The Task
The agent must collect Yellow bananas in a field full of yellow and blue bananas. The agent is rewarded a score of 1 for collecting a yellow banana and penalized a score of 1 for collecting a blue banana. The agent must learn to score 13 over 100 episodes, for the task to be complete.

## The Environment and the Agent
The environment is defined by a state vector composed of 37 variables. In response to the observed state, the agent can take an action to  move in four direction in 2-Dimensions space. 
What do the element of the state indicate? No clear answers from Unity. 
The good folks [here](https://github.com/Unity-Technologies/ml-agents/issues/1134) have figured some of it out (and have been kind enough to share, bless their hearts!). It seems that the agent looks 'around' in a visual field spanning 20 to 135 degrees in approximately 35 degree increment for total of 7 'sight rays'. If the 'sight' (a ray transmitted at one of these angles) falls on an object (Yellow Banana, Blue Banana, or a wall), then 5 attributes of the object are recorded: Binary identified for the object (YB,BB,Wall), agent, and distance to the object. I don't understand what the fourth attribute is. 
This makes for 35 variables. The velocity in left-right and forward-backward direction adds two additional state variables for a total of 37 variables. Does it matter  what the state variables mean? Not really. Not at least to get a working solution. Read the discussion towards the end.

## The Solution: Q learning via a Deep Neural Network
The primary solution is a plain vanilla DQN network, composed of three fully connected layers as follows:
|Layer|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|
|1|37|64| ReLu|
|2 |64|64|ReLu|
|3|64|4|Linear|

The network learns by minimising the MSE between the true Q value of a state and its current estimate. 
Apart from plain Vanilla, the following were tried.
1. Double DQN
2. Dueling DQN with the following architecture

|Layer|Type|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|--|
|1|Shared|37|128| ReLu|
|2 |Value n/w Layer 1|128|128|ReLu|
|3|Value n/w Layer 2|128|1|Linear|
|3|Advantage n/w Layer 1|128|128|Relu|
|3|Advantage n/w Layer 2|128|4|Linear|

3. Double Dueling DQN 

## Performance
Running for 2000 episodes, we get this:

1. DQN hits the target score of 13, the earliest, at around 400 episodes with double DQN close on its heels
2. Dueling DQN hits the passing score around the same number of episodes but then stumbles on without any score increment.
3. Double Dueling DQN stumbles early on and hits the score of 13 at ~650 episodes

So, what's going on? Some possible ad-hoc explanations:
1.  This is a single seed of environment, neural network and agents epsilon greedy actor.
2. The curves are tightly coupled. The models do not demonstrate a clear performance improvement over the others; all are within the variance due to seeds
3. There is correlation between the two non-double models   as well as between the double models. 

## Conclusion
Double DQN or Dueling DQN do not show any noticeable performance. In this one instance of the environment, the DQN is the best. However, since the results are very close to draw any conclusions on superiority. 
1. Run the models over a sweep of seeds of the environment
2. Try a larger networks

### State variable
The state vector might have redundancy, that may be exploited via a a lower dimension transformation of the state vector covariance matrix. The lower dimension would constitute an efficient representation of the data. But is it worth? Can you learn with fewer data? 
One variation of this project, along the line of the Atari game challenges to learn directly from the visual frames. In that case, I foresee a layer or two of  Convolutional Neural network followed by a layer or two of fully connected network. 
