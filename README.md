
# Unity-ML Banana Collector agent
The banana collection environment from Unitiy-ML is an emulation of an agent tasked with collecitng yellow bananas in a two dimensional enclosed space. The agent 'sees' the objects around it and can move in four directions. When it spots a yellow banana, it must take actions to move towards the banana. No action is needed to grab the banana. It must avoid any blue bananas. The emulation looks like this:  
![Unity-ML Banana Collector Agent in action](https://github.com/kpasad/DQN_navigation/blob/main/adjunct/trained-agent.gif)

See Report.md for detailed description

## Installation

* Navigation.py is the landing script
* Clone [this](https://github.com/kpasad/Value_methods ) repo
* The folder Value_methods contains :
	1. The agent (agent.py) that implements the RL agent functionalities (epsilon greedy action, learning  
	2. Deep Q network models : DQN(dqn_model.py ) and Dueling DQN (ddqn_model.py)
	3. Simulator helper (paramsutility.py) ; this is work in progress for a parameters manager utiity
* Folder,Value_methods, must be included in Navigation.py. Ensure that, in  
		     `sys.path.insert(0, '../Value_methods')` , the path to Vaue_methods is correctly set.
* File plotres.py is a utility to plot the results from multiple runs of Navigation.py

## Requirements:
	* Python 3.6 or greater
	* pythorch 1.7.1
	* Unity ML Agents, Banana Environment

## Running the code
* Set the parameters in the params object in Navigation.py
	* Set the name prefix of the output files. 
	* Two output files are generated. A pickle file containing a dump of the parameters object and raw scores, and network checkpoint weights compatible with pytorch
* Run the Navigation.py
* By default, the script ends when a score of 13 is met.
* Two files are generated:
	* The pickle file may be analyzed via the plotres.py utility
	* Checkpoint of the NN model.
## Baseline results
The default parameter/results are located in the folder '/results'. They can be analyzed with the plotres.py utility.
