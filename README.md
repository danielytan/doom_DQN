# doom_DQN
Final project for CSE 291.

## Setup
This is the implementation of an agent that plays a simple scenario (basic) from Doom using Deep Q-Learning.
There are 3 possible actions (moving left, moving right, or shooting), and an episode finishes either when the monster in the scenario is eliminated or after timeout. The reward for successfully eliminating the monster is +101, and the rewards for missing or living (without eliminating the monster) are -5 and -1, respectively.
 
Instructions for running:
```
  usage: doom.py [-h] t m ep

  train or test a DQN agent with a specified model

  positional arguments:
    t           specify mode for 'training' or 'testing'
    m           the model to train or test with
    ep          the number of episodes to run for

  optional arguments:
    -h, --help  show this help message and exit
```
If training is enabled, the agent will train for 'ep' episodes, and save the learned model into the dqn_model folder as 'm'. 

If testing is enabled, the agent will test a learned model 'm' from the dqn_model folder for 'ep' episodes.

## Results
Each agent is trained for 500 episodes and tested for 1000 episodes, with varying experience replay buffer sizes. 
The x-axis for each agent's graph is the number of episodes (1000), and the y-axis for the graph is the score (capped at 101.)

Based on the findings from a paper examining experience replay (https://arxiv.org/pdf/1712.01275.pdf), I would have expected that constraining the experience replay buffer might produce slightly better results (i.e. higher mean score), but it turns out that constraining the buffer lowers the mean progressively. This may indicate that a 1,000,000 experience replay buffer may be _too_ small, or is of just the right size (i.e. perhaps increasing the buffer size from this point would decrease the score.)

###### Agent with a 1,000,000 experience replay buffer
<img src="https://github.com/danielytan/doom_DQN/blob/master/results/500training%2C1000testing.png" alt="Graph (max score is 101)" width="500"/><img src="https://github.com/danielytan/doom_DQN/blob/master/results/500training%2C1000testing_stats.png" alt="Statistics" width="250"/>

###### Agent with a 100,000 experience replay buffer
<img src="https://github.com/danielytan/doom_DQN/blob/master/results/experiment1_test0.png" alt="Graph (max score is 101)" width="500"/><img src="https://github.com/danielytan/doom_DQN/blob/master/results/experiment1_test0_stats.png" alt="Statistics" width="250"/>

###### Agent with a 1,000 experience replay buffer
<img src="https://github.com/danielytan/doom_DQN/blob/master/results/experiment0_test0.png" alt="Graph (max score is 101)" width="500"/><img src="https://github.com/danielytan/doom_DQN/blob/master/results/experiment0_test0_stats.png" alt="Statistics" width="250"/>
