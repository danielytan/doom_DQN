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
These are the results of testing a DQN agent for 1000 episodes (trained for 500 episodes):

The x-axis for the graph is the number of episodes (1000), and the y-axis for the graph is the score (stretched to 1000, but capped at 101.)

![Graph (max score is 101)](https://github.com/danielytan/doom_DQN/blob/master/results/500training%2C1000testing.png)
![Statistics](https://github.com/danielytan/doom_DQN/blob/master/results/500training%2C1000testing_stats.png)
