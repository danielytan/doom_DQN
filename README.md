# doom_DQN
Final project for CSE 291.

This is the implementation of an agent that plays a simple scenario (basic) from Doom using Deep Q-Learning.
There are 3 possible actions (moving left, moving right, or shooting), and an episode finishes either when the monster in the scenario is eliminated or after timeout. The reward for successfully eliminating the monster is +101, and the rewards for missing or living (without eliminating the monster) are -5 and -1, respectively.

To run, type:
  python doom.py
 
If training is enabled in the python file, the agent will train for 500 episodes first; if not, it will directly use a model from the dqn_model folder.
