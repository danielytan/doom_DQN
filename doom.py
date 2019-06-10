import tensorflow as tf

import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

'''
Deep Q-Learning Neural Network model

How it works:
    1. Take a 4 frame stack as input
    2. Run the input through 3 convolutional neural networks
    3. Flatten
    4. Run through 2 fully connected layers
    5. Output a Q-value for each action
'''
class DQN:
    def __init__(self, state_size, action_size, learning_rate, name='DQN'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # use tensorflow library
        with tf.variable_scope(name):
            # create placeholders
            # *state_size -> take each element of state_size
            # [None, *state_size] = [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")

            # target_Q = R(s,a) + ymax Qhat(s',a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convolutional neural network:
            CNN
            BatchNormalization
            ELU
            """
            # input is 84x84x4 (4 84x84 preprocessed frames)
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs_,
                filters = 32,
                kernel_size = [8,8],
                strides = [4,4],
                padding = "VALID",
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = "conv1"
            )

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training = True,
                epsilon = 1e-5,
                name = "batch_norm1"
            )

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]

            """
            Second convolutional neural network:
            CNN
            BatchNormalization
            ELU
            """

            self.conv2 = tf.layers.conv2d(
                inputs = self.conv1_out,
                filters = 64,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = "conv2"
            )

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training = True,
                epsilon = 1e-5,
                name = "batch_norm2"
            )

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]

            """
            Third convolutional neural network:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(
                inputs = self.conv2_out,
                filters = 128,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = "conv3"
            )

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training = True,
                epsilon = 1e-5,
                name = "batch_norm3"
            )

            self.conv3_out = tf.nn.elu(
                self.conv3_batchnorm,
                name = "conv3_out"
            )
            ## --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.fc = tf.layers.dense(
                inputs = self.flatten,
                units = 512,
                activation = tf.nn.elu,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = "fc1"
            )

            self.output = tf.layers.dense(
                inputs = self.fc,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                units = 3,
                activation = None
            )

            # Q = predicted Q-value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # loss = difference between predicted Q-values and Q_target
            # sum(Q_target - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            
'''
Implementation of Experience Replay

Creates a Memory object with a deque buffer; using a deque allows us to automatically remove old frames
'''
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size = batch_size,
            replace = False
        )

        return [self.buffer[i] for i in index]

'''
Initialize the environment
'''
def initialize_environment():
    game = DoomGame()

    # load the correct config
    game.load_config("basic.cfg")

    # load the basic scenario
    game.set_doom_scenario_path("basic.wad")

    game.set_screen_format(ScreenFormat.GRAY8)

    game.init()

    # list possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

'''
Test the environment with random actions
'''
def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()

    # list possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [left, right, shoot]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

#test_environment()
game, possible_actions = initialize_environment()

'''
Preprocess frames to reduce state complexity by grayscaling, cropping, and normalizing pixel values
'''
def preprocess_frame(frame):
    # greyscale is done for us in vizdoom config

    # remove topbar
    cropped_frame = frame[30:-10, 30:-30]

    # normalize pixel values
    normalized_frame = cropped_frame/255.0

    # resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    
    return preprocessed_frame

'''
Training the agent
    1. Initialize weights, environment, and decay rate
    2. For episode to max_episode, do
        i. Make a new episode
        ii. Set step to 0
        iii. Observe the first state s_0

        iv. While step < max_steps, do
            1. Increase decay_rate
            2. Using epsilon, either select a random action a_t or pick the max Q-value a_t = argmax_a Q(s_t,a_t)
            3. Execute action a in the simulator and observe the reward r_t+1 and new state s_t+1
            4. Store transition $
            5. Sample random mini-batch from D: $$
            6. Set Qhat = r if the episode ends at +1, otherwise set Qhat = r + gamma * max_a * Q(s',a')
            7. Make a gradient descent step with loss (Qhat - Q(s,a))^2
'''
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## epsilon greedy strategy
    # choose an action a from state s using epsilon greedy.
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # make a random action (exploration)
        action = random.choice(possible_actions)
    
    else:
        # get action from DQN (exploitation) and estimate Qs values state
        Qs = sess.run(
            DQN.output, 
            feed_dict = {DQN.inputs_: state.reshape((1, *state.shape))}
        )

        # take the biggest Q value (effectively the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

'''
Stack frames to give the AI a sense of motion:
    1. Preprocess frame
    2. Append preprocessed frame to deque that will automatically remove the oldest frame
    3. Build the stacked state

How the stack works:
    1. First, feed four frames to the deque
    2. Add new frames to the deque at each time step to create a new stacked frame
    3. Each episode should start with a new stack with 4 new frames
'''
# we initially stack 4 frames
stack_size = 4

# initialize the deque with four arrays for each image (initialized to zero)
stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # clear the stacked frames
        stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(stack_size)], maxlen=4)

        # copy the same starting frame 4x
        for _ in range(4):
            stacked_frames.append(frame)
        
        # stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # append new frame to deque (automatically removing the oldest frame)
        stacked_frames.append(frame)

        # build the stacked state
        stacked_state = np.stack(stacked_frames, axis=2)
    
    return stacked_state, stacked_frames

# model hyperparameters
state_size = [84,84,4]                              # input is a stack of 4 preprocessed frames (width, height, channels)
action_size = game.get_available_buttons_size()     # 3 possible actions: move left, move right, shoot
learning_rate = 0.0002                              # alpha (learning rate)

# training hyperparameters
total_episodes = 500                                # total episodes for training
max_steps = 100                                     # max possible steps for one episode
batch_size = 64

# exploration parameters for exploration/exploitation
explore_start = 1.0                                 # starting exploration probability
explore_stop = 0.01                                 # minimum exploration probablity
decay_rate = 0.0001                                 # exploration probability for exponential decay rate

# Q-Learning hyperparameters
gamma = 0.95                                        # discount factor

# memory hyperparameters
pretrain_length = batch_size                        # number of experiences stored in memory when first initialized
memory_size = 1000000                               # number of experiences the memory can store in capacity

# toggle to see agent in training or not
training = True

# toggle to render the environment or not
episode_render = True

# reset the graph
tf.reset_default_graph()

# instantiate DQN
DQN = DQN(state_size, action_size, learning_rate)

# prepopulate memory by taking random actions and storing the gained experiences; deals with empty memory problem
# experience is (state, action, reward, new_state)
memory = Memory(max_size = memory_size)

# render the environment
game.new_episode()

for i in range(pretrain_length):
    # on the first step
    if i == 0:
        # get the state via screen buffer
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    # pick a random action
    action = random.choice(possible_actions)

    # get the rewards
    reward = game.make_action(action)

    # check if the episode is complete
    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state.shape)

        # add experience to memory
        memory.add((state, action, reward, next_state, done))

        # begin a new episode
        game.new_episode()

        # get state from screen buffer
        state = game.get_state().screen_buffer

        # stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # add experience to memory
        memory.add((state, action, reward, next_state, done))

        # state is now the next state
        state = next_state

# set up TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## losses
tf.summary.scalar("Loss", DQN.loss)

write_op = tf.summary.merge_all()

# save our model with Saver
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # initialize the decay rate (used to reduce epsilon)
        decay_step = 0

        # initailize the game
        game.init()

        for episode in range(total_episodes):
            # set step to 0
            step = 0

            # initialize the rewards of the episode
            episode_rewards = []

            # create a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer

            # calling the stack frame function will also call our preprocess function
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # increase decay step
                decay_step += 1

                # predict action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # take the action
                reward = game.make_action(action)

                # check the episode if complete
                done = game.is_episode_finished()

                # append reward to total rewards
                episode_rewards.append(reward)

                # if game is complete
                if done:
                    # no next state upon episode completion
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # set step = max_steps to end episode
                    step = max_steps

                    # get total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print(
                        "Episode: {}".format(episode),
                        "Total reward: {}".format(total_reward),
                        "Training loss: {:.4f}".format(loss),
                        "Explore P: {:.4f}".format(explore_probability)
                    )

                    memory.add((state, action, reward, next_state, done))
                
                else:
                    # get the next state
                    next_state = game.get_state().screen_buffer

                    # stack the frame of the next state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                # learning portion
                # obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # get Q values for next state
                Qs_next_state = sess.run(DQN.output, feed_dict = {DQN.inputs_: next_states_mb})

                # set Q_target = r if episode ends at s+1, otherwise set Q-target = r + gamma * maxQ(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # if we reach a terminal state, only equal to reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                
                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run(
                    [DQN.loss, DQN.optimizer],
                    feed_dict = {
                        DQN.inputs_: states_mb,
                        DQN.target_Q: targets_mb,
                        DQN.actions_: actions_mb
                    }
                )

                # write TF summaries
                summary = sess.run(
                    write_op, feed_dict= {
                        DQN.inputs_: states_mb,
                        DQN.target_Q: targets_mb,
                        DQN.actions_: actions_mb
                    }
                )

                writer.add_summary(summary, episode)
                writer.flush()

            # save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./dqn_model/dqn_model.ckpt")
                print("Model Saved")

# test the agent
with tf.Session() as sess:
    game, possible_actions = initialize_environment()

    total_score = 0

    # load the trained model
    saver.restore(sess, "./dqn_model/dqn_model.ckpt")
    game.init()
    for i in range(1):
        done = False
        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
            Qs = sess.run(
                DQN.output, 
                feed_dict = {
                    DQN.inputs_: state.reshape((1, *state.shape))
                }
            )

            # take the highest Q-value (effectively best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()

            if done:
                break

            else:
                print("next state")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        
        score = game.get_total_reward()
        print("Score: ", score)

    game.close()



    