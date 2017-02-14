'''
This code is based on
https://github.com/hunkim/DeepRL-Agents
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # default value = 0  From http://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information

import tensorflow as tf


env = gym.make('FrozenLake-v0')

# Input and output size based on the Env
input_size = env.observation_space.n;
output_size = env.action_space.n;
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)              # state input
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))   # weight

Qpred = tf.matmul(X, W)     # Out Q prediction
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)    # Y label

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning parameters
dis = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []

def one_hot(x):
    return np.identity(16)[x:x+1]

start_time = time.time()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedly (with a chance of random action)
            # from the Q-network
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)
            if done:
                # Update Q, and no Qs+1, since it's a termial state
                Qs[0, a] = reward
            else:
                # Obtain the Q_s` values by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                # Update Q
                Qs[0, a] = reward + dis*np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

            rAll += reward
            s = s1

        rList.append(rAll)

print("--- %s seconds ---" % (time.time() - start_time))

print("Success rate: " + str(sum(rList) / num_episodes))
#plt.bar(range(len(rList)), rList, color="blue")
plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()

