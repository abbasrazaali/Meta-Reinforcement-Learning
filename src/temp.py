#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : Reinforcement learning - the parent network
#==============================================================================

import tensorflow as tf
# import random
import numpy as np
from src.utils import logging

class Reinforce():
    def __init__(self, initial_learning_rate, num_layers, num_hidden, state_space, learning_rate_decay_factor, division_rate = 100.0, reg_param = 0.001, discount_factor = 0.99, exploration = 0.3, logger = None):
        self.sess = tf.Session()
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor=discount_factor
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.state_space = state_space
        self.embedding_dim = 20

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step, 500, learning_rate_decay_factor, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        self.cell_outputs = []

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

        self.logger = logger

    def policy_network(self, state, max_layers, logger=None):
        try:
            with tf.name_scope("policy_network"):
                nas_cell = tf.contrib.rnn.NASCell(4 * max_layers)
                outputs, state = tf.nn.dynamic_rnn(nas_cell, tf.expand_dims(state, -1), dtype=tf.float32)

                bias = tf.Variable([0.05] * 4 * max_layers)
                outputs = tf.nn.bias_add(outputs, bias)

                print("outputs: ", outputs, outputs[:, -1:, :],
                      tf.slice(outputs, [0, 4 * max_layers - 1, 0], [1, 1, 4 * max_layers]))

                # return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
                return outputs[:, -1:, :]
        except Exception as e:
            logging("Policy network failed - " + str(e), logger, 'error')

    def get_action(self, state):
        try:
            return self.sess.run(self.predicted_action, {self.states: state})
            # if random.random() < self.exploration:
            #     return np.array([[random.sample(range(1, 35), 4*self.max_layers)]])
            # else:
            #     return self.sess.run(self.predicted_action, {self.states: state})
        except Exception as e:
            logging("Get action failed - " + str(e), self.logger, 'error')

    def create_variables(self):
        try:
            with tf.name_scope("model_inputs"):
                self.states = tf.placeholder(tf.float32, [None, self.max_layers * 4], name="states")      # raw state representation

            with tf.name_scope("predict_actions"):
                with tf.variable_scope("policy_network"):       # initialize policy network
                    # state input is the first input fed into the controller RNN. the rest of the inputs are fed to the RNN internally
                    # with tf.name_scope('state_input'):
                    state_input = tf.placeholder(dtype=tf.int32, shape=(1, None), name='state_input')

                    # self.state_input = state_input

                    nas_cell = tf.nn.rnn_cell.LSTMCell(35)
                    cell_state = nas_cell.zero_state(batch_size=1, dtype=tf.float32)

                    embedding_weights = []

                    # for each possible state, create a new embedding. Reuse the weights for multiple layers.
                    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
                        # for i in range(len(self.state_space)):
                        for key, value, index in zip(self.state_space.items(), range(len(self.state_space))):
                            state_ = value
                            size = len(value)

                            # size + 1 is used so that 0th index is never updated and is "default" value
                            weights = tf.get_variable('state_embeddings_%d' % index, shape=[size + 1, self.embedding_dim], initializer=tf.initializers.random_uniform(-1., 1.))
                            embedding_weights.append(weights)

                        # initially, cell input will be 1st state input
                        embeddings = tf.nn.embedding_lookup(embedding_weights[0], state_input)

                    cell_input = embeddings

                    for i in range(self.num_layers):
                        for key, value in self.state_space.items():
                            state_id = i % len(self.state_space)
                            size = len(value)

                            with tf.name_scope('controller_output_%d' % i):
                                # feed the ith layer input (i-1 layer output) to the RNN
                                outputs, final_state = tf.nn.dynamic_rnn(nas_cell, cell_input, initial_state=cell_state, dtype=tf.float32)

                                # add a new classifier for each layers output
                                classifier = tf.layers.dense(outputs[:, -1, :], units=size, name='classifier_%d' % (i), reuse=False)
                                predictions = tf.nn.softmax(classifier)

                                # feed the previous layer (i-1 layer output) to the next layers input, along with state take the class label
                                cell_input = tf.argmax(predictions, axis=-1)
                                cell_input = tf.expand_dims(cell_input, -1, name='pred_output_%d' % (i))
                                cell_input = tf.cast(cell_input, tf.int32)
                                cell_input = tf.add(cell_input, 1)  # we avoid using 0 so as to have a "default" embedding at 0th index

                                # embedding lookup of this state using its state weights ; reuse weights
                                cell_input = tf.nn.embedding_lookup(embedding_weights[state_id], cell_input, name='cell_output_%d' % (i))
                                cell_state = final_state

                            # store the tensors for later loss computation
                            self.cell_outputs.append(cell_input)
                            self.policy_classifiers.append(classifier)
                            self.policy_actions.append(predictions)

                    # self.policy_outputs = self.policy_network(self.states, self.max_layers)

                # self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
                # self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name="predicted_action")

            policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")          # regularization loss

            # compute loss and gradients
            with tf.name_scope("compute_gradients"):
                self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")        # gradients for selecting action from policy network

                with tf.variable_scope("policy_network", reuse=True):
                    self.logprobs = self.policy_network(self.states, self.max_layers)

                # compute policy loss and regularization loss
                self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:, -1, :], labels=self.states)
                self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
                self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables]) # Regularization
                self.loss               = self.pg_loss + self.reg_param * self.reg_loss

                self.gradients = self.optimizer.compute_gradients(self.loss)        # compute gradients

                # compute policy gradients
                for i, (grad, var) in enumerate(self.gradients):
                    if grad is not None:
                        self.gradients[i] = (grad * self.discounted_rewards, var)

                # training update
                with tf.name_scope("train_policy_network"):
                    # apply gradients to update policy network
                    self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)
        except Exception as e:
            logging("Create variables failed - " + str(e), self.logger, 'error')

    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        rewars = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss], {self.states: states, self.discounted_rewards: rewars})

        return ls
