    #==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : Reinforcement learning - the parent network
#==============================================================================

import tensorflow as tf
import random
import numpy as np
from src.utils import logging
import time

class Reinforce():
    def __init__(self, model_dir, log_dir, initial_learning_rate, num_hidden, num_layers, search_space, num_steps_per_decay, learning_rate_decay_factor, optimizer, division_rate = 100.0,
                 reg_param = 0.001, discount_factor = 0.95, exploration = 0.3, logger = None):
        self.model_dir = model_dir
        self.log_dir = log_dir

        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor=discount_factor
        self.exploration = exploration
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.search_space = search_space
        self.search_space_size = len(search_space)

        self.activation_fn = {"adam": tf.train.AdamOptimizer, "rmsprop": tf.train.RMSPropOptimizer, "gd": tf.train.GradientDescentOptimizer}

        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step, num_steps_per_decay, learning_rate_decay_factor, staircase=True)
        # self.optimizer = self.activation_fn[optimizer](learning_rate=self.learning_rate)
        self.optimizer = self.activation_fn[optimizer](learning_rate=initial_learning_rate, beta1=learning_rate_decay_factor)

        # starter_learning_rate = 0.1
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 500, 0.95, staircase=True)
        # tf.summary.scalar('learning_rate', learning_rate)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        self.cell_outputs = []

        self.reward_buffer = []
        self.state_buffer = []
        self.action_buffer = []

        # self.states = tf.get_variable(name="states", shape=[None, self.search_space_size], dtype=tf.float32, initializer=tf.initializers.random_uniform(0., 1.))  # raw state representation

        self.logger = logger

        self.policy_session = tf.Session()
        self.create_variables()
        # var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # self.policy_session.run(tf.variables_initializer(var_lists))

    def policy_network(self, state, search_space_size):
        try:
            with tf.name_scope("policy_network"):
                nas_cell = tf.contrib.rnn.NASCell(search_space_size)

                outputs, state = tf.nn.dynamic_rnn(nas_cell, tf.expand_dims(state, -1), dtype=tf.float32)

                bias = tf.Variable([0.05] * search_space_size)
                outputs = tf.nn.bias_add(outputs, bias)

                # print("outputs: ", outputs, outputs[:, -1:, :], tf.slice(outputs, [0, 6 - 1, 0], [1, 1, 6]))
                # return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn

                return outputs[:, -1:, :]
        except Exception as e:
            logging("Policy network failed - " + str(e), self.logger, 'error')

    # Gets a one hot encoded action list, either from random sampling or from the Controller RNN
    def get_action(self, state, init = False):
        try:
            if random.random() < self.exploration or init:
                return np.array([[random.sample(range(1, self.num_hidden), self.search_space_size)]], dtype = np.int32)
            else:
                return self.policy_session.run(self.predicted_action, {self.states: state})
        except Exception as e:
            logging("Get action failed - " + str(e), self.logger, 'error')

    def create_variables(self):
        try:
            with tf.name_scope("model_inputs"):
                self.states = tf.placeholder(tf.float32, [None, self.search_space_size], name="states")

            with tf.name_scope("predict_actions"):
                # initialize policy network
                with tf.variable_scope("policy_network"):
                    self.policy_outputs = self.policy_network(self.states, self.search_space_size)

                self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
                self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name="predicted_action")

            # regularization loss
            policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

            # compute loss and gradients
            with tf.name_scope("compute_gradients"):
                self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")        # gradients for selecting action from policy network

                with tf.variable_scope("policy_network", reuse=True):
                    self.log_probs = self.policy_network(self.states, self.search_space_size)

                # compute policy loss and regularization loss
                self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.log_probs[:, -1, :], labels=self.states)
                self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
                self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])  # regularization
                self.loss = self.pg_loss + self.reg_param * self.reg_loss

                # compute gradients
                self.gradients = self.optimizer.compute_gradients(self.loss)

                # compute policy gradients
                for i, (grad, var) in enumerate(self.gradients):
                    if grad is not None:
                        self.gradients[i] = (grad * self.discounted_rewards, var)

                # training update
                with tf.name_scope("train_policy_network"):
                    self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)        # apply gradients to update policy network

            # vars = tf.trainable_variables()
            # print(vars)

            # tf.summary.scalar("controller_cross_entropy_loss", self.pg_loss)
            # tf.summary.scalar('controller_regularizer_loss', self.reg_loss)
            tf.summary.scalar('controller_discounted_reward', tf.reduce_sum(self.discounted_rewards))
            tf.summary.scalar("controller_loss", self.loss)
            # tf.summary.scalar("learning_rate", self.learning_rate)

            self.summaries_op = tf.summary.merge_all()
            filename = self.log_dir + '/controller/tb_logs/' #%s' % time.strftime("%Y-%m-%d-%H-%M-%S")

            self.summary_writer = tf.summary.FileWriter(filename, graph=self.policy_session.graph)

            self.policy_session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=1)

            ckpt = tf.train.latest_checkpoint(self.model_dir + 'controller/model.chkt')
            if ckpt and tf.train.checkpoint_exists(self.model_dir):
                self.saver.restore(self.policy_session, ckpt)
                logging(self.model_dir + 'controller/model.chkt' + " model loaded successfully", self.logger, 'info')
        except Exception as e:
            logging("Create variables failed - " + str(e), self.logger, 'error')

    def storeRollout(self, state, reward):
        try:
            self.reward_buffer.append(reward)
            self.state_buffer.append(state[0])
        except Exception as e:
            logging("Store rollout failed - " + str(e), self.logger, 'error')

    # perform a single train step on the controller RNN
    def train_step(self, steps_count):
        try:
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    print(self.gradients[i])

            # print('prev_reward: ' + str(self.reward_buffer[-steps_count:]))
            states = np.array(self.state_buffer[-steps_count:]) / self.division_rate
            # reward = self.reward_buffer[-steps_count:]
            reward = np.asarray([self.discount_reward_computation()]).astype('float32')

            # print('states: ' + str(states[0]))
            # print('rewards: ' + str(reward))

            _, loss, summary, log_probs, global_step = self.policy_session.run([self.train_op, self.loss, self.summaries_op, self.policy_outputs, self.global_step],
                                                                               {self.states: states, self.discounted_rewards: reward})
            log_probs = ['%.3f' % elem for elem in log_probs[0][0]]

            # print('' + str(log_probs))

            self.summary_writer.add_summary(summary, global_step)
            self.summary_writer.flush()
            self.saver.save(self.policy_session, save_path=self.model_dir + 'controller/model.chkt', global_step=self.global_step)

            # reduce exploration after many train steps
            if global_step != 0 and global_step % 20 == 0 and self.exploration > 0.5:
                self.exploration *= 0.99

            return loss, log_probs
        except Exception as e:
            logging("Train step failed - " + str(e), self.logger, 'error')

    # compute discounted rewards over the entire reward buffer
    def discount_reward_computation(self):
        try:
            rewards = np.asarray(self.reward_buffer)
            discounted_rewards = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, rewards.size)):
                if rewards[t] != 0.0:
                    running_add = 0
                running_add = running_add * self.discount_factor + rewards[t]
                discounted_rewards[t] = running_add

            return discounted_rewards[-1]
        except Exception as e:
            logging("Discount rewards failed - " + str(e), self.logger, 'error')