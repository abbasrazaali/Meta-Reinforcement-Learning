#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : the child network
#==============================================================================

import tensorflow as tf
import math
import numpy as np

from src.child_network import ChildNetwork
from src.utils import logging

class NetManager():
    def __init__(self, search_space, input_dimensions, num_classes, dataset, dataset_name, log_dir, train_batch_size = 100, test_batch_size = 1,
                 train_num_epochs = 5, max_depth = 18, num_child_steps_per_cycle = 20, initial_filters = 32, model_dir = None, logger = None):
        self.search_space = search_space
        self.input_dimensions = input_dimensions
        self.num_classes = num_classes
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.train_num_epochs = train_num_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.max_depth = max_depth
        self.initial_filters = initial_filters
        self.num_child_steps_per_cycle = num_child_steps_per_cycle

        self.data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.beta = 0.8
        self.beta_bias = 0.8
        self.moving_accuracy = 0.0
        self.clip_rewards = False

        self.log_dir = log_dir

        self.model_dir = model_dir
        # self.prev_step = 0

        self.best_accuracy = 0.0
        self.loss_value = 0.0

        # self.exclude_patterns = []

        self.activation_fn = {"1": tf.train.AdamOptimizer, "2": tf.train.RMSPropOptimizer, "3": tf.train.GradientDescentOptimizer, "4": tf.train.MomentumOptimizer}

        self.logger = logger

        self.initialize_graph()

    def initialize_graph(self): # , action, step, pre_acc, search_space_size):
        try:
            # creating graph
            # self.graph = tf.Graph().as_default()
            tf.reset_default_graph()

            print('Building graph...')

            # max_depth = 18
            # image_size = 228
            # channels = 3
            # batch_size = 32
            # num_classes = 10
            # learning_rate = 0.01
            self.input_dimensions = self.input_dimensions.split('x')
            # if('mnist' in self.dataset_name):
            #     self.inputs = tf.placeholder(tf.uint8, [None, int(self.input_dimensions[0]), int(self.input_dimensions[1]), int(self.input_dimensions[2])], name="inputs")
            # elif 'cifar' in self.dataset_name:
            if self.data_format == 'channels_last':
                self.inputs = tf.placeholder(tf.float32, shape=[None, int(self.input_dimensions[0]), int(self.input_dimensions[1]), int(self.input_dimensions[2])],  name="inputs")
            else:
                self.inputs = tf.placeholder(tf.float32, shape=[None, int(self.input_dimensions[2]), int(self.input_dimensions[0]), int(self.input_dimensions[1])], name="inputs")

            self.labels = tf.placeholder(tf.int32, shape=[None, self.num_classes], name='label')

            self.depth = tf.placeholder(tf.int32, shape=[], name='depth')
            self.dropout_rate = tf.placeholder(tf.float32, shape=[], name="dropout")
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')
            # self.nesterov = tf.placeholder(tf.bool, shape=(), name='nesterov')

            # labels_onehot = tf.one_hot(self.labels, 10)

            # child network
            self.model = ChildNetwork(self.inputs, self.depth, self.dropout_rate, self.num_classes, max_depth=self.max_depth, initial_filters=self.initial_filters, data_format = self.data_format)
            logits, self.probs = self.model.stochastic_depth_conv2d(mode = 'train')
            self.loss, self.accuracy = self.model.classification_loss(logits=logits, label=self.labels)
            _, self.validation_accuracy = self.model.classification_loss(logits=logits, label=self.labels)

            # print([t.name for op in self.graph.get_operations() for t in op.values()])
            # print([t for op in self.graph.get_operations() for t in op.values()])

            self.global_step = tf.Variable(0, trainable=False)

            self.optimizer = self.activation_fn['4'](learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)
            self.var_list = tf.trainable_variables()
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # self.all_trainable_vars = [np.product(list(map(int, v.shape))) * v.dtype.size for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
            self.all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            # np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])

            # if ('mnist' in self.dataset_name):
            #     self.train_batch_x, self.train_batch_y, self.test_batch_x, self.test_batch_y = self.dataset[0], self.dataset[1], self.dataset[2], self.dataset[3]
            # elif ('cifar' in self.dataset_name):
            self.train_batch_x, self.train_batch_y, self.test_batch_x, self.test_batch_y = self.dataset

            self.num_steps = int(math.ceil(len(self.train_batch_x) / self.train_batch_size))

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            tf.summary.scalar("network_trainable_variables", self.all_trainable_vars)
            tf.summary.scalar("network_depth", self.depth)
            tf.summary.scalar("network_loss", self.loss)
            tf.summary.scalar('network_training_accuracy', self.accuracy)
            tf.summary.scalar('network_validation_accuracy', self.validation_accuracy)
            self.summaries_op = tf.summary.merge_all()

            filename = self.log_dir + '/network/tb_logs/'
            self.summary_writer = tf.summary.FileWriter(filename, graph=self.sess.graph)

            # vars = tf.trainable_variables()
            # print(vars)  # some infos about variables...
            # vars_vals = self.sess.run(vars)
            # for var, val in zip(vars, vars_vals):
            #     print("var: {}, value: {}".format(var.name, val))

            self.saver = tf.train.Saver(max_to_keep=1)

            ckpt = tf.train.latest_checkpoint(self.model_dir + 'network/model.chkt')
            if ckpt and tf.train.checkpoint_exists(self.model_dir):
                self.saver.restore(self.sess, ckpt)
                logging(self.model_dir + 'network/model.chkt' + " model loaded successfully", self.logger, 'info')
        except Exception as e:
            logging("Get reward failed - " + str(e), self.logger, 'error')

    def get_reward(self, action, prev_step, prev_accuracy):     # prev_epoch
        try:
            # action = [action[0][0][x:x + search_space_size] for x in range(0, len(action[0][0]), search_space_size)]

            # drop_rate = [self.search_space['3'][c[3] % len(self.search_space['3'])] for c in action]
            # activation = self.network_structure[0][8]
            # drop_rate = [c[0] for c in action]
            depth = action[0]
            dropout = action[1]
            learning_rate = action[2]
            momentum = action[3]
            # nesterov = action[4]

            # depth = 101
            # dropout = 0.2
            # learning_rate = 0.001
            # momentum = 0.95

            summary, global_step = None, None
            # for epoch in range(prev_epoch, self.train_num_epochs + 1):
            iterations = 0
            for step_ in range(prev_step, self.num_steps + 1):
                iterations += 1
                curr_itr = prev_step + self.num_child_steps_per_cycle
                for step in range(prev_step, (curr_itr if curr_itr < self.num_steps else self.num_steps)):

                    train_batch_xi = self.train_batch_x[step * self.train_batch_size : (step + 1) * self.train_batch_size]
                    train_batch_yi = self.train_batch_y[step * self.train_batch_size : (step + 1) * self.train_batch_size]

                    _, summary, global_step = self.sess.run([self.train_op, self.summaries_op, self.global_step], feed_dict={self.inputs: train_batch_xi, self.labels: train_batch_yi, self.depth: depth, self.dropout_rate: dropout,
                                                                self.learning_rate: learning_rate, self.momentum: momentum})
                    if step % 10 == 0:     # calculate batch loss and accuracy
                        self.loss_value, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict={self.inputs: train_batch_xi, self.labels: train_batch_yi, self.depth: depth, self.dropout_rate: dropout,
                                                                self.learning_rate: learning_rate, self.momentum: momentum})

                        print("Training: | Step: " + str(step) + " | Training Loss: " + "{:.3f}".format(self.loss_value) + " | Training Accuracy: " + "{:.3f}".format(accuracy))
                        #  Epoch: " + str(epoch + 1) + "

                # validation
                validation_accuracy, probs, trainable_variables = self.sess.run([self.validation_accuracy, self.probs, self.all_trainable_vars], feed_dict={self.inputs: self.test_batch_x, self.labels: self.test_batch_y, self.depth: depth, self.dropout_rate: dropout,
                                                                self.learning_rate: learning_rate, self.momentum: momentum})

                print("Validation | Step: " + str(step) + " | Validation Accuracy: " + "{:.3f}".format(validation_accuracy))

                # print gradients
                # print(g)
                # for var, grad_value in zip(self.var_list, g):
                #     grad, value = grad_value
                #     print('', var.op.name, grad.squeeze(), sep='\n')

                # difference factor
                # different_factor = self.get_different_factor(accuracy * 100)
                # if accuracy * (1 + different_factor) <= self.best_accuracy:
                #     self.best_accuracy = accuracy
                #     step = prev_step
                # else:

                # if prev_accuracy * 1.2 <= accuracy:


                # if(prev_step == 0):
                #     prev_epoch += 1
                # else:
                #     prev_epoch = epoch

                # if (accuracy + different_factor) <= self.best_accuracy:
                    # compute the reward
                reward = validation_accuracy #- self.moving_accuracy)
                # if self.moving_accuracy == 0.0 or reward == 0.0:
                #     reward = 0.01

                if self.clip_rewards:
                    reward = np.clip(reward, -0.05, 0.05)

                # update moving accuracy with bias correction for 1st update
                if self.beta > 0.0 and self.beta < 1.0:
                    self.moving_accuracy = self.beta * self.moving_accuracy + (1 - self.beta) * validation_accuracy
                    self.moving_accuracy = self.moving_accuracy / (1 - self.beta_bias)
                    self.beta_bias = 0

                    # reward = np.clip(reward, -0.1, 0.1)
                    if reward <= 0.0:
                        reward = 0.01

                print("Evaluation accuracy: " + str(validation_accuracy) + " | moving accuracy: " + str(round(self.moving_accuracy, 4)) + " | previous accuracy: " + str(prev_accuracy))

                # if(self.moving_accuracy > validation_accuracy and (validation_accuracy - prev_accuracy) < 0.0): # if (validation_accuracy - prev_accuracy) < 0.0: #different_factor:
                self.summary_writer.add_summary(summary, global_step)
                self.summary_writer.flush()

                self.saver.save(self.sess, save_path=self.model_dir + 'network/model.chkt', global_step=tf.train.get_global_step())

                return reward, validation_accuracy, self.loss_value, prev_step, step, probs, iterations, self.moving_accuracy, trainable_variables
                # else:
                #     prev_accuracy = validation_accuracy
                #     prev_step = step % (self.num_steps - 1)

                # else:
                #     self.best_accuracy = accuracy

                # if accuracy - prev_accuracy <= 0.01: #and reward >= 0.0:
                #     return accuracy, accuracy, loss, prev_step, step, probs, iterations
                # else:
                #     return 0.01, accuracy, loss, prev_step, step, probs, iterations      # prev_epoch

                # if (accuracy - prev_accuracy) <= different_factor:
                # if (accuracy - prev_accuracy) <= 0.01:
                #     # # compute the reward
                #     # reward = (accuracy - self.moving_accuracy)
                #     #
                #     # # if rewards are clipped, clip them in the range -0.05 to 0.05
                #     # # if self.clip_rewards:
                #     # #     reward = np.clip(reward, -0.05, 0.05)
                #     #
                #     # # update moving accuracy with bias correction for 1st update
                #     # if self.beta > 0.0 and self.beta < 1.0:
                #     #     self.moving_accuracy = self.beta * self.moving_accuracy + (1 - self.beta) * accuracy
                #     #     self.moving_accuracy = self.moving_accuracy / (1 - self.beta_bias)
                #     #     self.beta_bias = 0
                #     #
                #     # # reward = np.clip(reward, -0.1, 0.1)
                #
                #     reward = accuracy
                #
                #     return reward, accuracy, loss, epoch, prev_step, step
                # else:
                #     return 0.01, accuracy, loss, epoch, prev_step, step

        except Exception as e:
            logging("Get reward failed - " + str(e), self.logger, 'error')

    def get_different_factor(self, accuracy):
        different_factor = 0.0
        if(accuracy >= 0.0 and accuracy <= 70.0):
            different_factor = 0.01
        elif(accuracy > 70.0 and accuracy <= 90.0):
            different_factor = 0.008
        elif(accuracy > 90.0 and accuracy <= 95.0):
            different_factor = 0.004
        elif(accuracy > 95.0 and accuracy <= 100.0):
            different_factor = 0.002

    # def get_different_factor(self, accuracy):
    #     different_factor = 0.0
    #     if (accuracy >= 0.0 and accuracy <= 10.0):
    #         different_factor = 0.07
    #     elif (accuracy > 10.0 and accuracy <= 25.0):
    #         different_factor = 0.06
    #     elif (accuracy > 25.0 and accuracy <= 50.0):
    #         different_factor = 0.05
    #     elif (accuracy > 50.0 and accuracy <= 70.0):
    #         different_factor = 0.04
    #     elif (accuracy > 70.0 and accuracy <= 80.0):
    #         different_factor = 0.03
    #     elif (accuracy > 80.0 and accuracy <= 90.0):
    #         different_factor = 0.02
    #     elif (accuracy > 90.0 and accuracy <= 95.0):
    #         different_factor = 0.01
    #     elif (accuracy > 95.0 and accuracy <= 100.0):
    #         different_factor = 0.08

        return different_factor