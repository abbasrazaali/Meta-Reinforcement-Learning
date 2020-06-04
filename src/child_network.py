#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : child network
#==============================================================================

import tensorflow as tf
import tensorflow.contrib as tf_contrib
from src.utils import get_residual_layer
import numpy as np

class ChildNetwork():
    def __init__(self, inputs, depth, dropout_rate, num_classes, max_depth, initial_filters, data_format):
        self.inputs = inputs
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.initial_filters = initial_filters
        self.data_format = data_format

    def stochastic_depth_conv2d(self, mode):
        try:
            layer_no = 0
            # pred = []

            with tf.variable_scope('resnet_model'):
                # if self.data_format == 'channels_first':
                #     self.inputs = tf.transpose(self.inputs, [0, 3, 1, 2])

                residual_list, kernels, _, strides = get_residual_layer(self.max_depth)

                network = self.conv2d_fixed_padding(inputs=self.inputs, filters=self.initial_filters, kernel_size=7, strides=2, data_format=self.data_format, layer_name=str(layer_no))
                network = tf.identity(network, 'initial_conv')

                for blocks in range(len(residual_list)):
                    self.initial_filters *= 2
                    for i in range(residual_list[blocks]):
                        layer_no += 1
                        network, pred_ = self.residual_block(network, filters=self.initial_filters, kernels=kernels, strides=strides, depth=self.depth, is_training=(mode == 'train'),
                                                             data_format=self.data_format, layer_no=str(layer_no), block_scope='resblock_' + str(layer_no))

                    network = tf.identity(network, name = 'conv_identity_' + str(i))

                # only apply the BN and ReLU for model that does pre_activation in each building/bottleneck block, eg resnet V2.
                network = self.batch_norm(network, training=(mode=='train'), data_format=self.data_format)
                network = tf.nn.relu(network)

                self.inputs = tf.Print(self.inputs, [tf.shape(self.inputs)], 'shape_1: ', summarize=32)

                # current top layer has shape `batch_size x pool_size x pool_size x final_size`.
                axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
                network = tf.reduce_mean(network, axes, keepdims=True)
                network = tf.identity(network, 'final_reduce_mean')

                network = tf.layers.flatten(network)

                network = tf.layers.dropout(network, rate=self.dropout_rate, training=(mode=='train'), name='Dropout')

                network = tf.layers.dense(inputs=network, units=self.num_classes, kernel_initializer=tf_contrib.layers.variance_scaling_initializer())
                # network = tf.identity(network, 'final_dense')

                probs = tf.nn.softmax(network)

                return network, probs
        except Exception as e:
            print("stochastic depth conv2d failed - " + str(e))

    # A single block for ResNet v2, without a bottleneck
    def residual_block(self, inputs, filters, kernels, strides, depth, is_training, data_format, layer_no, block_scope = ''):
        with tf.variable_scope(block_scope):
            batch_norm_ = self.batch_norm(inputs, is_training, data_format, layer_name='batch_norm_0_' + layer_no)
            batch_norm_ = tf.nn.relu(batch_norm_)

            # The projection shortcut should come after the first batch norm and ReLU since it performs a 1x1 convolution.
            # if (self.max_depth > 50):
            #     shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=4*filters, kernel_size=1, strides=2, data_format=data_format, layer_name='conv_0_' + layer_no)
            # else:
            shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=2, data_format=data_format, layer_name='conv_0_' + layer_no)

            cov_layer = self.conv2d_fixed_padding(inputs=batch_norm_, filters=filters, kernel_size=kernels[0], strides=strides[0], data_format=data_format, layer_name='conv_1_' + layer_no)

            batch_norm_ = self.batch_norm(cov_layer, is_training, data_format, layer_name='batch_norm_1_' + layer_no)
            batch_norm_ = tf.nn.relu(batch_norm_)
            cov_layer = self.conv2d_fixed_padding(inputs=batch_norm_, filters=filters, kernel_size=kernels[1], strides=strides[1], data_format=data_format, layer_name='conv_2_' + layer_no)

            # if(self.max_depth > 50):
            #     batch_norm_ = self.batch_norm(cov_layer, is_training, data_format, layer_name='batch_norm_2_' + layer_no)
            #     batch_norm_ = tf.nn.relu(batch_norm_)
            #     cov_layer = self.conv2d_fixed_padding(inputs=batch_norm_, filters=4*filters, kernel_size=kernels[2], strides=strides[2], data_format=data_format, layer_name='conv_3_' + layer_no)

            layer_number = tf.constant(int(cov_layer.name.split('/')[1].split('_')[-1]), name='layer_number')

            pred = tf.cast(tf.less(layer_number, tf.add(depth, 1)), tf.bool)
            # pred = tf.Print(pred, [pred, layer_number, depth], 'pred: ', summarize=32)

            # return tf.cond(pred, lambda: cov_layer + shortcut, lambda: shortcut)
            return tf.cond(pred, lambda: cov_layer, lambda: shortcut), pred

    # performs a batch normalization using a standard set of parameters
    def batch_norm(self, inputs, training, data_format, layer_name = ''):
        return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else -1, momentum=0.997, epsilon=1e-5, center=True,
            scale=True, training=training, fused=True, name = layer_name)

    # pads the input along the spatial dimensions independently of input size
    def fixed_padding(self, inputs, kernel_size, data_format):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    # Strided 2-D convolution with explicit padding
    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format, layer_name = ''):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(), data_format=data_format, name = layer_name)

    # loss function
    def classification_loss(self, logits, label):
        # prediction = tf.nn.softmax(logits, name="prediction")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label), name="loss")
        prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(label, -1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")

        return loss, accuracy
