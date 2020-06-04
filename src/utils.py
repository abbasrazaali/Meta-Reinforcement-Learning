#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : utility functions
#==============================================================================

import os
import numpy as np
import random
# # from keras.layers import Activation, Conv2D
# # from keras.layers.normalization import BatchNormalization
# # from keras import backend as K
from enum import Enum
import logging as _logging
from tensorflow.python.client import device_lib
from configparser import ConfigParser
import tensorflow as tf
from tensorflow.python.training.device_setter import _RoundRobinStrategy
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
import operator

# Load configuration
def getConfig(config_file):
    try:
        parser = ConfigParser()
        parser.read(config_file)

        # get the ints, floats and strings
        _conf_ints = [(key, int(value)) for (key, value) in parser.items('ints')]
        _conf_floats = [(key, float(value)) for (key, value) in parser.items('floats')]
        _conf_strings = [(key, str(value)) for (key, value) in parser.items('strings')]
        return dict(_conf_ints + _conf_floats + _conf_strings)
    except Exception as e:
        print("Failed to load configuration - " + str(e))

# check a path exists or not
def checkPathExists(path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

# devices enum
class DeviceCategory(Enum):
    CPU = 1
    GPU = 2

# Used with tf.device() to place variables on the least loaded GPU
class GpuParamServerDeviceSetter(object):
    """
      A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
      'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
      placed on the least loaded gpu. All other Ops, which will be the computation
      Ops, will be placed on the worker_device.
    """

    def __init__(self, worker_device, ps_devices):
        """Initializer for GpuParamServerDeviceSetter.
        Args:
          worker_device: the device to use for computation Ops.
          ps_devices: a list of devices to use for Variable Ops. Each variable is
          assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
            return self.worker_device

        # Gets the least loaded ps_device
        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


# get gpu devices
def get_gpu_devices():
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpu_devices


# get cpu devices
def get_cpu_devices():
    local_device_protos = device_lib.list_local_devices()
    cpu_devices = [x.name for x in local_device_protos if x.device_type == 'CPU']
    return cpu_devices


# Create device setter object
def create_device_setter(device_category: DeviceCategory, device: str, gpu_devices: list):
    if device_category == DeviceCategory.CPU:
        # tf.train.replica_device_setter supports placing variables on the CPU, all on one GPU, or on ps_servers defined in a cluster_spec.
        return tf.train.replica_device_setter(worker_device=device, ps_device='/cpu:0', ps_tasks=1)
    else:
        return GpuParamServerDeviceSetter(device, gpu_devices)


# device setter
def get_device_setter(device_category: DeviceCategory, device):
    if device_category == DeviceCategory.GPU:
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(get_gpu_devices()[0]),
                                                                      tf.contrib.training.byte_size_load_fn)
    else:
        ps_strategy = _RoundRobinStrategy(len(get_cpu_devices()[0]))

    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string('/{}:{}'.format(device_category.name, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser

# logging
def logging(message, logging, type='error'):
    try:
        if (type == 'debug'):
            _logging.debug(message)
        elif (type == 'info'):
            _logging.info(message)
        elif (type == 'warning'):
            _logging.warning(message)
        elif (type == 'error'):
            _logging.error(message)
        elif (type == 'critical'):
            _logging.critical(message)

        print(message)
    except Exception as e:
        print("Logging failed - " + str(e))

# get resnet layers structure
def get_residual_layer(res_n):
    layers, kernals = [], []

    if res_n == 9:
        layers = [2, 2]
        kernals = [3, 3]
        strides = [2, 1]
        min_depth = 2

    if res_n == 18:
        layers = [2, 2, 2, 2]
        kernals = [3, 3]
        strides = [2, 1]
        min_depth = 2

    if res_n == 34:
        layers = [3, 4, 6, 3]
        kernals = [3, 3]
        strides = [2, 1]
        min_depth = 8

    if res_n == 50:
        layers = [3, 4, 6, 3]
        kernals = [3, 3]
        strides = [2, 1]
        min_depth = 10

    if res_n == 101:
        layers = [3, 4, 23, 3]
        kernals = [3, 3]
        strides = [2, 1]
        min_depth = 24

    if res_n == 152:
        layers = [3, 8, 36, 3]
        kernals = [1, 3, 1]
        strides = [1, 2, 1]
        min_depth = 25

    return layers, kernals, min_depth, strides

# get resnet layers parameters
def get_residual_filters(max_depth, min_filters):
    filters_max = 0

    # get_residual_layers, _ = get_residual_layer(max_depth)
    for i in range(1, max_depth + 1):
        filters_max += min_filters
        if(i % 2 == 0):
            min_filters *= 2

    return filters_max

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

# def data_augmentation(batch, img_size, dataset_name):
#     if dataset_name == 'mnist':
#         batch = _random_crop(batch, [img_size, img_size], 4)
#
#     elif dataset_name =='tiny':
#         batch = _random_flip_leftright(batch)
#         batch = _random_crop(batch, [img_size, img_size], 8)
#
#     else:
#         batch = _random_flip_leftright(batch)
#         batch = _random_crop(batch, [img_size, img_size], 4)
#     return batch