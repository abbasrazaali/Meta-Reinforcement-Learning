#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : main function - starting point
#==============================================================================

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import logging as _log
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
from src.train import train
from src.analysis import load_dataset, plot, plot_cifar10
from src.datasets import cifar10, cifar100, tiny_imagenet
from src.utils import getConfig, checkPathExists, get_gpu_devices, get_cpu_devices, normalize

import matplotlib.pyplot as plt

# main function
def main():
    try:
        gConfig = getConfig('config/meta_rl.ini')  # get configuration
        # site = gConfig['site']

        mode = gConfig['mode']
        dataset_name = gConfig['dataset']

        data_dir = gConfig['data_dir']
        # features_dir = gConfig['features_dir']
        # infer_dir = gConfig['infer_dir']
        model_dir = gConfig['model_dir']
        output_dir = gConfig['output_dir']
        log_dir = gConfig['log_dir']

        train_num_epochs = gConfig['train_num_epochs']
        num_layers = gConfig['num_layers']
        num_hidden = gConfig['num_hidden']
        learning_rate = gConfig['learning_rate']
        learning_rate_decay_factor = gConfig['learning_rate_decay_factor']
        num_steps_per_decay = gConfig['num_steps_per_decay']
        num_episodes = gConfig['num_episodes']
        train_batch_size = gConfig['train_batch_size']
        exploration = gConfig['exploration']
        discount_factor = gConfig['discount_factor']
        num_child_steps_per_cycle = gConfig['num_child_steps_per_cycle']
        # max_depth = gConfig['max_depth']
        initial_filters = gConfig['initial_filters']
        # num_classes = gConfig['num_classes']

        optimizer = gConfig['optimizer']
        # dropout_keep_prob = gConfig['dropout_keep_prob']

        # port = gConfig['port']
        # certificate = gConfig['certificate']
        # resource_dir = gConfig['resources']

        if ('train' in mode):
            # specify GPU numbers to use get gpu and cpu devices
            cpu_devices = get_cpu_devices()
            gpu_devices = get_gpu_devices()
            if (len(gpu_devices) > 1):
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gConfig["gpu_to_use"])

                print("The available GPU devices: " + str(gpu_devices))

                # devices, device_category = (gpu_devices, DeviceCategory.GPU) if len(gpu_devices) > 1 else (cpu_devices, DeviceCategory.CPU)

                # desc = "A Meta-Reinforcement Learning Approach to Optimise Parameters and Hyper-parameters Simultaneously"
                # parser = argparse.ArgumentParser(description=desc)
                #
                # parser.add_argument('--max_layers', default=2)
                #
                # args = parser.parse_args()
                # args.max_layers = int(args.max_layers)

            for dataset_ in dataset_name.split(','):     # datasets
                checkPathExists([model_dir + '/' + dataset_ + '/', data_dir, log_dir + '/' + dataset_ + '/', output_dir + '/' + dataset_ + '/'])

                # create logger
                _log.basicConfig(filename=log_dir + "/" + "log.txt", level=_log.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                logger = _log.getLogger("VoiceNet")
                logger.setLevel(_log.DEBUG)
                console = _log.StreamHandler()
                console.setLevel(_log.DEBUG)

                formatter = _log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # create formatter
                console.setFormatter(formatter)
                logger.addHandler(console)

                data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

                # dataset preprocessing
                if 'mnist' == dataset_:
                    input_dimensions = '28x28x1'
                    num_classes = 10
                    max_depth = 9
                    train_batch_size = 32
                    # train_num_epochs = 20

                    # dataset = mnist_dataset.read_data_sets(data_dirs + '/', one_hot=True)
                    # train_x, train_y, test_x, test_y = np.reshape(mnist_dataset.train_images(data_dirs), [-1, 784]), mnist_dataset.train_labels(data_dirs), \
                    #                                    np.reshape(mnist_dataset.test_images(data_dirs), [-1, 784]), mnist_dataset.test_labels(data_dirs)
                    (train_x, train_y), (test_x, test_y) = mnist.load_data()

                    train_x = np.expand_dims(train_x, axis=-1)
                    test_x = np.expand_dims(test_x, axis=-1)
                        
                    train_x, test_x = normalize(train_x, test_x)

                    train_y = to_categorical(train_y, num_classes)
                    test_y = to_categorical(test_y, num_classes)

                    if('channels_first' in data_format):
                        train_x = train_x.transpose(0, 3, 1, 2)
                        test_x = test_x.transpose(0, 3, 1, 2)

                    num_episodes = (len(train_x) // train_batch_size) * 100         # episodes = num_steps * num_epochs

                elif 'fashion_mst' == dataset_:
                    input_dimensions = '28x28x1'
                    num_classes = 10
                    max_depth = 9
                    # num_episodes = 30000
                    train_batch_size = 32
                    # train_num_epochs = 25

                    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

                    train_x = np.expand_dims(train_x, axis=-1)
                    test_x = np.expand_dims(test_x, axis=-1)

                    train_x, test_x = normalize(train_x, test_x)

                    train_y = to_categorical(train_y, num_classes)
                    test_y = to_categorical(test_y, num_classes)

                    if('channels_first' in data_format):
                        train_x = train_x.transpose(0, 3, 1, 2)
                        test_x = test_x.transpose(0, 3, 1, 2)

                    num_episodes = (len(train_x) // train_batch_size) * 100  # episodes = num_steps * num_epochs
                elif 'cifar10' == dataset_:
                    input_dimensions = '32x32x3'
                    num_classes = 10
                    max_depth = 9
                    # num_episodes = 35000
                    train_batch_size = 32
                    # train_num_epochs = 25

                    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

                    train_x, test_x = normalize(train_x, test_x)

                    train_y = to_categorical(train_y, num_classes)
                    test_y = to_categorical(test_y, num_classes)

                    if('channels_last' in data_format):
                        train_x = train_x.transpose(0, 2, 3, 1)
                        test_x = test_x.transpose(0, 2, 3, 1)

                    num_episodes = (len(train_x) // train_batch_size) * 120  # episodes = num_steps * num_epochs
                elif 'cifar100' == dataset_:
                    input_dimensions = '32x32x3'
                    num_classes = 100
                    max_depth = 18
                    train_batch_size = 32
                    # num_episodes = 60000
                    # train_num_epochs = 35

                    (train_x, train_y), (test_x, test_y) = cifar100.load_data()

                    train_x, test_x = normalize(train_x, test_x)

                    train_y = to_categorical(train_y, num_classes)
                    test_y = to_categorical(test_y, num_classes)

                    if('channels_last' in data_format):
                        train_x = train_x.transpose(0, 2, 3, 1)
                        test_x = test_x.transpose(0, 2, 3, 1)

                    num_episodes = (len(train_x) // train_batch_size) * 150  # episodes = num_steps * num_epochs
                elif 'tiny_imagenet' == dataset_:
                    input_dimensions = '64x64x3'
                    num_classes = 200
                    max_depth = 18
                    train_batch_size = 32
                    # num_episodes = 80000
                    # train_num_epochs = 30

                    (train_x, train_y), (test_x, test_y) = tiny_imagenet.load_data(data_dir + '/' + dataset_)

                    train_x, test_x = normalize(train_x, test_x)

                    train_y = to_categorical(train_y, num_classes)
                    test_y = to_categorical(test_y, num_classes)

                    if('channels_last' in data_format):
                        train_x = train_x.transpose(0, 2, 3, 1)
                        test_x = test_x.transpose(0, 2, 3, 1)

                    num_episodes = (len(train_x) // train_batch_size) * 180  # episodes = num_steps * num_epochs

                np.random.seed(777)
                np.random.shuffle(train_x)
                np.random.seed(777)
                np.random.shuffle(train_y)

                dataset = [train_x, train_y, test_x, test_y]  # pack the dataset for the Network Manager

                train(dataset, dataset_name = dataset_, model_dir = model_dir + '/' + dataset_ + '/', num_episodes = num_episodes, max_depth = max_depth, initial_filters = initial_filters,
                      num_layers = num_layers, num_hidden = num_hidden, initial_learning_rate = learning_rate, learning_rate_decay_factor = learning_rate_decay_factor,
                      train_batch_size = train_batch_size, test_batch_size = 1, train_num_epochs = train_num_epochs, input_dimensions = input_dimensions,
                      num_classes = num_classes, optimizer = optimizer, num_steps_per_decay=num_steps_per_decay, num_child_steps_per_cycle=num_child_steps_per_cycle,
                      exploration = exploration, discount_factor = discount_factor, log_dir = log_dir + '/' + dataset_ + '/', output_dir = output_dir + '/' + dataset_ + '/', logger = logger)

        # elif ('test' in mode):
        #     # 61, 24, 60,  5, 57, 55, 59, 3
        #     evaluate("5, 32, 2,  5, 3, 64, 2, 3", "model", data_dirs)
        elif ('analysis' in mode):
            plt.figure(figsize=(10, 10))

            plt.rcParams.update({'font.size': 6})

            count = 1
            for dataset_ in dataset_name.split(','):  # datasets
                # checkPathExists(['plots/' + dataset_ + '/'])
                checkPathExists(['plots/'])
                dataset = load_dataset(output_dir + '/' + dataset_ + '/' + dataset_ + '_results.csv')
                plot(dataset['policy_episode'], dataset['policy_loss'], dataset['reward'], dataset['network_accuracy'], 0, 10000, "Episodes", "Policy Loss", dataset_.replace('_', ' '), count, "plots/" + dataset_ + "/episod_accuracy.png")
                count += 1

            # plt.savefig("plots/results.png")
            # plt.gca().yaxis.set_minor_formatter(NullFormatter())
            # Adjust the subplot layout, because the logit one may take more space
            # than usual, due to y-tick labels like "1 - 10^{-3}"
            plt.subplots_adjust(top=0.92, bottom=0.22, left=0.1, right=0.6, hspace=0.25, wspace=0.35)

            plt.savefig("plots/results.pdf", bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.rcParams.update({'font.size': 8})
            dataset = load_dataset(output_dir + '/cifar10/cifar10_results.csv')
            plot_cifar10(dataset['time_taken'], dataset['network_accuracy'], 0, 720, "Time (minutes)", "Network validation accuracy (%)", '', "plots/cifar10_time_accuracy.pdf")

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex