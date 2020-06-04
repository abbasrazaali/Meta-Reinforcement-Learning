#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : training main function
#==============================================================================

import numpy as np
import csv
import time
from src.utils import get_residual_layer

from src.net_manager import NetManager
from src.reinforce import Reinforce
from src.utils import logging
import scipy as sc

def train(dataset, dataset_name, model_dir, num_episodes, max_depth, initial_filters, num_layers, num_hidden, initial_learning_rate, learning_rate_decay_factor, train_batch_size,
          test_batch_size, train_num_epochs, input_dimensions, num_classes, optimizer, num_steps_per_decay, num_child_steps_per_cycle, exploration, discount_factor,
          log_dir, output_dir, logger = None):
    try:
        # loading log
        # logging_dict = {}
        # csv_fields = ['dataset','policy_episode', 'policy_layers', 'policy_neurons', 'policy_loss', 'search_space', 'max depth of child network', 'lr optimizer',
        #               'child steps/episod', 'state', 'steps', 'reward', 'time taken']

        # csv_fields = ['dataset','pg_episode','pg_layers','pg_neurons','network_loss','network_search_space','network_state','network_reward','time_stamp','time_taken']
        # if (os.path.exists(output_dir + '/' + dataset_name + '_results.csv')):
        #     with open(output_dir + '/' + dataset_name + '_results.csv', mode='r') as csv_file:
        #         csv_reader = csv.DictReader(csv_file)
        #         for row in csv_reader:
        #             logging_dict[row['time_stamp']] = row['dataset'] + "|" + row['pg_episode'] + "|" + row['pg_layers'] + "|" + row['pg_neurons'] + "|" + \
        #                               row['network_loss'] + "|" + row['network_search_space'] + "|" + row['network_state'] + "|" + row['network_reward'] + "|" + \
        #                               row['time_stamp'] + "|" + row['time_taken']

        csv_fields = ['dataset', 'policy_episode', 'policy_layers', 'policy_neurons', 'policy_loss', 'policy_probs', 'lr optimizer', 'search_space', 'state',
                      'max depth of child network', 'child_steps_episod', 'steps', 'reward', 'network_accuracy', 'moving_accuracy', 'total_reward', 'gradients_entropty_avg',
                      'gradients_entropty_std', 'trainable_variables', 'time_taken']
        with open(output_dir + '/' + dataset_name + '_results.csv', 'w') as target_csv_file:
            writer = csv.DictWriter(target_csv_file, fieldnames=csv_fields)
            writer.writeheader()

        search_space_fields = ['depth','dropout','learning_rate','momentum']

        max_depth_layers, _, min_depth_layers, _ = get_residual_layer(max_depth)

        search_space = {'0': list(np.arange(min_depth_layers, sum(max_depth_layers) + 1, 1)),  # depth
                        '1': list(np.arange(0.05, 0.35, 0.05)),                     # 'dropout'
                        '2': list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.001, 0.01, 0.001)), # list(np.arange(0.1, 0.99, 0.1)) + list(np.arange(0.0001, 0.001, 0.0001)),   # 'learning_rate'
                        '3': list(np.arange(0.7, 0.99, 0.05))}                  # 'momentum' -> 0.6-0.99
                        # '4': list(np.arange(0, 1.0, 0.01)),                     # 'total_reward' -> 0-1
                        # '5': list(np.arange(0, 1.0, 0.01)),                    # 'loss' -> 0-1
                        # '6': list(np.arange(0, 1.0, 0.01)),                     # average entropy
                        # '7': list(np.arange(0, 1.0, 0.01))}                     # std entropy

        reinforce = Reinforce(model_dir, log_dir, initial_learning_rate = initial_learning_rate, num_hidden = num_hidden, num_layers = num_layers, search_space = search_space, num_steps_per_decay=num_steps_per_decay,
                              learning_rate_decay_factor = learning_rate_decay_factor, optimizer = optimizer, exploration = exploration, discount_factor = discount_factor, logger = logger)        # manages the training and evaluation of the Controller RNN
        net_manager = NetManager(search_space, input_dimensions=input_dimensions, num_classes=num_classes, dataset = dataset, dataset_name=dataset_name, log_dir=log_dir, train_batch_size = train_batch_size,
                                 test_batch_size = test_batch_size, train_num_epochs = train_num_epochs, max_depth=max_depth, num_child_steps_per_cycle=num_child_steps_per_cycle,
                                 initial_filters=initial_filters, model_dir = model_dir, logger = logger)   # handles the training and reward computation of a model

        # print("Search Space: ", search_space)
        prev_accuracy, prev_step, total_rewards, reward, network_loss, ent_avg, ent_std, elapsed_time = 0.0, 0, 0.01, 0.01, 0.0, 0.0, 0.01, 0.01

        # get_residual_layers, _ = get_residual_layer(max_depth)
        # max_channels = get_residual_filters(sum(get_residual_layers), min_channels)
        state = np.array([[0, 0, 0, 0]], dtype=np.int32)      # max_channels, network_loss
        # state = np.array([[0.05, 0.1, 0.7, 0.0, 0.0]], dtype=np.float32)      # max_channels, network_loss
        # state = np.array([[min(search_space['0']), min(search_space['1']), min(search_space['2']), min(search_space['3']), max_channels, prev_accuracy]], dtype=np.int32) # min_filters, accuracy
        # entropy = lambda p: -np.sum(p * np.log2(p))
        total_rewards = 0.0
        action_in_search_space = []
        for i_episode in range(num_episodes):
            # state_in_search_space = get_state_search(search_space, state[0])
            # state = [abs(x) for x in state]
            if(i_episode != 0):
                action = reinforce.get_action(state, init=False)
            else:
                action = reinforce.get_action(state, init=True)
            #     action = [[list(state[0])]]

            if all(ai >= 0 for ai in action[0][0]):
                start_time = time.time()

                action_in_search_space = [get_state_search(search_space, action[0][0], network_loss, ent_avg, ent_std, total_rewards)]     # max_channels, network_loss
                print("Actions: ", action_in_search_space[0][0])
                # print(action[0][0])

                reward, prev_accuracy, network_loss, prev_step, steps, probs, iterations, moving_accuracy, trainable_variables = net_manager.get_reward(action_in_search_space[0][0], prev_step, prev_accuracy)

                ent_avg = sum(sc.stats.entropy(probs)) / num_classes #/ (train_batch_size * num_child_steps_per_cycle * iterations)
                ent_std = np.std(sc.stats.entropy(probs)) * 10

                elapsed_time += (time.time() - start_time)
                print("Reward: " + str(reward) + " | Accuracy: " + str(prev_accuracy))
            else:
                reward = 0.01

            total_rewards += reward
            print('Total Reward: ' + str(round(total_rewards, 4)))

            # state_in_search_space = get_state_search(search_space, action_in_search_space[0][0])
            # max_channels = get_residual_filters(action_in_search_space[0][0][0], min_channels)
            state = action[0]
            # state[0][0] = round(total_rewards * 10, 0) if round(network_loss * 10, 0) > 0 else 0.01
            # state[0][1] = round(network_loss * 10, 0) if round(network_loss * 10, 0) > 0 else 0.01
            # state[0][2] = round(ent_avg, 0) if round(ent_avg, 0) > 0 else 0.01
            # state[0][3] = round(ent_std * 10, 0) if round(ent_std * 10, 0) > 0 else 0.1
            reinforce.storeRollout(state, reward)

            loss, log_probs = reinforce.train_step(1)

            # logging
            log_str = "time taken: " + str(elapsed_time / 60) + " | problem: " + dataset_name + " | episode: " + str(i_episode) + " | steps: " + str(steps) + " | loss: " + str(round(loss, 3)) + \
                      " | log_probs: " + str(log_probs) + " | state: " + str(action[0]) + " | reward: " + str(round(reward, 2)) + " | network accuracy: " + str(round(prev_accuracy * 100, 2)) + "\n"
            # logging_dict[i_episode] = dataset_name + "|" + str(i_episode) + "|" + str(num_layers) + "|" + str(num_hidden) + "|" + str(loss) + "|" + str(search_space_fields) + "|" + \
            #                           str(state[0]) + "|" + str(max_depth) + "|" + optimizer + "|" + str(num_child_steps_per_cycle) + "|" + str(steps) + "|" + str(reward) + "|" + \
            #                           str(datetime.datetime.now().time()).split('.')[0] + "|" + str(elapsed_time)
            print(log_str)

            # writing logs
            with open(output_dir + '/' + dataset_name + '_results.csv', 'a') as target_csv_file:
                writer = csv.writer(target_csv_file)

                writer.writerows([[dataset_name, str(i_episode), str(num_layers), str(num_hidden), str(round(loss * 100, 3)), str(log_probs), str(optimizer),
                                   str(search_space_fields), str(action_in_search_space[0][0]), str(max_depth), str(num_child_steps_per_cycle), str(steps),
                                   str(round(reward, 2)), str(round(prev_accuracy * 100, 2)), str(round(moving_accuracy * 100, 2)), str(round(total_rewards, 2)),
                                   str(round(ent_avg * 100, 2)), str(round(ent_std * 100, 2)), str(round(trainable_variables, 2)), str(round(elapsed_time, 2))]])
    except Exception as e:
        logging("Meta-RL training failed - " + str(e), logger, 'error')
    # finally:
    #     # writing logs
    #     with open(output_dir + '/' + dataset_name + '_results.csv', 'w') as target_csv_file:
    #         writer = csv.DictWriter(target_csv_file, fieldnames=csv_fields)
    #         writer.writeheader()
    #
    #         for key in sorted(logging_dict, key=logging_dict.get):
    #             values = key.split('|')
    #             writer.writerow({'dataset': values[0], 'policy_episode': values[1], 'policy_layers': values[2], 'policy_neurons': values[3], 'policy_loss': values[4],
    #                              'search_space': values[5], 'max depth of child network': values[6], 'lr optimizer': values[7], 'child steps/episod': values[8], 'state': values[9],
    #                              'steps': values[10], 'reward': values[11], 'time taken': values[12]})

def get_state_search(search_space, state, network_loss, ent_avg, ent_std, total_rewards):  # , max_channels, loss):
    state_in_search_space = []
    # for layer in range(1, num_layers  + 1):
    #     for key, value in search_space.items():
    for index in range(len(state)):
        values = search_space[str(index % len(search_space))]
        if(index == 0):
            state_in_search_space.append(values[state[index] % len(values)])
        elif(index == 1):
            state_in_search_space.append(round(values[state[index] % len(values)], 5))
        elif(index == 2 or index == 3):
            state_in_search_space.append(round(values[state[index] % len(values)], 3))
        # elif (index == 4):
        #     state_in_search_space.append(round(total_rewards * 10, 0) if round(total_rewards * 10, 0) > 0 else 0.01)
        # elif (index == 5):
        #     state_in_search_space.append(round(network_loss * 10, 0) if round(network_loss * 10, 0) > 0 else 0.01)
        # elif (index == 6):
        #     state_in_search_space.append(round(ent_avg, 0) if round(ent_avg, 0) > 0 else 0.01)
        # elif (index == 7):
        #     state_in_search_space.append(round(ent_std * 10, 0) if round(ent_std * 10, 0) > 0 else 0.01)

    return [state_in_search_space]


