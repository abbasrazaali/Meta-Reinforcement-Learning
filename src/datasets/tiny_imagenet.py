"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import os


def load_data(path):
    """Loads tiny_imagenet dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    num_classes = 200
    num_train_samples = 100000

    print('Loading ' + str(num_classes) + ' classes')

    x_train = np.empty((num_train_samples, 3, 64, 64), dtype=np.float32)
    y_train = np.empty((num_train_samples), dtype=np.float32)

    print('loading training images...')

    i, j = 0, 0
    annotations = {}
    trainPath = path + '/train'
    for sChild in os.listdir(trainPath):
        if(not '.DS_Store' in sChild):
            sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
            annotations[sChild] = j
            for c in os.listdir(sChildPath):
                X = np.array(Image.open(os.path.join(sChildPath, c)))
                if len(np.shape(X)) == 2:
                    x_train[i] = np.array([X, X, X])
                else:
                    x_train[i] = np.transpose(X, (2, 0, 1))
                y_train[i] = j
                i += 1
            j += 1
            if (j >= num_classes):
                break

    print('finished loading training images')

    val_annotations_map = get_annotations_map(path)

    x_test = np.zeros((num_classes * 50, 3, 64, 64), dtype='uint8')
    y_test = np.zeros((num_classes * 50), dtype='uint8')

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(path + '/val/images'):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                x_test[i] = np.array([X, X, X])
            else:
                x_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    # y_train = np.reshape(y_train, (len(y_train), 1))
    # y_test = np.reshape(y_test, (len(y_test), 1))

    # x_train = x_train.transpose(0, 2, 3, 1)
    # x_test = x_test.transpose(0, 2, 3, 1)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    print('finished loading test images')

    return (x_train, y_train), (x_test, y_test)

def get_annotations_map(path):
    valAnnotationsFile = open(path + '/val/val_annotations.txt', 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations