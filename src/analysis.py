#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 01, 2018
# Description  : analysis
#==============================================================================

import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

# create and export a plot
def load_dataset(dataset_name):
    try:
        data = pd.read_csv(dataset_name)

        return data
    except Exception as e:
        print("Dataset loading failed - " + str(e))

# create and export a plot
def plot(x_axis, y_axis_1, y_axis_2, y_axis_3, x_min_range, x_max_range, xlabel, ylabel, title, count, filename):
    try:
        # plt.figure()
        plt.subplot(3, 2, count)

        plt.xlim(x_min_range, x_max_range)
        plt.ylim(0, 100)

        y_axis_1 = [i * 100 / max(y_axis_1.tolist()) for i in y_axis_1.tolist()]
        y_axis_2 = [i * 100 / max(y_axis_2.tolist()) for i in y_axis_2.tolist()]
        y_axis_3 = [i for i in y_axis_3.tolist()]
        # y_axis_4 = [i * 100 / max(y_axis_4.tolist()) for i in y_axis_4.tolist()]

        sigma = 10
        ysmoothed_1 = gaussian_filter1d(y_axis_1, sigma=sigma)
        ysmoothed_2 = gaussian_filter1d(y_axis_2, sigma=sigma)
        ysmoothed_3 = gaussian_filter1d(y_axis_3, sigma=sigma)
        # ysmoothed_4 = gaussian_filter1d(y_axis_4, sigma=sigma)

        plt.plot(x_axis.tolist()[x_min_range:x_max_range], ysmoothed_1[x_min_range:x_max_range])
        plt.plot(x_axis.tolist()[x_min_range:x_max_range], ysmoothed_2[x_min_range:x_max_range])
        plt.plot(x_axis.tolist()[x_min_range:x_max_range], ysmoothed_3[x_min_range:x_max_range])
        # plt.plot(x_axis.tolist()[x_min_range:x_max_range], ysmoothed_4[x_min_range:x_max_range])

        # x_sm = np.array(x_axis.tolist()[x_min_range:x_max_range])
        # y_sm = np.array(y_axis[x_min_range:x  _max_range])
        #
        # x_smooth = np.linspace(x_sm.min(), x_sm.max(), 300)
        # y_smooth = spline(x_axis.tolist()[x_min_range:x_max_range], y_axis.tolist()[x_min_range:x_max_range], x_smooth)
        #
        # plt.plot(x_smooth, y_smooth)
        y_ysmoothed_3_1 = list(ysmoothed_3)
        y_max = max(y_ysmoothed_3_1)
        y_index = y_ysmoothed_3_1.index(y_max)
        plt.vlines(x=x_axis[y_index], ymin=0, ymax=y_max, color='red', zorder=2, linestyles='dotted')

        plt.legend(['Policy Loss (%)', 'Reward (%)', 'Network Accuracy (%)', '(' + str(int(x_axis[y_index])) + ',' + str(round(max(y_axis_3), 1)) + ')'], loc='upper left', prop={'size': 4})
        # plt.legend()

        plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title, fontweight="bold", horizontalalignment='right', verticalalignment='center')
        plt.text(1.02, 0.5, title, horizontalalignment='left', fontweight="bold", verticalalignment='center', rotation=90, clip_on=False, transform=plt.gca().transAxes)

        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(filename)
        # plt.close()
    except Exception as e:
        print("Plot failed - " + str(e))

# create and export a plot
def plot_cifar10(x_axis, y_axis, x_min_range, x_max_range, xlabel, ylabel, title, filename):
    try:
        plt.figure()

        plt.xlim(x_min_range, x_max_range)
        plt.ylim(0, 100)

        x_axis = [i / 60 for i in x_axis.tolist()]
        y_axis = [i for i in y_axis.tolist()]

        sigma = 5
        ysmoothed = gaussian_filter1d(y_axis, sigma=sigma)

        # plt.plot(x_axis[x_min_range:x_max_range], ysmoothed[x_min_range:x_max_range])
        plt.plot(x_axis, ysmoothed)

        ysmoothed_1 = list(ysmoothed)
        y_max = max(ysmoothed_1)
        y_index = ysmoothed_1.index(y_max)
        plt.vlines(x=x_axis[y_index], ymin=0, ymax=y_max, color='red', zorder=2, linestyles='dotted')
        # plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.title(title, fontweight="bold")
        # plt.text(1.02, 0.5, title, horizontalalignment='left', fontweight="bold", verticalalignment='center', rotation=90, clip_on=False, transform=plt.gca().transAxes)

        plt.legend(['1 Nvidia 1080ti GPU', '(' + str(int(x_axis[y_index])) + ',' + str(round(max(y_axis), 1)) + ')'], loc='upper right')

        plt.grid(True)
        # plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Cifar10 Plot failed - " + str(e))