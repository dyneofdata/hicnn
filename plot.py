from __future__ import division
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as bstat
import sklearn as sk

def plotPearsonCoefficient(pairDistance, trueCount, predCount, output, title):
	pearson = []
	prev_idx = 0
	bins = np.linspace(pairDistance[0], pairDistance[-1], num = 11)
	for b in bins[1:-1]:
		idx = np.argmax(pairDistance > b)
		pearson.append(np.asscalar(np.corrcoef(trueCount[prev_idx:idx], predCount[prev_idx:idx])[:1,1:]))
		prev_idx = idx

	pearson.append(np.asscalar(np.corrcoef(trueCount[prev_idx:], predCount[prev_idx:])[:1,1:]))

	plt.xlim(0, bins[-1] + bins[1])
	plt.plot(bins[1:], pearson, marker = 'o', color = '#0E6655', label = 'CNN')
	plt.xlabel("Pairwise Distance (bp)")
	plt.ylabel("Pearson Coefficient")
	plt.legend(loc = 'best')
	plt.title(title)
	plt.savefig(output +'_pearson.png')
	plt.clf()

def plotScatter(pairDistance, trueCount, predCount, output, title):
	prev_idx = 0
	bins = np.linspace(pairDistance[0], pairDistance[-1], num = 11)
	binsize = bins[1] - bins[0]
	for b in bins[1:-1]:
                idx = np.argmax(pairDistance > b)
                plt.scatter(trueCount[prev_idx:idx], predCount[prev_idx:idx], label = str(int((b-binsize)/1000)) +'kbp - ' + str(int(b/1000)) + 'kbp')
                prev_idx = idx

	plt.xlabel("Actual Count")
	plt.ylabel("Predicted Count")
	plt.legend(loc='lower right')
	plt.title(title)
	plt.savefig(output +'_scatter.png')
	plt.clf()

def histogram(pairDistance, output, title):
        plt.hist(pair_dist)
        plt.xlabel('Pairwise Distance (bp)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.savefig(output + '_histogram')
        plt.clf()

def plotMeanStdev(pairDistance, trueCount, predCount, output, title):
        bins = np.linspace(pairDistance[0], pairDistance[-1], num = 11)
        true_count_mean, bin_edges, _  = bstat(pairDistance, trueCount, 'mean', bins = bins)
        true_count_stdev, _, _ = bstat(pairDistance, trueCount, 'std', bins = bins)
        pred_count_mean, _, _ = bstat(pairDistance, predCount, 'mean', bins = bins)
        pred_count_stdev, _, _ = bstat(pairDistance, predCount, 'std', bins = bins)
        plt.xlim(0, bins[-1] + bins[1])
        plt.errorbar(bins[1:], true_count_mean, true_count_stdev, linestyle = 'None', marker = 'o', color = 'g', label = 'Actual Count')
        plt.errorbar(bins[1:], pred_count_mean, pred_count_stdev, linestyle = 'None', marker = 'o', color = 'r', label = 'Predicted Count')
        plt.xlabel("Pairwise Distance (bp)")
        plt.ylabel("Interaction Count")
        plt.legend(loc = 'best')
        plt.title(title)
        plt.savefig(output +'_distribution.png')
        plt.clf()


