from __future__ import division
import numpy as np
import csv
import math
import sys
import scipy as sp
import matplotlib.pyplot as plt
import argparse
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.noise import GaussianNoise
from keras.layers import Dense, Flatten, Dropout, Concatenate
from keras.models import Model, load_model, Input
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
import plot

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

## Input & outpus:
chro = sys.argv[1]
cell = sys.argv[2]
fold = sys.argv[3]

input_seq = 'input/chr'+ chro + '/sequence.txt'
input_chr = 'input/chr'+ chro + '/' + cell + '/markers.txt'
input_train = 'input/chr'+ chro + '/' + cell + '/train' + fold + '.txt'
input_test = 'input/chr'+ chro + '/' + cell + '/test' + fold + '.txt'
output_name = 'output/chr' + chro + '_' + cell + '_f' + fold
image_title = 'Chromosome ' + chro + ' ' + cell + ' (Fold=' + fold + ')'

neighbors = 7
maxLength = neighbors * 2 + 1

## Sequence data & chromatin marker data (features):
with open(input_seq) as seqData:
	reader = csv.reader(seqData, delimiter= '\t')
	seqData = list(reader)
seqData.pop(0)

with open(input_chr) as chrData:
	reader = csv.reader(chrData, delimiter= '\t')
	chrData = list(reader)
chrData.pop(0)

numRegions = len(seqData) if len(seqData) > len(chrData) else len(chrData)

seq = np.zeros(shape=(numRegions,len(seqData[0])-1))
for region in seqData:
	seq[int(region[0])] = [float(i) for i in region[1:]]

chrom = np.zeros(shape=(numRegions,len(chrData[0])-1))
for region in chrData:
	chrom[int(region[0])] = [float(i) for i in region[1:]]

chrom = chrom / chrom.max(axis = 0)

features = np.concatenate((seq, chrom), axis = 1)

numChannels = len(features[0]) * 2
pad = np.zeros(shape = (1, int(numChannels/2)))

## Train data:
with open(input_train) as trainData:
	reader2 = csv.reader(trainData, delimiter = '\t')
	trainData = list(reader2)
trainData.pop(0)
numExamples = int(len(trainData))
	
shape = [numExamples] + [maxLength] + [numChannels]
X = np.zeros(shape = shape)
Y = np.array(trainData)[:,2].astype(float)
#W = np.zeros(numExamples)


for idx, example in enumerate(trainData):
        e = int(example[0])
        p = int(example[1])     
        e_start = 0 if (e - neighbors) < 0 else e - neighbors
        e_end = numRegions - 1 if (e + neighbors) >= numRegions else e + neighbors
        p_start = 0 if (p - neighbors) < 0 else p - neighbors
        p_end = numRegions - 1 if (p + neighbors) >= numRegions else p + neighbors 
        e_front_pad = -1 * (e - neighbors)
        e_back_pad  = (e + neighbors) - numRegions + 1 if (e + neighbors) >= numRegions else 0
        p_front_pad = -1 * (p - neighbors)
        p_back_pad  = (p + neighbors) - numRegions + 1 if (p + neighbors) >= numRegions else 0
        if e_front_pad > 0:
                E = np.concatenate((np.repeat(pad, e_front_pad, axis = 0), features[e_start:e_end+1,:]), axis=0)
        else:
                E = features[e_start:e_end+1,:]
        if e_back_pad > 0:
                E = np.concatenate((E, np.repeat(pad, e_back_pad, axis = 0)), axis = 0)
        if p_front_pad > 0:
                E = np.concatenate((np.repeat(pad, p_front_pad, axis = 0), features[p_start:p_end+1,:]), axis=0)
        else:
                P = features[p_start:p_end+1,:]
        if p_back_pad > 0:
                P = np.concatenate((P, np.repeat(pad, p_back_pad, axis = 0)), axis = 0)
        x = np.concatenate((E,P), axis = 1)
        X[idx] = x
        #W[idx] = sp.stats.norm.cdf(math.floor((p - e- 1)/4)) + 0.25

## Model:
filter_sizes = (3, 7)
num_filters = 10
dropout_prob = 0.5
hidden_dims = 50
shape = [maxLength] + [numChannels]

model_input = Input(shape=shape)
conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         strides=1)(model_input)
    conv = ELU()(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob)(z)
z = Dense(hidden_dims)(z)
z = ELU()(z)
model_output = Dense(1)(z)
model = Model(model_input, model_output)

adam = Adam(lr = 0.01)
model.compile(loss='mean_squared_error', optimizer = 'adam')

early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose = 1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, patience = 3, min_lr = 0.0001)
csv_logger = CSVLogger(output_name + '.log')
checkpoint = ModelCheckpoint(output_name + '.hdf5', monitor = 'val_loss', verbose = 0, save_best_only = True)
lrate = LearningRateScheduler(step_decay)

model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 2, callbacks = [early_stopping, lrate, csv_logger, checkpoint], validation_split = 0.1, shuffle = True)

## Test:
with open(input_test) as testData:
	reader3 = csv.reader(testData, delimiter = '\t')
	testData = list(reader3)
testData.pop(0)
numExamples = int(len(testData))
testData = sorted(testData, key = lambda x: int(x[3]))
pairDistance = np.array(testData)[:,3].astype(int)
shape = [numExamples] + [maxLength] + [numChannels]
X = np.zeros(shape = shape)
Y_actual = np.array(testData)[:,2].astype(float)

for idx, example in enumerate(testData):
        e = int(example[0])
        p = int(example[1])     
        e_start = 0 if (e - neighbors) < 0 else e - neighbors
        e_end = numRegions - 1 if (e + neighbors) >= numRegions else e + neighbors
        p_start = 0 if (p - neighbors) < 0 else p - neighbors
        p_end = numRegions - 1 if (p + neighbors) >= numRegions else p + neighbors 
        e_front_pad = -1 * (e - neighbors)
        e_back_pad  = (e + neighbors) - numRegions + 1 if (e + neighbors) >= numRegions else 0
        p_front_pad = -1 * (p - neighbors)
        p_back_pad  = (p + neighbors) - numRegions + 1 if (p + neighbors) >= numRegions else 0
        if e_front_pad > 0:
                e = np.concatenate((np.repeat(pad, e_front_pad, axis = 0), features[e_start:e_end+1,:]), axis=0)
        else:
                e = features[e_start:e_end+1,:]
        if e_back_pad > 0:
                e = np.concatenate((e, np.repeat(pad, e_back_pad, axis = 0)), axis = 0)
        if p_front_pad > 0:
                p = np.concatenate((np.repeat(pad, p_front_pad, axis = 0), features[p_start:p_end+1,:]), axis=0)
        else:
                p = features[p_start:p_end+1,:]
        if p_back_pad > 0:
                p = np.concatenate((p, np.repeat(pad, p_back_pad, axis = 0)), axis = 0)
        x = np.concatenate((e,p), axis = 1)
        X[idx] = x

model = load_model(output_name + '.hdf5')
Y_predict = model.predict(X)
Y_predict = Y_predict[:,0]

plot.plotPearsonCoefficient(pairDistance, Y_actual, Y_predict, output_name, image_title)
plot.plotScatter(pairDistance, Y_actual, Y_predict, output_name, image_title)
plot.plotMeanStdev(pairDistance, Y_actual, Y_predict, output_name, image_title)

print("Pearson Correlation:", np.asscalar(np.corrcoef(Y_actual, Y_predict)[:1,1:]))
#print("Explained Variance Score:", sk.metrics.explained_variance_score(Y_actual, Y_predict))
