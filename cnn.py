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
from keras.layers import Dense, Flatten, Dropout, Concatenate
from keras.models import Model, load_model, Input, Sequential
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
import plot
import bitstring
from string import maketrans

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

## Input & outputs:
chro = sys.argv[1]
cell = sys.argv[2]
fold = sys.argv[3]
mode = sys.argv[4]

input_seq = 'input/chr'+ chro + '/raw_sequence.txt'
input_chr = 'input/chr'+ chro + '/' + cell + '/markers.txt'
input_train = 'input/chr'+ chro + '/' + cell + '/train' + fold + '.txt'
input_test = 'input/chr'+ chro + '/' + cell + '/test' + fold + '.txt'
output_name = 'output/chr' + chro + '_' + cell + '_f' + fold + '_' + mode
image_title = 'Chromosome ' + chro + ' ' + cell + ' (Fold=' + fold + ')'

## Sequence data (features):
with open(input_seq) as seqData:
	reader = csv.reader(seqData, delimiter= '\t')
	seqData = list(reader)
seqData.pop(0)
numRegions = 50000
shape = (numRegions, 625)
print shape
A = np.zeros(shape = shape)
C = np.zeros(shape = shape)
T = np.zeros(shape = shape)
G = np.zeros(shape = shape)

o = 'AaTtCcGgN'
a = '110000000'
t = '001100000'
c = '000011000'
g = '000000110'
l = [a, t, c, g]

for region in seqData:
	idx = int(int(region[0].split('_')[1])/5000)
	for nt, out in enumerate(l):
		trans = maketrans(o, out)
		s = region[1][:-2].translate(trans)
		b = bitstring.BitArray(bin = s)
		i = [float(ord(x)) for x in b.tobytes()]
		if nt == 0:
			A[idx] = i
		elif nt == 1:
			T[idx] = i
		elif nt == 2:
			C[idx] = i
		else:
			G[idx] = i 

## Train data:
with open(input_train) as trainData:
	reader2 = csv.reader(trainData, delimiter = '\t')
	trainData = list(reader2)
trainData.pop(0)
numExamples = int(len(trainData))
	
shape = [numExamples] + [625] + [4]
X1 = np.zeros(shape = shape)
X2 = np.zeros(shape = shape)
Y = np.array(trainData)[:,2].astype(float)

for idx, example in enumerate(trainData):
        e = int(example[0])
        p = int(example[1])     
	X1[idx] = np.array((A[e], C[e], G[e], T[e])).T
	X2[idx] = np.array((A[p], C[p], G[p], T[p])).T

## Model:
num_filters = 10
kernel_size = 3
dropout_prob = 0.5
hidden_dims = 50
shape = [625] + [4] 

# input layer
input1 = Input(shape=shape, name='input1')
input2 = Input(shape=shape, name='input2')

# convolution layer for input1
conv1 = Conv1D(filters=num_filters,
				kernel_size=kernel_size,
				padding="valid",
				strides=1)(input1)
conv1 = ELU()(conv1)

# convolution layer for input2
conv2 = Conv1D(filters=num_filters,
				kernel_size=kernel_size,
				padding="valid",
				strides=1)(input2)
conv2 = ELU()(conv2)

# separate pooling layers for two inputs
conv1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = MaxPooling1D(pool_size=2)(conv2)

conv1 = Flatten()(conv1)
conv2 = Flatten()(conv2)

# merge two streams
z = Concatenate()([conv1, conv2])
z = Dropout(dropout_prob)(z)
z = Dense(hidden_dims)(z)
z = ELU()(z)
model_output = Dense(1)(z)

# fit model
early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose = 1)
csv_logger = CSVLogger(output_name + '.log')
checkpoint = ModelCheckpoint(output_name + '.hdf5', monitor = 'val_loss', verbose = 0, save_best_only = True)
lrate = LearningRateScheduler(step_decay)

model = Model(inputs=[input1, input2], outputs=[model_output])
adam = Adam(lr = 0.01)
model.compile(loss='mean_squared_error', optimizer = 'adam')

model.fit([X1, X2], Y, batch_size = 32, epochs = 100, verbose = 2, callbacks = [early_stopping, lrate, csv_logger, checkpoint], validation_split = 0.1, shuffle = True)

## Test:
with open(input_test) as testData:
	reader3 = csv.reader(testData, delimiter = '\t')
	testData = list(reader3)
testData.pop(0)
numExamples = int(len(testData))
testData = sorted(testData, key = lambda x: int(x[3]))
pairDistance = np.array(testData)[:,3].astype(int)

shape = [numExamples] + [625] + [4]
X1 = np.zeros(shape = shape)
X2 = np.zeros(shape = shape)
Y_actual = np.array(testData)[:,2].astype(float)
Y_ripple = np.array(testData)[:,4].astype(float)

for idx, example in enumerate(testData):
        e = int(example[0])
	p = int(example[1])
	X1[idx] = np.array((A[e], C[e], G[e], T[e])).T
	X2[idx] = np.array((A[p], C[p], G[p], T[p])).T

model = load_model(output_name + '.hdf5')
Y_predict = model.predict([X1, X2])
Y_predict = Y_predict[:,0]

# Output:
plot.plotPearsonCoefficient(pairDistance, Y_actual, Y_predict, Y_ripple, output_name, image_title)
plot.plotScatter(pairDistance, Y_actual, Y_predict, output_name, image_title)
plot.plotMeanStdev(pairDistance, Y_actual, Y_predict, output_name, image_title)

print("Pearson Correlation:", np.asscalar(np.corrcoef(Y_actual, Y_predict)[:1,1:]))
