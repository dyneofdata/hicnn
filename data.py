from __future__ import division
import numpy as np
import math
import bitstring
import csv
#from string import maketrans # for python 2.6

class Data:

    def __init__(self, chromosome, cell, batch_size):
        self.input_seq = 'input/chr' + chromosome + '/raw_sequence.txt'
        self.input_train = ['input/chr' + chromosome + '/' + cell + '/train' + str(fold) + '.txt' for fold in range(5)]
        self.input_tune = ['input/chr' + chromosome + '/' + cell + '/test' + str(fold) + '.txt' for fold in [0,2,4]]
        self.input_test = ['input/chr' + chromosome + '/' + cell + '/test' + str(fold) + '.txt' for fold in [1,3]]
        self.batch_size = batch_size
        self.distance_scaler = 40

    def generate_acgt(self):
        numRegions = 500000
        shape = (numRegions, 625, 4)

        self.acgt = np.zeros(shape = shape)
        
        o = 'AaCcGgTtN'
        a = '110000000'
        c = '001100000'
        g = '000011000'
        t = '000000110'
        l = [a,c,g,t]

        with open(self.input_seq) as seqData:
            for line in seqData:
                region = line.split('\t')
                idx = int(int(region[0].split('_')[1])/5000)
                bits = np.zeros(shape = (4,625))
                for nt, bit in enumerate(l):
                    trans = str.maketrans(o, bit) # remove str. for python 2
                    s = region[1][:-2].translate(trans)
                    b = bitstring.BitArray(bin = s)
                    bits[nt] = [float(x) for x in b.tobytes()] # ord(x) for python 2
                self.acgt[idx] = bits.T

    def generate_train(self):
        i = 0
        X1 = np.zeros(shape = (self.batch_size, 625, 4))
        X2 = np.zeros(shape = (self.batch_size, 625, 4))
        X3 = np.zeros(self.batch_size)
        Y = np.zeros(self.batch_size)
        while 1:
            for file in self.input_train:
                f = open(file)
                next(f)
                for line in f:
                    if i == self.batch_size:
                        i = 0
                        yield([X1, X2, X3], Y)
                    example = line.split('\t')
                    e = int(example[0])
                    p = int(example[1])
                    X1[i] = self.acgt[e]
                    X2[i] = self.acgt[p]
                    X3[i] = float(p - e) #/ self.distance_scaler
                    Y[i] = float(example[2])
                    i += 1
                f.close()

    def generate_tune(self):
        i = 0
        X1 = np.zeros(shape = (self.batch_size, 625, 4))
        X2 = np.zeros(shape = (self.batch_size, 625, 4))
        X3 = np.zeros(self.batch_size)
        Y = np.zeros(self.batch_size)
        while 1: 
            for file in self.input_tune:
                f = open(file)
                next(f)
                for line in f:
                    if i == self.batch_size:
                        i = 0
                        yield([X1,X2,X3], Y)
                    example = line.split('\t')
                    e = int(example[0])
                    p = int(example[1])
                    X1[i] = self.acgt[e]
                    X2[i] = self.acgt[p]
                    X3[i] = float(p - e) #/ self.distance_scaler
                    Y[i] = float(example[2])
                    i += 1
                f.close()

    def generate_test(self, steps):
        self.Y_ripple = np.zeros(self.batch_size * steps) 
        self.Y_actual = np.zeros(self.batch_size * steps)
        self.distance = np.zeros(self.batch_size * steps, dtype = np.int)
        i = 0
        cnt = 0
        X1 = np.zeros(shape = (self.batch_size, 625, 4))
        X2 = np.zeros(shape = (self.batch_size, 625, 4))
        X3 = np.zeros(self.batch_size)
        while 1:
            for file in self.input_test:
                f = open(file)
                next(f)
                for line in f:
                    if i == self.batch_size:
                        i = 0
                        yield([X1, X2, X3])    
                    example = line.split('\t')
                    e = int(example[0])
                    p = int(example[1])
                    X1[i] = self.acgt[e]
                    X2[i] = self.acgt[p]
                    X3[i] = float(p - e) #/ self.distance_scaler
                    if cnt < self.batch_size * steps:
                        self.Y_actual[cnt] = float(example[2])
                        self.Y_ripple[cnt] = float(example[4])
                        self.distance[cnt] = int(example[3])
                    i+= 1
                    cnt += 1
                f.close()

    def generate_test_whole(self):
        with open(self.input_test[0]) as testData:
            reader = csv.reader(testData, delimiter = '\t')
            testData = list(reader)
        testData.pop(0)
        numExamples = int(len(testData))
        #testData = sorted(testData, key = lambda x: int(x[3]))
        self.distance = np.array(testData)[:,3].astype(int)
        self.Y_actual = np.array(testData)[:,2].astype(float)
        self.Y_ripple = np.array(testData)[:,4].astype(float)

        shape = [numExamples] + [625] + [4]
        X1 = np.zeros(shape = shape)
        X2 = np.zeros(shape = shape)
        X3 = np.zeros(shape = numExamples)

        for idx, example in enumerate(testData):
            e = int(example[0])
            p = int(example[1])
            X1[idx] = self.acgt[e]
            X2[idx] = self.acgt[p]
            X3[idx] = float(p-e)

        return [X1, X2, X3]

