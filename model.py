from __future__ import division
import numpy as np
import keras
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Flatten, Dropout, Concatenate, Activation
from keras.models import Model, load_model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from data import Data
import plot

class Model:

    def __init__(self, chromosome, cell):
        self.num_conv_layers = 3
        self.kernel_size = [5,10,15]
        self.stride = [1,2,3]
        self.num_filters = [10,5,1]
        
        self.num_dense_layers = 3
        self.dropout_prob = [0.5, 0.3, 0.3]
        self.hidden_dims = [300, 100, 50]
        
        self.data = Data(chromosome, cell, 32)
        self.data.generate_acgt()
        self.input_shape = [625] + [4]

        self.output_name = 'output/chr' + chromosome + '_' + cell
        self.image_title = 'Chromosome' + chromosome + ' - Cell Type: ' + cell

    def train(self):
        
        input_X1 = Input(shape = self.input_shape)
        input_X2 = Input(shape = self.input_shape)
        lone_input_shape = [1]
        input_X3 = Input(shape = lone_input_shape)

        conv_X1 = Conv1D(filters = self.num_filters[0],
                             kernel_size = self.kernel_size[0],
                             strides = self.stride[0],
                             activation = 'elu')(input_X1)
        conv_X2 = Conv1D(filters = self.num_filters[0],
                             kernel_size = self.kernel_size[0],
                             strides = self.stride[0],
                             activation = 'elu')(input_X2)
        
        for i in range(self.num_conv_layers)[1:]:
            conv_X1 = Conv1D(filters = self.num_filters[i],
                             kernel_size = self.kernel_size[i],
                             strides = self.stride[i],
                             activation = 'elu')(conv_X1)
            conv_X2 = Conv1D(filters = self.num_filters[i],
                             kernel_size = self.kernel_size[i],
                             strides = self.stride[i],
                             activation = 'elu')(conv_X2)

        conv_X1 = Flatten()(conv_X1)
        conv_X2 = Flatten()(conv_X2)

        dense = Concatenate()([conv_X1, conv_X2, input_X3])
        dense = Dropout(0.5)(dense)

        for i in range(self.num_dense_layers):
            dense = Dense(self.hidden_dims[i], activation = 'elu')(dense)
            dense = Dropout(self.dropout_prob[i])(dense)
        
        output = Dense(1)(dense)
                
        model = keras.models.Model(inputs = [input_X1, input_X2, input_X3], outputs = output)
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')

        early_stopping = EarlyStopping(monitor = 'val_loss', 
                                       patience = 10, 
                                       verbose = 1)
        csv_logger = CSVLogger(self.output_name + '.log')
        checkpoint = ModelCheckpoint(self.output_name + '.hdf5', 
                                     monitor = 'val_loss', 
                                     verbose = 0, 
                                     save_best_only = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                      factor = 0.1,
                                      patience = 10,
                                      min_lr = 0.0001)
        callbacks = [early_stopping, csv_logger, checkpoint, reduce_lr]
        
        model.fit_generator(self.data.generate_train(),
                                 steps_per_epoch = 10000, 
                                 epochs = 100,
                                 verbose = 2,
                                 callbacks = callbacks,
                                 validation_data = self.data.generate_tune(),
                                 validation_steps = 1000)

    def test(self):
        model = load_model(output_name + '.hdf5')
        steps = 10000
        Y_predict = model.predict_generator(self.data.generate_test(steps),
                                            workers = 1,
                                            steps = steps)
        Y_predict = Y_predict[:,0]
        
        plot.generate_plots(self.data.distance, 
                            self.data.Y_actual, 
                            Y_predict, 
                            self.data.Y_ripple, 
                            self.output_name, 
                            self.image_title)

