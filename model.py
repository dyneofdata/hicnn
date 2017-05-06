from data import Data
import plot
#TODO: import all dependencies

class Model:

    def __init__(self, chromosome, cell):
        self.num_conv_layers = 3
        self.kernel_size = [5,10,15]
        self.stride = [1,2,3]
        self.num_filters = [10,5,1]
        
        self.num_dense_layers = 3
        self.dropout_prob = [0.8, 0.5, 0.5]
        self.hidden_dims = [300, 100, 50]
        
        self.data = Data(chromosome, cell)
        self.input_shape = [625] + [4]

        self.output_name = 'output/chr' + chromosome + '_' + cell
        self.image_title = 'Chromosome' + chromosome + ' - Cell Type: ' + cell

    def train(self):
        conv_X1 = Input(shape = self.input_shape)
        conv_X2 = Input(shape = self.input_shape)
        input_X3 = Input(shape = (1,1)

        for i in range(num_conv_layers):
            conv_X1 = Conv1D(filters = self.num_filters[i],
                             kernel_size = self.kernel_size[i],
                             strides = self.stride[i])(conv_X1)
            conv_X1 = ELU()(conv_X1)
            conv_X2 = ConvID(filters = self.num_filters[i],
                             kernel_size = self.kernel_size[i],
                             strides = self.stride[i])(conv_X2)
            conv_X2 = ELU()(conv_X2)

        conv_X1 = Flatten()(convX1)
        conv_X2 = Flatten()(convX2)

        dense = Concatenate()([conv_X1, conv_X2, input_x3])

        for i in range(num_dense_layers):
            dense = Dense(hidden_dims[i])(dense)
            dense = Dropout(dropout_prob[i])(dense)
            dense = ELU()(dense)

        output = Dense(1)(dense)

        early_stopping = EarlyStopping(monitor = 'val_loss', 
                                       patience = 10, 
                                       verbose = 1)
        csv_logger = CSVLogger(self.output_name + '.log')
        checkpoint = ModelCheckpoint(self.output_name + '.hdf5', 
                                     monitor = 'val_loss', 
                                     verbose = 0, 
                                     save_best_only = True)
        callbacks = [early_stopping, csv_logger, checkpoint]

        model = Model(inputs = [conv_X1, conv_X2, input_X3], 
                           outputs = [output])
        model.compile(loss = 'mean_squared_error', 
                           optimizer = 'adam')
        model.fit_generator(self.data.generate_train(),
                                 steps_per_epoch = 10000, 
                                 epochs = 100,
                                 verbose = 2,
                                 callbacks = callbacks,
                                 validation_data = self.data.generate_tune(),
                                 validation_step = 1000)

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

