
#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D 
import numpy as np
import keras
from keras.utils import plot_model
import coloredlogs
import logging
from DataDeal import *
import struct


l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
#install the coloredlogs
coloredlogs.install(fmt=fmt)

#for model
input_size  = 400
output_size = 65536
strides = 3
epochs = 1
batch_size = 200

#for collect
ignore_ts = 29
from_file = True


class Nmodel():
    def __init__(self, input_size, output_size, sample_size, strides=1, batch_size=200, epochs=50):
        self.input_size  =  input_size 
        self.output_size = output_size 
        self.sample_size =  sample_size 
        self.strides     = strides 
        self.epochs      = epochs
        self.batch_size  = batch_size

        self.inputs_with_label = None
        self.total_sample_number = 0

        self.model       = Sequential()

        self.inputs_train = None
        self.labels_train = None
        self.inputs_test  = None
        self.labels_test  = None

    def create_model(self):
        self.model.add(Conv1D(64, 9, activation='relu', strides= self.strides,  padding="same", input_shape=( self.input_size,1)))
        self.model.add(Conv1D(64, 9, activation='relu', strides= self.strides,  padding="same"))
        self.model.add(Conv1D(64, 9, activation='relu', strides= self.strides,  padding="same"))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense( self.output_size, name="full_connect_layer"))
        self.model.add(Dropout(0.25))
        self.model.add(Activation("sigmoid"))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def get_model_net(self):
        # print the struct of the model
        print(self.model.summary())
    

    def predict(self):
        # predict
        # print(data_1d.shape)
        # output = keras.Model(inputs=model.input, outputs=model.get_layer('full_connect_layer').output).predict(data_1d)
        pass
    
    def save_model(self):
        self.model.save("./model.h5")
        plot_model(self.model, to_file='model.png')
        
    def train_model(self):
        #self.model.fit(self.inputs_train, self.labels_train, batch_size=16, epochs=self.epochs)
        #score = self.model.evaluate(x_test, y_test, batch_size=16)
        
        self.model.fit_generator( self.generator_data_by_batch(), steps_per_epoch=self.sample_size/self.batch_size,
                                  epochs = self.epochs, verbose =1 )

    
    def set_all_data(self, inputs_with_label):
        self.inputs_with_label = inputs_with_label
        self.total_sample_number = len(self.inputs_with_label)

    #def set_train_data(self, inputs_train, labels_train):
    #    self.inputs_train = inputs_train 
    #    self.labels_train = labels_train

    #def set_test_data(self, inputs_test, labels_test):
    #    self.inputs_test = inputs_test
    #    self.labels_test = labels_test
    
    def _read_input_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()

        # control the length
        if len(content) >= self.input_size:
            content = content[0:self.input_size]
        else:
            content = content + b'\x00'*( self.input_size - len(content)) 

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple
        content = np.array(content)

        return content

    def _read_bitmap_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple
        content = np.array(content)

        return content

    
    def read_samples_by_size(self, size=10 , start_index=0):
        '''
        start_index : fromt start_index to start_index+size
        '''
        
        inputs_data = np.ones( (1, self.input_size ,1) )
        labels_data = np.ones( (1, self.output_size ) )
        
        for i in xrange(size):
            input_path = self.inputs_with_label[start_index +i][0]
            bitmap_path = self.inputs_with_label[start_index +i][1]
   
            input_content = self._read_input_content(input_path)
            input_content = np.reshape( input_content , (1, len(input_content),1) )
            inputs_data = np.append( inputs_data, input_content, axis=0) 

            bitmap_content = self._read_bitmap_content(bitmap_path)
            bitmap_content = np.reshape( bitmap_content , (1, len(bitmap_content)) )
            labels_data = np.append( labels_data, bitmap_content, axis=0)

        inputs_data = np.delete(inputs_data, 0, axis=0)
        labels_data = np.delete(labels_data, 0, axis=0)
        
        #l.info("get the data from %d to %d", start_index, start_index +size )
        #l.info("inputs_data shape %s", inputs_data.shape)
        #l.info("labels_data shape %s", labels_data.shape)
        
        return (inputs_data, labels_data)


    def generator_data_by_batch(self):
        '''
        it is the generator for the fit_generator function
        '''
        start_index = 0
        
        while True:
            inputs_data, labels_data = self.read_samples_by_size( size = self.batch_size, start_index =start_index )
            start_index += batch_size
            yield (inputs_data, labels_data)




#def try_test():
#    nmodel = Nmodel( input_size, output_size, sample_size, strides=3, epochs=10)
#    nmodel.create_model()
#    
#    x_train = np.random.randint( 2, size=(sample_size,input_size,1) )
#    y_train = np.random.randint( 2, size=(sample_size, output_size) )
#
#    nmodel.set_train_data( x_train, y_train )
#
#    nmodel.train_model()
#    nmodel.save_mode()


def test_part_data(  ):
    afl_work_dir = "/home/binzhang/NAFL/output-afl"
    binary_path = "/home/binzhang/NAFL/benchmark/size"
    
    # init the collect 
    collect = Collect( afl_work_dir =afl_work_dir, binary_path = binary_path, ignore_ts =ignore_ts, 
                        input_fix_len = input_size, from_file= from_file)
    
    #1. collect the path of each input
    collect.collect_by_path()
    sample_size = collect.get_total_samples() 
    sample_size = 10000 
    #2.init the model
    nmodel = Nmodel( input_size = input_size, output_size = output_size, sample_size = sample_size, 
                    strides = strides, epochs = epochs, batch_size = batch_size  )

    nmodel.create_model()
    
    #nmodel.get_model_net()
    
    #2. read the content of each input
    inputs_with_label = collect.get_data()
    nmodel.set_all_data( inputs_with_label)


    # for test
    nmodel.read_samples_by_size(1)


    # train the model
    nmodel.train_model()

    # save the model
    nmodel.save_model()


def main():
    test_part_data(  )
    #try_test()    
    #data_1d = np.random.randint( 2, size=(1,input_length,1) )

if __name__ == "__main__":
    main()
    



