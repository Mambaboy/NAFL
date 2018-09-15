
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

from sklearn.model_selection import train_test_split


l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
coloredlogs.install(fmt=fmt)

cur_dir = os.path.abspath(os.path.dirname(__file__))

#for model
max_input_size  = 400
max_output_size = 6000  # this is the max
strides = 3
epochs = 1 
batch_size = 40
use_rate = 0.005
valid_rate=0.25
test_rate =0
#for collect
ignore_ts = 20
from_file = True  # data infor from


class Nmodel():
    def __init__(self, input_size, output_size, strides=1, batch_size=200, epochs=50, use_rate=1, valid_rate=0.25):
        
        self.input_size  =  input_size 
        self.output_size = output_size 

        self.strides     = strides 
        self.epochs      = epochs
        self.batch_size  = batch_size
       
        self.valid_rate = valid_rate 
        self.use_rate = use_rate # cut the input for some rate for test

        self.all_inputs_with_label = None # the data path
        self.total_sample_number = 0  #all the sample size, including the train and test
        
        self.train_inputs_with_label = None # the train data
        self.train_sample_number = 0  
    
        self.test_inputs_with_label = None # the test data 
        self.test_sample_number = 0  

        self.model       = Sequential()

    def create_model(self):
        self.model.add( Conv1D(64, 9, strides=self.strides,  padding="same", input_shape=(self.input_size, 1) ) )
        self.model.add( Activation("relu",name="relu1") )
       
        self.model.add( Conv1D(64*2, 9, strides=self.strides,  padding="same") ) 
        self.model.add( Activation("relu",name="relu2") )

        self.model.add( Conv1D(64*2*2, 9, activation='relu', strides= self.strides, padding="same") )
        self.model.add( Activation("relu",name="relu3") )

        self.model.add( Dropout(0.25) )

        self.model.add( Flatten() )
        self.model.add( Dense(self.output_size*1, activation='relu', name="full_connect_layer1") )
        
        #self.model.add(Dropout(0.25))
        
        #self.model.add(Dense( self.output_size, name="full_connect_layer2"))

        self.model.add( Activation("sigmoid",name="sigmoid") )

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def get_model_net(self):
        l.info(self.model.summary())
    
    def save_model(self):
        self.model.save("./model.h5")
        plot_model(self.model, to_file='model.png')
        
    
    def train_model(self):
        l.info("training the modle")
        self.model.fit_generator( self.generator_train_data_by_batch(), 
                                  steps_per_epoch = self.train_sample_number/self.batch_size, 
                                  epochs = self.epochs, verbose=1,
                                  validation_data = self.generator_test_data_by_batch(),
                                  validation_steps = self.test_sample_number/self.batch_size 
 )
        
    def evaluate_model(self):
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
   
    def _split_data(self):
        l.info("the test rate is %f", self.valid_rate)
        train_inputs_with_label, test_inputs_with_label = train_test_split( self.all_inputs_with_label, test_size = self.valid_rate)

        # abandon some to fit the batch size
        self.train_sample_number = int(len(train_inputs_with_label)/self.batch_size)*self.batch_size
        self.test_sample_number = int(len(test_inputs_with_label)/self.batch_size)*self.batch_size

        if self.test_sample_number == 0:
            l.warn("there are too little data, exit")
            exit(0)
        
        # the last one is not included
        self.train_inputs_with_label = train_inputs_with_label[0: self.train_sample_number]
        self.test_inputs_with_label = test_inputs_with_label[0: self.test_sample_number]
        

    def set_all_data(self , all_inputs_with_label):
        
        l.info("use %f samples for speed", self.use_rate)
        self.all_inputs_with_label = all_inputs_with_label[0:int(len(all_inputs_with_label)*self.use_rate)]
        self.total_sample_number = len(self.all_inputs_with_label)
        
        # split the train data and test data
        self._split_data()


    def _read_input_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()

        # control the length
        if len(content) >= self.input_size:
            content = content[0:self.input_size]
        else:
            content = ''.join( [content, b'\x00' * (self.input_size-len(content) ) ] )

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple, b means sign

        #normalization
        content = np.array(content).astype(np.float64)/255
        
        return content

    def _read_reduced_bitmap_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()
        
        # control the length
        if len(content) >= self.output_size:
            content = content[0:self.output_size]
        else:
            content = ''.join( [content,  b'\x00'* ( self.output_size-len(content)) ])

        # transform
        content = struct.unpack('B'*len(content), content) ## it is a tuple, B means unsing
        content = np.array(content)
        
        # normalization
        content = content.astype(np.bool).astype(np.int8)

        return content

    
    def read_samples_by_size(self, size=10 , start_index=0, train = True):
        '''
        start_index : fromt start_index to start_index+size
        '''
        inputs_data = np.ones( (1, self.input_size ,1) )
        labels_data = np.ones( (1, self.output_size ) )
        
        for i in xrange(size):
            if train:
                #l.info("gest index  %d", start_index+i )
                input_path  = self.train_inputs_with_label[start_index +i][0]
                reduce_bitmap_path = self.train_inputs_with_label[start_index +i][1]
            else:
                input_path = self.test_inputs_with_label[start_index +i][0]
                reduce_bitmap_path = self.test_inputs_with_label[start_index +i][1]
   
            input_content = self._read_input_content(input_path)
            input_content = np.reshape( input_content , (1, len(input_content), 1) )
            inputs_data = np.append( inputs_data, input_content, axis=0) 

            reduce_bitmap_content = self._read_reduced_bitmap_content(reduce_bitmap_path)
            reduce_bitmap_content = np.reshape( reduce_bitmap_content , (1, len(reduce_bitmap_content)) )
            labels_data = np.append( labels_data, reduce_bitmap_content, axis=0)

        inputs_data = np.delete(inputs_data, 0, axis=0)
        labels_data = np.delete(labels_data, 0, axis=0)
        
        #l.info("the length of train is %d", self.train_sample_number)
        #l.info("get the data from [%d to %d)", start_index, start_index +size )
        #l.info("inputs_data shape %s", inputs_data.shape)
        #l.info("labels_data shape %s", labels_data.shape)
        #l.info("\n")
        
        return (inputs_data, labels_data)
    
    
    def generator_train_data_by_batch(self):
        '''
        it is the generator for the fit_generator function
        '''
        start_index = 0
        while True:
            start_index %= self.train_sample_number
            inputs_data, labels_data = self.read_samples_by_size( size = self.batch_size, start_index = start_index, train= True )
            start_index += batch_size
            #l.info("generator train data from [%d, to %d)", start_index, start_index + self.batch_size)
            yield (inputs_data, labels_data)
    
    def generator_test_data_by_batch(self):
        '''
        it is the generator for the fit_generator function
        '''
        
        start_index = 0
        while True:
            start_index %= self.test_sample_number
            inputs_data, labels_data = self.read_samples_by_size( size = self.batch_size, start_index = start_index, train=False)
            start_index += batch_size
            #l.info("generator test data from [%d, to %d)", start_index, start_index + self.batch_size)
            yield (inputs_data, labels_data)


def test_part_data(  ):
    
    #engine = "fair" 
    engine = "afl"
    l.info("using the data from %s", engine)

    afl_work_dir = os.path.join(cur_dir, "output-"+engine )
    binary_path =  os.path.join(cur_dir, "benchmark/size")
    
    # init the collect 
    collect = Collect( afl_work_dir = afl_work_dir, binary_path = binary_path, ignore_ts =ignore_ts, 
                         from_file = from_file, engine = engine)

    reduce_output_size = collect.get_length_reduce_bitmap()
    l.info("the length of reduce bitmap is %d", reduce_output_size)
    
    output_size =reduce_output_size
    if output_size > max_output_size:
        output_size = max_output_size
        l.info("use the max output_size of %d", max_output_size)

    l.info("use the output_size of %d", output_size)

     
    #1. collect the path of each input
    l.info("begine to collect the data from %s", engine)
    l.info("the ignore ts is %d", ignore_ts)
    collect.collect_by_path()

    #2.init the model
    nmodel = Nmodel( input_size = max_input_size, output_size = output_size, 
                    strides = strides, epochs = epochs, batch_size = batch_size ,
                    use_rate =use_rate, valid_rate=valid_rate )

    nmodel.create_model()
    nmodel.get_model_net()
    #exit(0)    
    
    #3. read the content of each input
    all_inputs_with_label = collect.get_data()
    nmodel.set_all_data( all_inputs_with_label)

    # for test
    nmodel.read_samples_by_size(10)
    #exit(0)

    # train the model
    nmodel.train_model()

    # save the model
    #nmodel.save_model()


def main():
    test_part_data(  )

if __name__ == "__main__":
    main()
    



