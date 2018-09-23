#coding=utf-8
import coloredlogs
import logging
from DataDeal import *
import struct
from collections import OrderedDict
import os

import keras
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras import  activations
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D 
import numpy as np

from sklearn.model_selection import train_test_split
from vis.visualization import visualize_saliency
from vis.utils import utils

from skimage import io

l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
coloredlogs.install(fmt=fmt)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cur_dir = os.path.abspath(os.path.dirname(__file__))

#for model
max_input_size  = 400
max_output_size = 6000  # this is the max
strides = 3
batch_size = 50
epochs = 30
use_rate = 1
valid_rate=0.25
test_rate =0.25
use_old_model=False # load model from file

#for collect
#engine = "fair" 
engine ="afl"
binary_path =  os.path.join(cur_dir, "benchmark/jhead-nb")
ignore_ts = 10  # if the number of samples for one class is smaller than it, ignore
from_file = False  # data infor from
reduce_use_old = False
l.info("using the data from %s", engine)

class Nmodel():
    def __init__(self, input_size, output_size,binary, strides=1, batch_size=200, epochs=50, use_rate=1, valid_rate=0.25, use_old_model=False):
        
        self.input_size  =  input_size 
        self.output_size = output_size 
        self.binary = binary

        self.strides     = strides 
        self.epochs      = epochs
        self.batch_size  = batch_size
       
        self.valid_rate = valid_rate 
        self.use_rate = use_rate # cut the input for some rate for test
        self.use_old_model = use_old_model # load model from file

        self.all_inputs_with_label = None # the data path
        self.total_sample_number = 0  #all the sample size, including the train and test
        
        self.train_inputs_with_label = None # the train data
        self.train_sample_number = 0  
    
        self.test_inputs_with_label = None # the test data 
        self.test_sample_number = 0  

        self.model       = Sequential()
        self.model_file_path =  os.path.join(cur_dir, self.binary+"-model.h5")

        self.useful_index = None

    def set_useful_index(self, useful_index):
        self.useful_index = np.array(useful_index)

    def create_model(self):
        self.model.add( Conv1D(32, 9, strides=self.strides,  padding="same", input_shape=(self.input_size, 1) ) )
        self.model.add( Activation("relu",name="relu1") )
       
        self.model.add( Conv1D(32*2, 9, strides=self.strides,  padding="same") ) 
        self.model.add( Activation("relu",name="relu2") )

        self.model.add( Conv1D(32*2*2, 9, activation='relu', strides= self.strides, padding="same") )
        self.model.add( Activation("relu",name="relu3") )

        self.model.add( Dropout(0.25) )

        self.model.add( Flatten() )
        
        self.model.add( Dense(self.output_size*2, activation='relu', name="full_connect_layer1") )
        
        self.model.add(Dropout(0.25))
        
        self.model.add(Dense( self.output_size, name="full_connect_layer2"))

        self.model.add( Activation("sigmoid",name="sigmoid") )

        # the accuracy and optimizer
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  

    def get_model_net(self):
        self.model.summary()
    
    def save_model(self):
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
        self.model.save(self.model_file_path)
        #plot_model(self.model, to_file='model.png')

    def load_model(self):
        self.model =load_model(self.model_file_path)
        
    def train_model(self):

        if self.use_old_model and os.path.exists(self.model_file_path):
            l.info("load old model from file")
            self.load_model()
            return

        l.info("training the modle")
        self.model.fit_generator( self.generator_train_data_by_batch(), 
                                  steps_per_epoch = self.train_sample_number/self.batch_size, 
                                  epochs = self.epochs, verbose=1,
                                  validation_data = self.generator_test_data_by_batch(),
                                  validation_steps = self.test_sample_number/self.batch_size )
        # save the model
        self.save_model()
        
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
        
        content = bytearray(content)
        # control the length
        if len(content) < self.input_size:
            content.extend( [0] * (self.input_size - len(content) ) )
        elif len(content) > self.input_size:
            content = content[0:self.input_size ] 

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple, b means sign

        #normalization
        content = np.array(content).astype(np.float64)/255 # between -1 1
        
        return content

    def _read_label_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()
        
        # control the length
        if len(content) < self.output_size:
            content.extend( [0] * ( self.output_size - len(content) ) )
        elif len(content)  > self.output_size :
            content = content[0:self.output_size ]

        # transform
        content = struct.unpack('B'*len(content), content) ## it is a tuple, B means unsing

        # normalization
        content = (np.array(content) >0 ).astype(np.int8)

        return content

    
    def read_samples_by_size(self, size=10 , start_index=0, train = True):
        '''
        start_index : fromt start_index to start_index+size
        '''
        inputs_data = np.ones( (1, self.input_size ,1) )
        labels_data = np.ones( (1, self.output_size ) )
        #l.info("self.input_size is %d", self.input_size) 
        for i in range(size):
            if train:
                #l.info("gest index  %d", start_index+i )
                input_path  = self.train_inputs_with_label[start_index +i][0]
                reduce_bitmap_path = self.train_inputs_with_label[start_index +i][1]
            else:
                input_path = self.test_inputs_with_label[start_index +i][0]
                reduce_bitmap_path = self.test_inputs_with_label[start_index +i][1]
  
            #for test
            if size ==1:
                l.info("input_path: %s",input_path)
                l.info("reduce_bitmap_path: %s", reduce_bitmap_path)

            input_content = self._read_input_content(input_path)
            input_content = np.reshape( input_content , (1, len(input_content), 1) )
            inputs_data = np.append( inputs_data, input_content, axis=0) 

            reduce_bitmap_content = self._read_label_content(reduce_bitmap_path)
            reduce_bitmap_content = np.reshape( reduce_bitmap_content , (1, len(reduce_bitmap_content)) )
            
            # for test
            #num= Collect.sum_non_values_in_content(reduce_bitmap_content[0], 0) 
            #l.info("there is %d 1 in the trace of %s", num, reduce_bitmap_path)
            
            labels_data = np.append( labels_data, reduce_bitmap_content, axis=0 )

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


    def turn_to_binary_value_by_ts(self, content, ts):
        '''
        value <= ts is 0
        value > tx is 1
        '''
        content = np.array( content)
        content = (content > ts).astype(np.int8) 
        return content

    def evaluate(self, size=10):
        inputs_data,labels_data = self.read_samples_by_size(size , train = False)
        result = self.model.evaluate(inputs_data, labels_data, verbose =1)
        return result

    def predict(self, size =30):
        inputs_data,labels_data = self.read_samples_by_size(size , train = False)
        result = self.model.predict( inputs_data, batch_size =5, verbose=1)
      
        for i in range(size):
            num=0
            a=list()
            for index in range(len(labels_data[i])):
                if labels_data[i][index] ==1:
                    num+=1
                    a.append(index)
            l.info("there is %d number 1 in the label", num)
            #l.info(a)
            for ts in [0.5]:
                num=0
                b=list()
                temp_result = self.turn_to_binary_value_by_ts(result[i], ts)
                for index in range(len(temp_result)):
                    if temp_result[index] == 1:
                        num+=1
                        b.append(index)
                #l.info(b)
                l.info("there is %d number large than %f", num, ts)

                d1=np.sum(np.abs(labels_data[i] - temp_result))
                l.info("the Manhattan Distance is %d", d1)
            l.info("")
   
    def plot_saliency(self,data):
        data_length = len(data)
        
        #add the second 
        temp=np.reshape(data, (data_length,1)  )
        temp=np.repeat(temp, data_length/2, axis=1)

        #add the third
        temp=np.reshape(temp, (data_length,data_length/2,1))
        temp=np.repeat(temp, 3,axis=2)

        # turn the color
        temp=255-temp
        dst=io.imshow(temp)
        io.show()

    @staticmethod
    def get_index_max_value( result):
        '''
        get the index for the k max value
        '''
        key_locations = OrderedDict()
        temp_result = np.copy(result)
        for i in xrange(30):
            max_value = np.max(temp_result)
            if max_value == 0:
                continue
            indexs = np.where( temp_result == max_value)
            #l.info("get the max value %s at %s", max_value, indexs)
            key_locations[max_value] = indexs
            temp_result[indexs] =0
            #l.info("now it is %s",  temp_result)

        #l.info(key_locations)
        for k, v in key_locations.items():
                print( k, v)

    def saliency(self, input_path, branch_id):
        if os.path.exists(input_path):
            l.info("grads for the %d transition for %s", branch_id, input_path)
            input_content = self._read_input_content(input_path)
            input_content = np.reshape(input_content,  (1, len(input_content), 1) )
        else:
            l.info("there is not %s, use one sample", input_path)
            input_content, _ = self.read_samples_by_size(1)

        output_index = np.where(self.useful_index == branch_id)
        if len(output_index[0]) == 0:
            l.warn("%s do not meet transition of %d", input_path, branch_id)
            return None

        layer_idx = utils.find_layer_idx(self.model, 'sigmoid')
        # Swap sigmoid with linear
        self.model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(self.model)

        #saliency
        result = visualize_saliency(model, layer_idx=layer_idx, filter_indices=output_index[0], seed_input = input_content, backprop_modifier=None,  grad_modifier="absolute")

        #plot the result
        self.plot_saliency(result) 
        
        #l.info(result)
        return result
        
def start():
   
    binary =os.path.basename(binary_path)
    afl_work_dir = os.path.join( "/tmp", "output-"+engine+'-'+binary )
    
    l.info("using the data from %s", engine)
    l.info("deal with the binary %s", binary)

    # 0.init the collect 
    collect = Collect( afl_work_dir = afl_work_dir, binary_path = binary_path, ignore_ts =ignore_ts, 
                         from_file = from_file, engine = engine, reduce_use_old =False)

    # 1.
    reduce_output_size = collect.get_length_reduce_bitmap()
    l.info("the length of reduce bitmap is %d", reduce_output_size)
    output_size =reduce_output_size
    if output_size > max_output_size:
        output_size = max_output_size
        l.info("use the max output_size of %d", max_output_size)
    l.info("at last, use the output_size of %d", output_size)
     
    # 2. collect the path of each input
    l.info("the ignore ts is %d", ignore_ts)
    l.info("begine to collect the data from %s", engine)
    collect.collect_by_path()

    # 3.init the model
    nmodel = Nmodel( input_size = max_input_size, output_size = output_size, binary=binary,
                    strides = strides, epochs = epochs, batch_size = batch_size ,
                    use_rate =use_rate, valid_rate=valid_rate, use_old_model =use_old_model )

    nmodel.create_model()
    #nmodel.get_model_net()
    
    # 4. send data to the model
    all_inputs_with_label = collect.get_data()
    nmodel.set_all_data( all_inputs_with_label)
    
    useful_index = collect.get_useful_index()
    nmodel.set_useful_index(useful_index)

    # for test
    nmodel.read_samples_by_size(1)

    # train the model
    nmodel.train_model( )

    l.info("begin to predict")
    nmodel.predict()

    #l.info("begin to evalute")
    #evaluate_result = nmodel.evaluate()

    #check_input_path= "/tmp/afl-nb/jhead/queue/id:000091,src:000026+000028,op:splice,rep:8,+cov"
    #result = nmodel.saliency(check_input_path, 10577 )
    #if not result is None:
    #    nmodel.get_index_max_value(result)

def main():
    start(  )

if __name__ == "__main__":
    main()
    



