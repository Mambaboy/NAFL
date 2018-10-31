#coding=utf-8
import coloredlogs
import logging
from cnnafl.datadeal import *
import struct
from collections import OrderedDict
import os

from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras import  activations
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D 
import numpy as np

from keras import backend as K
K.clear_session()

from sklearn.model_selection import train_test_split

from vis.visualization import visualize_saliency
from vis.utils import utils

from skimage import io
import time
import cProfile
import math

l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
coloredlogs.install(fmt=fmt)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cur_dir = os.path.abspath(os.path.dirname(__file__))

#for model
max_input_size  = 1000
max_output_size = 6000  # this is the max
strides = 3 
batch_size = 500
epochs = 10
use_rate = 1
valid_rate=0.25
test_rate =0.25
use_old_model= False # load model from file

#for collect
#engine = "fair" 
engine ="fair"
binary =  "3D_Image_Toolkit"
ignore_ts = 0   # if the number of samples for one class is smaller than it, ignore
from_file = True # data infor from
reduce_use_old = True

if use_old_model==True:
    from_file=True
    reduce_use_old=True
num=0
from keras.utils import Sequence
class SequenceData(Sequence):
    def __init__(self,  batch_size, input_size, output_size, inputs_with_label, tag):
        # 初始化所需的参数
        self.batch_size  = batch_size
        self.input_size  = input_size
        self.output_size = output_size
        self.tag = tag
        global num
        num=num+1
        self.num= num
        
        self.inputs_with_label = inputs_with_label
        if len(self.inputs_with_label) == 0:
            l.error("error inputs with label length")	
            exit()

        #cache
        self.bitmap_content_cache=dict()

    def __len__(self):
        # 让代码知道这个序列的长度
        return math.floor( len(self.inputs_with_label) / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        batch = self.inputs_with_label[idx*self.batch_size: (idx+1)*self.batch_size]
        x_arrays = np.array( [self.read_input_content(input_path) for input_path, bitmap_path in batch]).reshape(self.batch_size, self.input_size, 1).astype(np.float64)/255 
        y_arrays = (np.array( [self.read_label_content(bitmap_path) for input_path, bitmap_path in batch]) > 0).astype(np.int8) 
        #if "train" in self.tag:
        #    l.info("%s:%d, get data from [%d, %d)",self.tag,self.num, idx*self.batch_size, (idx+1)*self.batch_size)
        #l.info("the shape of x is %s", x_arrays.shape) 
        #l.info("the shape of y is %s", y_arrays.shape) 
        return x_arrays, y_arrays
   
    # 读取 input 文件 内容
    def read_input_content(self, file):
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

        return content

    # 读取 reduced bitmap 文件内容
    def read_label_content(self, file):
        
        if file in self.bitmap_content_cache:
            content=self.bitmap_content_cache[file]
            return content
        
        f = open(file, "rb")
        content = f.read()
        f.close()
        
        content = bytearray(content)
        # control the length
        if len(content) < self.output_size:
            content.extend( [0] * ( self.output_size - len(content) ) )
        elif len(content)  > self.output_size :
            content = content[0:self.output_size ]

        # transform
        content = struct.unpack('B'*len(content), content) ## it is a tuple, B means unsing
       
        # add to cache
        self.bitmap_content_cache[file]=content

        return content

    #服务于生成器 
    def read_samples_by_size(self, size=10 , start_index=0):
        '''
        start_index : fromt start_index to start_index+size
        '''
        batch = self.inputs_with_label[start_index*self.batch_size: (start_index+1)*self.batch_size]
        x_arrays = np.array( [self.read_input_content(input_path) for input_path, bitmap_path in batch]).reshape(self.batch_size, self.input_size, 1).astype(np.float64)/255 
        y_arrays = (np.array( [self.read_label_content(bitmap_path) for input_path, bitmap_path in batch]) > 0).astype(np.int8) 

        return (x_arrays, y_arrays)
   
    #生成器 
    def generator_batch_data(self):
        '''
        it is the generator for the fit_generator function
        '''
        start_index = 0
        while True:
            #start_index %= math.floor( len(self.inputs_with_label) / self.batch_size)
            inputs_data, labels_data = self.read_samples_by_size( size = self.batch_size, start_index = start_index )
            start_index += self.batch_size
            #l.info("generator train data from [%d, to %d)", start_index, start_index + self.batch_size)
            yield (inputs_data, labels_data)

   
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
        
        self.train_inputs_with_label = None # the train data
        self.train_sequence = None
    
        self.test_inputs_with_label = None # the test data 
        self.test_sequence = None

        self.model       = Sequential()
        self._create_model()
        self.model_file_path =  os.path.join(cur_dir, self.binary+"-model.h5")

        self.useful_index = None


    def _set_useful_index_to_model(self, useful_index):
        self.useful_index = np.array(useful_index)

    def _create_model(self):
        # 64 个输出通道，卷积大小9
        self.model.add( Conv1D(128, 9, strides=self.strides,  padding="same", input_shape=(self.input_size, 1) ) )
        self.model.add( Activation("relu",name="relu1") )
       
        self.model.add( Conv1D(128*2, 9, strides=self.strides,  padding="same") ) 
        self.model.add( Activation("relu",name="relu2") )

        self.model.add( Conv1D(128*2*2, 9, activation='relu', strides= self.strides, padding="same") )
        self.model.add( Activation("relu",name="relu3") )

        self.model.add( Flatten() )
        
        self.model.add( Dropout(0.25) )

        self.model.add( Dense(self.output_size*2, activation='relu', name="full_connect_layer1") )
        
        self.model.add(Dropout(0.25))
        
        self.model.add(Dense( self.output_size, name="full_connect_layer2"))

        self.model.add( Activation("sigmoid",name="sigmoid") )

        self.model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])  

    def get_model_net(self):
        self.model.summary()
    
    def save_model(self):
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
        self.model.save(self.model_file_path)
        #plot_model(self.model, to_file='model.png')

    def load_model(self):
        self.model =load_model(self.model_file_path)
        self.input_size = self.model.input_shape[1]
        l.info("using the input size of %s as the loaded model", self.input_size)
        self.output_size = self.model.output_shape[1]
        l.info("using the output size of %s as the loadedmodel", self.output_size)

    def train_model(self, all_inputs_with_label, useful_index):
        # set the useful index 
        self._set_useful_index_to_model(useful_index)
        
        # load from  old model 
        if self.use_old_model and os.path.exists(self.model_file_path):
            l.info("load old model from file")
            self.load_model()
            return
        
        # set some  metadata to model
        self._set_data_to_model(all_inputs_with_label)

        l.info("training the modle")
        l.info("the steps is %d", len(self.train_sequence))
        self.model.fit_generator( self.train_sequence, 
                                  steps_per_epoch = len(self.train_sequence), 
                                  epochs = self.epochs, verbose=1,
                                  max_queue_size=1, # 每次最多生成的批次i
                                  workers=3,         # 工作者的数量
                                  use_multiprocessing=True,  # 是否启用多进程
                                  validation_data = self.test_sequence,
                                  validation_steps = len(self.test_sequence) )
        # save the model
        self.save_model()
    
    # split data into train and test, and init the sequence data 
    def _split_data(self):
        l.info("the test rate is %f", self.valid_rate)
        self.train_inputs_with_label, self.test_inputs_with_label = train_test_split( self.all_inputs_with_label, test_size = self.valid_rate)

        if len(self.test_inputs_with_label) < self.batch_size:
            l.warn("there are too little data, exit")
            exit(0)

        #init the sequence data
        self.train_sequence = SequenceData(self.batch_size, self.input_size, self.output_size, self.train_inputs_with_label, "train:")
        self.test_sequence  = SequenceData(self.batch_size, self.input_size, self.output_size, self.test_inputs_with_label, "test:")
        
    # set the sample data to model
    def _set_data_to_model(self , all_inputs_with_label):
        l.info("use %f samples for speed", self.use_rate)
        self.all_inputs_with_label = all_inputs_with_label[0:int(len(all_inputs_with_label)*self.use_rate)]
        
        # split the train data and test data
        self._split_data()
    
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
    
    def predict(self, size =10):
        inputs_data,labels_data = self.read_samples_by_size(size , train = False)
        result = self.model.predict( inputs_data, batch_size =1, verbose=1)
      
        for i in range(size):
            num=0
            a=list()
            for index in range(len(labels_data[i])):
                if labels_data[i][index] ==1:
                    num+=1
                    a.append(index)
            l.info("there is %d number 1 in the label", num)
            #l.info(a)
            for ts in [0.5,0.51]:
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
        temp=np.repeat(temp, int(data_length/2), axis=1)

        #add the third
        temp=np.reshape(temp, (data_length, int(data_length/2), 1))
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
        l.info("the length of result is %d", len(result))
        for i in range( min(20, len(result)) ):
            max_value = np.max(temp_result)
            if max_value == 0:
                l.info("the max gradient is 0")
                break
                return
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
            input_content = self.read_input_content(input_path)
            input_content = np.reshape(input_content,  (1, len(input_content), 1) )
        else:
            l.info("there is not %s, use one  other sample", input_path)
            input_content, _ = self.read_samples_by_size(1)

        output_index = np.where(self.useful_index == branch_id)
        if len(output_index[0]) == 0:
            l.warn("%s do not meet transition of %d", input_path, branch_id)
            return None

        layer_idx = utils.find_layer_idx(self.model, 'full_connect_layer2')
        # Swap sigmoid with linear
        #self.model.layers[layer_idx].activation = activations.linear
        #model = utils.apply_modifications(self.model)

        #saliency
        result = visualize_saliency(self.model, layer_idx=layer_idx, filter_indices=output_index[0], seed_input = input_content, backprop_modifier=None,  grad_modifier="absolute")

        #plot the result
        #self.plot_saliency(result) 
        
        #l.info(result)
        return result * 255
        
def start():
   
    afl_work_dir = os.path.join( "/dev/shm", "output-"+engine+'-'+binary )
   
    l.info("using the data from %s", engine)
    l.info("deal with the binary %s\n", binary)

    # 0.init the collect 
    collect = Collect( afl_work_dir = afl_work_dir, binary = binary, ignore_ts =ignore_ts, 
                         from_file = from_file, engine = engine, reduce_use_old =reduce_use_old)
    #  collect the path of each input
    l.info("the ignore ts is %d", ignore_ts)
    l.info("begine to collect the data from %s", engine)
    collect.collect_by_path()
    
    # 1. get the size of output
    l.info("\n")
    reduce_output_size = collect.get_length_reduce_bitmap()
    l.info("the length of reduce bitmap is %d", reduce_output_size)
    output_size =reduce_output_size
    if output_size > max_output_size:
        output_size = max_output_size
    l.info("at last, use the output_size of %d\n", output_size)
    
    # get the size of input
    input_size = collect.input_mean_length
    l.info("the mean input size is %s", input_size)
    if input_size==0:
        input_size = max_input_size
    if input_size > max_input_size:
        input_size = max_input_size
    l.info("at last, use the input_size of %d\n", input_size)
    
    # init the model
    nmodel = Nmodel( input_size = input_size, output_size = output_size, binary=binary,
                    strides = strides, epochs = epochs, batch_size = batch_size ,
                    use_rate =use_rate, valid_rate=valid_rate, use_old_model =use_old_model )

    #nmodel.get_model_net()
    
    # 4. get the sample data 
    all_inputs_with_label = collect.get_data()
    useful_index = collect.get_useful_index()
    
    # train the model
    nmodel.train_model( all_inputs_with_label, useful_index)

    #l.info("begin to predict")
    #nmodel.predict()

    #l.info("begin to evalute")
    #evaluate_result = nmodel.evaluate()
    return
    check_input_path="/tmp/afl-nb/3D_Image_Toolkit/queue/id:000008,src:000000,op:havoc,rep:128"
    result = nmodel.saliency(check_input_path, 21799 )
    if not result is None:
        nmodel.get_index_max_value(result)
    
    result = nmodel.saliency(check_input_path, 57466)
    if not result is None:
        nmodel.get_index_max_value(result)


def main():
    start(  )

if __name__ == "__main__":
    main()
    



