import os
import shutil
import numpy as np

import logging
import coloredlogs
import time
import struct
from sklearn.utils import shuffle
import random

l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"

#scree handler
sh = logging.StreamHandler()# output to scree
#create formatter
formatter = logging.Formatter(fmt)
# add handler and formatter to logger
sh.setFormatter(formatter)
l.addHandler(sh)

#install the coloredlogs
coloredlogs.install(fmt=fmt,logger=l)



class Collect():
    def __init__(self, afl_work_dir, binary_path, ignore_ts, input_max_len=300):
        '''
        ignore_ts: ignore threshold, if the smaple number less than it, ignore
        input_max_len : the fixed length of the input; if less, add 0; if more ,cut
        '''
        self.afl_work_dir   = afl_work_dir
        self.all_data_dir   = os.path.join(afl_work_dir, "data")
        
        self.ignore_ts      =ignore_ts
        
        self.binary_path    = binary_path
        self.binary         = os.path.basename(binary_path)
        
        self.path_num_dict      = dict() #key is the hash of path, value is the samples of this path
        self.total_path_num       = 0
        

        self.input_max_len = input_max_len

        # all inputs it is huge 
        self.inputs_with_lable = list() #each element is a tuple, which is the inputs and bitmap paths


    def collect_by_path(self):
        for path_hash in os.listdir(self.all_data_dir):
            sole_data_dir = os.path.join(self.all_data_dir, path_hash)
            
            path = Path(path_hash, sole_data_dir, self.ignore_ts)
           
            # collect all input path
            input_paths = path.get_input_paths()
            bitmap_path = path.get_bitmap_path()
            for sole_input in input_paths:
                self.inputs_with_lable.append( (sole_input, bitmap_path) )

            # save the input number for each path
            self.path_num_dict.update({path_hash:path.inputs_num})

        # shuffle the list
        random.shuffle(self.inputs_with_lable)

        l.info("collect all inputs paths with their bitmap OK!")
   

    def get_total_samples(self):
        l.info("there are %s samples", len(self.inputs_with_lable))
        return len(self.inputs_with_lable)


    def change_bit_map(self):
        pass


    def read_input_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()

        # control the length
        if len(content) >= self.input_max_len:
            content = content[0:self.input_max_len]
        else:
            content = content + b'\x00'*( self.input_max_len - len(content)) 

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple
        content = np.array(content).reshape( len(content), 1 )

        return content

    def read_bitmap_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()

        # transform
        content = struct.unpack('b'*len(content), content) ## it is a tuple
        content = np.array(content).reshape( len(content), 1 )

        return content


    def read_samples_by_size(self, size=10):
        inputs_data = np.ones( (self.input_max_len ,1) )
        lables_data = np.ones( (65536 ,1) )
        
        for i in xrange(size):
            input_path = self.inputs_with_lable[i][0]
            bitmap_path = self.inputs_with_lable[i][1]
   
            input_content = self.read_input_content(input_path)
            inputs_data = np.append(inputs_data, input_content, axis=1) 

            bitmap_content = self.read_bitmap_content(bitmap_path)
            lables_data = np.append(lables_data, bitmap_content, axis=1)

        inputs_data = np.delete(inputs_data, 0, axis=1)
        lables_data = np.delete(lables_data, 0, axis=1)
        
        print(type(inputs_data.shape))
        l.info(inputs_data.shape)
        l.info(lables_data.shape)
        return inputs_data, lables_data
             
    def read_path_samples(self):
        for input in self.input_paths:
                   return self.inputs_num




'''
only collect the absolte path for each input
'''
class Path():
    def __init__(self, bitmap_hash, path_data_dir, ignore_ts=None):
        '''
        ignore_ts: ignore threshold, if the smaple number less than it, ignore
        '''
        self.bitmap_hash  = bitmap_hash;
        self.path_data_dir     = path_data_dir 
        self.ignore_ts  = ignore_ts
        
        self.bitmap_path  = os.path.join(path_data_dir, "trace-%s"%(bitmap_hash) )

        self.input_paths=set() #save all the inputs  absolute path
        self._get_input_paths()
        self.inputs_num   = len(self.input_paths) ## maybe 0
        
    
    def _get_input_paths(self):
        if not self.ignore_ts is None:
            if len(os.listdir(self.path_data_dir) ) < self.ignore_ts:
                # ignore this path
                return
        
        for item in os.listdir(self.path_data_dir):
            if "trace" in item:
                continue
            input_path = os.path.join(self.path_data_dir, item)
            self.input_paths.add(input_path)

    def get_input_paths(self):
        return self.input_paths

    def get_bitmap_path(self):
        return self.bitmap_path 



def main():
    afl_work_dir = "/home/binzhang/NAFL/output-fair"
    binary_path = "/home/binzhang/NAFL/benchmark/size"
    collect = Collect(afl_work_dir, binary_path, ignore_ts=20)
    
    #1. collect the path of each input
    collect.collect_by_path()
    collect.get_total_samples() 
    #2. read the content of each input, and deal with the length 
    collect.read_samples_by_size()



if __name__ == "__main__":
    main()
    





