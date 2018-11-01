import os
import shutil
import numpy as np

import logging
import coloredlogs
import time
import struct
#from sklearn.utils import shuffle
import random
import json

l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"

cur_dir = os.path.abspath(os.path.dirname(__file__))

#install the coloredlogs
coloredlogs.install(fmt=fmt)


'''
the default of total bitmap is 255 for each byte
the default of trace bits if 0 for each byte
'''

class Collect():
    def __init__(self, afl_work_dir, binary, ignore_ts, engine, from_file=False, reduce_use_old=False, info_files=None):
        '''
        ignore_ts: ignore threshold, if the smaple number less than it, ignore
        '''
        self.afl_work_dir   = afl_work_dir
        self.all_data_dir   = os.path.join(afl_work_dir, "data")
        self.total_bitmap_path   = os.path.join(afl_work_dir, "fuzz_bitmap")
        
        self.ignore_ts      =ignore_ts
        
        self.binary         = binary
        
        self.path_num_dict      = dict() #key is the hash of path, value is the samples of this path
        self.total_path_num       = 0
       
        self.from_file = from_file
        self.engine    =  engine
        self.json_file_path = engine+"-"+ self.binary+"-" +"data.json"

        self.input_mean_length = 0
        self.kind_num = 0

        # all inputs it is huge 
        self.all_inputs_with_label = list() #each element is a tuple, which is the inputs and bitmap paths

        # get the branch infor
        self.branch_info=dict()
        self.info_files =info_files
        self.read_branch_info()        

        # get the useful index from the total bitmap
        self.useful_index = None # it is should a sorted one %XX ; save the non -1 index of the total bitmap, meaning these index are used
        self._useful_index_from_total_bitmap()
       

        
        self.reduce_tail = "-reduce"
        self.reduce_use_old = reduce_use_old

      
    def reduce_trace_bitmap(self, old_bitmap_path):

        reduce_bitmap_path = old_bitmap_path + self.reduce_tail
        
        if os.path.exists( reduce_bitmap_path) and self.reduce_use_old:
            return reduce_bitmap_path

        # read the old bitmap
        with open(old_bitmap_path, "rb") as f:
            content = f.read()
            #for test
            #num = self.sum_non_values_in_content( content, b'\x00')
            #l.info("there is %d 1 in the trace of %s", num, old_bitmap_path)
            
        # read the data in useful index
        reduce_bitmap_content= bytearray()
        for index in self.useful_index:
            reduce_bitmap_content.append(content[index])
       
        # save the useful index with the content
        with open(reduce_bitmap_path, 'wb') as f:
            f.write(reduce_bitmap_content)
           
            # for test
            #num = self.sum_non_values_in_content( reduce_bitmap_content, 0)
            #l.info("there is %d 1 in the trace of %s", num, reduce_bitmap_path)
        
        # test for checking
        with open(reduce_bitmap_path, 'rb') as f:
            check_content=f.read()
        
        # for check
        temp_index = list(self.useful_index)
        for index in range(len(self.useful_index)):
            if  check_content[index] == content[ temp_index[index]]:
                continue
            else:
                l.info("there is some wrong")
                exit(1)
                break

        return reduce_bitmap_path

    @staticmethod
    def sum_non_values_in_content( content, value):
        num =0
        for index in range( len(content) ):
            if content[index] != value:
                num+=1
        return num

    def collect_by_path(self):
        if self.from_file: 
            if self.load_from_json():
                l.info("load from file")
                l.info("collect %d inputs with their bitmap!", len(self.all_inputs_with_label) )
                return
        l.info("begin to collect the input, wait for some time")
        l.info("reduce the bitmap")
       
        # collect the mean length in each kind
        kind_total_length=0
        kind_num=0

        for path_hash in os.listdir(self.all_data_dir):
            
            sole_data_dir = os.path.join(self.all_data_dir, path_hash)
            
            # collect all input path
            path = Path(path_hash, sole_data_dir, self.ignore_ts)
            input_paths = path.get_input_paths()
            bitmap_path = path.get_bitmap_path()

            #reduce the trace bitmap
            if len(input_paths) > 0:
                reduce_bitmap_path = self.reduce_trace_bitmap(bitmap_path)
            else:
                reduce_bitmap_path = None
            
            for sole_input in input_paths:
                if not reduce_bitmap_path is None and os.path.exists(reduce_bitmap_path):  
                    self.all_inputs_with_label.append( (sole_input, reduce_bitmap_path) )
                else:
                    l.info("reduce bitmap fail for %s", bitmap_path)
            
            # collect the mean input length for each kind
            if path.input_mean_length>0:
                kind_total_length+=path.input_mean_length
                self.kind_num+=1

            # save the input number for each path
            self.path_num_dict.update({path_hash:path.inputs_num})
       
        # set the total input mean length
        self.input_mean_length =int( kind_total_length/self.kind_num)+1
        l.warn("collect form %d kinds path\n", self.kind_num)

        # shuffle the list
        l.info("shuffle the inputs")
        random.shuffle(self.all_inputs_with_label)
    
        #save the collect file
        self.save_to_json()

        l.info("collect %d inputs with their bitmap!", len(self.all_inputs_with_label) )

    '''
    in the bitmap, each byte denote a transition
    just read do not transform
    '''
    def read_bitmap_content(self, file):
        f = open(file, "rb")
        content = f.read()
        f.close()
        
        return content
    
    def _useful_index_from_total_bitmap(self):

        content = self.read_bitmap_content(self.total_bitmap_path)  #the return type is tuple 

        #collect the activated index
        useful_index=set()
        for index in range( len(content) ):
            if not content[index] == 255:
                useful_index.add(index)
        l.info("%d activated index", len(useful_index))
        # collect out the bilateral branches
        copy_useful_index=useful_index.copy()
        for index in copy_useful_index: 
            if index in self.branch_info:
                if not self.branch_info[index] in useful_index:
                    useful_index.discard(index)
                    l.info("In the Intresting %d is not a bilateral branch, remove it", index)
                else:
                    l.info("%d is a bilateral covered one",index)
            else:
                l.info("%d is activated but not in the branch info", index)

        #transform to tuple
        self.useful_index = useful_index
        l.info("usefull indes has %d length", len(self.useful_index))
    
    def get_useful_index(self):
        return self.useful_index
    
    def get_total_samples(self):
        l.info("there are %s samples", len(self.all_inputs_with_label))
        return len(self.all_inputs_with_label)

    def get_length_reduce_bitmap(self):
        return len(self.useful_index)

    def get_data(self):
        return self.all_inputs_with_label

    def save_to_json(self):
        if os.path.exists(self.json_file_path):
            os.remove(self.json_file_path)

        with open(self.json_file_path, 'w') as outfile:
            json.dump(self.all_inputs_with_label, outfile)

    def load_from_json(self):
        if not os.path.exists(self.json_file_path):
            l.warn("there is no %s", self.json_file_path)
            return False
        with open(self.json_file_path, 'r') as outfile:
            self.all_inputs_with_label  = json.load( outfile )
        return True
   
    # get the branch info about the true and false
    def read_branch_info(self):
        for file in self.info_files:
            f = open(file, "r")
            for line in f:
                line=line.strip()
                result=line.split('|')
                true_branch  = int(result[3])
                false_branch = int(result[4] )
                if true_branch in self.branch_info:
                    l.warn("meeting reduandant branch at %d, for another, the record is %d, the new collect is %d", true_branch, self.branch_info[true_branch], false_branch)
                else:
                    self.branch_info[true_branch]=false_branch
                if false_branch in self.branch_info:
                    l.warn("meeting reduandant branch at %d, for another, the record is %d, the new collect is %d", false_branch, self.branch_info[false_branch], true_branch)
                else:
                    self.branch_info[false_branch]=true_branch
            f.close()



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
        
        self.bitmap_path  = os.path.join(path_data_dir, "trace-%s"%(bitmap_hash) ) # the ab path of the bitmap

        self.input_total_length = 0
        self.inputs_num = 0 
        self.input_mean_length = 0
        
        #collect the info
        self.input_paths=set() #save all the inputs  absolute path
        self._get_input_paths()
        
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
            
            # collect the input length
            input_length = os.path.getsize(input_path) 
            self.input_total_length+=input_length
            self.inputs_num+=1
        
        #set the mean input length
        self.input_mean_length = int(self.input_total_length/ self.inputs_num)+1


    def get_input_paths(self):
        return self.input_paths

    def get_bitmap_path(self):
        return self.bitmap_path 

        
def main():
    read_branch_info()
    return
    engine="afl"
    l.info("using the data from %s", engine)
    
    binary = "test"
    afl_work_dir = os.path.join(cur_dir, "output-"+engine+'-'+binary)
    
    collect = Collect(afl_work_dir, binary, ignore_ts=30, engine=engine, from_file =False, reduce_use_old =False)

    #1. collect the path of each input
    collect.collect_by_path()



if __name__ == "__main__":
    main()
    





