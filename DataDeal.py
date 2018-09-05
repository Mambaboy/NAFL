import os
import shutil
import numpy as np

import logging
import coloredlogs
import time
import struct
from sklearn.utils import shuffle
import random
import json

l=logging.getLogger("NEUZZ")
l.setLevel("INFO") 
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"

#install the coloredlogs
coloredlogs.install(fmt=fmt)

class Collect():
    def __init__(self, afl_work_dir, binary_path, ignore_ts, input_fix_len=300, from_file=True):
        '''
        ignore_ts: ignore threshold, if the smaple number less than it, ignore
        input_fix_len : the fixed length of the input; if less, add 0; if more ,cut
        '''
        self.afl_work_dir   = afl_work_dir
        self.all_data_dir   = os.path.join(afl_work_dir, "data")
        
        self.ignore_ts      =ignore_ts
        
        self.binary_path    = binary_path
        self.binary         = os.path.basename(binary_path)
        
        self.path_num_dict      = dict() #key is the hash of path, value is the samples of this path
        self.total_path_num       = 0
       
        self.from_file = from_file
        self.json_file_path = "data.json"

        self.input_fix_len = input_fix_len

        # all inputs it is huge 
        self.inputs_with_label = list() #each element is a tuple, which is the inputs and bitmap paths


    def collect_by_path(self):
        
        if self.from_file: 
            if self.load_from_json():
                l.info("load from file")
                return

        l.info("begin to collect the input, wait for some time")
        for path_hash in os.listdir(self.all_data_dir):
            sole_data_dir = os.path.join(self.all_data_dir, path_hash)
            
            path = Path(path_hash, sole_data_dir, self.ignore_ts)
           
            # collect all input path
            input_paths = path.get_input_paths()
            bitmap_path = path.get_bitmap_path()

            for sole_input in input_paths:
                self.inputs_with_label.append( (sole_input, bitmap_path) )

            # save the input number for each path
            self.path_num_dict.update({path_hash:path.inputs_num})

        # shuffle the list
        l.info("shuffle the inputs")
        random.shuffle(self.inputs_with_label)

        self.save_to_json()

        l.info("collect %d inputs with their bitmap!", len(self.inputs_with_label) )

    def get_total_samples(self):
        l.info("there are %s samples", len(self.inputs_with_label))
        return len(self.inputs_with_label)

    def get_data(self):
        return self.inputs_with_label

    def save_to_json(self):
        with open(self.json_file_path, 'w') as outfile:
            json.dump(self.inputs_with_label, outfile)

    def load_from_json(self):
        if not os.path.exists(self.json_file_path):
            l.warn("there is no %s", self.json_file_path)
            return False

        with open(self.json_file_path, 'r') as outfile:
            self.inputs_with_label  = json.load( outfile )

        return True


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
    afl_work_dir = "/home/binzhang/NAFL/output-afl"
    binary_path = "/home/binzhang/NAFL/benchmark/size"
    collect = Collect(afl_work_dir, binary_path, ignore_ts=30)
    
    #1. collect the path of each input
    collect.collect_by_path()
    collect.get_total_samples() 



if __name__ == "__main__":
    main()
    





