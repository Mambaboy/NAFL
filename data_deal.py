import os
import shutil
import numpy as np

import logging
import coloredlogs
import time

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
    def __init__(self, afl_work_dir, binary_path):
        self.afl_work_dir   = afl_work_dir
        self.all_data_dir   = os.path.join(afl_work_dir, "data")
        self.binary_path    = binary_path
        self.binary         = os.path.basename(binary_path)
        self.path_dict      = dict() #key is the hash of path, value is the samples of this path
        self.total_samples   = 0   

        self.paths          = list() #save all the object the path
    
    def collect_by_path(self):
        for path_hash in os.listdir(self.all_data_dir):
            sole_data_dir = os.path.join(self.all_data_dir, path_hash)
            path = Path(path_hash, sole_data_dir)
            self.paths.append(path)  # store the path object 
        #count the total 
        self.get_total_samples()
    
    def get_total_samples(self):
        for path in self.paths:
            self.path_dict.update( {path.bitmap_hash : path.inputs_num } )
            self.total_samples += path.inputs_num
        l.info("have some %d sample", self.total_samples)
        return self.total_samples

    def read_all_samples(self):
        read_num=0
        for path in self.paths:
            read_num += path.read_path_samples()
        l.info("read %s sample",read_num)



class Path():
    def __init__(self, bitmap_hash, data_dir):
        self.bitmap_hash  = bitmap_hash;
        self.data_dir     = data_dir 
        
        self.bitmap_path  = os.path.join(data_dir, "trace-%s"%(bitmap_hash) )
        
        self.inputs_path=set() #save all the inputs  absolute path
        self._get_inputs_path()
        self.inputs_num   = len(self.inputs_path)
    
    def _get_inputs_path(self):
        for item in os.listdir(self.data_dir):
            input_path = os.path.join(self.data_dir, item)
            self.inputs_path.add(input_path)

    def read_path_samples(self):
        for input in self.inputs_path:
            a = open(input, "rb")
        return self.inputs_num


def main():
    afl_work_dir = "/home/binzhang/NAFL/output-afl"
    binary_path = "/home/binzhang/NAFL/benchmark/size"
    collect = Collect(afl_work_dir, binary_path)
    
    collect.collect_by_path()
    collect.read_all_samples()



if __name__ == "__main__":
    main()
    





