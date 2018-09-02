#coding=utf-8
import os
import shutil
import numpy as np



#收集类
class Collect():
    def __init__(self, work_dir, binary_path):
        self.work_dir   = work_dir
        self.binary_path = binary_path
        self.binary      = os.path.basename(binary_path)
        self.inputs = dict()   # key is the input, value is the index of bitmap in the label
        self.label  = set()
    
    def read_by_path():
        pass


#数据类
#class Data(){


