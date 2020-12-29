# convert txt  file to npz file to speed up training
import numpy as np
import os
import re
import glob
import pathlib
from settings import Config
from multiprocessing import Process

def do_convert(idx,conf):
    DATA_DIR = conf.data_dir
    SAVE_PATH = conf.save_path
    MODES = conf.mode
    folder_list=conf.folder_list
    fold_name=folder_list[idx]
    m=re.search(r'subject(\d+)\_session(\d+)$',fold_name) # extract subject id and session via regexp matching from fold names
    subject_id=int(m[1])
    session=int(m[2])
    file_list=glob.glob(os.path.join(DATA_DIR,fold_name,f"{MODES}_preprocess_sample*.txt")) # use preprocessed files
    labels=np.loadtxt(os.path.join(DATA_DIR,fold_name,f"label_{MODES}.txt"),delimiter=",",dtype=np.long) # read labels
    for j, file_name in enumerate(file_list):
        label_idx=int(re.search(r'\D+(\d+)\.txt$',file_name)[1]) # determine label index for current file
        sig=np.loadtxt(file_name,delimiter=",",dtype=np.float32)  # read sinal array,shape:8196*256(maintenance)
        label=labels[label_idx-1]
        label=label-1 # we used to start with zero in python
        save_current_path=os.path.join(SAVE_PATH,MODES,f"subject_{subject_id-1}",f"session_{session-1}") # all id values minus one, because we used to start at index==0 in python
        pathlib.Path(save_current_path).mkdir(parents=True,exist_ok=True) # create save dir
        np.savez(os.path.join(save_current_path,f"{label_idx-1}.npz"),x=sig,y=label)
        print(f"{subject_id}-{j}_done")
if __name__ == '__main__':
    conf = Config()
    os.chdir(conf.os_dir)
    DATA_DIR = conf.data_dir
    SAVE_PATH = conf.save_path
    MODES = conf.mode  # MODES: "maintenance" | "dynamic"
    folder_list = os.listdir(DATA_DIR)
    folder_list.sort()
    subj=np.arange(0,20).reshape(10,2)
    # can choose to use mutiprocess functions
    for grous in subj:
        start=grous[0]
        end=grous[1]
        print(start,end)
        proc=Process(target=do_convert,args=(start,end+1))
        proc.start()