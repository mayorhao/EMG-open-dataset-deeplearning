import os
from settings import Config
conf=Config()
os.chdir(conf.os_dir)
from prepare_data.gen_stft_samples import gen_training_data
from prepare_data.extract_raw_data import do_convert
# read settings
conf = Config()
folder_list = os.listdir(conf.data_dir)
folder_list.sort()
conf.folder_list=folder_list
print("start convert txt file to npz")
for idx,fold_name in enumerate(folder_list[:1]):
    do_convert(idx,conf)
print("convert complete")
print("begin to generate training samples")
for subj_id in range(conf.num_sub):
    gen_training_data(subj_id,conf)
print("generation done")
