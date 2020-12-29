# prepare training data for CNN models
# step 1: data downsampling, from 2048 hz to 1024 hz
# step 2: data segmentation: signals were split to segmentations of 200 ms with 50% overlap
# step 3: save training data, orgnized as subj/session/sample, each sample was reshaped as time_points *16 * 16
import numpy as np
import os
import pathlib
from settings import Config
from scipy import signal
from sklearn.preprocessing import scale
from multiprocessing import Process
def gen_training_data(subj_id,conf):
    SAVE_PATH = conf.save_path
    MODES = conf.mode  # MODES: "maintenance" | "dynamic"
    FS=conf.fs
    SEGMENTATION_LENGTH= conf.segmentation_length
    OVERLAP_RATIO=conf.overlap_ratio
    DISCARD_LENGTH=conf.discard_length
    FFT_SIZE = 256
    for session_id in range(2):
        counter = [0] * 34
        file_list = os.listdir(os.path.join(SAVE_PATH, MODES, f"subject_{subj_id}", f"session_{session_id}"))
        file_list.sort(key=lambda x:int(x.replace(".npz","")))
        for idx , file_name in enumerate(file_list):
            with np.load(os.path.join(SAVE_PATH,MODES,f"subject_{subj_id}",f"session_{session_id}",file_name)) as f:
                sig=f["x"]
                sig=sig.transpose(1,0) # transpose to n_channel * time_point
                # sig=signal.decimate(sig,q=2,axis=1) # downsample, FS=2048/2
                n_channel,time_point=sig.shape
                sig=scale(sig,axis=1) # Z-score normalization
                label=f["y"]
                n_segments=int((time_point-SEGMENTATION_LENGTH*FS)/(OVERLAP_RATIO*SEGMENTATION_LENGTH*FS)+1)
                step=int((1-OVERLAP_RATIO)*SEGMENTATION_LENGTH*FS)
                real_start_idx=int(DISCARD_LENGTH*FS//step)
                for i in range(real_start_idx,n_segments):
                    seg=sig[:,i*step:i*step+int(SEGMENTATION_LENGTH*FS)] # split segmentse
                    nf, taxis, zxx = signal.stft(seg, nperseg=FFT_SIZE, fs=FS, noverlap=FFT_SIZE // 2, padded=False,axis=1)
                    zxx=abs(zxx)
                    # discard components higher than 500 Hz
                    zxx=zxx[:,0:64,:].reshape(16,16,-1).astype(np.float32)
                    zxx=np.transpose(zxx,(2,0,1))  # reshape to topographic images to train
                    # seg=seg.transpose(1,0).reshape((-1,16,16))
                    current_save_path=os.path.join(SAVE_PATH,f"{MODES}_STFT",f"subject_{subj_id}",f"session_{session_id}",f"{label}",f"{counter[label]}",f"{i-real_start_idx}")
                    pathlib.Path(current_save_path).mkdir(parents=True,exist_ok=True)
                    np.savez(os.path.join(current_save_path,"data.npz"),x=zxx,y=label)
                counter[label] = counter[label] + 1
                print(f"{file_name}-done")



if __name__ == '__main__':
    os.chdir("/home/fanjiahao/TNSRE")
    conf=Config()
    SAVE_PATH = conf.save_path
    MODES = conf.mode  # MODES: "maintenance" | "dynamic"
    FS=conf.FS
    SEGMENTATION_LENGTH= conf.segmentation_length
    OVERLAP_RATIO=conf.overlap_ratio
    DISCARD_LENGTH=conf.discard_length
    FFT_SIZE = 256
    subj_list=np.arange(0,20)
    # gen_training_data(0)
    # gen_training_data(0)
    for subj_id in subj_list:
        proc=Process(target=gen_training_data,args=(subj_id,))
        proc.start()

