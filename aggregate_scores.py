import numpy as np
from settings import Config
def cal_scores(fold_idx,trial_id,conf):
    with np.load(f"{conf.output_path}/trail_{trial_id}/fold_{fold_idx}/resut.npz") as f:
        y=f["y_true"]
        y_prob=f["y_prob"]
    seg_num=conf.seg_num
    y=y.reshape((-1,seg_num))[:,0]
    y_prob=y_prob.reshape((-1,seg_num,34)).mean(axis=1)
    y_pred=np.argmax(y_prob,axis=1)
    correct_l=np.where(y==y_pred)[0]
    acc=len(correct_l)/len(y)
    print(acc)
    return acc
if __name__ == '__main__':
    trail_id=0 # the obtiained trail_id
    conf=Config()
    fold_n=conf.fold_n
    result=np.zeros(fold_n)
    for i in range(fold_n):
        result[i]=cal_scores(i,trail_id)
    print(f"mean {result.mean()}\n std:{result.std()}")

