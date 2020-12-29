import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from argparse import ArgumentParser
import pathlib
import os
import argparse
import glob
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from settings import  Config
class EmgDataset(Dataset):
    def __init__(self,samples_list):
        self.files=[]
        for sample in samples_list:
            tem=os.listdir(sample)
            tem.sort(key=int)
            self.files+=[os.path.join(sample,x,"data.npz") for x in tem]
    def __getitem__(self,idx):
        sample_path = self.files[idx]
        with np.load(sample_path) as f:
            x=f["x"]
            y=f["y"]
            return {
                "x":x,
                "y":y
            }
    def __len__(self):
        return len(self.files)
def _gen_k_fold(args):
    # 6 repetation in total, random select one repeatation for test, the rest as training data, if one subj has not enough data, then in this validataion fold,its test data will be empty
    class_num=args.class_num
    SAVE_PATH=args.data_path
    subj_file_list=os.listdir(os.path.join(SAVE_PATH))
    subj_file_list.sort()
    train_samples=[]
    test_samples=[]
    np.random.seed(0)
    for subj_file_name in subj_file_list:
        for label in range(class_num):
            temp=glob.glob(os.path.join(SAVE_PATH,subj_file_name,f"session_{args.session_i}",str(label),"*"))
            np.random.shuffle(temp)
            test_repetation=[]
            if(len(temp)>args.fold_i):
                test_repetation+=[temp[args.fold_i]]
            test_samples+=test_repetation
            train_samples+=list(set(temp)-set(test_repetation))

    train_length=len(train_samples)
    test_length=len(test_samples)
    print(f"train length: {train_length} \n test length:{test_length} ")
    return train_samples,test_samples

def MakeDataloaders(opts):
    train_samples, test_samples=_gen_k_fold(opts)
    train_set=EmgDataset(samples_list=train_samples)
    test_dataset=EmgDataset(samples_list=test_samples)
    total_num = len(train_set)
    val_length = int(total_num * opts.val_ratio)
    train_dataset, val_dataset = random_split(train_set, [total_num - val_length, val_length])
    print(f"train:{len(train_dataset)} \n val: {len(val_dataset)} \n test:{len(test_dataset)}")
    train_data_loader=DataLoader(train_dataset,batch_size=opts.batch_size,num_workers=opts.num_workers,pin_memory=True,shuffle=True)
    val_data_loader=DataLoader(val_dataset,batch_size=opts.batch_size,num_workers=opts.num_workers,pin_memory=True,shuffle=True)
    test_data_loader=DataLoader(test_dataset,batch_size=opts.batch_size,num_workers=opts.num_workers,pin_memory=True,shuffle=False)
    return train_data_loader,val_data_loader,test_data_loader



    # shuffled_indice=

# define model structure
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, args.class_num)
        )

    def forward(self, x):
        x = self.model(x)
        x=self.fc(torch.flatten(x,start_dim=-3,end_dim=-1))
        return x
# train model
def train_c(iterator,model,optimizer,epoch):
    print(f"begin to train {epoch}")
    model.train()
    loop= tqdm(enumerate(iterator),total=len(iterator),leave=False)
    for idx,batch in loop:
        y_true = batch["y"].to(device)
        y_pred = model(batch["x"].type(torch.float32).to(device))
        loss = loss_criterion(y_pred, y_true)
        # wirte scalars to monitor
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item())
        writer.add_scalar('Loss/train', loss.item(), idx+epoch*len(loop))
# model validation
def val_c(iterator,model,epoch):
    model.eval()
    print("begin to validataion")
    val_loss=[]
    y_true_list=[]
    y_pred_list=[]
    with torch.no_grad():
        for idx,batch in enumerate(iterator):
            y_true=(batch["y"].to(device))
            y_pred=model(batch["x"].type(torch.float32).to(device))
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            loss = loss_criterion(y_pred, y_true)
            val_loss.append(loss)
    val_loss=torch.stack(val_loss).mean()
    val_loss=torch.stack([val_loss],dim=0).mean()
    y_true=torch.cat(y_true_list,dim=0)
    y_pred=torch.cat(y_pred_list,dim=0)
    y_pred=F.softmax(y_pred,dim=-1)
    y_pred=torch.argmax(y_pred,dim=-1)
    c = (y_true == y_pred).squeeze()
    acc = c.sum().type(torch.float32) / len(y_true)

    print("Epoch: {:.0f} Val Acc: {:.6f} Val_loss: {:.6f} ".format(epoch, acc, val_loss))
    writer.add_scalar("Loss/val",val_loss.item(),epoch)
    writer.add_scalar("Acc/val",acc.item(),epoch)
    print("validation end")
    del y_true
    del y_pred
    return val_loss
# test_model
def test_c(iterator, model,save_path):
    print("begin to test")
    model.eval()
    test_loss = []
    y_true_list=[]
    y_prob_list=[]
    for idx, batch in enumerate(iterator):
        y_true = (batch["y"].to(device))
        y_out = model(batch["x"].type(torch.float32).to(device))
        y_prob=F.softmax(y_out,dim=-1)
        y_true_list.append(y_true.detach().cpu().numpy())
        y_prob_list.append(y_prob.detach().cpu().numpy())
        loss = loss_criterion(y_out, y_true)
        test_loss.append(loss.item())
    test_loss=np.hstack(test_loss).mean()
    y_true=np.hstack(y_true_list)
    y_prob=np.vstack(y_prob_list)

    y_pred=np.argmax(y_prob,axis=-1)
    c = np.where(y_true==y_pred)[0]
    acc = len(c) / len(y_true)

    print("Epoch: {:.0f} Test Acc: {:.6f} Test_loss: {:.6f} ".format(epoch, acc, test_loss))
    writer.add_scalar("Loss/test",test_loss,epoch)
    writer.add_scalar("Acc/test",acc,epoch)
    np.savez(os.path.join(save_path,"resut.npz"),y_pred=y_pred,y_true=y_true,y_prob=y_prob)
    print("test end")
def save_checkpoint(model,epoch,opts):
    model_out_path = os.path.join(opts.model_path,"epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def create_dirs(dirs_list):
    for dir in dirs_list:
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
def config_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # set device

    VERSION=0
    parser=ArgumentParser()
    # paths
    # parser.add_argument("--data_path",type=str,default="/home/fanjiahao/TNSRE/data/dynamic_STFT_v2")
    parser.add_argument("--tb_log_dir", type=str, default="./tb_logs")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./outputs")

    # experiments settings
    parser.add_argument("--mode", type=str, default="maintance")
    parser.add_argument("--trial_id", type=int, default=4001)
    # parser.add_argument("--fold_n",type=int,default=5)
    # parser.add_argument("--fold_i",type=int,default=0)
    parser.add_argument("--val_ratio",type=float,default=0.1)
    # parser.add_argument("--n_session",type=int,default=1)
    parser.add_argument("--class_num",type=int,default=34)
    # process
    parser.add_argument("--num_workers",type=int,default=10)
    parser.add_argument("--gpu",type=int,default=0)
    # train parameters
    parser.add_argument("--batch_size",type=int,default=512)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--drop_out_prob', type=int, default=.2)
    # read global settings
    conf=Config()
    MODE=conf.mode
    SESSION_I=conf.session_i
    FOLD_N=conf.fold_n
    DATA_PATH=conf.data_path
    model_path=conf.model_path
    output_path=conf.output_path
    tb_log_dir=conf.tb_log_dir
    # read global settings end
    opts=parser.parse_args()
    opts.session_i = SESSION_I
    opts.data_path = DATA_PATH
    os.chdir(conf.os_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)  # device setting
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_random_seed(0)  # config random seed
    # create dirs
    for fold_i in range(FOLD_N):
        opts.fold_i=fold_i
        SUB_DIR=os.path.join(f"{MODE}",f"trail_{opts.trial_id}",f"fold_{opts.fold_i}")
        opts.model_path=os.path.join(model_path,SUB_DIR)
        opts.output_path=os.path.join(output_path,SUB_DIR)
        opts.tb_log_dir=os.path.join(tb_log_dir,SUB_DIR,str(VERSION))
        create_dirs([ opts.model_path, opts.output_path,opts.tb_log_dir])
        # create dirs end
        model=Classifier(opts).to(device)     # initialize models
        writer = SummaryWriter(log_dir=opts.tb_log_dir)  # initialize logger
        train_data_loader,val_data_loader,test_data_loader=MakeDataloaders(opts) # fetch data loaders
        optimizer=torch.optim.Adam(model.parameters(),lr=opts.lr,weight_decay=opts.weight_decay) # define optimizer
        scheduler=ReduceLROnPlateau(optimizer,patience=10,factor=0.1, verbose=True) # define scheduler to gradually decrease learning rate based on validation loss
        loss_criterion=nn.CrossEntropyLoss()  # define loss function
        for epoch in range(opts.max_epochs):
            train_c(train_data_loader,model,optimizer,epoch) # train
            val_loss=val_c(val_data_loader,model,epoch)  # validataion
            scheduler.step(val_loss) # update learning rate
        test_c(test_data_loader,model,opts.output_path) # test model
        save_checkpoint(model,epoch,opts) # save model
        print(f"{MODE}: session {opts.session_i},fold:{opts.fold_i} done")



