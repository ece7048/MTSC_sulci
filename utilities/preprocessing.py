"""General preprocessing helpers for checkpoints, labels, and datasets."""

from __future__ import division, print_function

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class AverageMeter(object):
    """Track running averages for scalar metrics on CPU."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class AverageMeter_cuda(object):
    """Track running averages for scalar metrics stored on CUDA tensors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0, device='cuda')
        self.avg = torch.tensor(0.0, device='cuda')
        self.sum = torch.tensor(0.0, device='cuda')
        self.count = torch.tensor(0, device='cuda')

    def update(self, val, n=1):
        self.val = torch.tensor(val.clone().detach(), device='cuda')
        self.sum += self.val * n
        self.count += n
        if  self.count==0:
            self.count=0.00000001
        self.avg = torch.where(self.count > 0, self.sum / self.count, self.sum)



def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, train_loss=10000, dir_add=''):
    """Save a model checkpoint, handling generator/discriminator pairs."""
    if len(model)==2:
        filename1='generator_'+filename
        filename2='discriminator_'+filename
        state_dict1 = model[0].state_dict()
        save_dict1 = {"epoch": epoch, "best_acc": best_acc, "train_loss": train_loss, "state_dict": state_dict1}
        filename1 = os.path.join(dir_add, filename1)
        torch.save(save_dict1, filename1)
        print("Saving checkpoint generator", filename1)
        state_dict2 = model[1].state_dict()
        save_dict2 = {"epoch": epoch, "best_acc": best_acc, "train_loss": train_loss, "state_dict": state_dict2}
        filename2 = os.path.join(dir_add, filename2)
        torch.save(save_dict2, filename2)
        print("Saving checkpoint", filename2)
    else:
        state_dict = model[0].state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc,"train_loss": train_loss, "state_dict": state_dict}
        filename = os.path.join(dir_add, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)


def excel_label(filenamep='example.csv', given_name='908', cell_col=2, list_name=['absent','prominent','present'],data='top',nclass=3):
    """Read a class label for a subject from a CSV metadata file."""
    with open(filenamep, 'r') as csvfile:
        reader = csv.reader(csvfile)
        cell_value = None
        if data=='top':
            for row in reader:
                if row[0] == given_name[:-3]: #_T1 TOP-OSLO with [:3], benemin without [:3]
                    cell_value = row[cell_col-1]  # adjust for 0-based indexing
                    break
        else:
            for row in reader:
                if row[0] == given_name: #_T1 TOP-OSLO with [:3], benemin without [:3]
                    cell_value = row[cell_col-1]  # adjust for 0-based indexing
                    break
    if nclass==3:
        if cell_value==list_name[0]:
            cell_value=0
        elif cell_value==list_name[1]:
            cell_value=1
        elif cell_value==list_name[2]:
            cell_value=2
        else:
            cell_value=cell_value
    else:
        if cell_value==list_name[0]:
            cell_value=0
        elif cell_value==list_name[1]:
            cell_value=1
        elif cell_value==list_name[2]:
            cell_value=1
        else:
            cell_value=cell_value
    csvfile.close()
    return cell_value


def normalize(volume):
    """Scale a volume to the [0, 1] range as float32."""
    min_value=np.min(volume)
    max_value=np.max(volume)
    volume[volume<min_value]=min_value
    volume[volume>max_value]=max_value
    volume=(volume-min_value)/(max_value-min_value)
    volume=volume.astype("float32")
    return volume

class GraphImageDataset(Dataset):
    """Dataset that maps subject folders to image, label, and class records."""

    def __init__(self,channels=1, data_path:str = "", data_path2:str = "",transform=None,inp=0,sx=112,sy=112,sz=112,excel=None,cs=["",""],afs=["",""],round=1,mean=0,std=1,pre=True, data='ben',nclass=3):
        self.case_file=cs
        self.after_sample=afs
        self.data_path = data_path
        self.data_path2 = data_path2
        self.transform = transform
        self.inp=inp
        self.sx,self.sy,self.sz=sx,sy,sz
        self.excel=excel
        list_sub=[]
        self.sub=[]
        list_sub2=[]
        self.sub2=[]
        self.mean=mean
        self.std=std
        items=os.listdir(self.data_path)
        items=sorted(items)
        self.round=round
        self.label_num=nclass
        for item in items:
            list_sub.append(self.data_path+item)
            self.sub.append(item)
        self.path=list_sub
        if self.data_path2!='None':
            items2=os.listdir(self.data_path2)
            items2=sorted(items2)
            for item2 in items2:
                list_sub2.append(self.data_path2+item2)
            self.path2=list_sub2
        self.sub2=self.sub
        self.pre=pre
        self.data=data
        self.channels=channels
        

    def __len__(self):
        """Return the number of available subject records."""
        return (len(self.path)-1)

    def __getitem__(self, index):
        """Build one MONAI transform input dictionary for a subject."""
        subpath=str(self.path[index])
        sub=self.sub[index]
        if self.channels==1:
                subpath_f=str(subpath)+"/t1mri/default_acquisition/default_analysis/segmentation/"+self.case_file[0]+str(sub)+self.after_sample[0]+".nii.gz"
                subpath2=str(self.path2[index])
        else:
                subpath_f=str(subpath)+"/t1mri/default_acquisition/default_analysis/segmentation/"+self.case_file[0]+str(sub)+self.after_sample[0]+".nii.gz"
                subpath2=str(self.path2[index])
                subpath_f2=str(subpath)+"/t1mri/default_acquisition/default_analysis/segmentation/"+self.case_file[2]+str(sub)+self.after_sample[2]+".nii.gz"
                
        sub2=self.sub2[index]
        if self.pre==True:
            subpath2_f=subpath_f
        else:
            subpath2_f=str(self.data_path2)+self.case_file[1]+str(sub2)+self.after_sample[1]+".nii"
        if self.excel!=None:
            val=excel_label(filenamep=self.excel, given_name=sub,data=self.data,nclass=self.label_num)
            if val==None:
                val=0    
            y=torch.tensor(val)
        else:
            y=torch.tensor(self.inp)
        if self.channels==1:
            data_dicts = {"image": str(subpath_f), "label": str(subpath2_f), "class": y}
        else:
            data_dicts = {"image_1": str(subpath_f),"image_2": str(subpath_f2), "label": str(subpath2_f), "class": y}
        if self.transform:
            data_dicts_transform = self.transform(data_dicts)
        else:
            data_dicts_transform= data_dicts
        return data_dicts_transform

