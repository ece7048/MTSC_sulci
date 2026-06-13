"""Data loading, MONAI transforms, and NIfTI export helpers."""

from __future__ import division, print_function
import os
import torch
import nibabel as nib
import numpy as np
from typing import Dict, Tuple
from functools import partial

from monai.data import decollate_batch
from monai import transforms

from MTSC_sulci.utilities import DiffModel, transformation_utils
from MTSC_sulci.utilities.monai_utils import sliding_window_inference
from MTSC_sulci.utilities.preprocessing import GraphImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

is_distributed=True

def save(store,volume,sp=(1,112,112,112)):
    """Save one or more channel-first volumes as NIfTI files."""
    if sp[0]==1:
        volume=np.squeeze(volume)
        if volume.shape[0]==1:
            volume=np.squeeze(volume)
        volume=255*volume
        volume=volume.astype(dtype=np.uint16)
        imgnthree1=nib.Nifti1Image(volume,affine=np.eye(4))
        imgnthree1.header.set_data_dtype(np.uint16)
        imgnthree1.header.set_sform(affine=np.eye(4),code='talairach')
        nib.save(imgnthree1,store)
    else:
        print('channels: ',sp[0])
        for i in range(sp[0]):
            if volume.shape[0]==1:
                volume=np.squeeze(volume)            
            #volumep=np.reshape(volume[i],(sp[1],sp[2],sp[3],1))
            volumet=volume[i]
            if volumet.shape[0]==1:
                volumet=np.squeeze(volumet)
            print(volumet.shape)
            #volumept=np.expand_dims(volumet,axis=0)
            volumet=255*volumet
            volumet=volumet.astype(dtype=np.uint16)
            imgnthree1=nib.Nifti1Image(volumet,affine=np.eye(4))
            imgnthree1.header.set_data_dtype(np.uint16)
            imgnthree1.header.set_sform(affine=np.eye(4),code='talairach')
            nib.save(imgnthree1,store[i])

def save_nii_valid(test_loader,net, root_dir,model_name,store_path,roi_size=64,batch_n=4,post_pred=None,scheduler=None):
    """Run validation/test inference and export predictions as NIfTI files."""

    net.load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(DEVICE))["state_dict"])
    print('load the best checkpoint!')
    net.to(DEVICE)
    print('Upload the test dataset!!')
    net.eval()
    model_inferer = partial(sliding_window_inference,roi_size=[roi_size,roi_size,roi_size],sw_batch_size=1,predictor=net,overlap=0.5,buffer_steps=1,buffer_dim=0)
    for idx, batch_data in enumerate(test_loader):
        datad, labelsd, targetd = batch_data['image'], batch_data['class'], batch_data['label']
        image,labels, target = datad.to(DEVICE), labelsd.to(DEVICE), targetd.to(DEVICE)
        target=torch.squeeze(target)
        if scheduler==None:
            logit,classes = model_inferer(image)
        else:
            noise = torch.ones_like(image).to(DEVICE)
            inferer = DiffModel.DiffusionInferer(scheduler)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (image.shape[0],), device=image.device).long()
            in_image=(0.65*image+0.35*noise)
            logit,classes = inferer.sample(input_noise=in_image, diffusion_model=model_inferer, scheduler=scheduler)

        print(image.shape,target.shape,logit.shape,classes.shape,labels.shape)
        val_outputs_list = decollate_batch(logit)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        pred=val_output_convert
        base=target
        
        print(pred[0].shape, target[0].shape)
        sp1=(pred[0].shape[0],image[0].shape[1],image[0].shape[2],image[0].shape[3])
        sp2=(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],pred[0].shape[3])
        if pred[0].shape[0]>1:
            store1=[(store_path+'_'+str(idx)+'_batch_first_prediction1.nii'),(store_path+'_'+str(idx)+'_batch_first_prediction2.nii'),(store_path+'_'+str(idx)+'_batch_first_prediction3.nii')]
            store2=[(store_path+'_'+str(idx)+'_batch_first_input1.nii'),(store_path+'_'+str(idx)+'_batch_first_input2.nii'),(store_path+'_'+str(idx)+'_batch_first_input3.nii')]
        else:
            store1=(store_path+'_'+str(idx)+'_batch_first_prediction.nii')
            store2=(store_path+'_'+str(idx)+'_batch_first_input.nii')
        save((store1),pred[0].cpu().detach().numpy(),sp2)
        save((store2),base.cpu().numpy(),sp1)

def save_nii_train(train_loader,net, root_dir,model_name,store_path,roi_size=64,batch_n=4,post_pred=None,scheduler=None,loop=90):
    """Run training-set inference and export predictions as NIfTI files."""

    net.load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(DEVICE))["state_dict"])
    print('load the best checkpoint!')
    net.to(DEVICE)
    print('Upload the test dataset!!')
    net.eval()
    model_inferer=net
    for idx, batch_data in enumerate(train_loader):
        for i in range(loop):
            datad, labelsd, targetd = batch_data[i]['image'], batch_data[i]['class'], batch_data[i]['label']
            image,labels, target = datad.to(DEVICE), labelsd.to(DEVICE), targetd.to(DEVICE)
            target=torch.squeeze(target)
            if scheduler==None:
                logit,classes = model_inferer(image)
            else:
                noise = torch.ones_like(image).to(DEVICE)
                inferer = DiffModel.DiffusionInferer(scheduler)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (image.shape[0],), device=image.device).long()
                in_image=(0.65*image+0.35*noise)
                logit,classes = inferer.sample(input_noise=in_image, diffusion_model=model_inferer, scheduler=scheduler)

            print(image.shape,target.shape,logit.shape,classes.shape,labels.shape)
            val_outputs_list = decollate_batch(logit)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            pred=val_output_convert
            val_list_base=decollate_batch(target)
            val_covert=[post_pred(val_pred_tensor2) for val_pred_tensor2 in val_list_base]
            base=target


            print(pred[0].shape, base[0].shape,target[0].shape)
            sp1=(pred[0].shape[0],image[0].shape[1],image[0].shape[2],image[0].shape[3])
            sp2=(pred[0].shape[0],pred[0].shape[1],pred[0].shape[2],pred[0].shape[3])
            if pred[0].shape[0]>1:
                store1=[(store_path+'_'+str(idx)+'_'+str(i)+'_batch_first_prediction1.nii'),(store_path+'_'+str(idx)+'_batch_first_prediction2.nii'),(store_path+'_'+str(idx)+'_batch_first_prediction3.nii')]
                store2=[(store_path+'_'+str(idx)+'_'+str(i)+'_batch_first_input1.nii'),(store_path+'_'+str(idx)+'_batch_first_input2.nii'),(store_path+'_'+str(idx)+'_batch_first_input3.nii')]
            else:
                store1=(store_path+'_'+str(idx)+'_'+str(i)+'_batch_first_prediction.nii')
                store2=(store_path+'_'+str(idx)+'_'+str(i)+'_batch_first_input.nii')
            save((store1),pred[0].cpu().detach().numpy(),sp2)
            save((store2),base[0].cpu().numpy(),sp1)

def load_data(channels=1,batch_size=1,sx=112,sy=112,sz=112,excel=None,DATA_ROOT1='None',DATA_ROOT2='None',cs=["",""],afs=["",""],label_num=3,case_training='pre',nclass=3,loops=24, aug=False, fabric=None,data='top') -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Build train, validation, and test loaders for pre-training, tuning, or classification."""
    portion=0.6
    portion2=0.2
    if data=='top':
        if cs[0][0]=='L':
            cx,cy,cz=45,122,171
            sx1,sy1,sz1=90,190,160
        elif cs[0][0]=='R':
            cx,cy,cz=115,35,171
            sx1,sy1,sz1=90,190,160
        else:
            cx,cy,cz=75,75,75
        print((cs[0]))
    elif sx==None:
        cx,cy,cz=0,0,0
        sx1,sy1,sz1=96,96,96
    else:
        if cs[0][0]=='L':
            cx,cy,cz=60,109,85
            sx1,sy1,sz1=90,190,160
            print('here........')
        elif cs[0][0]=='R':
            cx,cy,cz=130,9,85
            sx1,sy1,sz1=90,190,160
        else:
            cx,cy,cz=75,75,75
        print((cs[0]))


    if case_training=='class':
        aug_transform= transforms.Compose([
        transforms.LoadImaged(keys=["label"],image_only=False, ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
        transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),])

        pre_train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["label"],image_only=False, ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
        transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
             transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),])

        pre_val_transform = transforms.Compose(
        [transforms.LoadImaged(keys=["label"], image_only=False,ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
     transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
      transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
        ])
    else:
        aug_transform= transforms.Compose([
        transforms.LoadImaged(keys=["label"],image_only=False, ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
        transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
            transformation_utils.SpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=[sx,sy,sz],
                image_size=[sx1,sy1,sz1]),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),])

        pre_train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["label"],image_only=False, ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
            transformation_utils.SpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=[sx,sy,sz],
                image_size=[sx1,sy1,sz1]),
             transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
        ])

        pre_val_transform = transforms.Compose(
        [transforms.LoadImaged(keys=["label"], image_only=False,ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
     transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
      transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        transforms.NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
        ])
        train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["label"],ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        transformation_utils.ConvertToMultiChannelsulcalClassesd(keys="label"),
        transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
            transformation_utils.SpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=[sx,sy,sz],
                image_size=[90,190,160]
            ),
            transforms.Lambdad(keys='label', func=lambda x: 1 if int(x.max())==2 else 0, overwrite='class'),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
        val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["label"],ensure_channel_first=True),
        transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        transformation_utils.ConvertToMultiChannelsulcalClassesd(keys="label"),
        transforms.SpatialCropd(keys=["image","label"],roi_center=(cx,cy,cz), roi_size=(sx1,sy1,sz1)),
      transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        transforms.Lambdad(keys='label', func=lambda x: 1 if int(x.max())==2 else 0, overwrite='class'),
        transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
        
        )
    if case_training=='tuning':

        multitask = GraphImageDataset(channels=channels,data_path=DATA_ROOT1, data_path2=DATA_ROOT2, transform=train_transform, sx=sx,sy=sy,sz=sz,excel=excel, cs=cs, afs=afs,pre=False,data=data, nclass=nclass)
        whole=int(len(multitask))
        mid=int((len(multitask)*portion))
        mid2=mid+int((len(multitask)*portion2))
        gen1 = torch.Generator().manual_seed(whole)
        whole_sampler = torch.utils.data.RandomSampler(multitask,num_samples=whole, generator=gen1)
        indices=list(whole_sampler)
        train_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:mid])
        multitask2 = GraphImageDataset(channels=channels,data_path=DATA_ROOT1, data_path2=DATA_ROOT2, transform=val_transform, sx=sx,sy=sy,sz=sz,excel= excel,cs=cs,afs=afs,pre=False,data=data, nclass=nclass)
        whole_sampler2 = torch.utils.data.RandomSampler(multitask2,num_samples=whole, generator=gen1)
        indices2=list(whole_sampler2)
        valid_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid:mid2])
        test_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid2:])
        print('stat: ',whole,mid,mid2)
        trainloader = torch.utils.data.DataLoader(multitask, batch_size=batch_size,  sampler=train_sampler)
        validloader= torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=valid_sampler)
        testloader = torch.utils.data.DataLoader(multitask2, batch_size=1,  sampler=test_sampler)
        return trainloader, validloader, testloader

    if case_training=='pre' or case_training=='class':

        multitask_ = GraphImageDataset(channels=channels,data_path=DATA_ROOT1, data_path2=DATA_ROOT1, transform=pre_train_transform, sx=sx,sy=sy,sz=sz,excel= excel,cs=cs,afs=afs,data=data, nclass=nclass)
        whole=int(len(multitask_))
        mid=int((len(multitask_)*portion))
        mid2=mid+int((len(multitask_)*portion2))
        aug_case=mid+mid-1
        
        if aug==True:
            gen = torch.Generator().manual_seed(aug_case)
            aug_multitask = GraphImageDataset(channels=channels,data_path=DATA_ROOT1, data_path2=DATA_ROOT1, transform=aug_transform, sx=sx,sy=sy,sz=sz,excel= excel,cs=cs,afs=afs,data=data, nclass=nclass)
            multitask=torch.utils.data.ConcatDataset([multitask_,aug_multitask])
            whole_sampler = torch.utils.data.SequentialSampler(multitask)
            indices_conc=list(whole_sampler)
            mid_whole=whole+mid
            ind= [indices_conc[:mid],indices_conc[whole:mid_whole]]
            indices=[j for i in ind for j in i]
            print('the length of the training dataset is: ' ,len(indices))
            gen = torch.Generator().manual_seed(len(indices))
            if is_distributed:
                train_sampler=torch.utils.data.distributed.DistributedSampler(indices,num_replicas=fabric.world_size, rank=fabric.global_rank)
            else:
                train_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices, generator=gen)

        else:
            multitask=multitask_
            whole=int(len(multitask))
            mid=int((len(multitask)*portion))
            mid2=mid+int((len(multitask)*portion2))
            gen = torch.Generator().manual_seed(mid)
            whole_sampler = torch.utils.data.SequentialSampler(multitask)
            indices=list(whole_sampler)
            if is_distributed:
                train_sampler=torch.utils.data.distributed.DistributedSampler(indices[:mid],num_replicas=fabric.world_size, rank=fabric.global_rank)
            else:
                train_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:mid], generator=gen)

        multitask2 = GraphImageDataset(channels=channels,data_path=DATA_ROOT1, data_path2=DATA_ROOT1, transform=pre_val_transform, sx=sx,sy=sy,sz=sz,excel= excel,cs=cs,afs=afs,data=data, nclass=nclass)
        whole_sampler2 = torch.utils.data.SequentialSampler(multitask2)
        indices2=list(whole_sampler2)
        print('stat: ',whole,mid,mid2)
        valid=mid+mid2-1
        test=whole-mid2-1
        genv = torch.Generator().manual_seed(valid)
        gent = torch.Generator().manual_seed(test)
        if fabric!=None:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(indices2[mid:mid2],num_replicas=fabric.world_size, rank=fabric.global_rank)
            test_sampler=torch.utils.data.distributed.DistributedSampler(indices2[mid2:],num_replicas=fabric.world_size, rank=fabric.global_rank)
        else:
            valid_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid:mid2],generator=genv)
            test_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid2:], generator=gent)
            
        trainloader = torch.utils.data.DataLoader(multitask, batch_size=batch_size,  sampler=train_sampler)
        validloader= torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=valid_sampler)
        testloader = torch.utils.data.DataLoader(multitask2, batch_size=1,  sampler=test_sampler)

        return trainloader, validloader, testloader

def data_build(channels=1,bz=1,datan='None',data2n='None',excel='None',sp=[112,112,112],cs=["",""],afs=["",""],case_training='pre',nclass=3,loops=12, aug=False, fabric=None,data='top'):
    """Convenience wrapper around `load_data` with project naming defaults."""
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, validloader, testloader = load_data(channels=channels,batch_size=bz,sx=sp[0],sy=sp[1],sz=sp[2],excel=excel,DATA_ROOT1=datan,DATA_ROOT2=data2n,cs=cs,afs=afs,case_training=case_training,nclass=nclass,loops=loops, aug=aug, fabric=fabric,data=data)
    return trainloader, validloader, testloader
