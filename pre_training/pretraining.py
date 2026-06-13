"""Pre-training entry points for 3D sulcal MRI representation learning."""

from __future__ import division, print_function

import os

import monai
import torch
import torch.nn as nn
from generative.networks.schedulers import DDPMScheduler
from lightning.pytorch import seed_everything
from monai.losses import SSIMLoss
from monai.metrics import MultiScaleSSIMMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.utils.enums import MetricReduction
from functools import partial

from MTSC_sulci.utilities.CNN3D import DiffusionUnet, Discrim, SwinMT
from MTSC_sulci.utilities.DiffModel import DiffusionInferer
from MTSC_sulci.utilities.load_data import data_build, save_nii_valid
from MTSC_sulci.utilities.monai_utils import sliding_window_inference
from MTSC_sulci.utilities.trainer import device, trainer
from MTSC_sulci.utilities.trainer_lit import trainer_lit


seed_everything(42,workers=True)


def pre(excel=None, afs=["",""],bc=["",""],batch_n = 2, nclass=3, val_every = 2,num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="",model_name="pre_training_model_swift.pt",roi_size=64,l1=0.8,l2=0.2,save_nii=False,lr=1e-3,wd=1e-2,loops=24,upscale=64,light=False,aug_data=False,data='top',fabric1=None):    
    """Train the baseline self-supervised reconstruction model."""
    torch.backends.cudnn.benchmark = True
    ssim_loss=SSIMLoss(spatial_dims=3,data_range=1,win_size=8,reduction="mean")
    issm_acc=monai.metrics.LossMetric(loss_fn=ssim_loss,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    shapex=24
    if light:
        fabric=fabric1
    else:
        fabric=None

    train_loader, val_loader, test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,sp=[roi_size,roi_size,roi_size],cs=bc,afs=afs,case_training='pre',loops=loops, aug=aug_data,fabric=fabric,data=data)
    sw_batch_size = int(4*batch_n)
    infer_overlap = 0.5
    net = SwinMT(shapex,nclass,roi_size,upscale)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = net
    model_inferer = [roi_size,sw_batch_size,infer_overlap]
    post_label = AsDiscrete(argmax=False, to_onehot=nclass)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    start_epoch = 0
    if light==True:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer_lit(
    model=[net],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer],
    scheduler=[scheduler],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[ssim_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH,loops=loops,roi_size=roi_size,fabric=fabric)
    else:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer(
    model=[net],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer],
    scheduler=[scheduler],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[ssim_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH,loops=loops,roi_size=roi_size)

    if save_nii==True:
        prefix_store_path=PATH+'/test_samples/pre/'
        store_path=PATH+'/test_samples/pre/reco/'
        
        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)
        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        save_nii_valid(test_loader,net, PATH, model_name,store_path,upscale,1,post_pred)

def gan_pre(excel=None, afs=["",""],bc=["",""],batch_n = 2, nclass=3,val_every = 2, num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="",model_name="pre_training_model_swift.pt",roi_size=64,l1=0.8,l2=0.2, save_nii=False,lr=1e-3,loops=24,upscale=64,light=False,aug_data=False,data='top',fabric1=None):
    """Train the adversarial pre-training variant with a discriminator."""
    torch.backends.cudnn.benchmark = True
    ssim_loss=SSIMLoss(spatial_dims=3,data_range=1,reduction="mean")
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=1,kernel_size=4, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    shapex=24
    if light:
        fabric=fabric1
    else:
        fabric=None
    train_loader, val_loader, test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,sp=[roi_size,roi_size,roi_size],cs=bc,afs=afs,case_training='pre',loops=loops, aug=aug_data,fabric=fabric,data=data)
    sw_batch_size = int(4*batch_n)
    infer_overlap = 0.5
    net = SwinMT(shapex,nclass,roi_size,upscale)
    disc= Discrim(roi_size)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        disc = nn.DataParallel(disc)
    net = net
    disc = disc
    disc_loss = torch.nn.BCELoss()
    gen_loss = torch.nn.BCELoss()
    tot_loss= SSIMLoss(spatial_dims=3)
    model_inferer = [roi_size,sw_batch_size,infer_overlap]
    post_label = AsDiscrete(argmax=False, to_onehot=nclass)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    optimizer2 = torch.optim.AdamW(disc.parameters(), lr=(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)
    start_epoch = 0
    if light==True:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer_lit(
    model=[net,disc],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer,optimizer2],
    scheduler=[scheduler,scheduler1],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[disc_loss,gen_loss,tot_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH,loops=loops,roi_size=roi_size,fabric=fabric)
    else:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer(
    model=[net,disc],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer,optimizer2],
    scheduler=[scheduler,scheduler1],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[disc_loss,gen_loss,tot_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH,loops=loops,roi_size=roi_size)

    if save_nii==True:
        prefix_store_path=PATH+'/test_samples/'
        store_path=PATH+'/test_samples/pre/'
        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)
        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        save_nii_valid(test_loader,net, PATH, 'generator_'+model_name,store_path,upscale,batch_n,post_pred)
def con_pre(excel=None, afs=["",""],bc=["",""],batch_n = 2, nclass=3, val_every = 2, num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="",model_name="pre_training_model_swift.pt",roi_size=64,l1=0.8,l2=0.2,save_nii=False,lr=1e-3,loops=24,upscale=64,light=False,aug_data=False,data='top',fabric1=None):
    """Train the contrastive pre-training variant."""
    torch.backends.cudnn.benchmark = True
    ssim_loss=SSIMLoss(spatial_dims=3,data_range=1,reduction="mean")
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=1, kernel_size=4, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    shapex=24
    if light:
        fabric=fabric1
    else:
        fabric=None
    train_loader, val_loader, test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,sp=[upscale,upscale,upscale],cs=bc,afs=afs,case_training='pre',loops=loops, aug=aug_data, fabric=fabric,data=data)
    sw_batch_size = int(4*batch_n)
    infer_overlap = 0.5
    net = SwinMT(shapex,nclass,roi_size,upscale)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)
    model_inferer = [roi_size,sw_batch_size,infer_overlap]
    post_label = AsDiscrete(argmax=False, to_onehot=nclass)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    start_epoch = 0

    if light==True:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer_lit(
    model=[net],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer],
    scheduler=[scheduler],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[ssim_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH, constract=True,loops=loops,roi_size=roi_size,fabric=fabric)
    else:        
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer(
    model=[net],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer],
    scheduler=[scheduler],
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[ssim_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH, constract=True,loops=loops,roi_size=roi_size)
    if save_nii==True:
        prefix_store_path=PATH+'/test_samples/pre/'
        store_path=PATH+'/test_samples/pre/constract/'

        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)
        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        save_nii_valid(test_loader,net, PATH, model_name,store_path,upscale,batch_n,post_pred)

def diff_pre(excel=None, afs=["",""],bc=["",""],batch_n = 2, nclass=3, val_every = 2, num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="",model_name="pre_training_model_swift.pt",roi_size=32,l1=0.8,l2=0.2, save_nii=False,lr=1e-3,loops=24,time=1000,upscale=64,data='top',light=False,fabric1=None):
    """Train the diffusion-based pre-training variant."""
    torch.backends.cudnn.benchmark = True
    ssim_loss=SSIMLoss(spatial_dims=3,data_range=1,reduction="mean")
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=1,kernel_size=4, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    shapex=36
    tim=time
    if light:
        fabric=fabric1
    else:
        fabric=None
    train_loader, val_loader, test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,sp=[roi_size,roi_size,roi_size],cs=bc,afs=afs,case_training='pre',loops=loops,fabric=fabric,data=data)
    sw_batch_size = 2
    infer_overlap = 0
    net = DiffusionUnet(shapex,nclass)
    net = net
    model_inferer = partial(sliding_window_inference,roi_size=[upscale,upscale,upscale],sw_batch_size=sw_batch_size,predictor=net,overlap=infer_overlap)
    post_label = AsDiscrete(argmax=False, to_onehot=nclass)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=tim, schedule="scaled_linear_beta", beta_start=(0.005), beta_end=0.02)
    inferer = DiffusionInferer(scheduler)
    start_epoch = 0
    (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer(
    model=[net],
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=[optimizer],
    scheduler=[scheduler],
    inferer=inferer,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred,
    nclass=nclass,
    batch_size=batch_n,
    val_every = val_every ,
    max_epochs=num_epochs,
    loss_seg=[ssim_loss],
    model_name=model_name,
    metric_seg=issm_acc,
    l1=l1,
    l2=l2,
    model_name_pre= None,
    PATH=PATH,timestep=tim,loops=loops,roi_size=roi_size)
    if save_nii==True:
        prefix_store_path=PATH+'/test_samples/'
        store_path=PATH+'/test_samples/diff/'
        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)
        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        save_nii_valid(test_loader,net, PATH, model_name,store_path,upscale,batch_n,post_pred,scheduler)
