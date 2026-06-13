"""Fine-tuning entry point for adapting pretrained 3D models."""

from __future__ import division, print_function

import os

import monai
import torch
from functools import partial
from generative.networks.schedulers import DDPMScheduler
from lightning.pytorch import seed_everything
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.utils.enums import MetricReduction

from MTSC_sulci.utilities import DiffModel, SwiftUnet3D
from MTSC_sulci.utilities.CNN3D import DiffusionUnet, SwinMT
from MTSC_sulci.utilities.load_data import data_build, save_nii_valid
from MTSC_sulci.utilities.monai_utils import sliding_window_inference
from MTSC_sulci.utilities.trainer import device, trainer
from MTSC_sulci.utilities.trainer_lit import trainer_lit

torch.multiprocessing.set_sharing_strategy('file_system')


seed_everything(42,workers=True)

def fine(excel=None, afs=["",""],bc=["",""], batch_n = 2, nclass=3, val_every = 2, num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="", model_name_pre= "pre_training_model_swift.pt", model_name="fine_tuning_model_swift.pt",roi_size=64,l1=0.8,l2=0.2,save_nii=False,lr=5e-3,case='pre',focus_label='first',loops=24,kind='step',upscale=64,light=False,aug_data=False,data='top',fabric=None):
    """Fine-tune a pretrained SwinMT or diffusion model for sulcal labels."""

    torch.backends.cudnn.benchmark = True
    gdfocal=monai.losses.GeneralizedDiceFocalLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=False, other_act=None, smooth_nr=1e-05, smooth_dr=1e-05, batch=False, gamma=2.0, focal_weight=None, weight=None, lambda_gdl=1.0, lambda_focal=1.0)
    _loss=gdfocal
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False,num_classes=None)
    shapex=24
    train_loader, val_loader, test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,sp=[roi_size,roi_size,roi_size],cs=bc,afs=afs,case_training='tuning',loops=loops, aug=aug_data,fabric=fabric,data=data)
    sw_batch_size = int(4*batch_n)
    infer_overlap = 0.5
    if case !='diff/':
        net = SwinMT(shapex,nclass,roi_size)
        inferer = None
    else:
        shapex=36
        net = DiffusionUnet(shapex,nclass)
        scheduler = DDPMScheduler(num_train_timesteps=20, schedule="scaled_linear_beta", beta_start=(0.005), beta_end=0.02)
        inferer = DiffModel.DiffusionInferer(scheduler)
        kind='diff'

    net = net.to(device)
    model_inferer = partial(sliding_window_inference,roi_size=[upscale,upscale,upscale],sw_batch_size=sw_batch_size,predictor=net,overlap=infer_overlap,buffer_steps=1,buffer_dim=0)
    post_label = AsDiscrete(argmax=False, to_onehot=2)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
    wd_optimizer=1e-4
    opt1=50
    opt2=0.1
    start_epoch = 0
    if light==True:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer_lit(model=[net],train_loader=train_loader,val_loader=val_loader,optimizer=[lr,wd_optimizer],scheduler=[kind,opt1,opt2],inferer=inferer,model_inferer=[roi_size,sw_batch_size,infer_overlap],start_epoch=start_epoch,post_label=post_label,post_pred=post_pred,nclass=nclass,batch_size=batch_n,val_every = val_every,max_epochs=num_epochs,loss_seg=[_loss],model_name=model_name,metric_seg=dice_acc,l1=l1,l2=l2,model_name_pre= model_name_pre,PATH=PATH,constract=False,pre=False,fine_tuning=focus_label,loops=loops,roi_size=roi_size,fabric=fabric)
    else:
        (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer(model=[net],train_loader=train_loader,val_loader=val_loader,optimizer=[lr,wd_optimizer],scheduler=[kind,opt1,opt2],inferer=inferer,model_inferer=[roi_size,sw_batch_size,infer_overlap],start_epoch=start_epoch,post_label=post_label,post_pred=post_pred,nclass=nclass,batch_size=batch_n,val_every = val_every, max_epochs=num_epochs,loss_seg=[_loss],model_name=model_name,metric_seg=dice_acc,l1=l1,l2=l2,model_name_pre= model_name_pre,PATH=PATH,constract=False,pre=False,fine_tuning=focus_label,loops=loops,roi_size=roi_size)

    ext=SwiftUnet3D.Extend(nclass).to(device)
    net2=torch.nn.Sequential(net,ext)
   
    if save_nii==True:
        print('save')
        prefix_store_path=PATH+'/test_samples/'
        store_path1=PATH+'/test_samples/fine/'
        store_path=PATH+'/test_samples/fine/'+case
        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)

        if os.path.isdir(store_path1):
            print('folders exist')
        else:
            os.mkdir(store_path1)

        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        save_nii_valid(test_loader,net2,PATH, model_name,store_path,upscale,1,post_pred)
