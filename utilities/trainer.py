"""Core PyTorch training loops for pre-training and fine-tuning."""

from __future__ import division, print_function

from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import os
import wandb
import torchvision
import torchvision.transforms as T
import torch.nn.functional as nnf
import torch
import time
import monai
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from MTSC_sulci.utilities import SwiftUnet3D, monai_utils
from MTSC_sulci.utilities.SwiftUnet3D import *
from MTSC_sulci.utilities.monai_utils import *
from functools import partial
from monai.losses import DiceLoss,SSIMLoss
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from MTSC_sulci.utilities import load_data, DiffModel
from monai.config import print_config
from monai.metrics import DiceMetric,ROCAUCMetric,HausdorffDistanceMetric,ConfusionMatrixMetric,MSEMetric,MultiScaleSSIMMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial
from MTSC_sulci.utilities.preprocessing import *
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora
Device = torch.device( "cpu")
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class ContrastiveLoss(torch.nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, temperature=1.0):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return nnf.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]

       z_i = nnf.normalize(proj_1, p=2, dim=1)
       z_j = nnf.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)
           
       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)
       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)

       return loss


def discriminator_loss(disc_loss,gen_images, real_images,gen_images_d, real_images_d):
    """
    The discriminator loss if calculated by comparing its
    prediction for real and generated images.

    """
    real = real_images.new_full((real_images.shape[0], 1), 1)
    gen = gen_images.new_full((gen_images.shape[0], 1), 0)

    realloss = disc_loss(real_images_d, real)
    genloss = disc_loss(gen_images_d, gen)

    return (realloss + genloss) / 2


def generator_loss(gen_loss,output):
    """
    The generator loss is calculated by determining how well
    the discriminator was fooled by the generated images.

    """
    gen = output.new_full(output.shape, 1)
    return gen_loss(output, gen)


def trainer(model,train_loader,val_loader,optimizer,scheduler,inferer=None,model_inferer=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    nclass=2,
    batch_size=1,
    max_epochs=10,
    loss_seg=None,
    model_name="_model.pt",
    metric_seg=None,
    l1=0.7,
    l2=0.3,
    model_name_pre= None,
    PATH="/content/gdrive/MyDrive/Colab/Workshop/"
    ,constract=False,pre=True, timestep=100, loops=12, fine_tuning='first',attention_encoder=False,roi_size=32):
    """Coordinate setup, checkpoint loading, training, validation, and W&B logging."""
    val_acc_max = 0.0
    metric_seg1_ = []
    haus_ = []
    f1_ = []
    roc_ = []
    val_avg_acc_=[]
    loss_epochs = []
    trains_epoch = []
    val_every = 2
    root_dir=PATH
    loss=100000
    focus_label=0
    print('pre status: ',pre)
    if pre==True:
        project_n="Three_class_Normalize_Pre_training_SwinUnet_36_crop"
    else:
        project_n="Three_class_Normalized_Fine_training_SwinUnet_crop_36"

    # Public releases must not ship a personal W&B key. Set WANDB_API_KEY in
    # your environment, or set WANDB_MODE=offline/disabled before training.
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
    #    # Set the project where this run will be logged
         project=project_n, #"prova-project-ovarian"
    #    # Track hyperparameters and run metadata
         config={"learning_rate": 1e-4,"epochs": max_epochs, "batch" : batch_size })


    if (model_name_pre!= None) and (len(model)<2):
        if fine_tuning=='decoder':
            mod=model[0]
            mod.load_state_dict(torch.load(os.path.join(root_dir, model_name_pre),map_location=torch.device(device))["state_dict"], strict=False)
            mod.to(device)
            if os.path.exists(root_dir+model_name):
                mod.load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))["state_dict"], strict=False)
                mod.to(device)
                checkpoint2 = torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))
                #val_acc_max=checkpoint2["best_acc"]
                #loss=checkpoint2["train_loss"]
            stop=0
            for param in mod.parameters():
                stop=stop+1
            o=0
            print('number of parameter of model: ,',stop)
            for param in mod.parameters():
                if o <= int((stop)/2) and o!=0:
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
            focus_label=0
            if attention_encoder==True:
                ext=SwiftUnet3D.ExtendAttention(nclass).to(device)
            else:
                ext=SwiftUnet3D.Extend(nclass).to(device)
            if inferer!=None:
                diff_model=model[0]
                mod1=ext
            else:
                mod1=torch.nn.Sequential(model[0],ext)
            model[0]=mod1
            parameters=model[0].parameters()
        else:
            if attention_encoder==True:
                ext=SwiftUnet3D.ExtendAttention(nclass).to(device)
            else:
                ext=SwiftUnet3D.Extend(nclass).to(device)

            mod=model[0].to(device)
            mod.load_state_dict(torch.load(os.path.join(root_dir, model_name_pre),map_location=torch.device(device))["state_dict"], strict=False)
            #mod.to(device)
            if os.path.exists(root_dir+model_name):
                mod.load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))["state_dict"], strict=False)
                mod.to(device)
                checkpoint2 = torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))
                #val_acc_max=checkpoint2["best_acc"]
                #loss=checkpoint2["train_loss"]
            stop=0
            for param in mod.parameters():
                stop=stop+1
            o=0

            if fine_tuning=='top':
                for param in model[0].parameters():
                    if o <= int(90*(stop)/100) and o!=0:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                focus_label=0
                if inferer!=None:
                    diff_model=model[0]
                    mod=ext
                else:
                    mod=torch.nn.Sequential(model[0],ext)
               
                mod1=mod
                model[0]=mod1.to(device)
                parameters=model[0].parameters()

            elif fine_tuning=='LoRA':
                print('LoRA is coming!!')
                if inferer!=None:
                    diff_model=model[0]
                    mod1=ext
                else:
                    mod1=torch.nn.Sequential(mod.to(Device),ext.to(Device)).to(Device)
                model[0]=mod1.to(Device)
                add_lora(mod1)
                parameters=[{"params": list(get_lora_params(mod1))},]
                focus_label=0
            elif fine_tuning=='full':
                for param in model[0].parameters():
                    param.requires_grad = True
                focus_label=0
                if inferer!=None:
                    diff_model=model[0]
                    mod1=ext
                    model[0]=mod1.to(device)
                    parameters=model[0].parameters()
                    parameters2=diff_model.parameters()
                else:
                    mod1=torch.nn.Sequential(mod,ext)
                    model[0]=mod1.to(device)
                    parameters=model[0].parameters()
            else:
                print("Please use : 'full', 'top' or 'LoRA' ")
                            
        lr=optimizer[0]
        wd=optimizer[1] 
        optimizer=[]
        optimizer.append(torch.optim.AdamW(parameters, lr=lr, weight_decay=wd))
        if inferer!=None:
            optimizer.append(torch.optim.AdamW(parameters2, lr=(lr*0.1), weight_decay=wd))
            model=[]
            model.append(ext)
            model.append(diff_model)
        kind=scheduler[0]
        opt1=scheduler[1]
        opt2=scheduler[2]
        scheduler=[]
        if kind=='step':
            scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=opt1, gamma=opt2))
        elif kind=='cosin':
            scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=opt1))
        elif kind=='diff':
            scheduler.append(DDPMScheduler(num_train_timesteps=25, schedule="scaled_linear_beta", beta_start=(0.005), beta_end=0.02))
        else:
            print("No predifine common schedule")
        print('Test: ',len(scheduler), len(optimizer))
        #model_inferer=model_inferer(predictor=(model[0]))
        roi_size,sw_batch_size,infer_overlap=model_inferer[0],model_inferer[1],model_inferer[2]
        model_inferer = partial(sliding_window_inference,roi_size=[roi_size,roi_size,roi_size],sw_batch_size=sw_batch_size,predictor=model[0].to(device),overlap=infer_overlap,buffer_steps=1,buffer_dim=0)

    elif (model_name_pre==None) and len(scheduler)==3:
        mod=model[0].to(device)
        if attention_encoder==True:
            ext=SwiftUnet3D.ExtendAttention(nclass).to(device)
        else:
            ext=SwiftUnet3D.Extend(nclass).to(device)
        if inferer!=None:
            diff_model=model[0]
            model[0]=ext
        else:
            model[0]=torch.nn.Sequential(mod,ext).to(device)
        if os.path.exists(root_dir+model_name):
            model[0].load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))["state_dict"], strict=False)
            model[0].to(device)
            checkpoint2 = torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))
            val_acc_max=checkpoint2["best_acc"]
            loss=checkpoint2["train_loss"]
        stop=0
        for param in model[0].parameters():
            stop=stop+1
        o=0
        print('number of parameter of model: ,',stop)
        if fine_tuning=='decoder':
            for param in model[0].parameters():
                if o <= int((stop)/2) and o!=0:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            focus_label=0
            parameters=model[0].parameters()
            print(parameters)
        elif fine_tuning=='top':
            for param in model[0].parameters():
                if o <= int(90*(stop)/100) and o!=0:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            focus_label=0
            parameters=model[0].parameters()

        elif fine_tuning=='LoRA':
            print('LoRA is coming!!')
            add_lora(model[0].to(Device))
            parameters=[{"params": list(get_lora_params(model[0]))},]
            focus_label=0

        elif fine_tuning=='full':
            for param in model[0].parameters():
                param.requires_grad = True
            focus_label=0
            if inferer!=None:
                parameters=model[0].parameters()
                parameters2=diff_model.parameters()
            else:
                parameters=model[0].parameters()
        else:
            
            print("Please use : 'decoder', 'top' or 'LoRA' ")

        lr=optimizer[0]
        wd=optimizer[1]
        optimizer=[]
        optimizer.append(torch.optim.AdamW(parameters, lr=lr, weight_decay=wd))
        if inferer!=None:
            optimizer.append(torch.optim.AdamW(parameters2, lr=(lr*0.1), weight_decay=wd))
            model=[]
            model.append(ext)
            model.append(diff_model)
        kind=scheduler[0]
        opt1=scheduler[1]
        opt2=scheduler[2]
        scheduler=[]
        if kind=='step':
            scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=opt1, gamma=opt2))
        elif kind=='cosin':
            scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=opt1))
        elif kind=='diff':
            scheduler.append(DDPMScheduler(num_train_timesteps=20, schedule="scaled_linear_beta", beta_start=(0.005), beta_end=0.02))
        else:
            print("No predifine common schedule")
        print('Test: ',len(scheduler), len(optimizer))
        #model_inferer=model_inferer(predictor=(model[0]))
        roi_size,sw_batch_size,infer_overlap=model_inferer[0],model_inferer[1],model_inferer[2]
        model_inferer = partial(sliding_window_inference,roi_size=[roi_size,roi_size,roi_size],sw_batch_size=sw_batch_size,predictor=model[0].to(device),overlap=infer_overlap,buffer_steps=1,buffer_dim=0)

    elif (model_name_pre== None) and (len(model)==2):
        filename1='generator_'+model_name
        filename2='discriminator_'+model_name
        loss1=loss
        loss2=loss
        if os.path.exists(root_dir+filename1):
            model[0].load_state_dict(torch.load(os.path.join(root_dir, filename1),map_location=torch.device(device))["state_dict"])
            model[0].to(device)
            checkpoint2 = torch.load(os.path.join(root_dir, filename1),map_location=torch.device(device))
            val_acc_max=checkpoint2["best_acc"]
            loss1=checkpoint2["train_loss"]
        if os.path.exists(root_dir+filename2):
            model[1].load_state_dict(torch.load(os.path.join(root_dir, filename2),map_location=torch.device(device))["state_dict"])
            model[1].to(device)
            checkpoint2 = torch.load(os.path.join(root_dir, filename2),map_location=torch.device(device))
            val_acc_max=checkpoint2["best_acc"]
            loss2=checkpoint2["train_loss"]
        loss=(loss1+loss2)/2

    else:
        if os.path.exists(root_dir+model_name):
            model[0].load_state_dict(torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))["state_dict"])
            model[0].to(device)
            checkpoint = torch.load(os.path.join(root_dir, model_name),map_location=torch.device(device))
            val_acc_max=checkpoint["best_acc"]
            loss=checkpoint["train_loss"]
            print('MAX acc: ', val_acc_max)

    print('The analysis will start with max accuracy: ', val_acc_max)
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss_t,loss1,loss2 = train_epoch(
            model,
            inferer,
            train_loader,
            optimizer,
            epoch=epoch,
            nclass=nclass,
            loss_seg=loss_seg,
            max_epochs=max_epochs,
            l1=l1,
            l2=l2,
            constract=constract,
            batch_size= batch_size,
            loops=loops,
            scheduler=scheduler,pre=pre
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss_t),
            "loss_1: {:.4f}".format(loss1),
            "loss_2: {:.4f}".format(loss2),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss_t)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc6 = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
                nclass=nclass,
                max_epochs=max_epochs,
                metric_seg=metric_seg,
                inferer=inferer,
                batch_size= batch_size,
                timestep=timestep,
                focus_label=focus_label,pre=pre,roi_size=roi_size,
            )
            metric_seg1 = val_acc
            f1 = val_acc1
            roc = val_acc2
            haus=val_acc3
            val_acc_class=np.nanmean([val_acc1,val_acc2,val_acc4,val_acc5,val_acc6])
            val_avg_acc = 100*np.sum([l1*val_acc, l2*val_acc_class])
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", Total_:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            metric_seg1_.append(metric_seg1)
            f1_.append(f1)
            roc_.append(roc)
            haus_.append(haus)
            val_avg_acc_.append(val_avg_acc)
            if val_avg_acc >= val_acc_max: #and train_loss_t <= loss:
                print("SAVE NEW BEST!!!!!! ({:.6f} ---> {:.6f}). {:.6f} ---> {:.6f}).)".format(val_acc_max, val_avg_acc,loss,train_loss_t))
                val_acc_max = val_avg_acc
                loss= train_loss_t
                save_checkpoint(
                    model,
                    epoch,
                    filename=model_name,
                    best_acc=val_acc_max,
                    train_loss=loss,
                    dir_add=root_dir
                )

            wandb.log({"loss_total": train_loss_t, "loss_1":loss1,"loss_2":loss2,"acc_val": val_acc, "acc_val1": val_acc1, "acc_val2": val_acc2,"acc_val3": val_acc3, "acc_val4": val_acc4, "acc_val5": val_acc5, "acc_val6": val_acc6})

            if scheduler==None:
                print('no scheduler strategy for lr used fix lr.')
            elif inferer!=None:
                scheduler[0].set_timesteps(num_inference_steps=timestep)
            elif len(scheduler)==2:
                if epoch%2==0:
                    scheduler[0].step()
                else:
                    scheduler[1].step()
            else:
                scheduler[0].step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        metric_seg1_,
        f1_,
        roc_,
        haus_,
        val_avg_acc_,
        loss_epochs,
        trains_epoch,
    )

def train_epoch(model,inferer, loader, optimizer, epoch, nclass,max_epochs=10,loss_seg=None, l1=0.6, l2=0.4,constract=False,batch_size=1,loops=24,scheduler=None,pre=True):
    """Run one epoch for reconstruction, adversarial, contrastive, or diffusion training."""
    model_save=[]
    post_sigmoid = Activations(sigmoid=True)
    loss_function=torch.nn.CrossEntropyLoss()
    CLR_aug = transforms.Compose([
            transforms.RandFlip( prob=0.5, spatial_axis=1),
            transforms.RandFlip(  prob=0.5, spatial_axis=2),
            transforms.RandFlip( prob=0.5, spatial_axis=0),
            transforms.RandAdjustContrast(prob=0.5),
            transforms.RandScaleIntensity(factors=0.1, prob=0.5),
            transforms.RandShiftIntensity(offsets=0.1, prob=0.5),])
    if len(model)==2: #len(loss_seg)==2:
        model[0].train()
        model[1].train()
        model_save=model
    else:
        model[0].train()
        model_save=model
    g=True
    d=False
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss1 = AverageMeter()
    run_loss2 = AverageMeter()
    post_pred = Compose([Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=nclass)])
    loss=0
    lossd=0
    lossg=0
    count1=0
    losst1=0
    lossg1=0
    lossd1=0
    lossdiff_total=0
    scaler = GradScaler('cuda')
    scaler2 = GradScaler('cuda')
    print('memory allocate:')
    print(torch.cuda.memory_allocated(0))
    print('max memory allocate:')
    print(torch.cuda.max_memory_allocated(0))
    print('the loss sem length is: ',len(loss_seg))
    for idx, batch_data in enumerate(loader):
        loss1=0
        loss_g=0
        loss_d=0
        loss2diff=0
        for i in range(loops):
             datad, labelsd, targetd = batch_data[i]['image'], batch_data[i]['class'], batch_data[i]['label']
             data,labels, target = datad.to(device), labelsd.to(device), targetd.to(device)
             #target=torch.squeeze(target)
           #  model[0]=model[0].to(device)
             model=model_save
             
             if len(loss_seg)==3:
                 model[1]=model[1].to(device)
                 model[0]=model[0].to(device)
                 latent = torch.randn(data.shape).to(device)
                 gen,classes = model[0](data)
                 fake1=model[1](gen)
                 real=model[1](data)
                 loss_disc=discriminator_loss(loss_seg[0],gen, target, fake1, real)
                 loss_gen=generator_loss(loss_seg[1],fake1)
                 loss=loss_seg[2](gen,target.detach())
                 #losst=((loss_disc+loss_gen)/2)
                 loss_g=loss_g+loss_gen.detach()
                 loss_d=loss_d+loss_disc.detach()
                 if epoch%2==0:
                     loss_gen.backward()
                     optimizer[0].step()
                 elif (epoch)%3==0:
                     loss.backward()
                     optimizer[1].step()
                     optimizer[0].step()
                 elif (epoch+1)%2==0:
                     loss_disc.backward()
                     optimizer[1].step()
                 else:
                     print('error no epoch backprop...............')
                 loss1=loss.detach()+loss1
                 del gen,classes,fake1,real

             elif constract==True:
                 model[0]=model[0].to(device)
                 f1= monai.engines.utils.engine_apply_transform(batch=data,output=data,transform=CLR_aug)
                 f2= monai.engines.utils.engine_apply_transform(batch=data,output=data,transform=CLR_aug)
                 x1,x2=f1[1],f2[1]
                 z1,c1 = model[0](x1)
                 z2,c2 = model[0](x2)
                 loss_g = ContrastiveLoss(data.shape[0])(c1, c2)
                 if pre==True:
                     loss_bce1=(loss_seg[0](z1,x1))
                     loss_bce2=(loss_seg[0](z2,x2))
                 else:
                     loss_bce1=0.6*(loss_seg[0](z1,x1))+0.4*(loss_seg[1](z1,x1))
                     loss_bce2=0.6*(loss_seg[0](z2,x2))+0.4*(loss_seg[1](z2,x2))

                 loss_d=abs(0.5*loss_bce1+0.5*loss_bce2)
                 losst=0.6*loss_d+0.4*loss_g
                 if epoch%2==0:
                     loss_g.backward()
                     optimizer[0].step()
                 elif (epoch)%3==0:
                     losst.backward()
                     #optimizer[1].step()
                     optimizer[0].step()
                 elif (epoch+1)%2==0:
                     loss_d.backward()
                     optimizer[0].step()
                 else:
                     print('error no epoch backprop...............')
                 loss1=losst.detach()+loss1
                 del z1,c1,z2,c2,x1,x2,f1,f2

             elif inferer!=None:
                 time_step=0
                 losstep=0
                 lossdiff_t=0
                 with autocast('cuda',enabled=True):
                     noise = torch.randn_like(data).to(device)
                     #noise = torch.ones_like(data).to(device)
                     timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (data.shape[0],), device=data.device).long() 
                     #datai=data.detach().clone()
                     #data=data[torch.randper]
                     in_image=(0.65*data+0.35*noise) #if normalized both need to have 1 max value so if there are overlabs of 1 and 1 need to go 1
                     #    change the data order to put characteristics of other samples :)
                     if pre==True:
                         model[0]=model[0].to(device)                     
                         logits,classes = inferer(inputs=data.detach(), diffusion_model=model[0], noise=noise.detach(), timesteps=timesteps,condition=data.detach())
                         lossdiff_t=0
                         
                     else:
                         model[0]=model[0].to(device)
                         model[1]=model[1].to(device)
                         logit,classe = inferer(inputs=data, diffusion_model=model[1], noise=noise, timesteps=timesteps,condition=data)
                         logits,classes = model[0]([logit,classe])
                         lossdiff=loss_seg[0](logit,noise)
                         lossdiff_t+=lossdiff.detach()
                         del logit,classe, noise
                     loss_classifier = loss_function(classes,labels.long())                     
                     if pre==True:
                         loss_segmentation = loss_seg[0](logits,target)
                     else:
                         loss_segmentation = 0.6*loss_seg[0](logits,target)+0.4*loss_seg[1](logits,target)
                     losst=l2*loss_classifier+l1*loss_segmentation
                     if pre==True:
                         scaler.scale(losst).backward()
                         scaler.step(optimizer[0])
                         scaler.update()
                     else:
                         if epoch%2==0:
                             scaler.scale(losst).backward()
                             scaler.step(optimizer[0])
                             scaler.update()
                         else:
                             scaler2.scale(lossdiff).backward()
                             scaler2.step(optimizer[1])
                             scaler2.update()

                     losstep=losst.detach()+losstep
                     time_step+=1
                 loss1=(losstep.detach()/time_step)+loss1
                 loss2diff=(lossdiff_t.detach()/time_step)+loss2diff
                 del noise, in_image, logits,classes, losst,
             else:
                 model[0]=model[0].to(device)
                 logits,classes = model[0](data)
                 loss_classifier = loss_function(classes,labels.long())
                 if pre==True:
                     loss_segmentation = loss_seg[0](logits,target)
                 else:
                     lseg=loss_seg[0](logits,target)
                     loss_segmentation = lseg #0.6*lseg[0]+0.4*lseg[1]
                 losst=l2*loss_classifier+l1*loss_segmentation
                 losst.backward()
                 optimizer[0].step()
                 loss1+=(losst.detach())
                 del logits,classes
             del model
             #data=data.to('cpu')
             #datad=datad.to('cpu')
             #labels=labels.to('cpu')
             #labelsd=labelsd.to('cpu')
             #target=target.to('cpu')
             #targetd=targetd.to('cpu')
        loss2=loss1/loops
        lossd2=loss_d/loops
        lossg2=loss_g/loops
        loss2diff=loss2diff/loops
        losst1+=loss2
        lossd1+=lossd2
        lossg1+=lossg2
        lossdiff_total+=loss2diff
        count1=+1
        del loss2, lossd2, lossg2, loss1, loss_d, loss_g, loss2diff
    loss=losst1/count1
    lossd=lossd1/count1
    lossg=lossg1/count1
    lossdiff_total=lossdiff_total/count1
    #if (len(loss_seg)==3):
    #    if epoch%2==0:
    ##        lossg.backward()
    ##        optimizer[0].step()              
    #    elif (epoch)%3==0:
    #        loss.backward()
    #        optimizer[1].step()
    #        optimizer[0].step()

     #   elif (epoch+1)%2==0:
     #       lossd.backward()
     #       optimizer[1].step()
     #   else:
    #elif (constract==True):
    #    if epoch%2==0:
    #        lossg.backward()
    #        optimizer[0].step()

     #   elif (epoch)%3==0:
     #       loss.backward()
     #       optimizer[0].step()

      #  elif (epoch+1)%2==0:
      #      lossd.backward()
      #      optimizer[0].step()
      #  else:
   # elif inferer!=None:
   #     if pre==True:
   #         scaler.scale(loss).backward()
   #         scaler.step(optimizer[0])
   #         scaler.update()
   #     else:
   #         scaler.scale(loss).backward()
   #         scaler.step(optimizer[0])
   #         scaler.update()
   #         scaler2.scale(lossdiff_total).backward()
   #         scaler2.step(optimizer[1])
   #         scaler2.update()
    #else:
        #loss.backward()
        #optimizer[0].step()

    #run_loss.update(loss.item(), n=1)
    run_loss.update(loss.cpu(), n=1)
    #if len(loss_seg)==3:
    #    run_loss1.update(lossd, n=1) #run_loss1.update(lossd.item(), n=1)
    #    run_loss2.update(lossg, n=1) #run_loss2.update(lossg.item(), n=1)
    #    loss1=run_loss1.avg
    #    loss2=run_loss2.avg

    #elif (constract==True):
    #    run_loss1.update(lossd, n=1) #run_loss1.update(lossd.item(), n=1)
    #    run_loss2.update(lossg, n=1) #run_loss2.update(lossg.item(), n=1)
    #    loss1=run_loss1.avg
    #    loss2=run_loss2.avg
    #else:
    loss1=0
    loss2=0
    print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
    start_time = time.time()
    return run_loss.avg, loss1, loss2


def val_epoch(
    model,
    loader,
    epoch,
    model_inferer=None,
    post_label=None,
    post_pred=None,
    nclass=2,
    max_epochs=10,
    metric_seg=None,
    inferer=None,
    batch_size=1,
    timestep=100,
    focus_label=0,
    pre=True,
    roi_size=32,
):
    """Evaluate a model and aggregate reconstruction plus classification metrics."""
    if len(model)==1:
        model[0].eval()
    else:
        model[0].eval()
        model[1].eval()
        if pre==False:
            model_inferer2 = partial(sliding_window_inference,roi_size=[roi_size,roi_size,roi_size],sw_batch_size=int(4*batch_size),predictor=model[0],overlap=0.5,buffer_steps=1,buffer_dim=0)

    start_time = time.time()
    run_acc = AverageMeter()
    run_acc1 = AverageMeter()
    run_acc2 = AverageMeter()
    run_acc3 = AverageMeter()
    run_acc4 = AverageMeter()
    run_acc5 = AverageMeter()
    run_acc6 = AverageMeter()
    
    haus_acc=HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=True, directed=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    roc=ROCAUCMetric(average="macro")
    f1=ConfusionMatrixMetric(metric_name="f1 score",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    sens=ConfusionMatrixMetric(metric_name="sensitivity",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    spec=ConfusionMatrixMetric(metric_name="specificity",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    prec=ConfusionMatrixMetric(metric_name="precision",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    post_sigmoid = Activations(sigmoid=True)
    f1.reset()
    spec.reset()
    sens.reset()
    prec.reset()
    roc.reset()
    metric_seg.reset()
    haus_acc.reset()
    model_inferer=model_inferer
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            loss1=0
            targets_total=torch.tensor([])
            logits_total=torch.tensor([])
            labels_total=torch.tensor([])
            classes_total=torch.tensor([])
            datad, labelsd, targetd = batch_data['image'], batch_data['class'], batch_data['label']
            data,labels, target = datad.to(device), labelsd.to(device), targetd.to(device)
            target=torch.squeeze(target)
            if inferer==None:
                logit,classes = model_inferer(data)
            else:
                targ=target.type(torch.FloatTensor)
                noise = torch.randn_like(input=data, device=device)
                #noise = torch.ones_like(data).to(device)
                scheduler = DDPMScheduler(num_train_timesteps=timestep, schedule="scaled_linear_beta", beta_start=0.005, beta_end=0.02)
                inferer = DiffModel.DiffusionInferer(scheduler)
                #in_image=(0.65*data+0.35*noise)

                if pre==True:
                    logit,classes = inferer.sample(input_noise=noise, diffusion_model=model_inferer, scheduler=scheduler,conditioning=data)
                    #val_outputs_list = decollate_batch(logit)
                    #val_output_convert = [(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                else:
                    logits,classe = inferer.sample(input_noise=noise, diffusion_model=model_inferer, scheduler=scheduler,conditioning=data)
                    logit,classes = model_inferer2([logits,classe])
                    #val_outputs_list = decollate_batch(logit)
                    #val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                del noise,logits,classe
            #val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logit)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_labels_convert = target #[(val_pred_tensorgt) for val_pred_tensorgt in val_labels_list]
            if len(val_labels_convert.shape)==4:
                val_labels_convert=torch.unsqueeze(val_labels_convert,dim=1)
            val_output_convert=torch.stack(val_output_convert,dim=0)
             #metric_seg.reset()
             #roc.reset()
             #haus_acc.reset()
             #f1.reset()
             #spec.reset()
             #sens.reset()
             #prec.reset()
            classes_t=classes #torch.argmax(classes,dim=1)
            labels_t=post_label(torch.unsqueeze(labels,dim=0))
            labels_t=torch.permute(labels_t,(1,0))
            metric_seg(y_pred=val_output_convert , y=val_labels_convert)
            haus_acc(y_pred=val_output_convert , y=val_labels_convert)
            acc3, not_nans3 = haus_acc.aggregate()
            roc(y_pred=classes_t, y=labels_t)
            f1(y_pred=classes_t, y=labels_t)
            spec(y_pred=classes_t, y=labels_t)
            sens(y_pred=classes_t, y=labels_t)
            prec(y_pred=classes_t, y=labels_t)
            acc, not_nans = metric_seg.aggregate()
            acc1 = roc.aggregate()
            acc2_total = (f1.aggregate())
            acc4_total = (spec.aggregate())
            acc5_total = (sens.aggregate())
            acc6_total = (prec.aggregate())
            acc2=torch.nanmean(acc2_total[0])
            acc4=torch.nanmean(acc4_total[0])
            acc5=torch.nanmean(acc5_total[0])
            acc6=torch.nanmean(acc6_total[0])
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            run_acc2.update(acc2.cpu().numpy())
            run_acc1.update(acc1)
            run_acc3.update(acc3.cpu().numpy(), n=not_nans3.cpu().numpy())
            run_acc4.update(acc4.cpu().numpy())
            run_acc5.update(acc5.cpu().numpy())
            run_acc6.update(acc6.cpu().numpy())
            dice_v = run_acc.avg
            f1_v = run_acc2.avg
            roc_v= run_acc1.avg
            haus_v = run_acc3.avg
            specif=run_acc4.avg
            scens=run_acc5.avg
            precis=run_acc6.avg
            del logit, classes 
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice:",
                dice_v,
                ", f1:",
                f1_v,
                ", hausdorff:",
                haus_v,
                 ", ROC:",
                roc_v,
                                 ", specificity:",
                specif,
                                 ", sensitivity:",
                scens,
                                 ", precision:",
                precis,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg[focus_label], run_acc1.avg, run_acc2.avg, run_acc3.avg[focus_label], run_acc4.avg, run_acc5.avg, run_acc6.avg
