"""Training and validation loops for classification models."""

from __future__ import division, print_function

import os
import time

import torch
import wandb
from generative.networks.schedulers import DDPMScheduler
from monai.metrics import ConfusionMatrixMetric, ROCAUCMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
)
from monai.utils.enums import MetricReduction

from MTSC_sulci.utilities.preprocessing import AverageMeter_cuda, save_checkpoint
 

def trainer_class(model1,train_loader,val_loader,optimizer1,scheduler1,
    start_epoch=0,  nclass=2,
    batch_size=1,
    max_epochs=10,
    model_name="_model.pt",
    PATH="/content/gdrive/MyDrive/Colab/Workshop/",fabric=None):
    """Train a classifier with Fabric and save the best validation checkpoint."""
    
    kind=scheduler1[0]
    opt1=scheduler1[1]
    opt2=scheduler1[2]
    val_acc_max = 0.0
    f1_ = []
    roc_ = []
    val_avg_acc_=[]
    loss_epochs = []
    trains_epoch = []
    val_every = 2
    root_dir=PATH
    loss=100000
    project_n="Classification_benemin"
    # Public releases must not ship a personal W&B key. Set WANDB_API_KEY in
    # your environment, or set WANDB_MODE=offline/disabled before training.
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project=project_n)
    model, optimizer= fabric.setup(model1, optimizer1)
    if kind=='step':
        scheduler=(torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt1, gamma=opt2))
    elif kind=='cosin':
        scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt1))
    elif kind=='diff':
        scheduler=(DDPMScheduler(num_train_timesteps=20, schedule="scaled_linear_beta", beta_start=(0.005), beta_end=0.02))
    else:
        print("No predifine common schedule")
        
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    best_model_state = model.state_dict() 
    print('The analysis will start with max accuracy: ', val_acc_max)
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss_t = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            fabric=fabric
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss_t),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss_t)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            
            val_acc1, val_acc2,val_acc4, val_acc5, val_acc6 = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                nclass=nclass,
                max_epochs=max_epochs,
                fabric=fabric
            )
            f1 = val_acc1
            roc = val_acc2
            l1=0.6
            l2=0.4
            val_acc_class=torch.tensor([val_acc4,val_acc2,val_acc5,val_acc6]).nanmean()
            val_avg_acc = 100*torch.sum(torch.tensor([l1*val_acc1, l2*val_acc_class]))
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", Total_:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            f1_.append(f1)
            roc_.append(roc)
            val_avg_acc_.append(val_avg_acc)
            save_point=False
            print('gather all the results')
            all_criteria = fabric.all_gather(val_avg_acc)
            print('Fine-tuning average errors for saving the weights')
            if (val_avg_acc == torch.max(all_criteria)):
                if val_avg_acc >= val_acc_max: #and train_loss_t <= loss:
                    save_point=True
                    val_acc_max = val_avg_acc
                    best_model_state = model.state_dict()  # Store the best model state
                else:
                    save_point=False
            else:
                save_point=False
            # Broadcast best model state across all GPUs if updated
            if fabric.global_rank == 0:  # Let rank 0 distribute the model
                fabric.broadcast(best_model_state, src=0)
            model.load_state_dict(best_model_state)  # Load the updated state
            # Ensure all GPUs are updated with the best criterion
            fabric.barrier()

            if save_point==True:        
                print("SAVE NEW BEST!!!!!! ({:.6f} ---> {:.6f}). {:.6f} ---> {:.6f}).)".format(val_acc_max, val_avg_acc,loss,train_loss_t))
                val_acc_max = val_avg_acc
                val_acc_min=val_avg_acc
                loss= train_loss_t
                 #torch.save(best_model_state, os.path.join(root_dir, "../../rds/hpc-work/best_rot_end.pth"))
                save_checkpoint(
                    [model],
                    epoch,
                    filename=model_name,
                    best_acc=val_acc_max,
                    train_loss=loss,
                    dir_add=root_dir
                )
            
            wandb.log({"loss_total": train_loss_t, "ROC": val_acc1, "F1": val_acc2, "Sensitivity": val_acc4, "Specificity": val_acc5, "Precision": val_acc6})
            if scheduler==None:
                print('no scheduler strategy for lr used fix lr.')
            else:
                scheduler.step()
    wandb.finish()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        f1_,
        roc_,
        val_avg_acc_,
        loss_epochs,
        trains_epoch,
    )

def train_epoch(model, loader, optimizer, epoch, fabric=None):
    """Run one classification training epoch."""
    loss_function=torch.nn.CrossEntropyLoss()
    model.train()
    start_time = time.time()
    run_loss = AverageMeter_cuda()
    loss2=0
    count1=0
    for idx, batch_data in enumerate(loader):
        data, labels = batch_data['image'], batch_data['class']
        classes = model(data)
        loss_classifier = loss_function(classes,labels.long())
        losst=loss_classifier
        losst1=losst.detach()
        optimizer.zero_grad()
        fabric.backward(losst)
        optimizer.step()
        loss2=loss2+losst1
        count1 += 1
    loss=loss2/count1
    run_loss.update(loss)
    print(
            "Epoch {}  {}/{}".format(epoch, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    nclass=2,
    max_epochs=10,
    fabric=None
):
    """Evaluate a classifier and return ROC, F1, specificity, sensitivity, and precision."""

    model.eval()
    post_label = AsDiscrete(argmax=False, to_onehot=nclass)
    post_label2=Activations(softmax=True)
    start_time = time.time()
    run_acc1 = AverageMeter_cuda()
    run_acc2 = AverageMeter_cuda()
    run_acc4 = AverageMeter_cuda()
    run_acc5 = AverageMeter_cuda()
    run_acc6 = AverageMeter_cuda() 
    
    roc=ROCAUCMetric()
    f1=ConfusionMatrixMetric(metric_name="f1 score",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    sens=ConfusionMatrixMetric(metric_name="sensitivity",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    spec=ConfusionMatrixMetric(metric_name="specificity",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    prec=ConfusionMatrixMetric(metric_name="precision",compute_sample=True,reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
    f1.reset()
    spec.reset()
    sens.reset()
    prec.reset()
    roc.reset()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, labels = batch_data['image'], batch_data['class']
            classes = model(data)
            classes_t=post_label2(classes)
            labels_t=post_label(torch.unsqueeze(labels,dim=0))
            labels_t=torch.transpose(labels_t, 0, 1)
            r=roc(y_pred=classes_t, y=labels_t)
            f=f1(y_pred=classes_t, y=labels_t)
            sp=spec(y_pred=classes_t, y=labels_t)
            se=sens(y_pred=classes_t, y=labels_t)
            pr=prec(y_pred=classes_t, y=labels_t)
            acc1_total = r
            acc2_total = f
            acc4_total = sp
            acc5_total = se
            acc6_total = pr
            acc1_t=torch.nanmean((acc1_total[0])).detach()
            acc2=torch.nanmean((acc2_total[0])).detach()
            acc4=torch.nanmean((acc4_total[0])).detach()
            acc5=torch.nanmean((acc5_total[0])).detach()
            acc6=torch.nanmean((acc6_total[0])).detach()
            acc1=torch.nanmean((acc1_t)).detach()
            run_acc2.update(acc2)
            run_acc1.update(acc1)
            run_acc4.update(acc4)
            run_acc5.update(acc5)
            run_acc6.update(acc6)

            f1_v = run_acc2
            roc_v= run_acc1
            specif=run_acc4
            scens=run_acc5
            precis=run_acc6
            del data, labels, classes 
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", f1:",
                acc2,
                 ", ROC:",
                acc1,
                                 ", specificity:",
                acc4,
                                 ", sensitivity:",
                acc5,
                                 ", precision:",
                acc6,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

        return roc_v.avg, f1_v.avg,  specif.avg, scens.avg, precis.avg
