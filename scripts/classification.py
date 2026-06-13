"""Classification workflow for 3D sulcal MRI models."""

from __future__ import division, print_function

import argparse
import sys
import types
from pathlib import Path

if "MTSC_sulci" not in sys.modules:
    package_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("MTSC_sulci")
    package.__path__ = [str(package_root)]
    sys.modules["MTSC_sulci"] = package

from MTSC_sulci.utilities.config import cli_overrides, load_config, merge_parameters, section


DEFAULTS = {
    "case": "simple_MHL",
    "excel": None,
    "afs": ["", ""],
    "bc": ["", ""],
    "batch_n": 2,
    "nclass": 3,
    "num_epochs": 5,
    "DATA_ROOT1": "",
    "DATA_ROOT2": "",
    "PATH": "",
    "name": "Lsk_simple3d_new_data_align",
    "model_name": "class_model_benemin.pt",
    "save_nii": False,
    "lr": 5e-3,
    "kind": "step",
    "light": True,
    "aug_data": True,
    "data": "top",
    "backbone_weights": None,
    "pr": "off",
    "back_w": "",
}


def classif(case="simple_MHL",excel=None, afs=["",""],bc=["",""], batch_n = 2, nclass=3, num_epochs = 5, DATA_ROOT1="", DATA_ROOT2="", PATH="", name="Lsk_simple3d_new_data_align", model_name="class_model_benemin.pt",save_nii=False,lr=5e-3,kind='step',light=True,aug_data=True,data='top',fabric=None,backbone_weights=None,pr='off',back_w=''):    
    """Train a 3D CNN classifier from cropped sulcal MRI volumes."""
    import lightning as L
    import torch
    from lightning.pytorch import seed_everything

    from MTSC_sulci.utilities import create_3Dnet
    from MTSC_sulci.utilities.CNN3D import pytorch_model
    from MTSC_sulci.utilities.load_data import data_build
    from MTSC_sulci.utilities.trainer_class import trainer_class

    seed_everything(42,workers=True)

    if save_nii:
        raise NotImplementedError("NIfTI export is not implemented for classification-only runs.")
    if fabric is None:
        fabric = L.Fabric(accelerator="auto", devices=1)
        fabric.launch()

    train_loader, val_loader, _test_loader= data_build(channels=1,bz=batch_n,datan=DATA_ROOT1,data2n=DATA_ROOT2,excel=excel,cs=bc,afs=afs,case_training='class', nclass=nclass,aug=aug_data,fabric=fabric,data=data)
    wd_optimizer=1e-4

    tensorfl=create_3Dnet.create_3Dnet(model=case,height=90, width=190, depth=160, channels=1, classes=nclass,name=name,do=0.3,backbone=backbone_weights,paral=pr,b_w=back_w)
    ten=tensorfl.model_builder()
    model=pytorch_model(ten)
    net=model
    param=net.parameters()
    optimizer=torch.optim.AdamW(param, lr=lr, weight_decay=wd_optimizer)
    kind=kind
    opt1=int(0.5*num_epochs)
    opt2=0.1
    start_epoch = 0
    (val_acc_max,dices_,f1_,roc_,haus_,aver_,loss_epochs,trains_epoch,) = trainer_class(model1=net,train_loader=train_loader,val_loader=val_loader,optimizer1=optimizer,scheduler1=[kind,opt1,opt2],start_epoch=start_epoch,nclass=nclass,batch_size=batch_n,max_epochs=num_epochs,model_name=model_name,PATH=PATH,fabric=fabric)


def build_parser():
    """Create the classification CLI parser."""
    parser = argparse.ArgumentParser(description="Run 3D MRI classification training.")
    parser.add_argument("--config", help="Path to a YAML/JSON configuration file.")
    parser.add_argument("--case")
    parser.add_argument("--excel")
    parser.add_argument("--afs", help="After-sample suffix list, e.g. \"_a,_b\" or \"['_a','_b']\".")
    parser.add_argument("--bc", help="Case filename prefix list, e.g. \"L_,R_\" or \"['L_','R_']\".")
    parser.add_argument("--batch-n", dest="batch_n", type=int)
    parser.add_argument("--nclass", type=int)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int)
    parser.add_argument("--data-root1", dest="DATA_ROOT1")
    parser.add_argument("--data-root2", dest="DATA_ROOT2")
    parser.add_argument("--path", dest="PATH")
    parser.add_argument("--name")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--save-nii", dest="save_nii", action="store_true", default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--kind", choices=["step", "cosin", "diff"])
    parser.add_argument("--light", action="store_true", default=None)
    parser.add_argument("--aug-data", dest="aug_data", action="store_true", default=None)
    parser.add_argument("--data")
    parser.add_argument("--backbone-weights", dest="backbone_weights")
    parser.add_argument("--pr")
    parser.add_argument("--back-w", dest="back_w")
    return parser


def main(argv=None):
    """Parse config/CLI values and run classification training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    params = merge_parameters(DEFAULTS, section(config, "classification"), cli_overrides(args))
    return classif(**params)


if __name__ == "__main__":
    main()
