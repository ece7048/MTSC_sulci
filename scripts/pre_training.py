"""Command-line entry point for unsupervised pre-training."""

from __future__ import annotations

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
    "method": "pre",
    "excel": None,
    "afs": ["", ""],
    "bc": ["", ""],
    "batch_n": 2,
    "nclass": 3,
    "val_every": 2,
    "num_epochs": 5,
    "DATA_ROOT1": "",
    "DATA_ROOT2": "",
    "PATH": "",
    "model_name": "pre_training_model_swift.pt",
    "roi_size": 64,
    "l1": 0.8,
    "l2": 0.2,
    "save_nii": False,
    "lr": 1e-3,
    "wd": 1e-2,
    "loops": 24,
    "upscale": 64,
    "light": False,
    "aug_data": False,
    "data": "top",
    "time": 1000,
}


METHODS = ("pre", "gan", "contrastive", "diffusion")


def build_parser():
    """Create the pre-training CLI parser."""
    parser = argparse.ArgumentParser(description="Run unsupervised 3D MRI pre-training.")
    parser.add_argument("--config", help="Path to a YAML/JSON configuration file.")
    parser.add_argument("--method", choices=METHODS, help="Pre-training method to run.")
    parser.add_argument("--excel", help="CSV metadata file with class labels.")
    parser.add_argument("--afs", help="After-sample suffix list, e.g. \"_a,_b\" or \"['_a','_b']\".")
    parser.add_argument("--bc", help="Case filename prefix list, e.g. \"L_,R_\" or \"['L_','R_']\".")
    parser.add_argument("--batch-n", dest="batch_n", type=int)
    parser.add_argument("--nclass", type=int)
    parser.add_argument("--val-every", dest="val_every", type=int)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int)
    parser.add_argument("--data-root1", dest="DATA_ROOT1")
    parser.add_argument("--data-root2", dest="DATA_ROOT2")
    parser.add_argument("--path", dest="PATH")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--roi-size", dest="roi_size", type=int)
    parser.add_argument("--l1", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--save-nii", dest="save_nii", action="store_true", default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--loops", type=int)
    parser.add_argument("--upscale", type=int)
    parser.add_argument("--light", action="store_true", default=None)
    parser.add_argument("--aug-data", dest="aug_data", action="store_true", default=None)
    parser.add_argument("--data")
    parser.add_argument("--time", type=int, help="Diffusion timesteps for --method diffusion.")
    return parser


def run_from_params(params):
    """Dispatch to the selected pre-training function."""
    from MTSC_sulci.pre_training.pretraining import con_pre, diff_pre, gan_pre, pre

    methods = {
        "pre": pre,
        "gan": gan_pre,
        "contrastive": con_pre,
        "diffusion": diff_pre,
    }
    params = dict(params)
    method = params.pop("method")
    train_fn = methods[method]

    if method != "pre":
        params.pop("wd", None)
    if method != "diffusion":
        params.pop("time", None)
    if method == "diffusion":
        params.pop("wd", None)
        params.pop("aug_data", None)

    return train_fn(**params)


def main(argv=None):
    """Parse config/CLI values and run pre-training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    params = merge_parameters(DEFAULTS, section(config, "pre_training"), cli_overrides(args))
    return run_from_params(params)


if __name__ == "__main__":
    main()
