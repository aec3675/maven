import os, sys
import wandb
from ruamel.yaml import YAML
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np

import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

from src.models_multimodal import (
    LightCurveImageCLIP,
    load_pretrain_clip_model,
    initialize_model,
    ClipMLP,
)
from src.utils import (
    set_seed,
    get_valid_dir,
    LossTrackingCallback,
    plot_ROC_curves,
    plot_loss_history,
    get_embs,
)
from src.dataloader import (
    load_data,
    NoisyDataLoader,
)
from src.wandb_utils import continue_sweep, schedule_sweep



if __name__ == "__main__":
    wandb.login()

    arg = sys.argv[1]
    analysis_path = "./analysis/"

    if arg.endswith(".yaml"):
        config = arg
        sweep_id, model_path, cfg = schedule_sweep(config, analysis_path)
    else:
        sweep_id = os.path.basename(arg)
        model_path = os.path.join(analysis_path, sweep_id)
        cfg = continue_sweep(model_path)

    print("model path: " + model_path, flush=True)

    set_seed(0)
    # define constants
    val_fraction = cfg["extra_args"]["val_fraction"]

    # Data preprocessing

    data_dirs = [
        "/Users/pnr5sh/Documents/phd/maven/data/test/iib",
        "data/test/iib",
        "test/iib",
        "iib",
    ]
    # [
    #     "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
    #     "ZTFBTS/",
    #     "data/ZTFBTS/",
    #     "/ocean/projects/phy230064p/shared/ZTFBTS/",
    #     "/n/home02/gemzhang/repos/Multimodal-hackathon-2024/data/ZTFBTS/",
    # ]

    # Get the first valid directory
    data_dir = get_valid_dir(data_dirs)

    # Get what data combinations are used
    combinations = cfg["extra_args"]["combinations"]
    regression = cfg["extra_args"]["regression"]
    classification = cfg["extra_args"]["classification"]

    if classification:
        n_classes = cfg["extra_args"]["n_classes"]
    else:
        n_classes = 1 #5

    pretrain_path = cfg["extra_args"].get("pretrain_path")
    freeze_backbone = cfg["extra_args"].get("freeze_backbone")

    # Check if the config file has a spectra key
    if "spectral" in combinations:
        data_dirs = ["iib_spectra/", "data/test/iib_spectra/"] #["ZTFBTS_spectra/", "data/ZTFBTS_spectra/"]
        spectra_dir = get_valid_dir(data_dirs)
    else:
        spectra_dir = None

    max_spectral_data_len = cfg["extra_args"][
        "max_spectral_data_len"
    ]  # Spectral data is cut to this length
    dataset, nband, filenames, stratifiedkfoldindices = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=max_spectral_data_len,
        combinations=combinations,
        n_classes=n_classes,
        spectral_rescalefactor=cfg["extra_args"]["spectral_rescalefactor"],
        kfolds=cfg["extra_args"].get("kfolds", None),
    )

    print(filenames)

    # wandb.agent(
    #     sweep_id=sweep_id,
    #     entity=cfg["entity"],
    #     project=cfg["project"],
    #     function=train_sweep,
    #     count=cfg["extra_args"]["nruns"],
    # )