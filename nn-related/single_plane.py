#!~/venv/bin/python

import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import weave
import random
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

#
sys.path.append("./src")
sys.path.append("./src/net_cfg.json")
sys.path.append("./src/cls.py")

from src import fns as fns
from src import cls as cls

# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#

with open("./src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

plane = 0

train_frac = config["dataset"]["train_fraction"]
val_frac = config["dataset"]["val_fraction"]
generator_seed = config["dataset"]["generator_seed"]

# dataloader configuration ----------
batch_size = config["dataloader"]["batch_size"]
num_workers = config["dataloader"]["num_workers"]
shuffle = config["dataloader"]["shuffle"]

# training configuration ------------
optimizer = config["training"]["optimizer"]
scheduler = config["training"]["scheduler"]
criterion = config["training"]["criterion"]

lr = config["training"]["learning_rate"]
momentum = config["training"]["momentum"]  # only for SGD
weight_decay = config["training"]["weight_decay"]  # only for Adam and RAdam
gamma = config["training"]["gamma"]  # only for StepLR and ExponentialLR
step_size = config["training"]["step_size"]  # only for StepLR
end_factor = config["training"]["end_factor"]  # only for LinearLR
num_epochs = config["training"]["num_epochs"]
patience = config["training"]["patience"]

# ==================================#
# ======== SET UP DIRECTORIES ======#
# ==================================#

data_dir = "/mnt/lustre/helios-shared/GAMS/dune/pdk-root"
print(f"Data dir: {data_dir}")

# ==================================#
# ============ W&B SETUP ===========#
# ==================================#
datetime = time.strftime("%d-%m_%H-%M")
wandb_run_name = f"{datetime}_resnet18_plane{plane}"
print(f"WB run name: {wandb_run_name} ")
run = wandb.init(
    project="pdecay-masters-single-w-decay-modes",
    name=wandb_run_name,
    config={
        "learning_rate": lr,
        "architecture": config["model"]["architecture"],
        "epochs": num_epochs,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "additional_info": config["model"]["remarks"],
        "patience": patience,
        "batch_size": batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "step_size": step_size,
        "end_factor": end_factor,
        "PLANE": plane,
        "sgn": sgn,
    },
)
weave_run = weave.init("pdecay-masters-single-w-decay-modes")

table = wandb.Table(columns=["ground truth", "prediction"])
val_table = wandb.Table(columns=["ground truth", "prediction"])

train_df = pd.DataFrame(columns=["ground_truth", "output"])
val_df = pd.DataFrame(columns=["ground_truth", "output"])
# ==================================#
# model = cls.ModifiedResNet()
# ==================================#
model = cls.ModifiedMobileNetV3()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distribute model across all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model.to(device)

# for name, param in model.named_parameters():
#     print(f"{name} is on {param.device}")

# Use optimizer and scheduler from config file
# Optimizer:
if optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer == "SGD":
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
elif optimizer == "Adadelta":
    optimizer = optim.Adadelta(
        model.parameters(), lr=lr, weight_decay=weight_decay, rho=0.9
    )
elif optimizer == "RAdam":
    optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Scheduler:
if scheduler == "StepLR":
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma, verbose=True
    )
elif scheduler == "LinearLR":
    scheduler = lr_scheduler.LinearLR(
        optimizer, end_factor=end_factor, total_iters=num_epochs, verbose=True
    )
elif scheduler == "ExponentialLR":
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
elif scheduler == "ReduceLROnPlateau":
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.1)
elif scheduler == "CosineAnnealingLR":
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, verbose=True, eta_min=1e-8
    )

# Use criterion from config file
if criterion == "BCEWithLogitsLoss":
    criterion = nn.BCEWithLogitsLoss()

# ==================================#
# ======== DATA PREPARATION ========#
# ==================================#

print("================")
print("Creating dataset.")
print("================")

print("Plane {}".format(plane))

signal_dir = os.path.join(
    data_dir, "pdk-npz"
)  # evt dirs of structure: metadata in csv format & plane dirs -> npz file
background_dir = os.path.join(data_dir, "atmonu-npz")

# event_dirs = signal_dirs.result() + background_dirs.result()
signal_decay_dirs = [
    os.path.join("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk_decays", dir)
    for dir in os.listdir("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk_decays")[:2]
    if os.path.isdir(
        os.path.join("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk_decays", dir)
    )
]
background_decay_dirs = [
    os.path.join("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu_decays", dir)
    for dir in os.listdir("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu_decays")[
        :2
    ]
    if os.path.isdir(
        os.path.join("/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu_decays", dir)
    )
]

decay_dirs = signal_decay_dirs + background_decay_dirs
dataset = cls.SparseMatrixDatasetMeta(decay_dirs=decay_dirs, plane_idx=plane)

print("==================")
print("Splitting dataset.")
print("==================")
train, val, test = fns.strat_meta_split(
    dataset=dataset,
    labels=dataset.labels,
    train_frac=train_frac,
    val_frac=val_frac,
    generator_seed=735,
)
print("Dataset split.")

dataloaders = {
    "train": DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    ),
    "test": DataLoader(
        test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    ),
    "val": DataLoader(
        val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    ),
}
# plot decay mode distribution
# fns.eval_decay_distrib(train, val, test, dataset, output_dir=".")

# ==================================#
# =========== TRAINING =============#
# ==================================#
start_time = time.time()

print("====================================")
print(
    "Training model. Started at {}".format(
        time.strftime("%H:%M:%S", time.localtime(start_time))
    )
)
print("====================================")

(
    trained_model,
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    precision_vals,
    recall_vals,
    f1_vals,
    best_ep,
) = fns.train_model(
    model=model,
    train_loader=dataloaders["train"],
    val_loader=dataloaders["val"],
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    num_epochs=num_epochs,
    device=device,
    patience=patience,
    table=table,
    val_table=val_table,
    df_train=train_df,
    df_val=val_df,
    plane=plane,
    seed=generator_seed,
)

print("Training complete. Took {} minutes.".format((time.time() - start_time) / 60))
run.log({f"train_table_{sgn}_plane{plane}": table})
run.log({f"val_table_{sgn}_plane{plane}": val_table})

np.save(
    f"/mnt/lustre/helios-home/gartmann/venv/diagnostics/resnet18-plane{plane}-train-loss.npy",
    train_loss,
)
np.save(
    f"/mnt/lustre/helios-home/gartmann/venv/diagnostics/resnet18-plane{plane}-train-acc.npy",
    train_acc,
)

np.save(
    f"/mnt/lustre/helios-home/gartmann/venv/diagnostics/resnet18-plane{plane}-val-loss.npy",
    val_loss,
)
np.save(
    f"/mnt/lustre/helios-home/gartmann/venv/diagnostics/resnet18-plane{plane}-val-acc.npy",
    val_acc,
)

run.finish()
