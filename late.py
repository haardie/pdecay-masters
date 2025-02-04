import os
import sys
import torch
import torch.nn as nn
import wandb
import weave
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
from src import fns as fns
from src import cls as cls

sys.path.append('./src/fns.py')
sys.path.append('./src/net_cfg.json')
sys.path.append('./src/cls.py')

_path_cache = {}

# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#

with open("./src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

generator_seed = config['dataset']['generator_seed']
train_frac = config['dataset']['train_fraction']
val_frac = config['dataset']['val_fraction']

# dataloader configuration ----------
batch_size = config['dataloader']['batch_size']
num_workers = config['dataloader']['num_workers']
shuffle = config['dataloader']['shuffle']

# training configuration ------------
optimizer = config['training']['optimizer']
scheduler = config['training']['scheduler']

lr = config['training']['learning_rate']
momentum = config['training']['momentum']
weight_decay = config['training']['weight_decay']
gamma = config['training']['gamma']
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']
step_size = config['training']['step_size']

# ==================================#
# ======== INITIALIZE W&B ==========#
# ==================================#

run = wandb.init(
    project="mpdecay-lf",
    config={
        "learning_rate": lr,
        "architecture": "fusion with gate",
        "dataset": "",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "additional_info": config["model"]["remarks"]
    }
)
weave_run = weave.init("mpdecay-lf")

# ==================================#
# ============ SETUP ===============#
# ==================================#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is {}'.format(device))
print()

data_dir_pos = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-pos/'

signal_dirs_pos = [os.path.join(data_dir_pos, f'plane{i}', 'signal') for i in range(3)]
background_dirs_pos = [os.path.join(data_dir_pos, f'plane{i}', 'background') for i in range(3)]

data_dir_neg = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-neg/'

signal_dirs_neg = [os.path.join(data_dir_neg, f'plane{i}', 'signal') for i in range(3)]
background_dirs_neg = [os.path.join(data_dir_neg, f'plane{i}', 'background') for i in range(3)]

# ==================================#
# ============ DATASET =============#
# ==================================#

pdatasets = {}
ndatasets = {}

for plane in range(3):
    ppaths = fns.get_sparse_matrix_paths_cached(signal_dirs_pos[plane], background_dirs_pos[plane])
    npaths = fns.get_sparse_matrix_paths_cached(signal_dirs_neg[plane], background_dirs_neg[plane])

    psubset = cls.SparseMatrixDataset(ppaths)
    nsubset = cls.SparseMatrixDataset(npaths)

    pdatasets[plane] = psubset
    ndatasets[plane] = nsubset


psplits = {}
nsplits = {}
for plane in range(3):
    ptrain, pval, ptest = fns.split_dset(dataset=pdatasets[plane], train_frac=train_frac, val_frac=val_frac,
                                         generator_seed=generator_seed)
    psplits[plane] = {'ptrain': ptrain, 'pval': pval, 'ptest': ptest}

    ntrain, nval, ntest = fns.split_dset(dataset=ndatasets[plane], train_frac=train_frac, val_frac=val_frac,
                                         generator_seed=generator_seed)
    nsplits[plane] = {'ntrain': ntrain, 'nval': nval, 'ntest': ntest}

dataloaders = {}
for plane in range(3):
    ptrain_loader = DataLoader(psplits[plane]['ptrain'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    pval_loader = DataLoader(psplits[plane]['pval'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ptest_loader = DataLoader(psplits[plane]['ptest'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    ntrain_loader = DataLoader(nsplits[plane]['ntrain'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    nval_loader = DataLoader(nsplits[plane]['nval'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ntest_loader = DataLoader(nsplits[plane]['ntest'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    dataloaders[plane] = {
        'ptrain': ptrain_loader, 'pval': pval_loader, 'ptest': ptest_loader,
        'ntrain': ntrain_loader, 'nval': nval_loader, 'ntest': ntest_loader
    }

# ==================================#
# ============ MODEL ===============#
# ==================================

checkpoint_dir = "./checkpoints"

trained_model_paths = [os.path.join(checkpoint_dir, 'resnet18-pos', "resnet18_03-05_16-06_pos_0.pt"),
                       os.path.join(checkpoint_dir, 'resnet18-pos', "resnet18_03-05_19-48_pos_1.pt"),
                       os.path.join(checkpoint_dir, 'resnet18-pos', "resnet18_04-05_22-46_pos_2.pt"),
                       os.path.join(checkpoint_dir, 'resnet18-neg', "resnet18_05-05_21-21_neg_0.pt"),
                       os.path.join(checkpoint_dir, 'resnet18-neg', "resnet18_05-05_22-57_neg_1.pt"),
                       os.path.join(checkpoint_dir, 'resnet18-neg', "resnet18_05-05_01-22_neg_2.pt")
                       ]

models = [cls.ModifiedResNet() for _ in range(6)]

for model in models:
    fns.load_checkpoint(model, trained_model_paths[models.index(model)])

fused_model = cls.LateFusedModel(models=models)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    fused_model = nn.DataParallel(fused_model)

fused_model = fused_model.to(device)

# ==================================#
if optimizer == 'Adam':
    opt = torch.optim.Adam(fused_model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer == 'SGD':
    opt = torch.optim.SGD(fused_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    print('Optimizer not recognized.')
    raise NotImplementedError

if scheduler == 'ExponentialLR':
    sched = lr_scheduler.ExponentialLR(opt, gamma=gamma, last_epoch=-1)
elif scheduler == 'StepLR':
    sched = lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma, last_epoch=-1)
elif scheduler == 'CosineAnnealingLR':
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs - 1, eta_min=1e-8)
else:
    print('Scheduler not recognized.')
    raise NotImplementedError

criterion = nn.BCEWithLogitsLoss()

# ==================================#
# ============ TRAIN ===============#
# ==================================#

log_df_val_local_save = pd.DataFrame(columns=['ground_truth', 'ensemble_output'])
log_df_train_local_save = pd.DataFrame(columns=['ground_truth', 'ensemble_output'])
log_df_test_local_save = pd.DataFrame(columns=['ground_truth', 'ensemble_output'])

trained_lfm, train_loss_values, val_loss_values, train_acc_values, val_acc_values, best_epoch = fns.train_lfm(
    model=fused_model,
    dataloaders=dataloaders, optimizer=opt, criterion=criterion, scheduler=sched, device=device,
    num_epochs=num_epochs, patience=patience, df_train=log_df_train_local_save, df_val=log_df_val_local_save,
    df_test=log_df_test_local_save)

np.save('./diagnostics/lfm6-train-loss.npy', train_loss_values)
np.save('./diagnostics/lfm6-train-acc.npy', train_acc_values)

np.save('./diagnostics/lfm6-val-loss.npy', val_loss_values)
np.save('./diagnostics/lfm6-val-acc.npy', val_acc_values)

run.finish()
