import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import weave
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import json

if len(sys.argv) != 5:
    print("Usage: python script.py <config_path> <signal_dir> <background_dir> <data_dir>")
    sys.exit(1)

cfg_path, sig_dir, bkg_dir, data_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.append('./src')
from src import fns as fns
from src import cls as cls



with open(cfg_path, "r") as f:
    cfg = json.load(f)

plane = cfg['dataset']['current_plane']
train_frac = cfg['dataset']['train_fraction']
val_frac = cfg['dataset']['val_fraction']
seed = cfg['dataset']['generator_seed']
batch_size = cfg['dataloader']['batch_size']
workers = cfg['dataloader']['num_workers']
shuffle = cfg['dataloader']['shuffle']
opt = cfg['training']['optimizer']
sched = cfg['training']['scheduler']
crit = cfg['training']['criterion']
lr = cfg['training']['learning_rate']
mom = cfg['training']['momentum']
wd = cfg['training']['weight_decay']
gamma = cfg['training']['gamma']
step_size = cfg['training']['step_size']
end_fact = cfg['training']['end_factor']
epochs = cfg['training']['num_epochs']
pat = cfg['training']['patience']

sig_files = [os.path.join(sig_dir, f) for f in os.listdir(sig_dir) if f.endswith('.h5')]
bkg_files = [os.path.join(bkg_dir, f) for f in os.listdir(bkg_dir) if f.endswith('.h5')]

print("Signal HDF5 Files:")
for f in sig_files:
    print(f)

dt = time.strftime('%d-%m_%H-%M')
run_name = f'{dt}_resnet18_h5_plane{plane}'
print(f'WB run name: {run_name} ')
run = wandb.init(
    project="pdecay-masters-single",
    name=run_name,
    config={
        "learning_rate": lr,
        "architecture": cfg["model"]["architecture"],
        "epochs": epochs,
        "optimizer": opt,
        "scheduler": sched,
        "additional_info": cfg["model"]["remarks"],
        "patience": pat,
        "batch_size": batch_size,
        "momentum": mom,
        "weight_decay": wd,
        "gamma": gamma,
        "step_size": step_size,
        "end_factor": end_fact,
        "PLANE": plane
    }
)
weave.init("pdecay-masters-single")

tbl = wandb.Table(columns=["ground truth", "prediction"])
val_tbl = wandb.Table(columns=["ground truth", "prediction"])

train_df = pd.DataFrame(columns=['ground_truth', 'output'])
val_df = pd.DataFrame(columns=['ground_truth', 'output'])

model = cls.ModifiedResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model.to(device)

for name, param in model.named_parameters():
    print(f"{name} is on {param.device}")

opt_map = {
    'Adam': (optim.Adam, {'lr': lr, 'weight_decay': wd}),
    'SGD': (optim.SGD, {'lr': lr, 'momentum': mom, 'weight_decay': wd}),
    'Adadelta': (optim.Adadelta, {'lr': lr, 'weight_decay': wd, 'rho': 0.9}),
    'RAdam': (optim.RAdam, {'lr': lr, 'weight_decay': wd})
}

sched_map = {
    'StepLR': (lr_scheduler.StepLR, {'step_size': step_size, 'gamma': gamma, 'verbose': True}),
    'LinearLR': (lr_scheduler.LinearLR, {'end_factor': end_fact, 'total_iters': epochs, 'verbose': True}),
    'ExponentialLR': (lr_scheduler.ExponentialLR, {'gamma': gamma, 'verbose': True}),
    'ReduceLROnPlateau': (lr_scheduler.ReduceLROnPlateau, {'mode': 'min', 'factor': 0.1}),
    'CosineAnnealingLR': (lr_scheduler.CosineAnnealingLR, {'T_max': epochs, 'verbose': True, 'eta_min': 1e-8})
}

crit_map = {
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss
}

opt_class, opt_params = opt_map[opt]
optimizer = opt_class(model.parameters(), **opt_params)

sched_class, sched_params = sched_map[sched]
scheduler = sched_class(optimizer, **sched_params)

criterion = crit_map[crit]()

print('================')
print('Creating dataset.')
print('================')
print(f'Plane {plane}')

dataset = cls.HDF5SparseDataset(signal_files=sig_files, background_files=bkg_files, plane_idx=plane)
train, val, test = fns.split_dset(dataset=dataset, train_frac=train_frac, val_frac=val_frac, generator_seed=seed)
train_loader = cls.create_dataloader(dataset=train, target_batch_size=batch_size, shuffle=shuffle, nworkers=workers)
val_loader = cls.create_dataloader(dataset=val, target_batch_size=batch_size, shuffle=shuffle, nworkers=workers)
test_loader = cls.create_dataloader(dataset=test, target_batch_size=batch_size, shuffle=shuffle, nworkers=workers)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

start_time = time.time()
print('====================================')
print(f'Training model. Started at {time.strftime("%H:%M:%S", time.localtime(start_time))}')
print('====================================')

results = fns.train_model(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    optimizer=optimizer, scheduler=scheduler,
    criterion=criterion, num_epochs=epochs,
    device=device,
    patience=pat, table=tbl, val_table=val_tbl, df_train=train_df, df_val=val_df,
    plane=plane)

print(f'Training complete. Took {(time.time() - start_time) / 60} minutes.')
run.log({f"train_table_h5_plane{plane}": tbl})
run.log({f"val_table_h5_plane{plane}": val_tbl})

np.save(f'./diagnostics/resnet18-h5-plane{plane}-train-loss.npy', results[1])
np.save(f'./diagnostics/resnet18-h5-plane{plane}-train-acc.npy', results[3])
np.save(f'./diagnostics/resnet18-h5-plane{plane}-val-loss.npy', results[2])
np.save(f'./diagnostics/resnet18-h5-plane{plane}-val-acc.npy', results[4])

run.finish()
