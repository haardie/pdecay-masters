import json
import weave
import os
import sys
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.append('./src')
sys.path.append('./src/net_cfg.json')
sys.path.append('./src/cls.py')

with open("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)


def split_dset(dataset, train_frac, val_frac, generator_seed):
    train_len = int(len(dataset) * train_frac)
    val_len = int(len(dataset) * val_frac)
    test_len = len(dataset) - train_len - val_len
    print(f'Lengths: (train, val, test): ({train_len}, {val_len}, {test_len})')

    train, val, test = random_split(dataset, [train_len, val_len, test_len],
                                    torch.Generator().manual_seed(generator_seed))
    return train, val, test


def get_sparse_matrix_paths_cached(signal_dir, background_dir, _path_cache={}):
    if signal_dir not in _path_cache:
        signal_paths = [entry.path for entry in sorted(os.scandir(signal_dir), key=lambda x: x.name) if entry.is_file()]
        _path_cache[signal_dir] = signal_paths

    if background_dir not in _path_cache:
        background_paths = [entry.path for entry in sorted(os.scandir(background_dir), key=lambda x: x.name) if
                            entry.is_file()]
        _path_cache[background_dir] = background_paths

    return _path_cache[signal_dir] + _path_cache[background_dir]


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Remove the "module." prefix from keys in the checkpoint
    checkpoint_state_dict = {key.replace("module.", ""): value for key, value in
                             checkpoint['model_state_dict'].items()}

    missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
    unexpected_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())

    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys}")

    model.load_state_dict(checkpoint_state_dict, strict=True) 


@weave.op()
def one_epoch_lfm(model, phase, dataloaders, optimizer, criterion, scheduler, threshold, device, log_df, is_best_epoch):
    temp_log_data = []
    all_predictions = []
    all_responses = []
    all_labels = []

    model.train() if phase == 'train' else model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    pos_key = f'p{phase}'
    neg_key = f'n{phase}'

    for data_planes in zip(*[dataloaders[i][pos_key] for i in range(3)],
                           *[dataloaders[i][neg_key] for i in range(3)]):
        optimizer.zero_grad()
        inputs, labels = [], []

        for data in data_planes[:3]:
            inputs.append(data[0].to(device))
        for data in data_planes[3:]:
            inputs.append(data[0].to(device))
        labels = data_planes[0][1].unsqueeze(1).float().to(device)
        with torch.set_grad_enabled(phase == 'train'):
            output = model(*inputs)
            loss = criterion(output, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        predictions = (output > threshold).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_responses.extend(torch.sigmoid(output).detach().cpu().numpy())

        running_loss += loss.item()
        total += labels.size(0)
        running_corrects += (predictions == labels).sum().item()

        if is_best_epoch:
            for i, label in enumerate(labels.cpu().numpy()):
                temp_log_data.append({'ground_truth': label, 'ensemble_output': torch.sigmoid(output[i]).item()})

    if phase == 'train':
        scheduler.step()

    if is_best_epoch:
        tempdf = pd.DataFrame(temp_log_data)
        log_df = pd.concat([log_df, tempdf], ignore_index=True)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions)

    wandb.log({
        f"{phase} loss": epoch_loss,
        f"{phase} accuracy": epoch_acc,
        f"{phase} precision": precision,
        f"{phase} recall": recall,
        f"{phase} f1": f1
    })

    return epoch_loss, epoch_acc, log_df


@weave.op()
def train_lfm(model, dataloaders, optimizer, criterion, scheduler, device, num_epochs, patience, df_train, df_val,
              df_test):
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict())
    no_improvement_epochs = 0
    best_epoch = None

    for epoch in range(num_epochs):
        is_current_best = (epoch == best_epoch)

        train_loss, train_acc, df_train = one_epoch_lfm(model, 'train', dataloaders, optimizer, criterion, scheduler,
                                                        0.5, device, df_train, is_current_best)
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, df_val = one_epoch_lfm(model, 'val', dataloaders, optimizer, criterion, scheduler,
                                                  0.5, device, df_val, is_current_best)
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_wts = deepcopy(model.state_dict())
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'No improvement for {patience} epochs, stopping')
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)
        _, _, df_train = one_epoch_lfm(model, 'train', dataloaders, optimizer, criterion, scheduler,
                                       0.5, device, df_train, True)

        _, _, df_val = one_epoch_lfm(model, 'val', dataloaders, optimizer, criterion, scheduler,
                                     0.5, device, df_val, True)

        _, _, df_test = one_epoch_lfm(model, 'test', dataloaders, optimizer, criterion, scheduler,
                                      0.5, device, df_test, True)

        save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion/late_fusion_{}.pt'.format(
            time.strftime('%d-%m_%H-%M'))

        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

        date_time = time.strftime('%d-%m_%H-%M')

        df_train.to_csv(f'./diagnostics/metrics-df/lfm/df_train_{date_time}_lfm.csv')
        df_val.to_csv(f'./diagnostics/metrics-df/lfm//df_val_{date_time}_lfm.csv')
        df_test.to_csv(f'./diagnostics/metrics-df/lfm/df_test_{date_time}_lfm.csv')

    return model, train_loss_values, val_loss_values, train_acc_values, val_acc_values, best_epoch


@weave.op()
def train_one_epoch(epoch_idx, model, train_loader, optimizer, criterion, device, scheduler, table, df, is_best_epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    temp_train_data = []

    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if is_best_epoch:
            for lab, pred in zip(labels, torch.sigmoid(outputs)):
                table.add_data(lab.item(), pred.item())
                new_row = {'ground_truth': lab.item(), 'output': pred.item()}
                temp_train_data.append(new_row)

    if is_best_epoch:
        tempdf = pd.DataFrame(temp_train_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    epoch_loss = running_loss / len(train_loader)
    acc = correct / total
    scheduler.step() 

    wandb.log({f"train_acc": acc, f"train_loss": epoch_loss})
    print(f'Epoch: {epoch_idx} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}')

    epoch_end = time.time()
    print(f'Epoch time: {(epoch_end - epoch_start) / 60:.2f} minutes')

    return epoch_loss, acc, df


@weave.op()
def validate(model, val_loader, criterion, device, table, df, epoch_idx, is_best_epoch):
    running_loss = 0.0
    all_predictions = []
    all_responses = []
    all_labels = []
    temp_val_data = []

    model.eval() 
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            sigmoid_out = torch.sigmoid(outputs)
            predictions = torch.round(torch.sigmoid(outputs))
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_responses.extend(sigmoid_out.cpu().numpy())

    average_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    wandb.log({
        "val_loss": average_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    })

    if is_best_epoch:
        for lab, resp in zip(all_labels, all_responses):
            new_row = {'ground_truth': lab.item(), 'output': resp.item()}
            temp_val_data.append(new_row)
            table.add_data(lab.item(), resp.item())
    if is_best_epoch:
        tempdf = pd.DataFrame(temp_val_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    return average_loss, accuracy, precision, recall, f1, df


@weave.op()
def test_model(model, sgn, plane, checkpoint_pth, test_loader, device, df, test_table):
    temp_test_data = []
    all_responses = []
    all_labels = []

    load_checkpoint(model, checkpoint_pth)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            outputs = model(inputs)

            all_labels.extend(labels.cpu().numpy())
            all_responses.extend(torch.sigmoid(outputs).cpu().numpy())

    for lab, resp in zip(all_labels, all_responses):
        new_row = {'ground_truth': lab.item(), 'output': resp.item()}
        temp_test_data.append(new_row)
        test_table.add_data(lab.item(), resp.item())

    tempdf = pd.DataFrame(temp_test_data)
    df = pd.concat([df, tempdf], ignore_index=True)

    date_time = time.strftime('%d-%m_%H-%M')
    df.to_csv(f'./diagnostics/metrics-df/df_test_{date_time}_resnet18_{sgn}_plane{plane}.csv')


@weave.op()
def train_model(model, train_loader, optimizer, criterion, scheduler, val_loader, device, num_epochs, patience, table,
                val_table, df_train, df_val, plane, sgn):
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    precision_vals = []
    recall_vals = []
    f1_vals = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict()) 
    no_improvement_epochs = 0

    best_epoch = None
    for epoch in range(num_epochs):
        is_current_best = (epoch == best_epoch)
        train_loss, train_acc, df_train = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device,
                                                          scheduler, table, df_train, is_current_best)
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, val_precision, recall, f1, df_val = validate(model, val_loader, criterion, device, val_table,
                                                                        df_val, epoch, is_current_best)

        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        precision_vals.append(val_precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_wts = deepcopy(model.state_dict())
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'No improvement for {patience} epochs, stopping')
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)
        
        _, _, df_train = train_one_epoch(best_epoch, model, train_loader, optimizer, criterion, device,
                                         scheduler, table, df_train, True)
        _, _, _, _, _, df_val = validate(model, val_loader, criterion, device, val_table, df_val, best_epoch, True)

    save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/resnet18-{}/resnet18_{}_{}_{}.pt'.format(
        sgn, time.strftime('%d-%m_%H-%M'), sgn, plane)

    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print('Checkpoint saved at {}'.format(save_path))
    date_time = time.strftime('%d-%m_%H-%M')

    wandb.log({f"train_table_{date_time}": table})
    wandb.log({f"val_table_{date_time}": val_table})

    df_train.to_csv(f'./diagnostics/metrics-df/df_train_{date_time}_resnet18_{sgn}_plane{plane}.csv')
    df_val.to_csv(f'./diagnostics/metrics-df/df_val_{date_time}_resnet18_{sgn}_plane{plane}.csv')

    return model, train_loss_values, val_loss_values, train_acc_values, val_acc_values, precision_vals, recall_vals, f1_vals, best_epoch
