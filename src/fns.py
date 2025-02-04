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
from torch.utils.data import Subset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import logging
import sklearn
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import concurrent.futures

# ==================================#
# == SET UP LOGGING CONFIGURATION ==#
# ==================================#

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
loggy = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
sys.path.append("./src")
sys.path.append("./src/net_cfg.json")
sys.path.append("./src/cls.py")

# ===================== #
# ==== LOAD CONFIG ==== #
# ===================== #

sys.path.append("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json")

with open("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)


def get_event_dirs(base_dir):
    event_dirs = []
    print(f"The base dir is {base_dir}, len: {len(os.listdir(base_dir))}")

    if "atmonu" in base_dir:
        all_batches = os.listdir(base_dir)
        batch_to_take = random.sample(all_batches, int(0.05 * len(all_batches)))

        print(
            f"Taking {len(batch_to_take)} out of {len(all_batches)} batches for atmonu."
        )
        print(f"Taking batches: {batch_to_take}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process each batch in parallel
            futures = [
                executor.submit(
                    _get_event_dirs_from_batch, os.path.join(base_dir, batch_dir)
                )
                for batch_dir in batch_to_take
                if os.path.isdir(os.path.join(base_dir, batch_dir))
            ]

            for future in concurrent.futures.as_completed(futures):
                event_dirs.extend(future.result())

    elif "pdk" in base_dir:
        all_batches = os.listdir(base_dir)[:20]
        print(f"Taking {len(all_batches)} batches for pdk.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _get_event_dirs_from_batch, os.path.join(base_dir, batch_dir)
                )
                for batch_dir in all_batches
                if os.path.isdir(os.path.join(base_dir, batch_dir))
            ]

            for future in concurrent.futures.as_completed(futures):
                event_dirs.extend(future.result())
    else:
        print("No such dataset.")

    return event_dirs


def _get_event_dirs_from_batch(batch_path):
    event_dirs = []
    for evt in os.listdir(batch_path):
        evt_path = os.path.join(batch_path, evt)
        if os.path.isdir(evt_path):
            event_dirs.append(evt_path)
    return event_dirs


def split_dset(dataset, train_frac, val_frac, generator_seed):
    train_len = int(len(dataset) * train_frac)
    val_len = int(len(dataset) * val_frac)
    test_len = len(dataset) - train_len - val_len
    print(f"Lengths: (train, val, test): ({train_len}, {val_len}, {test_len})")

    train, val, test = random_split(
        dataset,
        [train_len, val_len, test_len],
        torch.Generator().manual_seed(generator_seed),
    )

    # Print signal (label=1) and background (label=0) counts for each split
    train_labels = [dataset[i][1] for i in train.indices]
    val_labels = [dataset[i][1] for i in val.indices]
    test_labels = [dataset[i][1] for i in test.indices]

    print(
        f"Train: signal: {sum(train_labels)/len(train_labels)}, background: {(len(train_labels) - sum(train_labels)/len(train_labels))}"
    )
    print(
        f"Val: signal: {sum(val_labels)/len(val_labels)}, background: {(len(val_labels) - sum(val_labels)/len(val_labels))}"
    )
    print(
        f"Test: signal: {sum(test_labels)/len(test_labels)}, background: {(len(test_labels) - sum(test_labels)/len(test_labels))}"
    )

    return train, val, test


def extract_labels(dataset, indices):
    return [dataset[i][1] for i in indices]


def strat_meta_split(dataset, labels, train_frac, val_frac, generator_seed):
    logger.info("Starting stratified split")
    
    # Convert to NumPy arrays for faster indexing
    class_labels = np.array(dataset.labels)
    decays = np.array(dataset.decay_labels)
    
    # Create a combined stratification key using tuples (class, decay)
    stratify_labels = np.array([f"{c}_{d}" for c, d in zip(class_labels, decays)])

    # First split: train_val (95%) and test (5%)
    train_val_indices, test_indices = train_test_split(
        np.arange(len(class_labels)),
        test_size=0.05,
        stratify=stratify_labels,
        random_state=generator_seed
    )
    
    train_val_data = Subset(dataset, train_val_indices)
    test_data = Subset(dataset, test_indices)

    logger.info(f"Initial train&val and test split completed")
    logger.info(f"Train&Val size: {len(train_val_data)}, Test size: {len(test_data)}")

    # Second split: train (90% of train_val) and val (10% of train_val)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.10,
        stratify=stratify_labels[train_val_indices],
        random_state=generator_seed
    )

    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)

    logger.info("Train and val split completed")
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Count decay modes in train, val, test
    train_counts = Counter(decays[train_indices])
    val_counts = Counter(decays[val_indices])
    test_counts = Counter(decays[test_indices])

    logger.info(f"Decay counts in Train: {train_counts}")
    logger.info(f"Decay counts in Val: {val_counts}")
    logger.info(f"Decay counts in Test: {test_counts}")

    logger.info("Stratified meta split completed successfully")
    return train_data, val_data, test_data


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Remove the "module." prefix from keys in the checkpoint
    checkpoint_state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }

    missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
    unexpected_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())

    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys}")

    model.load_state_dict(
        checkpoint_state_dict, strict=False
    )  # strict=False to ignore missing and unexpected keys


@weave.op()
def one_epoch_lfm(
    model,
    phase,
    dataloaders,
    optimizer,
    criterion,
    scheduler,
    threshold,
    device,
    log_df,
    is_best_epoch,
):
    start_time = time.time()
    loggy.info(f"<fn: one_epoch_lfm>, phase: {phase} ")
    temp_log_data = []
    all_predictions = []
    all_responses = []
    all_labels = []

    model.train() if phase == "train" else model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    pos_key = f"p{phase}"

    ################################################################
    ## This is only applicable if we choose to use signed images ##
    # neg_key = f'n{phase}'
    # for data_planes in zip(*[dataloaders[i][pos_key] for i in range(3)],
    #                        *[dataloaders[i][neg_key] for i in range(3)]):
    ################################################################

    for batch_idx, data_planes in enumerate(
        zip(*[dataloaders[i][pos_key] for i in range(3)])
    ):
        batch_start_time = time.time()
        loggy.debug(f"Processing batch {batch_idx + 1}")
        optimizer.zero_grad()
        inputs, labels = [], []

        for data in data_planes[:3]:
            inputs.append(data[0].to(device))
        # for data in data_planes[3:]:
        #     inputs.append(data[0].to(device))
        labels = data_planes[0][1].unsqueeze(1).float().to(device)

        forward_pass_start_time = time.time()
        with torch.set_grad_enabled(phase == "train"):
            output = model(*inputs)
            loss = criterion(output, labels)
            loggy.info(
                f"Forward pass done in {time.time() - forward_pass_start_time:.4f} s"
            )

            if phase == "train":
                backward_pass_start_time = time.time()
                loss.backward()
                optimizer.step()
                loggy.info(
                    f"Backward pass done in {time.time() - backward_pass_start_time:.4f} s"
                )

        predictions = (output > threshold).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_responses.extend(torch.sigmoid(output).detach().cpu().numpy())

        running_loss += loss.item()
        total += labels.size(0)
        running_corrects += (predictions == labels).sum().item()

        if is_best_epoch:
            for i, label in enumerate(labels.cpu().numpy()):
                temp_log_data.append(
                    {
                        "ground_truth": label,
                        "ensemble_output": torch.sigmoid(output[i]).item(),
                    }
                )
        loggy.info(f"Batch processed in {time.time() - batch_start_time:.4f} s")

    if phase == "train":
        scheduler.step()

    if is_best_epoch:
        tempdf = pd.DataFrame(temp_log_data)
        log_df = pd.concat([log_df, tempdf], ignore_index=True)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions)

    wandb.log(
        {
            f"{phase} loss": epoch_loss,
            f"{phase} accuracy": epoch_acc,
            f"{phase} precision": precision,
            f"{phase} recall": recall,
            f"{phase} f1": f1,
        }
    )

    loggy.info(f"Phase {phase} completed in {time.time() - start_time:.4f} seconds")

    return epoch_loss, epoch_acc, log_df


@weave.op()
def train_lfm(
    model,
    dataloaders,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs,
    patience,
    df_train,
    df_val,
    df_test,
):
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict())
    no_improvement_epochs = 0
    best_epoch = None

    for epoch in range(num_epochs):
        is_current_best = epoch == best_epoch

        train_loss, train_acc, df_train = one_epoch_lfm(
            model,
            "train",
            dataloaders,
            optimizer,
            criterion,
            scheduler,
            0.5,
            device,
            df_train,
            is_current_best,
        )
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, df_val = one_epoch_lfm(
            model,
            "val",
            dataloaders,
            optimizer,
            criterion,
            scheduler,
            0.5,
            device,
            df_val,
            is_current_best,
        )
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
            loggy.info(f"No improvement for {patience} epochs, stopping")
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)
        _, _, df_train = one_epoch_lfm(
            model,
            "train",
            dataloaders,
            optimizer,
            criterion,
            scheduler,
            0.5,
            device,
            df_train,
            True,
        )

        _, _, df_val = one_epoch_lfm(
            model,
            "val",
            dataloaders,
            optimizer,
            criterion,
            scheduler,
            0.5,
            device,
            df_val,
            True,
        )

        _, _, df_test = one_epoch_lfm(
            model,
            "test",
            dataloaders,
            optimizer,
            criterion,
            scheduler,
            0.5,
            device,
            df_test,
            True,
        )

        save_path = "/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion/late_fusion_{}.pt".format(
            time.strftime("%d-%m_%H-%M")
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )

        date_time = time.strftime("%d-%m_%H-%M")

        df_train.to_csv(f"./diagnostics/metrics-df/lfm/df_train_{date_time}_lfm.csv")
        df_val.to_csv(f"./diagnostics/metrics-df/lfm//df_val_{date_time}_lfm.csv")
        df_test.to_csv(f"./diagnostics/metrics-df/lfm/df_test_{date_time}_lfm.csv")

    return (
        model,
        train_loss_values,
        val_loss_values,
        train_acc_values,
        val_acc_values,
        best_epoch,
    )


@weave.op()
def train_one_epoch(
    epoch_idx,
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scheduler,
    table,
    df,
    is_best_epoch,
):
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    temp_train_data = []
    model.train()

    for inputs, labels in train_loader:

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
                new_row = {"ground_truth": lab.item(), "output": pred.item()}
                temp_train_data.append(new_row)

        # batch_times.append((batch_end - batch_start)/ 60)
        # print(f'Batch {batch_idx} time {(batch_end - batch_start)/ 60:.2f} minutes')

    if is_best_epoch:
        tempdf = pd.DataFrame(temp_train_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    epoch_loss = running_loss / len(train_loader)
    acc = correct / total
    scheduler.step()

    wandb.log({f"train_acc": acc, f"train_loss": epoch_loss})
    print(f"Epoch: {epoch_idx} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}")

    epoch_end = time.time()
    print(f"Epoch time: {(epoch_end - epoch_start) / 60:.2f} minutes")
    print(f"Intermediate save at {epoch_idx}")
    # Save checkpoint
    save_path = f"/mnt/lustre/helios-home/gartmann/venv/checkpoints_wt_avg/resnet18_{epoch_idx}_at{time.strftime('%d-%m_%H-%M')}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )

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

    wandb.log(
        {
            "val_loss": average_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }
    )

    if is_best_epoch:
        for lab, resp in zip(all_labels, all_responses):
            new_row = {"ground_truth": lab.item(), "output": resp.item()}
            temp_val_data.append(new_row)
            table.add_data(lab.item(), resp.item())
    if is_best_epoch:
        tempdf = pd.DataFrame(temp_val_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    return average_loss, accuracy, precision, recall, f1, df


@weave.op()
def test_model(model, plane, checkpoint_pth, test_loader, device, df, test_table):
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
        new_row = {"ground_truth": lab.item(), "output": resp.item()}
        temp_test_data.append(new_row)
        test_table.add_data(lab.item(), resp.item())

    tempdf = pd.DataFrame(temp_test_data)
    df = pd.concat([df, tempdf], ignore_index=True)

    date_time = time.strftime("%d-%m_%H-%M")
    df.to_csv(
        f"./diagnostics/metrics-df/df_test_{date_time}_resnet18_h5_plane{plane}.csv"
    )


@weave.op()
def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    val_loader,
    device,
    num_epochs,
    patience,
    table,
    val_table,
    df_train,
    df_val,
    plane,
    seed,
):
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
        is_current_best = epoch == best_epoch
        train_loss, train_acc, df_train = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler,
            table,
            df_train,
            is_current_best,
        )
        # [Rest of the training code]
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, val_precision, recall, f1, df_val = validate(
            model,
            val_loader,
            criterion,
            device,
            val_table,
            df_val,
            epoch,
            is_current_best,
        )

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
            print(f"No improvement for {patience} epochs, stopping")
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)

        # Re-run training and validation for the best epoch to log the metrics
        _, _, df_train = train_one_epoch(
            best_epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler,
            table,
            df_train,
            True,
        )
        _, _, _, _, _, df_val = validate(
            model, val_loader, criterion, device, val_table, df_val, best_epoch, True
        )

    # Save checkpoint
    save_path = "/mnt/lustre/helios-home/gartmann/venv/checkpoints_wt_avg/resnet18_{}_{}.pt".format(
        seed, plane
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    print("Checkpoint saved at {}".format(save_path))
    date_time = time.strftime("%d-%m_%H-%M")

    # Log tables to W&B and save locally
    wandb.log({f"train_table_{date_time}": table})
    wandb.log({f"val_table_{date_time}": val_table})

    df_train.to_csv(
        f"./diagnostics/metrics-df/df_train_{date_time}_resnet18_h5_plane{plane}.csv"
    )
    df_val.to_csv(
        f"./diagnostics/metrics-df/df_val_{date_time}_resnet18_h5_plane{plane}.csv"
    )

    return (
        model,
        train_loss_values,
        val_loss_values,
        train_acc_values,
        val_acc_values,
        precision_vals,
        recall_vals,
        f1_vals,
        best_epoch,
    )


def eval_decay_distrib(
    train_subset, val_subset, test_subset, dataset, output_dir="plots"
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_decays = [dataset.decay[i] for i in train_subset.indices]
    val_decays = [dataset.decay[i] for i in val_subset.indices]
    test_decays = [dataset.decay[i] for i in test_subset.indices]

    train_labels = [dataset.labels[i] for i in train_subset.indices]
    val_labels = [dataset.labels[i] for i in val_subset.indices]
    test_labels = [dataset.labels[i] for i in test_subset.indices]

    def plot_decays(decay_modes, labels, split_name, output_dir):
        decay_mode_counter_signal = Counter(
            [mode for mode, label in zip(decay_modes, labels) if label == 1]
        )
        decay_mode_counter_background = Counter(
            [mode for mode, label in zip(decay_modes, labels) if label == 0]
        )

        sorted_decay_modes = sorted(set(decay_modes))
        signal_counts = []
        background_counts = []

        plt.figure(figsize=(10, 6), dpi=120)
        x = range(len(sorted_decay_modes))
        bar_width = 0.4

        plt.bar(x, signal_counts, width=bar_width, label="Signal")
        plt.bar(
            [p + bar_width for p in x],
            background_counts,
            width=bar_width,
            label="Background",
        )

        plt.xlabel("Decay Modes")
        plt.ylabel("Counts")
        plt.title(f"{split_name} Decay Mode Distribution")
        plt.xticks([p + bar_width / 2 for p in x], sorted_decay_modes, rotation=45)
        plt.legend()

        plot_path = os.path.join(
            output_dir, f"{split_name}_decay_mode_distribution.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {split_name} decay mode distribution plot at {plot_path}")

    plot_decays(train_decays, train_labels, "Train", output_dir)
    plot_decays(val_decays, val_labels, "Validation", output_dir)
    plot_decays(test_decays, test_labels, "Test", output_dir)

    print("Decay distribution eval and plots complete.")
