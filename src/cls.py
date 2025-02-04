import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import h5py
import weave
from functools import lru_cache
import functools
import multiprocessing as mp
import os
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import logging
from multiprocessing import Pool, cpu_count
import time
import itertools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class ModifiedMobileNetV3(nn.Module):
    def __init__(self, device="cuda", num_classes=1):
        super(ModifiedMobileNetV3, self).__init__()
        self.device = device
        self.model = models.mobilenet_v3_small()
        self.model.conv_stem = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x


class SparseMatrixDataset(Dataset):
    def __init__(self, event_paths, plane_idx):
        self.event_paths = event_paths
        self.plane_idx = plane_idx
        self.transform = transforms.Compose(
            [transforms.Resize((500, 500)), transforms.ToTensor()]
        )

        print("[INFO] Initializing dataset...")
        start_time = time.time()

        collected_data = list(self._collect_sparse_matrices())
        self.file_paths, self.labels = (
            zip(*collected_data) if collected_data else ([], [])
        )

        print(f"[INFO] Dataset initialized in {time.time() - start_time:.2f} seconds.")

    def _collect_sparse_matrices(self):

        for evt_path in self.event_paths:
            plane_path = evt_path / f"plane{self.plane_idx}"
            if plane_path.is_dir():
                decay_mode = evt_path.parent.name

                yield from (
                    (str(f), (1 if "pdk_decays" in str(evt_path) else 0, decay_mode))
                    for f in plane_path.iterdir()
                    if f.suffix == ".npz"
                )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        if idx >= len(self.file_paths):
            raise IndexError("Index out of range")

        sparse_matrix_path = self.file_paths[idx]
        label = self.labels[idx]

        image = self.process_sparse_matrix(sparse_matrix_path)
        return image, label

    def process_sparse_matrix(self, sparse_matrix_path):

        try:
            sparse_matrix = np.load(sparse_matrix_path)
            sparse_matrix = csr_matrix(
                (
                    sparse_matrix["data"],
                    sparse_matrix["indices"],
                    sparse_matrix["indptr"],
                ),
                shape=sparse_matrix["shape"],
            )
            image = Image.fromarray(sparse_matrix.toarray().astype(np.uint8), mode="L")
            image = self.transform(image)
        except Exception as e:
            print(
                f"[WARNING] Failed to load sparse matrix: {sparse_matrix_path}. Error: {e}"
            )
            image = torch.zeros((1, 500, 500))
        return image


class ModifiedResNet(nn.Module):
    def __init__(self, device="cuda", num_classes=1):
        super(ModifiedResNet, self).__init__()
        self.device = device
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(512, 512)
        # self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.model.to(device)
        # self._init_rand_weights()

    def forward(self, x, allow_drop=False):
        x = self.model(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        if allow_drop:
            x = F.dropout(x, p=0.3, training=True)
        x = self.fc4(x)
        return x

    def _init_rand_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class ModifiedEfficientNet(nn.Module):
    def __init__(self, dropout, device, num_classes=1):
        super(ModifiedEfficientNet, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, dropout_rate=0.2)
        self.model = EfficientNet.from_name(
            "efficientnet-b0", num_classes=num_classes, dropout_rate=0.2
        )
        # modify the first convolutional layer to accept single channel input
        self.model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.model.to(device)
        self.model._dropout = nn.Dropout(p=dropout, inplace=True)
        self.model._swish = nn.Sigmoid()

    def forward(self, x, drop_enabled=False):
        x = self.model(x)
        if drop_enabled:
            x = F.dropout(x, p=0.2, training=True)
        return x


class WeightAveragedEnsemble(nn.Module):
    def __init__(self, model_class, models_list):
        """Initialize the ensemble model with averaged weights from the given models.

        Args:
            model_class (nn.Module): The class of the base model to initialize.
            models_list (list): The list of model instances to average weights from.
        """
        super(WeightAveragedEnsemble, self).__init__()
        assert len(models_list) > 0, "The models_list cannot be empty."
        self.ensemble_model = model_class()
        self.average_weights(models_list)

    def average_weights(self, models_list):
        """Averages the weights of all models in models_list and assigns to ensemble_model."""
        with torch.no_grad():
            for name, param in self.ensemble_model.named_parameters():

                # Stack parameters from each model in the ensemble
                stack = torch.stack(
                    [model.state_dict()[name] for model in models_list], dim=0
                )
                # Take the mean of these parameters along the stacked dimension
                param.copy_(torch.mean(stack, dim=0))

    def forward(self, x, allow_drop=False):
        return self.ensemble_model(x, allow_drop=allow_drop)


class GatedFusion(nn.Module):
    def __init__(self, input_dim, gating_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gating_dim),
            nn.ReLU(),
            nn.Linear(gating_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate = self.gate(x)
        return x * gate


class LateFusedModel(nn.Module):
    def __init__(self, models, device="cuda"):
        super(LateFusedModel, self).__init__()
        self.device = device
        self.models = nn.ModuleList(models)

        for i in range(len(self.models)):
            self.models[i] = nn.Sequential(*list(self.models[i].children())[:-1])

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.input_dim = 512 * len(models)
        self.gating_dim = 512

        self.gated_fusion = GatedFusion(self.input_dim, self.gating_dim)
        self.classifier = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        inputs_plane0_pos,
        inputs_plane1_pos,
        inputs_plane2_pos,
        inputs_plane0_neg,
        inputs_plane1_neg,
        inputs_plane2_neg,
        drop_enabled=False,
    ):

        inputs_list = [
            inputs_plane0_pos.to(self.device),
            inputs_plane1_pos.to(self.device),
            inputs_plane2_pos.to(self.device),
            inputs_plane0_neg.to(self.device),
            inputs_plane1_neg.to(self.device),
            inputs_plane2_neg.to(self.device),
        ]

        outputs = [model(input) for model, input in zip(self.models, inputs_list)]

        for i in range(len(outputs)):
            outputs[i] = outputs[i].view(outputs[i].size(0), -1)

        fused_output = torch.cat(outputs, dim=1)
        gated_output = self.gated_fusion(fused_output)

        if drop_enabled:
            gated_output = F.dropout(gated_output, p=0.3, training=True)
        output = self.classifier(gated_output)

        return output


class LateFusedModel3(nn.Module):
    def __init__(self, models, device="cuda"):
        super(LateFusedModel3, self).__init__()
        self.device = device
        self.models = nn.ModuleList(models)

        for i in range(len(self.models)):
            self.models[i] = nn.Sequential(*list(self.models[i].children())[:-1])

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.input_dim = 512 * len(models)
        self.gating_dim = 512

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_plane0, inputs_plane1, inputs_plane2, drop_enabled=False):

        inputs_list = [
            inputs_plane0.to(self.device),
            inputs_plane1.to(self.device),
            inputs_plane2.to(self.device),
        ]

        outputs = [model(input) for model, input in zip(self.models, inputs_list)]

        for i in range(len(outputs)):
            outputs[i] = outputs[i].view(outputs[i].size(0), -1)

        fused_output = torch.cat(outputs, dim=1)
        fused_output = self.fc1(fused_output)
        fused_output = self.relu(fused_output)
        fused_output = self.fc2(fused_output)

        # if drop_enabled:
        #     gated_output = F.dropout(gated_output, p=0.2, training=True)
        output = self.classifier(fused_output)

        return output
