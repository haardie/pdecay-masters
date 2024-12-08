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
import multiprocessing as mp
import os
from scipy import sparse as sp
from scipy.sparse import csr_matrix


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
    # @weave.op()
    def __init__(self, evt_dirs):
        self.evt_dirs = evt_dirs
        self.labels = self.assign_labels()
        self.transform = transforms.Compose(
            [transforms.Resize((500, 500)), transforms.ToTensor()]
        )

        # self.mean, self.std = self._compute_mean_std_parallel()

        self.transform = transforms.Compose(
            [
                transforms.Resize((500, 500)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )

    def __len__(self):
        return len(self.evt_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        current_path = self.evt_dirs[idx]
        image = self.process_sparse_matrix(current_path)

        return image, label

    def assign_labels(self):
        labels = []
        for sparse_matrix_path in self.evt_dirs:
            if "signal" in sparse_matrix_path:
                labels.append(1)
            elif "background" in sparse_matrix_path:
                labels.append(0)
        print(
            f"Signal / background ratio = {sum(labels) / (len(labels) - sum(labels))}"
        )
        return labels

    def process_sparse_matrix(self, sparse_matrix_path):
        sparse_matrix = sp.sparse.load_npz(sparse_matrix_path)
        image = Image.fromarray(sparse_matrix.toarray().astype(np.uint8), mode="L")
        image = self.transform(image)
        return image

    # @weave.op()
    # def _compute_mean_std_parallel(self):
    #     num_processes = mp.cpu_count()
    #     print(f"Number of rinning processes: {num_processes}")

    #     chunk_size = len(self.evt_dirs) // num_processes
    #     print(f"Creating chunks of size {chunk_size}")

    #     chunks = [
    #         self.evt_dirs[i : i + chunk_size]
    #         for i in range(0, len(self.evt_dirs), chunk_size)
    #     ]

    #     with mp.Pool(processes=num_processes) as pool:
    #         results = pool.map(self._process_chunk, chunks)

    #     total_sum = np.sum([result[0] for result in results])
    #     total_sum_of_squares = np.sum([result[1] for result in results])
    #     num_samples = np.sum([result[2] for result in results])

    #     mean = total_sum / num_samples
    #     std = np.sqrt(total_sum_of_squares / num_samples - mean**2)

    #     print(f"Computed mean: {mean}, std: {std}")
    #     return mean, std

    @staticmethod
    def _process_chunk(chunk):
        total_sum = 0.0
        total_sum_of_squares = 0.0
        num_samples = 0

        for path in chunk:
            sparse_matrix = sp.sparse.load_npz(path)
            data = sparse_matrix.toarray().astype(np.float32)
            total_sum += np.sum(data)
            total_sum_of_squares += np.sum(data**2)
            num_samples += data.size

        return total_sum, total_sum_of_squares, num_samples


class SparseMatrixDatasetMeta(Dataset):
    def __init__(self, decay_dirs, plane_idx):
        self.plane_idx = plane_idx
        # self.evt_dirs = evt_dirs
        self.decay_dirs = decay_dirs
        self.labels, self.decay = self.assign_labels()
        # self.mu, self.sigma = self.compute_mean_std()
        self.transform = transforms.Compose(
            [
                transforms.Resize((500, 500)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[self.mu], std=[self.sigma]),
            ]
        )

    def __len__(self):
        if not hasattr(self, "_len"):
            self._len = sum(
                len(os.listdir(dir)) for dir in self.decay_dirs if os.path.isdir(dir)
            )
        return self._len

    def __getitem__(self, idx):
        if not hasattr(self, "_file_map"):
            self._file_map = []
            for decay_dir in self.decay_dirs:
                for evt_dir in os.listdir(decay_dir):
                    if os.path.isdir(os.path.join(decay_dir, evt_dir)):
                        for plane_dir in os.listdir(os.path.join(decay_dir, evt_dir)):
                            if f"plane{self.plane_idx}" in plane_dir:
                                files = os.listdir(
                                    os.path.join(decay_dir, evt_dir, plane_dir)
                                )
                                self._file_map.extend(
                                    [
                                        (decay_dir, evt_dir, plane_dir, file)
                                        for file in files
                                    ]
                                )

        # Get the file corresponding to idx
        try:
            decay_dir, evt_dir, plane_dir, file = self._file_map[idx]
        except IndexError:
            raise IndexError(
                f"Index {idx} out of range for dataset with length {len(self)}"
            )

        # Process the specific file
        label = self.labels[idx]
        filepath = os.path.join(decay_dir, evt_dir, plane_dir, file)
        sparse_data = np.load(filepath)
        sparse_matrix = csr_matrix(
            (sparse_data["data"], sparse_data["indices"], sparse_data["indptr"]),
            shape=sparse_data["shape"],
        )
        image = Image.fromarray(sparse_matrix.toarray().astype(np.uint8), mode="L")
        image = self.transform(image)

        return image, label

    def _get_single_item(self, idx):
        label = self.labels[idx]
        image = self.process_sparse_matrix()
        return image, label

    def process_sparse_matrix(self):
        for decay_dir in self.decay_dirs:
            for evt_dir in os.listdir(decay_dir):
                if os.path.isdir(evt_dir):
                    for plane_dir in os.listdir(evt_dir):
                        if f"plane{self.plane_idx}" in plane_dir:
                            filepath = os.listdir(
                                os.path.join(decay_dir, evt_dir, plane_dir)
                            )[0]
                            filepath = os.path.join(
                                decay_dir, evt_dir, plane_dir, filepath
                            )
                            sparse_data = np.load(filepath)
                            sparse_matrix = csr_matrix(
                                (
                                    sparse_data["data"],
                                    sparse_data["indices"],
                                    sparse_data["indptr"],
                                ),
                                shape=sparse_data["shape"],
                            )
                            image = Image.fromarray(
                                sparse_matrix.toarray().astype(np.uint8), mode="L"
                            )
                            image = self.transform(image)
                            return image

    def assign_labels(self):
        labels = []
        print(self.decay_dirs)
        for decay_dir in self.decay_dirs:
            decay_mode = os.path.basename(decay_dir)
            for evt_dir in os.listdir(decay_dir):
                label = 1 if "pdk_decays" in decay_dir else 0
                labels.append((label, decay_mode))
        signal_count = sum(label[0] for label in labels)
        oh_labels = [label[0] for label in labels]
        decay_labels = [label[1] for label in labels]
        print(decay_labels)
        print(
            f"Signal / background ratio = {signal_count / (len(labels) - signal_count) if len(labels) > signal_count else 0}"
        )
        return oh_labels, decay_labels


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
