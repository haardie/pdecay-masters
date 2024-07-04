import numpy as np
import scipy as sp
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

class ModifiedEfficientNet(nn.Module):
    def __init__(self, dropout, device, num_classes=1):
        super(ModifiedEfficientNet, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, dropout_rate=0.2)
        self.model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes, dropout_rate=0.2)
        self.model.to(device)
        self.model._dropout = nn.Dropout(p=dropout, inplace=True)
        self.model._swish = nn.Sigmoid()

    def forward(self, x, drop_enabled=False):
        x = self.model(x)
        if drop_enabled:
            x = F.dropout(x, p=0.2, training=True)
        return x


class SparseMatrixDataset(Dataset):
    def __init__(self, sparse_matrix_paths):
        self.sparse_matrix_paths = sparse_matrix_paths
        self.labels = self.assign_labels()
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.0035], std=[0.85])  # The normalization values are calculated from the
            # training set
        ])

    def __len__(self):
        return len(self.sparse_matrix_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        current_path = self.sparse_matrix_paths[idx]
        image = self.process_sparse_matrix(current_path)

        return image, label

    def assign_labels(self):
        labels = []
        for sparse_matrix_path in self.sparse_matrix_paths:
            if 'signal' in sparse_matrix_path:
                labels.append(1)
            elif 'background' in sparse_matrix_path:
                labels.append(0)
        print(f'Signal / background ratio = {sum(labels) / (len(labels) - sum(labels))}')

        return labels

    def process_sparse_matrix(self, sparse_matrix_path):
        sparse_matrix = sp.sparse.load_npz(sparse_matrix_path)
        image = Image.fromarray(sparse_matrix.toarray().astype(np.uint8), mode='L')
        image = self.transform(image)
        return image


class EarlyFusionDataset(Dataset):
    def __init__(self, dataset_list):
        assert all(len(dataset) == len(dataset_list[0]) for dataset in dataset_list), 'All datasets must be same length'
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list[0])

    def __getitem__(self, idx):
        images, labels = zip(*[dataset[idx] for dataset in self.dataset_list])
        image = torch.cat(images, dim=0)
        label = labels[0]
        return image, label


class ModifiedResNet(nn.Module):
    def __init__(self, device='cuda', num_classes=1):
        super(ModifiedResNet, self).__init__()
        self.device = device
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.model.to(device)
        self._init_rand_weights()

    def forward(self, x, allow_drop=False):
        x = self.model(x)
        x = self.fc2(x)
        if allow_drop:
            x = F.dropout(x, p=0.3, training=True)
        x = self.fc3(x)
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


class GatedFusion(nn.Module):
    def __init__(self, input_dim, gating_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gating_dim),
            nn.ReLU(),
            nn.Linear(gating_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        return x * gate


class LateFusedModel(nn.Module):
    def __init__(self, models, device='cuda'):
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

    def forward(self, inputs_plane0_pos, inputs_plane1_pos, inputs_plane2_pos,
                inputs_plane0_neg, inputs_plane1_neg, inputs_plane2_neg, drop_enabled=False):

        inputs_list = [inputs_plane0_pos.to(self.device), inputs_plane1_pos.to(self.device),
                       inputs_plane2_pos.to(self.device),
                       inputs_plane0_neg.to(self.device), inputs_plane1_neg.to(self.device),
                       inputs_plane2_neg.to(self.device)]

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
    def __init__(self, models, device='cuda'):
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

        self.gated_fusion = GatedFusion(self.input_dim, self.gating_dim)
        self.classifier = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_plane0, inputs_plane1, inputs_plane2, drop_enabled=False):

        inputs_list = [inputs_plane0.to(self.device), inputs_plane1.to(self.device), inputs_plane2.to(self.device)]

        outputs = [model(input) for model, input in zip(self.models, inputs_list)]

        for i in range(len(outputs)):
            outputs[i] = outputs[i].view(outputs[i].size(0), -1)

        fused_output = torch.cat(outputs, dim=1)
        gated_output = self.gated_fusion(fused_output)

        if drop_enabled:
            gated_output = F.dropout(gated_output, p=0.3, training=True)
        output = self.classifier(gated_output)

        return output


class HDF5SparseDataset(Dataset):
    @weave.op()
    def __init__(self, signal_files, background_files, plane_idx):
        self.signal_files = signal_files
        self.background_files = background_files
        self.plane_idx = plane_idx
   
        self.keys = []
        self.labels = []

        self._load_data_multiprocess(signal_files, background_files)
        # self.mean, self.std = self._compute_mean_std_parallel()

        self.transform = transforms.Compose([
        transforms.Resize(500),
        transforms.Normalize(mean=[20.19365526422530], std=[32.858101401293254])
        ])

    @weave.op()
    def _load_data_multiprocess(self, signal_files, background_files):
        print(f"Using {mp.cpu_count()} CPUs.")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            signal_results = pool.map(self._process_file, [(file, 1, self.plane_idx) for file in signal_files])
            background_results = pool.map(self._process_file, [(file, 0, self.plane_idx) for file in background_files])

        signal_count = sum(len(result[0]) for result in signal_results)
        background_count = sum(len(result[0]) for result in background_results)

        for result in signal_results:
            self.keys.extend(result[0])
            self.labels.extend(result[1])

        for result in background_results:
            self.keys.extend(result[0])
            self.labels.extend(result[1])

        if background_count > 0:
            ratio = signal_count / background_count
            print(f"Signal-to-Background Ratio: {ratio:.2f}")
        else:
            print("No background samples found.")

    @staticmethod
    def _process_file(args):
        file_path, label, plane_idx = args
        keys = []
        labels = []
        with h5py.File(file_path, 'r') as f:
            for batch in f.keys():
                for evt in f[batch].keys():
                    key_path = f'{batch}/{evt}/plane{plane_idx}'
                    if key_path in f:
                        keys.append((file_path, key_path))
                        labels.append(label)
        return keys, labels
    
    ##========================================================
    ## ======== COMMENT OUT TO COMPUTE ON NEW DATASET ========
    # @weave.op()
    # def _compute_mean_std_parallel(self):
    #     num_processes = mp.cpu_count()  # Use all available CPUs
    #     chunk_size = len(self.keys) // num_processes

    #     chunks = [self.keys[i:i + chunk_size] for i in range(0, len(self.keys), chunk_size)]

    #     with mp.Pool(processes=num_processes) as pool:
    #         results = pool.map(self._process_chunk, chunks)

    #     total_sum = np.sum(result[0] for result in results)
    #     total_sum_of_squares = np.sum(result[1] for result in results)
    #     num_samples = np.sum(result[2] for result in results)

    #     mean = total_sum / num_samples
    #     std = np.sqrt(total_sum_of_squares / num_samples - mean ** 2)

    #     print(f"Computed mean: {mean}, std: {std}")
    #     return mean, std

    #========================================================
    # @staticmethod
    # def _process_chunk(chunk):
    #     total_sum = 0
    #     total_sum_of_squares = 0
    #     num_samples = 0

    #     for file_path, key in chunk:
    #         with h5py.File(file_path, 'r') as f:
    #             data = np.array(f[key])
    #             for entry in data:
    #                 x, y, value = entry
    #                 total_sum += value
    #                 total_sum_of_squares += value ** 2
    #                 num_samples += 1

    #     return total_sum, total_sum_of_squares, num_samples
    #========================================================

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        file_path, key = self.keys[idx]
        data2d = np.zeros((1000, 1000))

        with h5py.File(file_path, 'r') as f:
            data = np.array(f[key])
            for entry in data:
                x, y, value = entry
                data2d[x, y] = value

        data2d = np.expand_dims(data2d, axis=0)
        tensor = torch.tensor(data2d, dtype=torch.float32)
        tensor = self.transform(tensor)
        label = self.labels[idx]
        return tensor, label
