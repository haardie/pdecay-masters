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
    def __init__(self, device='cpu', num_classes=1):
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

        signal_count = self._load_keys(self.signal_files, label=1)
        background_count = self._load_keys(self.background_files, label=0)

        if background_count > 0:
            ratio = signal_count / background_count
            print(f"Signal-to-Background Ratio: {ratio:.2f}")
        else:
            print("No background samples found.")

    @weave.op()
    def _load_keys(self, files, label):
        count = 0
        for file_path in files:
            with h5py.File(file_path, 'r') as f:
                for batch in f.keys():
                    for evt in f[batch].keys():
                        if f'{batch}/{evt}/plane{self.plane_idx}' in f:
                            self.keys.append((file_path, f'{batch}/{evt}/plane{self.plane_idx}'))
                            self.labels.append(label)
                            count += 1
        return count

    def __len__(self):
        return len(self.keys)

    @weave.op()
    def __getitem__(self, idx):
        file_path, key = self.keys[idx]
        with h5py.File(file_path, 'r') as f:
            data2d = np.zeros((1000, 1000))
            data = np.array(f[key])
            for entry in data:
                x, y, value = entry
                data2d[x, y] = value

        data2d = np.expand_dims(data2d, axis=0)  # Shape: (1, 1000, 1000)
        tensor = torch.tensor(data2d, dtype=torch.float32)
        label = self.labels[idx]
        return tensor, label


@weave.op()
def create_dataloader(dataset, target_batch_size, shuffle, nworkers):
    initial_batch_size = 64

    if target_batch_size % initial_batch_size != 0:
        raise ValueError("Target batch size must be a multiple of initial batch size")

    loader = DataLoader(dataset, batch_size=initial_batch_size, shuffle=shuffle, num_workers=nworkers, pin_memory=True)

    combined_batches = []
    batch_list = []

    for i, (data, label) in enumerate(loader):
        combined_batches.append((data, label))
        if (i + 1) % (target_batch_size // initial_batch_size) == 0:
            combined_data = torch.cat([batch[0] for batch in combined_batches])
            combined_labels = torch.cat([batch[1] for batch in combined_batches])
            batch_list.append((combined_data, combined_labels))
            combined_batches = []

    return batch_list
