import scipy.sparse as sp
import os
import h5py
import shutil
import numpy as np
from PIL import Image


def traverse_dirs(root):
    npz_dict = {'plane0': {'PDK': [], 'AtmoNu': []},
                'plane1': {'PDK': [], 'AtmoNu': []},
                'plane2': {'PDK': [], 'AtmoNu': []}}

    for plane_dir in os.listdir(root):
        plane_path = os.path.join(root, plane_dir)

        if os.path.isdir(plane_path):
            for cls_dir in os.listdir(plane_path):
                cls_path = os.path.join(plane_path, cls_dir)

                if os.path.isdir(cls_path):
                    for npz_file in os.listdir(cls_path):
                        npz_file_path = os.path.join(cls_path, npz_file)
                        if npz_file.endswith('.npz'):

                            # basic filename: files_{cls}_BATCH_larcv_plane{plane}_EVT
                            filename_parts = npz_file.split('_')
                            if len(filename_parts) > 0:
                                cls = filename_parts[1]
                                plane = f'{filename_parts[4]}'

                                npz_dict[f'{plane}'][cls].append(npz_file_path)

    return npz_dict


def create_hdf5_struct(npz_dict, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for plane, classes in npz_dict.items():
            for cls, files in classes.items():
                class_group_path = f"{plane}/{cls}"
                class_group = hdf5_file.create_group(class_group_path)

                labels_dataset = class_group.create_dataset("labels", (len(files),), dtype='i8', compression='gzip')

                # store components of a sparse matrix
                for idx, file_path in enumerate(files):
                    basename = os.path.splitext(os.path.basename(file_path))[0]
                    csr_matrix = sp.load_npz(file_path)

                    for component in ['data', 'indices', 'indptr']:
                        component_data = getattr(csr_matrix, component)
                        dataset_name = f"{basename}_{component}"
                        class_group.create_dataset(dataset_name, data=component_data, compression='gzip')

                    labels_dataset[idx] = 1 if cls == 'PDK' else 0


def create_dummy_data(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    classes = ['PDK', 'AtmoNu']
    planes = range(3)
    for plane in planes:
        for cls in classes:
            for i in range(100):
                random_matrix = np.random.binomial(1, 0.2, size=(1000, 1000))
                sparse_matrix = sp.csr_matrix(random_matrix)
                file_name = f'files_{cls}_0_dummy_plane{plane}_{i}.npz'
                dir_path = os.path.join(root_dir, f'plane{plane}', cls)
                os.makedirs(dir_path, exist_ok=True)
                sp.save_npz(os.path.join(dir_path, file_name), sparse_matrix)


def load_sparse_matrix(hdf5_file, group_path, basename):
    data = hdf5_file[f'{group_path}/{basename}_data'][:]
    indices = hdf5_file[f'{group_path}/{basename}_indices'][:]
    indptr = hdf5_file[f'{group_path}/{basename}_indptr'][:]
    return sp.csr_matrix((data, indices, indptr))


def cleanup_data(root_dir):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)


if __name__ == '__main__':

    dummy_root = '/Users/hardie/dummy-npz-dataset/'
    # create_dummy_data(dummy_root)

    create_hdf5_struct(traverse_dirs(dummy_root), 'dummy.hdf5')

    # with h5py.File('dummy.hdf5', 'r') as hdf:
    #     def print_details(name, obj):
    #         if isinstance(obj, h5py.Dataset):
    #             print(f"Dataset (=file): {name}")
    #             print(f"Shape: {obj.shape}, Type: {obj.dtype}")
    #             # print("Data:", obj[:])
    #         else:
    #             print(f"Group (=folder): {name}")
    #
    #     hdf.visititems(print_details)

    # example matrix reconstruction & plotting
    with h5py.File('dummy.hdf5', 'r') as hdf5_file:
        # Load the sparse matrix
        csr_matrix = load_sparse_matrix(hdf5_file, 'plane0/AtmoNu', 'files_AtmoNu_0_dummy_plane0_0')
        dense = csr_matrix.todense()
        min = dense.min()
        max = dense.max()
        norm_matrix = (dense - min) / (max - min) * 255
        norm_matrix = norm_matrix.astype(np.uint8)
        img = Image.fromarray(norm_matrix)
        img.show()
        
    # cleanup_data(dummy_root)
