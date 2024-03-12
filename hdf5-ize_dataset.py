import scipy.sparse as sp
import os
import h5py
import shutil
import numpy as np
import pandas as pd


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

def create_hdf5_struct_batched(npz_dict, hdf5_path, batch_size=20):
    dt = np.dtype([
        ('data', h5py.special_dtype(vlen=np.float32)),
        ('indices', h5py.special_dtype(vlen=np.int32)),
        ('indptr', h5py.special_dtype(vlen=np.int32)),
        ('label', np.int8)
    ])

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for plane, classes in npz_dict.items():
            for cls, files in classes.items():
                num_batches = len(files) // batch_size + (1 if len(files) % batch_size > 0 else 0)
                for batch_num in range(num_batches):
                    batch_files = files[batch_num * batch_size: (batch_num + 1) * batch_size]
                    batch_data = []

                    for file_path in batch_files:
                        csr_matrix = sp.load_npz(file_path)
                        batch_data.append((csr_matrix.data, csr_matrix.indices, csr_matrix.indptr, 1 if cls == 'PDK' else 0))

                    struct_arr = np.array(batch_data, dtype=dt)
                    batch_dataset_name = f"{plane}/{cls}/batch_{batch_num + 1}" 
                    batch_dataset = hdf5_file.create_dataset(batch_dataset_name, data=struct_arr, compression='gzip')


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

def convert_batch_to_dataframe(hdf5_path, dataset_path):
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf:
        if dataset_path in hdf:
            dataset = hdf[dataset_path]
            data = dataset[:]
            df = pd.DataFrame(data)
            df['data'] = df['data'].apply(np.array)
            df['indices'] = df['indices'].apply(np.array)
            df['indptr'] = df['indptr'].apply(np.array)

            return df
        else:
            print("Dataset not found in the file.")
            return None


if __name__ == '__main__':

    dummy_root = '/Users/hardie/dummy-npz-dataset/'
    # create_dummy_data(dummy_root)

    create_hdf5_struct_batched(traverse_dirs(dummy_root), 'dummy_batched.hdf5')
    # cleanup_data(dummy_root)
