import scipy.sparse as sp
import os
import h5py
import shutil
import numpy as np


def traverse_dirs(root):
    npz_dict = {}

    for plane_dir in os.listdir(root):
        plane_path = os.path.join(root, plane_dir)

        if os.path.isdir(plane_path):
            for cls_dir in os.listdir(plane_path):
                if not cls_dir.startswith('.'):
                    cls_path = os.path.join(plane_path, cls_dir)

                    if os.path.isdir(cls_path):
                        for npz_file in os.listdir(cls_path):
                            if npz_file.endswith('.npz'):

                                filename_parts = npz_file.split('_')
                                if len(filename_parts) > 4:
                                    cls = filename_parts[1]
                                    plane = filename_parts[4][-1]
                                    evt = filename_parts[-1].split('.')[0]

                                    evt_dir_name = f'{cls}_{filename_parts[2]}_larcv_evtno{evt}'

                                    if cls not in npz_dict:
                                        npz_dict[cls] = {}

                                    if evt_dir_name not in npz_dict[cls]:
                                        npz_dict[cls][evt_dir_name] = {'files': {}}

                                    npz_path = os.path.join(cls_path, npz_file)
                                    npz_dict[cls][evt_dir_name]['files'][f'plane{plane}'] = npz_path

    return npz_dict


def create_hdf5_struct(npz_dict, hdf5_path, batch_size=20):

    dt = np.dtype([
        ('data_plane0', h5py.special_dtype(vlen=np.float32)),
        ('indices_plane0', h5py.special_dtype(vlen=np.int32)),
        ('indptr_plane0', h5py.special_dtype(vlen=np.int32)),
        ('data_plane1', h5py.special_dtype(vlen=np.float32)),
        ('indices_plane1', h5py.special_dtype(vlen=np.int32)),
        ('indptr_plane1', h5py.special_dtype(vlen=np.int32)),
        ('data_plane2', h5py.special_dtype(vlen=np.float32)),
        ('indices_plane2', h5py.special_dtype(vlen=np.int32)),
        ('indptr_plane2', h5py.special_dtype(vlen=np.int32)),
        ('label', np.int8)  # label is the same for all plane imgs
    ])

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for cls, events in npz_dict.items():

            all_event_data = []
            for evt_dir_name, details in events.items():
                files = details.get('files', {})

                event_data = [None, None, None]  # placeholders for plane data
                for plane in ['plane0', 'plane1', 'plane2']:
                    file_path = files.get(plane)
                    if file_path:
                        csr_matrix = sp.load_npz(file_path)
                        event_data[int(plane[-1])] = (csr_matrix.data, csr_matrix.indices, csr_matrix.indptr)
                all_event_data.append(event_data)

            num_batches = len(all_event_data) // batch_size + (1 if len(all_event_data) % batch_size > 0 else 0)

            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                batch_event_data = all_event_data[start_idx:end_idx]
                batch_data = []

                for event_data in batch_event_data:
                    batch_entry = ()
                    for plane_data in event_data:
                        if plane_data:
                            batch_entry += plane_data
                        else:
                            batch_entry += (np.array([], dtype=np.float32), np.array([], dtype=np.int32),
                                            np.array([], dtype=np.int32))
                    # add label
                    batch_entry += (1 if cls == 'PDK' else 0,)
                    batch_data.append(batch_entry)

                # only create a dataset if there is data
                if batch_data:
                    struct_arr = np.array(batch_data, dtype=dt)
                    batch_dataset_name = f"{cls}/batch_{batch_num + 1}"
                    if cls not in hdf5_file:
                        hdf5_file.create_group(cls)
                    hdf5_file.create_dataset(batch_dataset_name, data=struct_arr)


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


def cleanup_data(root_dir):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)


def explore_hdf5_file(hdf5_path):

    if not os.path.exists(hdf5_path):
        print(f"No file found at {hdf5_path}")
        return

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if not list(hdf5_file.keys()):
            print("The HDF5 file is empty.")
            return

        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, Type: {obj.dtype}")

        hdf5_file.visititems(print_attrs)


def print_hdf5_dirtree(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as file:
        def recurse_through_group(group, indent=0):
            for key, item in group.items():
                print('│   ' * indent + '├── ' + key)
                if isinstance(item, h5py.Group):
                    recurse_through_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print('│   ' * (indent + 1) + '└── {} (shape: {}, dtype: {})'.format(key, item.shape, item.dtype))
                    num_events = item.shape[0] 
                    for i in range(num_events):
                        print('│   ' * (indent + 2) + f'├── event_{i + 1}')
                        for plane in range(3):
                            print('│   ' * (indent + 3) + f'├── plane{plane}')
                            print('│   ' * (indent + 4) + f'├── data_plane{plane}')
                            print('│   ' * (indent + 4) + f'├── indices_plane{plane}')
                            print('│   ' * (indent + 4) + f'├── indptr_plane{plane}')
                        print('│   ' * (indent + 3) + f'└── label')

        recurse_through_group(file)


if __name__ == '__main__':
    dummy_root = '/Users/hardie/dummy-npz-dataset/'
    npz_struct = traverse_dirs(dummy_root)
    # print(npz_struct)
    # create_dummy_data(dummy_root)

    create_hdf5_struct(npz_struct, 'dummy_batched_new.hdf5')
    # print_hdf5_contents_with_events('dummy_batched_new.hdf5')
    # # cleanup_data(dummy_root)
