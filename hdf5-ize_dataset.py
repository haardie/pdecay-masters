import h5py
import numpy as np
import uproot as ur
import awkward as ak
import torch
import matplotlib.pyplot as plt


def get_meta(branch):
    image_meta = {'image_id': '',
                  'origin.x': '',
                  'origin.y': '',
                  'width': '',
                  'height': '',
                  'col_count': '',
                  'row_count': '',
                  'plane': ''}

    for key in branch.keys():
        if '_meta.' in key:
            if 'x' not in key and 'y' not in key:
                key_parts = key.split('.')
                image_meta[key_parts[-1].lstrip('_')] = key
            else:
                key_parts = key.split('_')
                image_meta[key_parts[-1]] = key

    return image_meta


def get_comp_from_root(rootf, prec, next_):
    tree = rootf['image2d_tpc_tree']
    image_sup_branch = tree['_image_v']
    image_branch = tree['_image_v/_image_v._img']
    branch_entries = image_branch.array(entry_start=prec, entry_stop=next_)
    linear_data = branch_entries[0]
    image_meta = get_meta(image_sup_branch)
    col_count = image_sup_branch[image_meta['col_count']].array(entry_start=prec, entry_stop=next_)[0]
    row_count = image_sup_branch[image_meta['row_count']].array(entry_start=prec, entry_stop=next_)[0]
    nz_indices_evt = {'0': {'x': [], 'y':  []},
                      '1': {'x': [], 'y': []},
                      '2': {'x': [], 'y': []}}
    nz_values_evt = {'0': [], '1': [], '2': []}

    for plane_idx in range(3):
        mdata = ak.to_numpy(linear_data[plane_idx]).reshape(col_count[plane_idx], row_count[plane_idx])
        nz_indices = np.transpose(np.nonzero(mdata))
        nz_x, nz_y = nz_indices[:, 0], nz_indices[:, 1]
        nz_values = mdata[nz_x, nz_y]

        nz_indices_evt[f'{plane_idx}']['x'] = nz_x
        nz_indices_evt[f'{plane_idx}']['y'] = nz_y

        nz_values_evt[f'{plane_idx}'] = nz_values

    return nz_indices_evt, nz_values_evt


def coo_to_torch(hdf5_path, event_id, plane_id):

    image_key = f'evt{event_id}/plane{plane_id}'

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if image_key in hdf5_file:
            dset = hdf5_file[image_key]

            nz_x = dset['nz_x'][:]
            nz_y = dset['nz_y'][:]
            values = dset['value'][:]

            values = torch.FloatTensor(values)
            indices = torch.LongTensor([nz_x, nz_y])

            rows, cols = max(nz_x) + 1, max(nz_y) + 1
            sp_tensor = torch.sparse_coo_tensor(indices, values, torch.Size([rows, cols]))

            return sp_tensor
        else:
            print(f"key {image_key} not found in HDF5 file")
            return None

def create_hdf5_from_root(rootf_path, hdf5_path, batch_size=128):
    dtype = np.dtype([
        ('nz_x', np.uint32),
        ('nz_y', np.uint32),
        ('value', np.float32)])

    with ur.open(rootf_path) as rootf:
        tree = rootf['image2d_tpc_tree']
        entries = tree.num_entries
        num_batches = entries // batch_size + (1 if entries % batch_size > 0 else 0)

        with h5py.File(hdf5_path, 'w') as hdf5_file:
            for batch_num in range(num_batches):
                batch_group_name = f"batch{batch_num + 1}"
                batch_group = hdf5_file.create_group(batch_group_name)
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, entries)

                for evt_idx in range(start_idx, end_idx):
                    nz_indices_evt, nz_values_evt = get_comp_from_root(rootf, evt_idx, evt_idx+1)
                    event_group = batch_group.create_group(f"evt{evt_idx - start_idx + 1}")

                    for plane_idx in range(3):
                        nz_x, nz_y = nz_indices_evt[str(plane_idx)]['x'], nz_indices_evt[str(plane_idx)]['y']
                        nz_values = nz_values_evt[str(plane_idx)]
                        components = np.array(list(zip(nz_x, nz_y, nz_values)), dtype=dtype)

                        dset_name = f"evt{evt_idx + 1}/plane{plane_idx}"

                        event_group.create_dataset(dset_name, data=components, compression='gzip', compression_opts=1, chunks=True)

def print_hdf5_dirtree(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        def recurse_through_group(group, indent=0):
            for key, item in group.items():
                print('│   ' * indent + '├── ' + key)
                if isinstance(item, h5py.Group):
                    recurse_through_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):

                    if item.dtype.names:
                        print('│   ' * (indent + 1) + '└── {} (shape: {}, dtype: {})'.format(key, item.shape, item.dtype))
                        for name in item.dtype.names:
                            print('│   ' * (indent + 2) + '├── {} (field of {})'.format(name, key))
                    else:
                        print('│   ' * (indent + 1) + '└── {} (shape: {}, dtype: {})'.format(key, item.shape, item.dtype))

        recurse_through_group(file)


def plot_im_from_hdf5(hdf5_path, event_id, plane_idx):

    image_key = f'evt{event_id}/plane{plane_idx}'

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if image_key in hdf5_file:
            dset = hdf5_file[image_key]
            nz_x = dset['nz_x'][:]
            nz_y = dset['nz_y'][:]
            values = dset['value'][:]

            rows, cols = max(nz_x) + 1, max(nz_y) + 1

            densem = np.zeros((rows, cols))
            densem[nz_x, nz_y] = values
            
            plt.imshow(densem, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        else:
            print(f"key {image_key} not found in HDF5 file")


if __name__ == '__main__':
    rootf_path = '/Users/hardie/research/ROOT/protondecay_hA_BodekRitchie_dune10kt_1x2x6_54474279_179_20220423T063923Z_gen_g4_detsim_reco_65804491_0_20230126T175412Z_reReco_larcv.root'
    hdf5_path = '/Users/hardie/sample_root.hdf5'
    create_hdf5_from_root(rootf_path, hdf5_path, batch_size=64)
    print_hdf5_dirtree(hdf5_path)
    # plot_im_from_hdf5(hdf5_path, 1, 2)
    # sp_tensor = coo_to_torch(hdf5_path, 1, 2)
