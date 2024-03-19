import h5py
import numpy as np
import uproot as ur
import awkward as ak
import torch


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


def get_comp_from_root(rootf, prec, next):
    tree = rootf['image2d_tpc_tree']
    image_sup_branch = tree['_image_v']
    image_branch = tree['_image_v/_image_v._img']

    branch_entries = ak.to_numpy(image_branch.array(entry_start=prec, entry_stop=next))
    linear_data = branch_entries[0][0]

    image_meta = get_meta(image_sup_branch)
    col_count = image_sup_branch[image_meta['col_count']].array(entry_start=prec, entry_stop=next)[0][0]
    row_count = image_sup_branch[image_meta['row_count']].array(entry_start=prec, entry_stop=next)[0][0]
    mdata = linear_data.reshape(col_count, row_count)
    nz_indices = np.transpose(np.nonzero(mdata))
    nz_values = mdata[nz_indices[:, 0], nz_indices[:, 1]]

    return nz_indices, nz_values


def coo_to_torch(nz_ins, nz_vals, rows, cols):
    ins_torch = torch.LongTensor(nz_ins.T)
    vals_torch = torch.FloatTensor(nz_vals)
    sp_tensor = torch.sparse_coo_tensor(ins_torch, vals_torch, torch.Size([cols, rows]))

    return sp_tensor


def create_hdf5_from_root(root_file_path, hdf5_path):
    dt = np.dtype([
        ('nz_indices', h5py.special_dtype(vlen=np.uint16)),
        ('nz_values', h5py.special_dtype(vlen=np.float32))])

    with ur.open(root_file_path) as rootf:
        # 10 images in the ROOT file
        for i in range(10):
            nz_indices, nz_values = get_comp_from_root(rootf, i, i + 1)
            components = np.array([(nz_indices, nz_values)], dtype=dt)

            with h5py.File(hdf5_path, 'a') as hdf5_file:
                dset_name = f"image_{i + 1}"
                if dset_name not in hdf5_file:
                    hdf5_file.create_dataset(dset_name, data=components, compression='gzip',
                                             compression_opts=9, chunks=True)


def print_hdf5_dirtree(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as file:
        def recurse_through_group(group, indent=0):
            for key, item in group.items():
                print('│   ' * indent + '├── ' + key)
                if isinstance(item, h5py.Group):
                    recurse_through_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print('│   ' * (indent + 1) + '└── {} (shape: {}, dtype: {})'.format(key, item.shape, item.dtype))

                    print('│   ' * (indent + 2) + '├── nz_indices')
                    print('│   ' * (indent + 2) + '└── nz_values')

        recurse_through_group(file)


if __name__ == '__main__':
    rootf_path = '/Users/hardie/research/protondecay_hA_BodekRitchie_dune10kt_1x2x6_54474279_179_20220423T063923Z_gen_g4_detsim_reco_65804491_0_20230126T175412Z_reReco_larcv_10.root'
    hdf5_path = '/Users/hardie/sample_root.hdf5'
    create_hdf5_from_root(rootf_path, hdf5_path)
    # print_hdf5_dirtree(hdf5_path)
