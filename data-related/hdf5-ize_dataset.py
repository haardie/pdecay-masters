import h5py
import numpy as np
import uproot as ur
import awkward as ak


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


def find_max_intensity_patch(arr, out_shape):
    int_img = np.cumsum(np.cumsum(arr, axis=0), axis=1)

    sum_patches = int_img[out_shape[0] - 1:, out_shape[1] - 1:] - \
                  np.pad(int_img[out_shape[0] - 1:, :-out_shape[1]], ((0, 0), (1, 0)), 'constant', constant_values=0) - \
                  np.pad(int_img[:-out_shape[0], out_shape[1] - 1:], ((1, 0), (0, 0)), 'constant', constant_values=0) + \
                  np.pad(int_img[:-out_shape[0], :-out_shape[1]], ((1, 0), (1, 0)), 'constant', constant_values=0)

    max_pos = np.unravel_index(np.argmax(sum_patches), sum_patches.shape)

    return arr[max_pos[0]:max_pos[0] + out_shape[0], max_pos[1]:max_pos[1] + out_shape[1]]


def center_nz_pattern(patch, out_shape):
    nz_x, nz_y = np.nonzero(patch)
    if not nz_x.size:
        return np.zeros(out_shape)

    # center nz indices around the geometric center of the out_shape
    meanx = np.mean(nz_x).astype(int)
    meany = np.mean(nz_y).astype(int)

    centx = nz_x - meanx + out_shape[1] // 2
    centy = nz_y - meany + out_shape[0] // 2

    # filter out indices that fall outside the out_shape dimensions
    valid_mask = (centx >= 0) & (centx < out_shape[0]) & (centy >= 0) & (centy < out_shape[1])
    valid_x = centx[valid_mask]
    valid_y = centy[valid_mask]
    valid_values = patch[nz_x[valid_mask], nz_y[valid_mask]]

    cpatch = np.zeros(out_shape)
    cpatch[valid_y, valid_x] = valid_values
    return cpatch


def get_comp_from_root(rootf, prec, next_):
    tree = rootf['image2d_tpc_tree']
    image_sup_branch = tree['_image_v']
    image_branch = tree['_image_v/_image_v._img']
    branch_entries = image_branch.array(entry_start=prec, entry_stop=next_)
    linear_data = branch_entries[0]
    image_meta = get_meta(image_sup_branch)
    col_count = image_sup_branch[image_meta['col_count']].array(entry_start=prec, entry_stop=next_)[0]
    row_count = image_sup_branch[image_meta['row_count']].array(entry_start=prec, entry_stop=next_)[0]
    nz_indices_evt = {'0': {'x': [], 'y': []},
                      '1': {'x': [], 'y': []},
                      '2': {'x': [], 'y': []}}
    nz_values_evt = {'0': [], '1': [], '2': []}

    for plane_idx in range(3):
        mdata = ak.to_numpy(linear_data[plane_idx]).reshape(col_count[plane_idx], row_count[plane_idx])
        mdata[mdata < 0] = 0

        # === ROI extraction ===
        max_int_patch = find_max_intensity_patch(mdata, (1000, 200))
        cmdata = center_nz_pattern(max_int_patch, (1000, 1000))
        # ======================

        nz_indices = np.transpose(np.nonzero(cmdata))
        nz_x, nz_y = nz_indices[:, 0], nz_indices[:, 1]
        nz_values = cmdata[nz_x, nz_y]

        nz_indices_evt[f'{plane_idx}']['x'] = nz_x
        nz_indices_evt[f'{plane_idx}']['y'] = nz_y

        nz_values_evt[f'{plane_idx}'] = nz_values

    return nz_indices_evt, nz_values_evt


def create_hdf5_from_root(rootf_path, hdf5_path, batch_size, nevents=-1):
    dtype = np.dtype([
        ('nz_x', np.uint32),
        ('nz_y', np.uint32),
        ('value', np.float32)])

    with ur.open(rootf_path) as rootf:
        tree = rootf['image2d_tpc_tree']
        entries = tree.num_entries
        if nevents > -1 :
            entries = nevents
        num_batches = entries // batch_size + (1 if entries % batch_size > 0 else 0)

        with h5py.File(hdf5_path, 'w') as hdf5_file:
            for batch_num in range(num_batches):
                batch_group_name = f"batch{batch_num + 1}"
                batch_group = hdf5_file.create_group(batch_group_name)
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, entries)

                for evt_idx in range(start_idx, end_idx):
                    nz_indices_evt, nz_values_evt = get_comp_from_root(rootf, evt_idx, evt_idx + 1)
                    event_group = batch_group.create_group(f"evt{evt_idx - start_idx + 1}")

                    for plane_idx in range(3):
                        nz_x, nz_y = nz_indices_evt[str(plane_idx)]['x'], nz_indices_evt[str(plane_idx)]['y']
                        nz_values = nz_values_evt[str(plane_idx)]
                        components = np.array(list(zip(nz_x, nz_y, nz_values)), dtype=dtype)

                        dset_name = f"evt{evt_idx + 1}/plane{plane_idx}"

                        event_group.create_dataset(dset_name, data=components, chunks=True, compression_opts=1,
                                                   compression='gzip')


if __name__ == '__main__':
    import sys
    rootf_path = sys.argv[1]
    #'/Users/hardie/research/ROOT/protondecay_hA_BodekRitchie_dune10kt_1x2x6_54474279_179_20220423T063923Z_gen_g4_detsim_reco_65804491_0_20230126T175412Z_reReco_larcv.root'
    hdf5_path = sys.argv[2]
    nevts=-1
    if len(sys.argv) > 3 :
        nevts=int(sys.argv[3])
    #'/Users/hardie/sample_root.hdf5'
    create_hdf5_from_root(rootf_path, hdf5_path, batch_size=64, nevents=nevts)
