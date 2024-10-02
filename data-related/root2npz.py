import uproot as ur
import awkward as ak
import numpy as np
import os
from scipy.sparse import csr_matrix
import logging
import argparse


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
        # mdata[mdata < 0] = 0

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


def root2npz(rootf_path, loggy, output_dir='.'):
    with ur.open(rootf_path) as rootf:
        loggy.info(f'Opened ROOT file: {rootf_path}')
        tree = rootf['image2d_tpc_tree']

        entries = tree.num_entries
        loggy.info('Accessed  tree: image2d_tpc_tree, number of entries: {entries}')

        for evt_idx in range(entries):
            loggy.info('-' * 50)
            loggy.info(f'Processing event {evt_idx}')

            nz_indices_evt, nz_values_evt = get_comp_from_root(rootf, evt_idx, evt_idx + 1)
            loggy.info(f'Extracted non-zero indices and values for event {evt_idx}')

            for plane_idx in range(3):
                loggy.info(f'\n')
                loggy.info(f'Processing plane {plane_idx}')
                nz_x, nz_y = nz_indices_evt[str(plane_idx)]['x'], nz_indices_evt[str(plane_idx)]['y']
                nz_values = nz_values_evt[str(plane_idx)]

                sparse_matrix = csr_matrix((nz_values, (nz_x, nz_y)), shape=(1000, 1000))
                loggy.info(f'Created sparse matrix for plane {plane_idx}')

                base_name = os.path.basename(rootf_path).split('.')[0]

                evt_dir = os.path.join(output_dir, base_name, f'{base_name}_{evt_idx}')
                plane_dirs = [os.path.join(evt_dir, f'plane{plane_idx}') for plane_idx in range(3)]

                for plane_dir in plane_dirs:
                    loggy.info(f'Creating directory: {plane_dir}')
                    os.makedirs(plane_dir, exist_ok=True)

                output_path = os.path.join(plane_dirs[plane_idx], f'{base_name}_{evt_idx}_plane{plane_idx}.npz')

                if not os.path.exists(output_path):
                    np.savez(output_path, data=sparse_matrix.data, indices=sparse_matrix.indices,
                             indptr=sparse_matrix.indptr, shape=sparse_matrix.shape)
                    loggy.info(f'Successfully saved npz file: {output_path}.')
                else:
                    loggy.info(f'File already exists: {output_path}. Skipping...')


def main_(src, dest):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    loggy = logging.getLogger('root2npz')
    loggy.info(f'Source directory: {src}')
    loggy.info(f'Destination directory: {dest}')

    for root_file in [file for file in os.listdir(src) if file.endswith('.root')]:
        loggy.info(f'Processing ROOT file: {root_file}')
        root2npz(os.path.join(src, root_file), loggy, dest)


#---------------------------------------------- MAIN ---------------------------------------------- #

parser = argparse.ArgumentParser(description='Convert ROOT files to NPZ files.')
parser.add_argument('src', type=str, help='The source directory containing ROOT files.')
parser.add_argument('dest', type=str, help='The destination directory to save NPZ files.')
args = parser.parse_args()

main_(args.src, args.dest)
