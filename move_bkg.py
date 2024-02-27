import shutil
import os

src_dir = '/mnt/lustre/helios-home/gartmann/venv/pdecay_sparse_copy/'
targ_dir = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-upd/'

bkg_srcs = {}
bkg_targs = {}

for plane in range(3):
    bkg_srcs[plane] = os.path.join(src_dir, f'plane{plane}', 'background')
    bkg_targs[plane] = os.path.join(targ_dir, f'plane{plane}', 'background')

for plane in range(3):
    for file in os.listdir(bkg_srcs[plane]):
        if file.endswith('.npz'):
            src_file_path = os.path.join(bkg_srcs[plane], file)
            shutil.copy(src_file_path, bkg_targs[plane])
