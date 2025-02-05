#!~/venv/bin/python

import os
import time
from multiprocessing import Pool, Manager

import cv2 as cv
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.morphology import dilation, rectangle
from skimage.segmentation import flood_fill

import wandb


def aggressive_dilation(binary_mask, iterations=4, structuring_element_size=(4, 2)):
    structuring_element = rectangle(*structuring_element_size)
    for _ in range(iterations):
        binary_mask = dilation(binary_mask, structuring_element)
    return binary_mask


def sparse_convolution(image, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    convolved_image = convolve(image.astype(float), kernel, mode='nearest')

    downsampled_image = convolved_image[::kernel_size, ::kernel_size]
    return downsampled_image


def image_batch_generator(path_list, batch_size=256):
    for i in range(0, len(path_list), batch_size):
        batch_paths = path_list[i:i + batch_size]
        batch_images = {}
        for path in batch_paths:
            if path.endswith('.npz'):
                key = os.path.splitext(os.path.basename(path))[0]
                image = sp.load_npz(path).toarray()
                image = np.uint8(image)
                image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
                batch_images[key] = image
        yield i // batch_size, batch_images


def split_path(path):
    filename = os.path.basename(path)
    if 'larcv' in filename:
        type = filename.split('_')[1]
        batch = filename.split('_')[2]
        plane = int(filename.split('_')[4][5:])
        file_num = int(filename.split('_')[5].split('.')[0])

        return type, batch, plane, file_num

    else:
        type = filename.split('_')[0]
        plane = int(filename.split('_')[1][5:])
        file_num = int(filename.split('_')[2].split('.')[0])

        return type, 0, plane, file_num


def draw_bboxes_on_image(image, regions):
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cv.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
    return image


def process_batch(batch_number, image_dict, min_reg_area, max_hdist, tol, aeps, wandb_table):
    two_line_keys = []
    empty_keys = []
    found_in_batch = 0
    for key, img in image_dict.items():
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        conv_img = sparse_convolution(img, kernel_size=5)
        conv_img = np.uint8(conv_img)
        _, binary_mask = cv.threshold(conv_img, 0, 255, cv.THRESH_BINARY)
        binary_mask = aggressive_dilation(binary_mask)
        nonzero_pixels = np.transpose(np.nonzero(binary_mask))
        if len(nonzero_pixels) > 0:
            rand_pix_idx = np.random.choice(len(nonzero_pixels))
            x_seed, y_seed = nonzero_pixels[rand_pix_idx]
            labeled_mask = flood_fill(binary_mask, (x_seed, y_seed), 255, tolerance=tol)
        else:
            empty_keys.append(key)
            continue
        labeled_img = label(labeled_mask)
        regions = regionprops(labeled_img)
        large_regions = [region for region in regions if region.area >= min_reg_area]
        found_two_lines = False
        dists = []
        for i, region1 in enumerate(large_regions):
            if found_two_lines:
                break
            for region2 in large_regions[i + 1:]:
                cent1 = np.array(region1.centroid)
                cent2 = np.array(region2.centroid)
                hdist = abs(cent1[1] - cent2[1])
                adif = abs(region2.area - region1.area)
                dists.append(hdist)
                if hdist > max_hdist and adif <= aeps:
                    found_two_lines = True
                    two_line_keys.append(key)
                    found_in_batch += 1
                    break
        if len(dists) > 0:
            wandb_table.add_data(key, wandb.Image(img), max(dists), found_two_lines)
        else:
            wandb_table.add_data(key, wandb.Image(img), -7, found_two_lines)
    return two_line_keys, empty_keys, found_in_batch


def constrained_flood(image_gen, min_reg_area, max_hdist, tol=10, aeps=150):
    with Manager() as manager:
        wandb_table = manager.list(wandb.Table(columns=['key', 'image', 'max dist', 'found ghosts (bool)']))
        pool = Pool(processes=os.cpu_count())
        results = []
        for batch_number, image_dict in image_gen:
            result = pool.apply_async(process_batch,
                                      (batch_number, image_dict, min_reg_area, max_hdist, tol, aeps, wandb_table))
            results.append(result)
        pool.close()
        pool.join()
        two_line_keys = []
        empty_keys = []
        for result in results:
            two_line_keys_batch, empty_keys_batch, found_in_batch = result.get()
            two_line_keys.extend(two_line_keys_batch)
            empty_keys.extend(empty_keys_batch)
            print(f'In batch {batch_number}, found {found_in_batch} images with two lines meeting criteria')
            print('=' * 30)
        print(f'Empty image count: {len(empty_keys)}')
        return two_line_keys


if __name__ == '__main__':
    run = wandb.init(project="mpdecay-cff")
    wandb_table = wandb.Table(columns=['key', 'image', 'max dist', 'found ghosts (bool)'])

    data_dir = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-pos/'

    plane0 = os.path.join(data_dir, f'plane0', 'signal')
    plane1 = os.path.join(data_dir, f'plane1', 'signal')

    evt_files0 = [os.path.join(plane0, file) for file in os.listdir(plane0) if '.npz' in file]
    evt_files1 = [os.path.join(plane1, file) for file in os.listdir(plane1) if '.npz' in file]

    print('applying cff to plane0 {}'.format(time.strftime('%d-%m_%H-%M')))
    two_lines0 = constrained_flood(image_batch_generator(evt_files0), min_reg_area=400, max_hdist=12, tol=10)
    print(
        f'Found {len(two_lines0)}/{len(evt_files0)} or {len(two_lines0) / len(evt_files0) * 100:.2f}% two lines in plane0')

    print('applying cff to plane1 {}'.format(time.strftime('%d-%m_%H-%M')))
    two_lines1 = constrained_flood(image_batch_generator(evt_files1), min_reg_area=400, max_hdist=12, tol=10)

    print(
        f'Found {len(two_lines1)}/{len(evt_files1)} or {len(two_lines1) / len(evt_files1) * 100:.2f}% two lines in plane1')

    print('done {}'.format(time.strftime('%d-%m_%H-%M')))
    print(
        f'Totally found {len(two_lines0) + len(two_lines1)}/{len(evt_files0) + len(evt_files1)} or {(len(two_lines0) + len(two_lines1)) / (len(evt_files0) + len(evt_files1)) * 100:.2f}% two lines')
    run.log({'ghosts_tab_{}'.format(time.strftime('%d-%m_%H-%M')): wandb_table})
    run.finish()

    # bin_path = '/mnt/lustre/helios-home/gartmann/cff-bin'
    # for path in two_lines0:
    #     type, batch, plane, file_number = split_path(path)

    #     path1 = os.path.join(data_dir, f'plane1', f'files_{type}_{batch}_larcv_plane1_{file_number}.npz')
    #     path2 = os.path.join(data_dir, f'plane2', f'files_{type}_{batch}_larcv_plane2_{file_number}.npz')

    #     if os.path.exists(path1) and os.path.exists(path2):
    #         shutil.move(path, os.path.join(bin_path, 'plane0', 'signal'))
    #         shutil.move(path1, os.path.join(bin_path, 'plane1', 'signal'))
    #         shutil.move(path2, os.path.join(bin_path, 'plane2', 'signal'))
    # os.remove(path) #double-track-bkg-bin
    # os.remove(path1)
    # os.remove(path2)
    #
    # else:
    #     os.remove(path)

    # for path in two_lines1:
    #     type, batch, plane, file_number = split_path(path)

    #     path0 = os.path.join(data_dir, f'plane0', f'files_{type}_{batch}_larcv_plane0_{file_number}.npz')
    #     path2 = os.path.join(data_dir, f'plane2', f'files_{type}_{batch}_larcv_plane2_{file_number}.npz')

    #     if os.path.exists(path0) and os.path.exists(path2):
    #         shutil.move(path, os.path.join(bin_path, 'plane1', 'signal'))
    #         shutil.move(path0, os.path.join(bin_path, 'plane0', 'signal'))
    # shutil.move(path2, os.path.join(bin_path, 'plane2', 'signal'))
    # os.remove(path)
    # os.remove(path0)
    # os.remove(path2)
    #
    # else:
    #     os.remove(path)
