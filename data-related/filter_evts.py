import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.measure import label, regionprops
from skimage.segmentation import flood_fill
import find_ghosts as find2


def detect_regions(img, tol, detectable_area):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    conv_img = find2.sparse_convolution(img, kernel_size=5)
    conv_img = np.uint8(conv_img)

    _, binary_mask = cv.threshold(conv_img, 0, 255, cv.THRESH_BINARY)
    binary_mask = find2.aggressive_dilation(binary_mask, iterations=4, structuring_element_size=(4, 2))

    nonzero_pixels = np.transpose(np.nonzero(binary_mask))
    if len(nonzero_pixels) > 0:

        rand_pix_idx = np.random.choice(len(nonzero_pixels))
        x_seed, y_seed = nonzero_pixels[rand_pix_idx]
        labeled_mask = flood_fill(binary_mask, (x_seed, y_seed), 255, tolerance=tol)
    else:
        print('No detectable regions found.')
        return [], conv_img

    labeled_img = label(labeled_mask)

    regions = regionprops(labeled_img, intensity_image=conv_img)
    detectable_regions = [region for region in regions if region.area >= detectable_area]

    return detectable_regions, conv_img


def calculate_geometric_properties(regions):
    properties_list = []
    for region in regions:
        properties = {
            'Area': region.area,
            'Perimeter': region.perimeter,
            'Major Axis Length': region.major_axis_length,
            'Minor Axis Length': region.minor_axis_length,
            'Eccentricity': region.eccentricity,
            'Orientation': region.orientation,
            'Centroid': region.centroid,
            'Bounding Box': region.bbox,
            'Convex Area': region.convex_area,
            'Solidity': region.solidity
        }
        properties_list.append(properties)

    return properties_list


def filter_regions(regions, threshold_dict):
    if len(regions) > 5:
        print('Reject')
        return 'Reject'

    if len(regions) == 1:
        region = regions[0]
        if region.area >= threshold_dict['Area'] and \
                region.eccentricity >= threshold_dict['Ecc'] and \
                (region.major_axis_length / region.minor_axis_length) >= threshold_dict['Maj/min axis ratio']:
            print('Accept')
            return 'Accept'

    elif len(regions) >= 2:
        filtered_regions = []
        for region in regions:
            if region.eccentricity >= threshold_dict['Ecc'] and \
                    (region.major_axis_length / region.minor_axis_length) >= threshold_dict['Maj/min axis ratio']:
                filtered_regions.append(region)

        if len(filtered_regions) >= 2:
            for i, region1 in enumerate(filtered_regions):
                for region2 in filtered_regions[i + 1:]:
                    cent1 = np.array(region1.centroid)
                    cent2 = np.array(region2.centroid)
                    vdist = np.abs(cent1[0] - cent2[0])
                    hdist = np.abs(cent1[1] - cent2[1])
                    adif = abs(region2.area - region1.area)

                    if vdist > 50 and hdist <= 12 and \
                            region1.area > threshold_dict['Area multiple'] and \
                            region2.area > threshold_dict['Area multiple']:
                        print('Accept')
                        return 'Accept'
                    if hdist >= threshold_dict['Max hdist'] and adif <= threshold_dict['Area dif']:
                        print('Ghosts detected.')

            filtered_regions = [region for region in filtered_regions if region.area >= threshold_dict['Area multiple']]

        if filtered_regions:
            print('Accept')
            return 'Accept'

    print('Reject')
    return 'Reject'

if __name__ == '__main__':

    data_dir_pos = '/Users/hardie/research/sample-pos'
    imlist = [os.path.join(data_dir_pos, file) for file in os.listdir(data_dir_pos) if '.npz' in file]
    imgen = find2.image_batch_generator(imlist, batch_size=10)

    thresholds = {'Area': 400,
                  'Area dif': 150,
                  'Max hdist': 12,
                  'Area multiple': 350,
                  'Ecc': 0.96,
                  'Maj/min axis ratio': 3}

    for batchn, imdict in imgen:
        for key, img in imdict.items():
            detectable_regions, conv_img = detect_regions(img, tol=1, detectable_area=65)

            geometric_properties = calculate_geometric_properties(detectable_regions)
            str = filter_regions(detectable_regions, thresholds)
