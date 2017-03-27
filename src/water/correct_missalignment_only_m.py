# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import cv2
import tifffile as tiff
from joblib import Parallel, delayed

global warp_matrices


def _align_two_rasters(channel_1, channel_2, warp_mode):

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200,  1e-7)
    (cc, warp_matrix) = cv2.findTransformECC(channel_1, channel_2,warp_matrix, warp_mode, criteria)
    return warp_matrix, cc


def stretch_n(bands, channels, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands).astype(np.float32)
    # n = bands.shape[2]
    for i in range(channels):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        if channels > 1:
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
        else:
            c = np.percentile(bands[:, :], lower_percent)
            d = np.percentile(bands[:, :], higher_percent)
            t = a + (bands[:, :] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :] = t

    return out.astype(np.float32)


test_ids = ('6120_2_4',
        '6120_2_3',
        '6120_2_2',
        '6120_2_1',
        '6120_2_0',
        '6120_1_3',
        '6120_1_2',
        '6120_1_1',
        '6120_1_0',
        '6080_2_0',
        '6080_4_0',
        '6080_0_4',
        '6070_3_2',
        '6050_4_1',
        '6080_1_1',
        '6070_4_3',
        '6080_0_0',
        '6080_4_4',
        '6050_3_2',
        '6050_3_3',
        '6070_3_4',
        '6070_1_4',
        '6070_1_0',
        '6090_3_1',
        '6050_4_2')


def process_images(image_id):
    print('Processing ImageId: {}'.format(image_id))
    t0 = time.time()
    img_3_original = np.transpose(tiff.imread("three_band/{}.tif".format(image_id)), (1, 2, 0)).astype(np.float32)
    img_m_original = np.transpose(tiff.imread("sixteen_band/{}_{}.tif".format(image_id, 'M')), (1, 2, 0)).astype(np.float32)
    raster_size = img_m_original.shape
    img_3_original = cv2.resize(img_3_original, (raster_size[1], raster_size[0]), interpolation=cv2.INTER_CUBIC)

    channel_g = img_3_original[:, :, 1]
    channel_m = img_m_original[:, :, 2]

    t1 = time.time()
    print('Loading and resizing time: {}'.format(t1-t0))

    cc_3_m = 0
    warp_mode = cv2.MOTION_TRANSLATION
    # TODO: simplify this and make separate function to avoid repetition

    warp_matrix_3_m = np.eye(2, 3, dtype=np.float32)
    try:
        warp_matrix_3_m, cc_3_m = _align_two_rasters(channel_g, channel_m, warp_mode)
        # warp channel_m_to_a before going to the next step
        # img_m_aligned = cv2.warpAffine(img_m_original, warp_matrix_3_m, (raster_size[1], raster_size[0]),
        #                                flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        #                                borderMode=cv2.BORDER_TRANSPARENT)
    except:
        print('ImageId={} 3 to M align failed'.format(image_id))

    print('ImageId={}, cc_3_m={:.2f}'.format(image_id, cc_3_m))
    t2 = time.time()
    print('Aligning time: {}'.format(t2-t1))

    t3 = time.time()
    print('Full 20 image dumping time: {}'.format(t3-t2))

    t4 = time.time()
    print('Test image saving time: {}'.format(t4-t3))
    print('Total time: {}'.format(t4-t0))
    return image_id, warp_matrix_3_m, cc_3_m, t4 - t0

if __name__ == '__main__':
    GS = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    # test case
    # (results) = Parallel(n_jobs=16)(delayed(process_images)(image_id) for image_id in test_ids) # GS['ImageId'])
    (results) = Parallel(n_jobs=16)(delayed(process_images)(image_id) for image_id in GS['ImageId'])
    # results = []
    # for image_id in test_ids[9:10]:
    #     result = process_images(image_id)
    #     results.append(result)
    #     print(result)
    df = pd.DataFrame(results, columns=['image_id', 'warp_matrix_3_m',  'cc_3_m', 'time'])

    df.to_pickle('data/warp_matrices_translate_3_m_{}.pkl'.format(time.time()))
    print('Success!')
