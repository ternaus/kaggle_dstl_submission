from __future__ import division
import numpy as np
import tifffile as tiff

from skimage.transform import resize
import pandas as pd

import shapely
import extra_functions
import os
from tqdm import tqdm


def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(3):
        a = 0
        b = 1
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] =t
    return out.astype(np.float32)


sample = pd.read_csv('../data/sample_submission.csv')
data = pd.read_csv('../data/train_wkt_v4.csv')
data = data[data.MultipolygonWKT != 'MULTIPOLYGON EMPTY']
grid_sizes_fname = '../data/grid_sizes.csv'
wkt_fname = '../data/train_wkt_v4.csv'
image_fname = '../data/three_band/'
data_path = '../data'
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']


result = []


def CCCI_index(m, rgb):
    RE = resize(m[5, :, :], (rgb.shape[0], rgb.shape[1]))
    MIR = resize(m[7, :, :], (rgb.shape[0], rgb.shape[1]))
    R = rgb[:, :, 0]

    # canopy chloropyll content index
    CCCI = (MIR - RE) / (MIR + RE) * (MIR - R) / (MIR + R)
    return CCCI


def mask2poly(predicted_mask, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask, epsilon=0, min_area=10000)
    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons)


result = []
for image_id in tqdm(test_ids):
    rgb = tiff.imread('../data/three_band/{}.tif'.format(image_id))
    _, height, width = rgb.shape
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('../data/sixteen_band/{}_M.tif'.format(image_id))
    # get our index
    CCCI = CCCI_index(m, rgb)

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)
    x_scaler, y_scaler = extra_functions.get_scalers(height, width, x_max, y_min)

    # you can look on histogram and pick your favorite threshold value(0.11 is my best)
    predicted_mask = (CCCI > 0.11).astype(np.float32)

    if predicted_mask.sum() <= 500000:
        result += [(image_id, 7, 'MULTIPOLYGON EMPTY')]
    else:
        result += [(image_id, 7, mask2poly(predicted_mask, x_scaler, y_scaler))]

submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])


sample = sample.drop('MultipolygonWKT', 1)
submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')

submission.to_csv('temp_water_fast.csv', index=False)
