from __future__ import division


import extra_functions
import numpy as np
import shapely.wkt
import shapely.affinity
from numba import jit
import pandas as pd
import os
import sys
from shapely.geometry import MultiPolygon
from tqdm import tqdm

predictions = pd.read_csv(sys.argv[1])


real_test_ids = ['6080_4_4', '6080_4_1', '6010_0_1', '6150_3_4', '6020_0_4', '6020_4_3',
                 '6150_4_3', '6070_3_4', '6020_1_3', '6060_1_4', '6050_4_4', '6110_2_3',
                 '6060_4_1', '6100_2_4', '6050_3_3', '6100_0_2', '6060_0_0', '6060_0_1',
                 '6060_0_3', '6060_2_0', '6120_1_4', '6160_1_4', '6120_3_3', '6140_2_3',
                 '6090_3_2', '6090_3_4', '6170_4_4', '6120_4_4', '6030_1_4', '6120_0_2',
                 '6030_1_2', '6160_0_0']


def generate_mask(image_id, predictions, num_mask_channels=10):
    """

    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels:
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """
    height, width = extra_functions.get_shape(image_id)

    mask = np.zeros((num_mask_channels, height, width))

    for mask_channel in range(num_mask_channels):
        poly = predictions.loc[(predictions['ImageId'] == image_id) & (
        predictions['ClassType'] == mask_channel + 1), 'MultipolygonWKT'].values[0]
        polygons = shapely.wkt.loads(poly)
        if polygons.type == 'Polygon':
           polygons = MultiPolygon([polygons])
        mask[mask_channel, :, :] = extra_functions.polygons2mask_layer(height, width, polygons, image_id)
    return mask


@jit
def fix_pred(image_id, predictions):
    # read image and transform 2 multilayer mask
    mask = generate_mask(image_id, predictions)
    # if image_id in victim list => replace all slow water by fast
    building_index = 0
    road_index = 2
    tree_index = 4
    crop_index = 5
    fast_water_index = 6
    slow_water_index = 7

    # Let's remove everything from the fast water
    fast_water = (mask[fast_water_index] == 1)
    for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
        mask[i][fast_water] = 0

    # For some strange reason there is a lot of predicted roads near the water edge
    slow_water = (mask[slow_water_index] == 1)
    mask[road_index][slow_water] = 0

    # Let's remove everything from buildings (We trust building predictions)
    buildings = (mask[building_index] == 1)

    for i in range(1, 10):
        mask[i][buildings] = 0

    # Let's remove slow water from roads
    roads = (mask[road_index] == 1)
    mask[slow_water_index][roads] = 0

    #
    # # those that are all crops - make all crops:
    # if image_id in all_crops:
    #     mask[crop_index] = 1
    #
    # # those that are all trees - make all trees:
    # if image_id in all_trees:
    #     mask[tree_index] = 1

    # remove everything from fast_water
    # fast_water = (mask[fast_water_index] == 1)
    # for index in [0, 1, 2, 3, 4, 5, 8, 9]:
    #     mask[index][fast_water] = 0

    # Remove all slow water from buildings
    # mask[slow_water_index][buildings] = 0

    # # zero out crops that Sergey said are zero
    # if image_id in zero_crops:
    #     mask[crop_index] = 0
    #
    # # zero out roads from visual inspections
    # if image_id in zero_road:
    #     mask[road_index] = 0
    #
    # # take union of mapped and not mapped predictions for all classes except cars
    # if image_id in mapped_prediction['ImageId'].unique():
    #     mapped_mask = generate_mask(image_id, mapped_prediction)
    #     for i in range(8):
    #         if mapped_mask[i].sum() == 0:
    #             mask[i] = 0
    #         else:
    #             mask[i] += mapped_mask[i]

    return (mask > 0).astype(int)

@jit
def mask2poly(predicted_mask, x_scaler, y_scaler):
    if mask_channel == 7:
        min_area = 100
    elif mask_channel == 6:
        min_area = 5000
    else:
        min_area = 10

    polygons = extra_functions.mask2polygons_layer(predicted_mask, epsilon=0, min_area=min_area)

    if image_id == '6100_0_2' and mask_channel == 1:
        polygons = polygons.buffer(0.5)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))

    return shapely.wkt.dumps(polygons)


sample = pd.read_csv('../data/sample_submission.csv')

data_path = '../data'
num_mask_channels = 10

three_band_path = os.path.join(data_path, 'three_band')

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']


result = []

for image_id in tqdm(real_test_ids):

    height, width = extra_functions.get_shape(image_id)

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    predicted_mask = fix_pred(image_id, predictions)

    x_scaler, y_scaler = extra_functions.get_scalers(height, width, x_max, y_min)

    for mask_channel in range(num_mask_channels):
        result += [(image_id, mask_channel + 1, mask2poly(predicted_mask[mask_channel], x_scaler, y_scaler))]

submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])


sample = sample.drop('MultipolygonWKT', 1)
submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')

submission.to_csv('cleaned_' + sys.argv[1], index=False)
