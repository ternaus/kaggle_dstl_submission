from collections import defaultdict

import matplotlib.pyplot as plt
import cv2
import shapely
import time
import pandas as pd
import numpy as np
import os
from shapely.geometry import MultiPolygon
import shapely.affinity
import shapely.wkt
import tifffile as tiff

from mask_and_raster_functions import mask_from_polygons, mask_to_polygons, get_scalers, \
    generate_mask_for_image_and_class

from config import *

indices = ['ndvi', 'ndvi2', 'rei']

scores = []


def find_index_and_threshhold():
    best_scores = {}
    for scene_id in scenes_in_train:
        most_pixels = 0
        best_index_type = 'rei'
        best_thr = -100
        for image_id in train_ids:
            if image_id[:4] == scene_id:
                # check for big water
                pixel_num, score_max, index_type, thr = check_water(image_id, scores, class_i=7)
                if pixel_num > most_pixels:
                    most_pixels = pixel_num
                    best_index_type = index_type
                    best_thr = thr
                pixel_num, score_max, index_type, thr = check_water(image_id, scores, class_i=8)
                if pixel_num > most_pixels:
                    most_pixels = pixel_num
                    best_index_type = index_type
                    best_thr = thr
        if best_thr > -100:
            best_scores[scene_id] = (best_index_type, best_thr)
    print(best_scores)
    return best_scores

def check_water(image_id, scores, class_i):
    img_m_aligned = load_m(image_id)
    size_y = img_m_aligned.shape[0]
    size_x = img_m_aligned.shape[1]
    mask = generate_mask_for_image_and_class((size_y, size_x), image_id, class_i)
    pixel_num = np.sum(mask)
    if pixel_num > 0:
        score_max = 0
        best_thr = 0
        best_index_type = 'rei'
        for index_type in indices:
            index = calc_index(img_m_aligned, index_type=index_type)
            for threshold in range(-200, 200, 1):
                threshold_float = threshold / 100
                msk = np.where((index < threshold_float), 1, 0)
                score = np_jaccard(mask, msk)
                if score > score_max:
                    score_max = score
                    best_thr = threshold_float
                    best_index_type = index_type
                scores.append((image_id, 'fast_water' if class_i == 7 else 'slow_water', index_type, threshold_float, 'score', score, pixel_num))
                # print('Image:{}, index:{}, threshold:{}, score:{:.2f}, pixel_num:{}'.format(image_id, index_type, threshold_float,
                #                                                               score, pixel_num), ' fast_water' if class_i == 7 else 'slow_water')
    else:
        score_max = -1
        best_thr = -100
        best_index_type = 'rei'

        scores.append((image_id, 'fast_water' if class_i == 7 else 'slow_water', 'no_index', -2, 'score', -1, 0))
        # print('ImageId {} doesn''t have enough water pixels of fast type. Water pixels {}'.format(image_id, pixel_num))

    return pixel_num,score_max, best_index_type, best_thr


def np_jaccard(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def threshold_index(image_id, index_type, threshold, less=True):
    img_m_aligned = load_m(image_id)
    index = calc_index(img_m_aligned, index_type=index_type)
    if less:
        msk = np.where((index < threshold), 1, 0)
    else:
        msk = np.where((index > threshold), 1, 0)
    # pad array with single line of zeroes
    y, x = msk.shape
    return msk


def load_m(image_id):
    #TODO: warp_matrices_translate_3_m.pkl provided as part of the solution, to reproduce use correct_misalignment_only_m.py
    warp_matrices = pd.read_pickle('warp_matrices_translate_3_m.pkl')
    img_m = np.transpose(tiff.imread("sixteen_band/{}_{}.tif".format(image_id, 'M')), (1, 2, 0))
    raster_size = img_m.shape
    border_mode = cv2.BORDER_REFLECT
    warp_matrix_3_m = warp_matrices[warp_matrices.image_id == image_id].warp_matrix_3_m.values[0]
    img_m_aligned = cv2.warpAffine(img_m, warp_matrix_3_m, (raster_size[1], raster_size[0]),
                                   flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                   borderMode=border_mode)
    return img_m_aligned


def calc_index(raster, index_type='ndvi'):
    if index_type == 'ndvi':
        nir = raster[..., 6] / 2047
        red = raster[..., 4] / 2047
        index = (nir - red) / (nir + red)
    elif index_type == 'ndvi2':
        nir = raster[..., 7] / 2047
        red = raster[..., 4] / 2047
        index = (nir - red) / (nir + red)
    elif index_type == 'evi':
        l = 1
        c1 = 6
        c2 = 7.5
        g = 2.5
        nir = raster[..., 6] / 2047
        red = raster[..., 4] / 2047
        blue = raster[..., 1] / 2047
        index = np.zeros_like(nir, dtype=np.float32)
        index = g * (nir - red) / (nir + c1*red -c2*blue + l)
    elif index_type == 'rei':
        # REI =  NIR2 -  B/NIR2 + B*NIR2
        nir2 = raster[..., 7] / 2047
        blue = raster[..., 1] / 2047
        index = nir2 - blue/nir2 + blue*nir2
    else:
        print('Please provide index_type')
        index = raster[..., 4] / 2047
    return index

def stich_5_by_5_native_size(scene_id):
    size_x = np.zeros((5, 5), dtype=np.uint)
    size_y = np.zeros((5, 5), dtype=np.uint)
    tmp_row = []
    for j in range(0,5):
        for i in range(0, 5):
            image_id = '{}_{}_{}'.format(scene_id, j,i)
            tmp_3 = np.transpose(tiff.imread("sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0))
            size_x[j, i] = tmp_3.shape[1]
            size_y[j, i] = tmp_3.shape[0]
            tmp_row.append(tmp_3)

    return tmp_row, size_x[0], size_y[:, 0]

def big_pic(scene_id):
    tmp_row, size_x, size_y = stich_5_by_5_native_size(scene_id)
    big_pic = np.zeros((np.sum(size_y), np.sum(size_x), 8), dtype=np.uint16)
    for i in range(5):
        y_start = np.sum(size_y[:i])
        y_end = np.sum(size_y[:(i+1)])
#         print(y_start, y_end)
        for j in range(5):
            x_start = np.sum(size_x[:j])
            x_end = np.sum(size_x[:(j+1)])
#             print(x_start, x_end)
#             print('big_pic[y_start:y_end, x_start:x_end] shape:', big_pic[y_start:y_end, x_start:x_end].shape)
            big_pic[y_start:y_end, x_start:x_end] = tmp_row[i*5 + j]
    return big_pic


def save_5_5_index(scene_id, index_type='ndvi'):
    image = big_pic(scene_id)
    index = calc_index(image, index_type)
    plt.figure(figsize=(20, 20))
    plt.imshow(index, cmap='RdYlGn')
    plt.savefig('scenes/{}_{}.png'.format(scene_id, index_type))
    plt.clf()

def calc_and_save_hists(scene_id, index_type='ndvi', print_min_max = False):
    image = big_pic(scene_id)
    index = calc_index(image, index_type)
    if print_min_max:
        print(scene_id,
              'index:{}, min:{}, max:{}, percentile:{}, mean:{}'.format(index_type,
                                                         np.min(index),
                                                         np.max(index),
                                                         np.percentile(index, 1),
                                                         np.mean(index)))
        return
    plt.figure(figsize=(20, 20))
    # plt.imshow(index, cmap='RdYlGn')
    plt.hist(index.flatten(), bins=255, range=(-2, 2) )
    plt.savefig('scenes/hist{}_{}.png'.format(scene_id, index_type))
    plt.close()


def calc_hists_all_scenes():
    for scene_id in scene_ids:
        for index_type in indices:
            calc_and_save_hists(scene_id, index_type, print_min_max=True)




def calc_thresholds_from_train():
    global scores
    find_index_and_threshhold()
    scores = pd.DataFrame(scores, columns=('image_id', 'water_type', 'index_type', 'threshold', 'score_name', 'score', 'pixels'))
    scores.to_csv('water_thresholds.csv')


def heur_water(image_id):

    # thresholds calculated from train, using jaccard between thresholded index and mask
    # train_thresholds = find_index_and_threshhold()
    # hardcoded results from above to save computation time
    train_thresholds = {
        '6060': ('ndvi2', -0.2),
        '6070': ('ndvi2', -0.02),
        '6090': ('rei', -0.73),
        '6100': ('ndvi2', -0.05),
        '6110': ('rei', -1.7),
        '6120': ('rei', -0.61),
        '6140': ('rei', -0.67),
        '6150': ('rei', -0.6),
        '6170': ('ndvi2', -0.06)
    }


    # list of scenes without water in train images, assume no water on them
    no_water_in_train = ['6010', '6040', '6160']

    # three very similar scenes, where green pixels > 1e7 and water pixels > 1e6
    river_scenes = ['6050', '6070', '6080']

    scene_id = image_id[:4]
    if scene_id in scenes_in_train or scene_id in river_scenes:
        if scene_id in no_water_in_train:
            # quick and dirty to select no pixels
            index_type = 'rei'
            threshold = -100
        elif scene_id in river_scenes:
            index_type = train_thresholds['6070'][0]
            threshold = train_thresholds['6070'][1]
        else:
            index_type = train_thresholds[scene_id][0]
            threshold = train_thresholds[scene_id][1]
    else:
        img_m_aligned = big_pic(scene_id)

        size_y = img_m_aligned.shape[0]
        size_x = img_m_aligned.shape[1]
        # calculate rei index which works best for general image here
        index_type = 'rei'
        index = calc_index(img_m_aligned, index_type)
        # there are two different types of images: with distribution skewed left and right
        # and from train we see that the best threshold for them is about -1.7 and (-0.7, -0.6)
        if np.min(index) < -1.7:
            threshold = -1.7
        else:
            threshold = -0.6 # lean to higher threashold because FP pixels are cheaper then FN in metric

    return index_type, threshold


def make_water_submit_heur():
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))

    for idx, row in df.iterrows():
        image_id = row[0]
        class_i = row[1] - 1
        if class_i == 7:
            index_type, threshold = heur_water(image_id)
            if threshold == -100:
                df.iloc[idx, 2] = 'MULTIPOLYGON EMPTY'
                continue
            mask = threshold_index(image_id, index_type, threshold)
            if np.sum(mask) > 10 and np.sum(mask) < 7000 or image_id[:4] == '6100':
                pred_polygons = mask_to_polygons(mask, epsilon=0, buffer_amount=3)
                print(image_id, np.sum(mask), index_type, threshold)
            else:
                df.iloc[idx, 2] = 'MULTIPOLYGON EMPTY'
                continue
        elif class_i == 6:
            index_type, threshold = heur_water(image_id)
            if threshold == -100:
                df.iloc[idx, 2] = 'MULTIPOLYGON EMPTY'
                continue
            mask = threshold_index(image_id, index_type, threshold)
            if np.sum(mask) > 7000 and image_id[:4] != '6100':
                # for some reason '6150_4_3' don't want to become nice polygon without epsilon
                if image_id == '6150_4_3':
                    pred_polygons = mask_to_polygons(mask, epsilon=3, min_area=15)
                else:
                    pred_polygons = mask_to_polygons(mask, epsilon=0, min_area=15)

                print(image_id, np.sum(mask), index_type, threshold)
            else:
                df.iloc[idx, 2] = 'MULTIPOLYGON EMPTY'
                continue
        else:
            df.iloc[idx, 2] = 'MULTIPOLYGON EMPTY'
            continue

        x_max = GS.loc[GS['ImageId'] == image_id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == image_id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(mask.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))
        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)

    df.to_csv('../{}_indices.csv'.format('water_auto'), index=False)

scenes_in_train = set()
for image_id in train_ids:
    scene_id = image_id[:4]
    scenes_in_train.add(scene_id)

if __name__ == '__main__':
    make_water_submit_heur()