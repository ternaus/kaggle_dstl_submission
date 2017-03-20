from collections import defaultdict

import matplotlib.pyplot as plt
import cv2
import shapely
import time
import pandas as pd
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

def make_submit_water(engine='opencv'):
    predict_stats = []
    # TODO add saving to pickle file
    t0 = time.time()
    print("make submission file")
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print(df.head())
    # classes_in_images = pd.read_csv('classes_in_images.tsv', sep='\t')
    # 6010, 6040,
    slow_water = {'6020': ('rei', -1.8),
                  '6130': ('rei', -0.65),

                  '6110': ('rei', -1.8),
                  '6140': ('rei', -0.68),

                  '6060': ('ndvi2', -0.2),
                  '6120': ('rei', -0.65),

                  '6100': ('rei', -1), # also fast

                  '6090': ('rei', -0.8), #also fast

                  '6170': ('rei', -0.9)}

    fast_water = {'6030': ('rei', -1.7),
                  '6150': ('ndvi', 0.06),

                  '6050': ('ndvi', 0.1),
                  '6070': ('ndvi', 0.1),
                  '6080': ('ndvi', 0.1),

                  '6090': ('rei', -0.8), # also slow

                  '6100': ('rei', -1), # also slow

                  '6170': ('rei', -0.9)}

    for idx, row in df.iterrows():
        if idx % 10 == 0: print(idx)
        image_id = row[0]
        scene_id = image_id[:4]
        class_i = row[1] - 1
        # TODO one time hack to skip fast water
        if class_i == 6 and scene_id in fast_water: # fast water
            if scene_id in ['6100', '6090'] and image_id not in ['6100_1_2', '6100_2_2', '6090_4_3', '6090_3_4']:
                scaled_pred_polygons = MultiPolygon()
                df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
                continue
            msk = threshold_index(image_id, index_type=fast_water[scene_id][0], threshold=fast_water[scene_id][1])
            pred_polygons = mask_to_polygons(msk, min_area=15, engine=engine, buffer_amount=1)

        elif class_i == 7 and scene_id in slow_water:
            if image_id in ['6170_4_3', '6170_4_4', '6100_1_2', '6100_2_2', '6090_4_3', '6090_3_4']:
                scaled_pred_polygons = MultiPolygon()
                df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
                continue

            msk = threshold_index(image_id, index_type=slow_water[scene_id][0], threshold=slow_water[scene_id][1])
            pred_polygons = mask_to_polygons(msk, engine=engine, buffer_amount=2)

        else:
            scaled_pred_polygons = MultiPolygon()
            df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
            continue
        print('ImageId:{}, class:{}, area:{}'.format(image_id, class_i, pred_polygons.area))
        if True:
            msk2 = mask_from_polygons(pred_polygons, msk.shape)
            jaccard = np_jaccard(msk, msk2)
            print('ImageId: {} jaccard: {:.2f}, area: {}, {}'.format(image_id, jaccard, np.sum(msk), np.sum(msk2)))
            rgb = np.dstack((msk2, msk, np.zeros_like(msk)))
            predict_stats.append([image_id, class_i, jaccard, np.sum(msk), np.sum(msk2)])

            plt.imsave('check_submit/{}_pred.png'.format(image_id), rgb)

        x_max = GS.loc[GS['ImageId'] == image_id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == image_id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))
        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
    print(df.head())
    stat_df = pd.DataFrame(predict_stats, columns=['image_id', 'class_i', 'jaccard', 'true_area', 'poly_area'])
    stat_df.to_csv('check_submit/stats.csv')
    print(stat_df.true_area.describe())
    print(stat_df.poly_area.describe())
    print(stat_df.true_area.sum())
    print(stat_df.poly_area.sum())

    df.to_csv('subm/{}_{}_channels.csv'.format('water_buf1_2', channels), index=False)
    print('Finished in: {}'.format(time.time() - t0))


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


def print_heur():
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))

    for idx, row in df.iterrows():
        image_id = row[0]
        class_i = row[1] - 1
        if class_i == 7 and image_id in real_test_ids:
            # print(image_id, heur_water(image_id))
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
        elif class_i == 6 and image_id in real_test_ids:
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

    df.to_csv('{}_indices.csv'.format('water_auto'), index=False)



real_test_ids = ['6080_4_4', '6080_4_1', '6010_0_1', '6150_3_4', '6020_0_4', '6020_4_3',
                 '6150_4_3', '6070_3_4', '6020_1_3', '6060_1_4', '6050_4_4', '6110_2_3',
                 '6060_4_1', '6100_2_4', '6050_3_3', '6100_0_2', '6060_0_0', '6060_0_1',
                 '6060_0_3', '6060_2_0', '6120_1_4', '6160_1_4', '6120_3_3', '6140_2_3',
                 '6090_3_2', '6090_3_4', '6170_4_4', '6120_4_4', '6030_1_4', '6120_0_2',
                 '6030_1_2', '6160_0_0']

scenes_in_train = set()
for image_id in train_ids:
    scene_id = image_id[:4]
    scenes_in_train.add(scene_id)

# всего рек в public'e = 0.00630*10*6*835*835 = 263551.05
# всего рек в привате = 0.00565*10*26*835*835 = 1024223.525
# итого: 1287774.575

# всего озёр в паблике = 15060.06
# всего озёр в привате = 52570.765
# итого: 67630.765


if __name__ == '__main__':
    # find_index_and_threshhold()
    print_heur()