from __future__ import division

from shapely.wkt import loads as wkt_loads

import os
import shapely
import shapely.geometry
import shapely.affinity
import h5py
import pandas as pd
import tifffile as tiff
from numba import jit, njit
from tqdm import tqdm
from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

import skimage.color as color
from skimage.transform import rescale

# dirty hacks from SO to allow loading of big cvs's
# without decrement loop it crashes with C error
# http://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

data_path = '../data'
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

epsilon = 1e-15


def get_scalers(height, width, x_max, y_min):
    """

    :param height:
    :param width:
    :param x_max:
    :param y_min:
    :return: (xscaler, yscaler)
    """
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min


def polygons2mask_layer(height, width, polygons, image_id):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = _get_xmax_ymin(image_id)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def polygons2mask(height, width, polygons, image_id):
    num_channels = len(polygons)
    result = np.zeros((num_channels, height, width))
    for mask_channel in range(num_channels):
        result[mask_channel, :, :] = polygons2mask_layer(height, width, polygons[mask_channel], image_id)
    return result


def generate_mask(image_id, height, width, num_mask_channels=10, train=train_wkt):
    """

    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels: numbers of channels in the desired mask
    :param train: polygons with labels in the polygon format
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """

    mask = np.zeros((num_mask_channels, height, width))

    for mask_channel in range(num_mask_channels):
        poly = train.loc[(train['ImageId'] == image_id)
                         & (train['ClassType'] == mask_channel + 1), 'MultipolygonWKT'].values[0]
        polygons = shapely.wkt.loads(poly)
        mask[mask_channel, :, :] = polygons2mask_layer(height, width, polygons, image_id)
    return mask


def mask2polygons_layer(mask, epsilon=1.0, min_area=10.0):
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    if epsilon != 0:
        approx_contours = simplify_contours(contours, epsilon)
    else:
        approx_contours = contours

    if not approx_contours:
        return MultiPolygon()

    all_polygons = find_child_parent(hierarchy, approx_contours, min_area)

    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def find_child_parent(hierarchy, approx_contours, min_area):
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            holes = [c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area]
            contour = cnt[:, 0, :]

            poly = Polygon(shell=contour, holes=holes)

            if poly.area >= min_area:
                all_polygons.append(poly)

    return all_polygons


def simplify_contours(contours, epsilon):
    return [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]


def fix_invalid_polygons(all_polygons):
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def _get_xmax_ymin(image_id):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin


# def class_dist():
#     """
#
#     :return: dataframe with columns: [image_id, class, num_pixels]
#     """
#     data_path = '../data'
#     f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')
#
#     y = np.array(f['train_mask'])
#     train_ids = np.array(f['train_ids'])
#     f.close()
#     temp = []
#
#     for i in tqdm(range(y.shape[0])):
#         for j in range(y.shape[1]):
#             temp += [(train_ids[i], j, y[i, j, ...].sum())]
#
#     return pd.DataFrame(temp, columns=['image_id', 'class', 'num_pixels'])
#
#
# def train_3_mean():
#     return np.array([426.729, 464.685, 329.116], dtype=np.float32)
#
#
# def train_11_range():
#     return np.array([  2.31600000e+03,   2.32000000e+03,   2.27200000e+03,
#          2.29400000e+03,   1.00000000e+00,   2.17200000e+03,
#          1.00000000e+00,   1.00000000e+00,   2.04700000e+03,
#          7.79785156e-01,   7.43652344e-01], dtype=np.float16)
#
#
# def train_11_mean():
#     return np.array([0.03653806, 0.04173296, 0.3003644, 0.06254322, 0.27435446,
#         0.0670274 ,  0.08590008,  0.32844734,  0.24594332, -0.1377002,
#         0.03925277], dtype=np.float32)
#
#
# def train_3_max():
#     return np.array([2047.0, 2047.0, 2047.0])
#
#
# def train_19_mean():
#     return np.array([4297.52294922,  4552.06835938,  4244.44726562,  3777.89379883,
#         2949.22705078,  2646.66113281,  2582.85253906,  2482.5715332 ,
#          296.50283813,   329.24719238,   464.77896118,   487.25497437,
#          426.97521973,   530.96075439,   689.1484375,   530.78143311,
#          426.72903442,   464.6829834,   329.11608887], dtype=np.float32)
#
#
# @jit(nopython=True)
# def train_20_range():
#     return np.array([16528., 16688., 18016., 17056., 16544., 16384., 16416.,
#            17024., 2302., 2234., 2282., 2300., 2300., 2258.,
#            2282., 2274., 2078., 2047., 2047., 2047.], dtype=np.float32)
#
#
# @jit(nopython=True)
# def train_20_mean():
#     return np.array([0.25999427,  0.27280301,  0.23560654,  0.22151501,  0.17829731,
#         0.16156755,  0.15736707,  0.14585659,  0.12880161,  0.14738023,
#         0.20367183,  0.21184967,  0.18564145,  0.23514642,  0.30198848,
#         0.23341164,  0.24221851,  0.20851426,  0.22705624,  0.16081564], dtype=np.float32)
#
#
# def train_12_range():
#     return np.array([2302., 2234., 2282., 2300., 2300., 2258., 2282., 2274., 2078., 2047., 2047., 2047.], dtype=np.float32)
#
#
# def train_12_mean():
#     return np.array([0.12880161,  0.14738023, 0.20367183,  0.21184967,  0.18564145,  0.23514642,
#                      0.30198848, 0.23341164,  0.24221851,  0.20851426,  0.22705624,  0.16081564], dtype=np.float32)
#
#
# def train_10_range():
#     return np.array([  2.31600000e+03,   2.32000000e+03,   2.27200000e+03,
#          2.29400000e+03,   1.00000000e+00,   2.17200000e+03,
#          1.00000000e+00,   2.04700000e+03,   2.04700000e+03,
#          7.68554688e-01], dtype=np.float16)
#
#
# def train_10_mean():
#     return np.array([ 0.12802237,  0.21002401,  0.23369755,  0.30041313,  0.51965165,
#         0.23173374,  0.43949682,  0.22705658,  0.16081622, -0.1410438 ], dtype=np.float32)
#
#
# def train_19_max():
#     return np.array([16532., 16694., 18023., 17067., 16548., 16388., 16423.,
#            17030., 2316., 2239., 2290., 2321., 2318., 2273.,
#            2294., 2293., 2047., 2047., 2047.], dtype=np.float32)
#
#
# def train_M_max():
#     return np.array([2047., 2080.,  2220., 2216., 2200., 2140., 2164., 2180.], dtype=np.float32)
#
#
# def train_M_mean():
#     return np.array([ 0.14488664,  0.15829249,  0.20936057,  0.21988007,  0.19407853, 0.24811149,  0.31845763,  0.24347587], dtype=np.float32)
#
#
# def train_3_mean_ones(min_train_height, min_train_width):
#     temp = []
#     num_channels = 3
#     for i in range(num_channels):
#         temp += [train_3_mean()[i] * np.ones((min_train_height, min_train_width))]
#     return np.array(temp)
#
#
# def read_image_3(image_id):
#     num_channels = 3
#     data_path = '../data'
#     image_3 = tiff.imread(os.path.join(data_path, 'three_band', image_id + '.tif')).astype(np.float32)
#
#     for i in range(num_channels):
#         image_3[i, ...] -= train_3_mean()[i]
#         image_3[i, ...] /= train_3_max()[i]
#
#     return image_3
#
#
# def pansharpen(m, pan, method='browley', W=0.1, all_data=False):
#     # get m_bands
#     rgbn = np.empty((m.shape[1], m.shape[2], 4))
#     rgbn[:, :, 0] = m[4, :, :]  # red
#     rgbn[:, :, 1] = m[2, :, :]  # green
#     rgbn[:, :, 2] = m[1, :, :]  # blue
#     rgbn[:, :, 3] = m[6, :, :]  # NIR-1
#
#     # scaled them
#     rgbn_scaled = np.empty((m.shape[1] * 4, m.shape[2] * 4, 4))
#
#     for i in range(4):
#         img = rgbn[:, :, i]
#         scaled = rescale(img, (4, 4))
#         rgbn_scaled[:, :, i] = scaled
#
#     # check size and crop for pan band
#     if pan.shape[0] < rgbn_scaled.shape[0]:
#         rgbn_scaled = rgbn_scaled[:pan.shape[0], :, :]
#     else:
#         pan = pan[:rgbn_scaled.shape[0], :]
#
#     if pan.shape[1] < rgbn_scaled.shape[1]:
#         rgbn_scaled = rgbn_scaled[:, :pan.shape[1], :]
#     else:
#         pan = pan[:, :rgbn_scaled.shape[1]]
#
#     R = rgbn_scaled[:, :, 0]
#     G = rgbn_scaled[:, :, 1]
#     B = rgbn_scaled[:, :, 2]
#     I = rgbn_scaled[:, :, 3]
#
#     image = None
#
#     if method == 'simple_browley':
#         all_in = R + G + B
#         prod = np.multiply(all_in, pan)
#
#         r = np.multiply(R, pan / all_in)[:, :, np.newaxis]
#         g = np.multiply(G, pan / all_in)[:, :, np.newaxis]
#         b = np.multiply(B, pan / all_in)[:, :, np.newaxis]
#
#         image = np.concatenate([r, g, b], axis=2)
#
#     if method == 'sample_mean':
#         r = 0.5 * (R + pan)[:, :, np.newaxis]
#         g = 0.5 * (G + pan)[:, :, np.newaxis]
#         b = 0.5 * (B + pan)[:, :, np.newaxis]
#
#         image = np.concatenate([r, g, b], axis=2)
#
#     if method == 'esri':
#         ADJ = pan - rgbn_scaled.mean(axis=2)
#         r = (R + ADJ)[:, :, np.newaxis]
#         g = (G + ADJ)[:, :, np.newaxis]
#         b = (B + ADJ)[:, :, np.newaxis]
#         i = (I + ADJ)[:, :, np.newaxis]
#
#         image = np.concatenate([r, g, b, i], axis=2)
#
#     if method == 'browley':
#         DNF = (pan - W * I) / (W * R + W * G + W * B)
#
#         r = (R * DNF)[:, :, np.newaxis]
#         g = (G * DNF)[:, :, np.newaxis]
#         b = (B * DNF)[:, :, np.newaxis]
#         i = (I * DNF)[:, :, np.newaxis]
#
#         image = np.concatenate([r, g, b, i], axis=2)
#
#     if method == 'hsv':
#         hsv = color.rgb2hsv(rgbn_scaled[:, :, :3])
#         hsv[:, :, 2] = pan - I * W
#         image = color.hsv2rgb(hsv)
#
#     if all_data:
#         return rgbn_scaled, image, I
#     else:
#         return image
#
#
# def read_image_prgb(image_id):
#     M = tiff.imread(os.path.join(data_path, 'sixteen_band', '{image_id}_M.tif'.format(image_id=image_id)))
#     P = tiff.imread(os.path.join(data_path, 'sixteen_band', '{image_id}_P.tif'.format(image_id=image_id)))
#     M = M.astype(np.float32)
#     M /= 2047.0
#     P = P.astype(np.float32)
#     P /= 2047.0
#     height, width = P.shape
#     rescaled_M = cv2.resize(np.transpose(M, (1, 2, 0)), (width, height), interpolation=cv2.INTER_CUBIC)
#
#     rgbn = np.empty((height, width, 3))
#     rgbn[:, :, 0] = rescaled_M[:, :, 7]  # red
#     rgbn[:, :, 1] = rescaled_M[:, :, 2]  # green
#     rgbn[:, :, 2] = rescaled_M[:, :, 1]  # blue
#
#     rgbn[rgbn > 1] = 1
#     rgbn[:, :, 0] -= 7.70855093
#     rgbn[:, :, 1] -= 7.80433559
#     rgbn[:, :, 2] -= 7.69763994
#
#     return (rgbn * 255.0).astype(np.float16)
#
#
# def read_image_19(image_id):
#     num_channels = 19
#     image_3 = tiff.imread(os.path.join(data_path, 'three_band', image_id + '.tif'))
#     image_A = tiff.imread(os.path.join(data_path, 'sixteen_band', image_id + '_A.tif'))
#     image_M = tiff.imread(os.path.join(data_path, 'sixteen_band', image_id + '_M.tif'))
#
#     height = image_3.shape[1]
#     width = image_3.shape[2]
#
#     rescaled_A = np.transpose(
#         cv2.resize(np.transpose(image_A, (1, 2, 0)), (width, height), interpolation=cv2.INTER_CUBIC), (2, 0, 1))
#     rescaled_M = np.transpose(
#         cv2.resize(np.transpose(image_M, (1, 2, 0)), (width, height), interpolation=cv2.INTER_CUBIC), (2, 0, 1))
#
#     result = np.vstack([rescaled_A, rescaled_M, image_3]).astype(np.float32)
#
#     for i in range(num_channels):
#         result[i, ...] -= train_19_mean()[i]
#         result[i, ...] /= train_19_max()[i]
#
#     return result.astype(np.float16)
#
#
# def _align_two_rasters(img1, img2, band='M'):
#     if band == 'P':
#         p1 = img1[300:1900, 300:2200, 1].astype(np.float32)
#         p2 = img2[300:1900, 300:2200].astype(np.float32)
#     elif band == 'M':
#         p1 = img1[300:1900, 300:2200, 1].astype(np.float32)
#         p2 = img2[300:1900, 300:2200, 2].astype(np.float32)
#     elif band == 'A':
#         p1 = img1[300:1900, 300:2200, 6].astype(np.float32) + img1[300:1900, 300:2200, 7].astype(np.float32) / 2
#         p2 = img2[300:1900, 300:2200, 0].astype(np.float32)
#
#     # warp_mode = cv2.MOTION_EUCLIDEAN
#     warp_mode = cv2.MOTION_TRANSLATION
#     # warp_mode = cv2.MOTION_HOMOGRAPHY
#     warp_matrix = np.eye(2, 3, dtype=np.float32)
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-7)
#     (cc, warp_matrix) = cv2.findTransformECC(p1, p2, warp_matrix, warp_mode, criteria)
#     print("_align_two_rasters: cc:{}".format(cc))
#
#     img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
#     img3[img3 == 0] = np.average(img3)
#
#     return img3
#
#
# @jit
# def read_image_20(image_id, cached=True):
#     if cached:
#         f = h5py.File(os.path.join(data_path, 'hdf5', image_id + '.hdf5'))
#         img = np.array(f['X'])
#         f.close()
#         return img
#
#     num_channels = 20
#
#     img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0))#.astype(np.float32)
#     img_a = np.transpose(tiff.imread("../data/sixteen_band/{}_A.tif".format(image_id)), (1, 2, 0))#.astype(np.float32)
#     img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0))#.astype(np.float32)
#     img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id))#.astype(np.float32)
#
#     height = img_3.shape[0]
#     width = img_3.shape[1]
#
#     rescaled_A = cv2.resize(img_a, (width, height), interpolation=cv2.INTER_CUBIC)
#     rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)
#     rescaled_P = np.expand_dims(cv2.resize(img_p, (width, height), interpolation=cv2.INTER_CUBIC), 2)
#
#     # img_P_new = np.expand_dims(_align_two_rasters(img_3, rescaled_P, 'P'), 2)
#     #
#     # img_M_new = _align_two_rasters(img_3, rescaled_M, 'M')
#     #
#     # img_A_new = _align_two_rasters(img_M_new, rescaled_A, 'A')
#
#     # result = np.transpose(np.concatenate([img_A_new, img_M_new, img_P_new, img_3], axis=2), (2, 0, 1))
#     result = np.transpose(np.concatenate([rescaled_A, rescaled_M, rescaled_P, img_3], axis=2), (2, 0, 1)).astype(np.float32)
#
#     for i in range(num_channels):
#         result[i, ...] /= train_20_range()[i]
#         result[i, ...] -= train_20_mean()[i]
#     return result.astype(np.float16)
#
#
# def get_shape(image_id, band=3):
#     if band == 3:
#         height = shapes.loc[shapes['image_id'] == image_id, 'height_3'].values[0]
#         width = shapes.loc[shapes['image_id'] == image_id, 'width_3'].values[0]
#         return height, width


def read_image_16(image_id):
    img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0)) / 2047.0
    img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0
    img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id)).astype(np.float32) / 2047.0

    height, width, _ = img_3.shape

    rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)
    rescaled_P = cv2.resize(img_p, (width, height), interpolation=cv2.INTER_CUBIC)

    rescaled_M[rescaled_M > 1] = 1
    rescaled_M[rescaled_M < 0] = 0

    rescaled_P[rescaled_P > 1] = 1
    rescaled_P[rescaled_P < 0] = 0

    image_r = img_3[:, :, 0]
    image_g = img_3[:, :, 1]
    image_b = img_3[:, :, 2]
    nir = rescaled_M[:, :, 7]
    re = rescaled_M[:, :, 5]

    L = 1.0
    C1 = 6.0
    C2 = 7.5
    evi = (nir - image_r) / (nir + C1 * image_r - C2 * image_b + L)
    evi = np.expand_dims(evi, 2)

    ndwi = (image_g - nir) / (image_g + nir)
    ndwi = np.expand_dims(ndwi, 2)

    savi = (nir - image_r) / (image_r + nir)
    savi = np.expand_dims(savi, 2)

    ccci = (nir - re) / (nir + re) * (nir - image_r) / (nir + image_r)
    ccci = np.expand_dims(ccci, 2)

    rescaled_P = np.expand_dims(rescaled_P, 2)

    result = np.transpose(np.concatenate([rescaled_M, rescaled_P, ndwi, savi, evi, ccci, img_3], axis=2), (2, 0, 1))
    return result.astype(np.float16)


# def read_image_16_small(image_id):
#     img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0)) / 2047.0
#     img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0
#     img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id)).astype(np.float32) / 2047.0
#
#     height, width, _ = img_3.shape
#
#     rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)
#     rescaled_P = cv2.resize(img_p, (width, height), interpolation=cv2.INTER_CUBIC)
#
#     rescaled_M[rescaled_M > 1] = 1
#     rescaled_M[rescaled_M < 0] = 0
#
#     rescaled_P[rescaled_P > 1] = 1
#     rescaled_P[rescaled_P < 0] = 0
#
#     image_r = img_3[:, :, 0]
#     image_g = img_3[:, :, 1]
#     image_b = img_3[:, :, 2]
#     nir = rescaled_M[:, :, 7]
#     re = rescaled_M[:, :, 5]
#
#     L = 1.0
#     C1 = 6.0
#     C2 = 7.5
#     evi = (nir - image_r) / (nir + C1 * image_r - C2 * image_b + L)
#     evi = np.expand_dims(evi, 2)
#
#     ndwi = (image_g - nir) / (image_g + nir)
#     ndwi = np.expand_dims(ndwi, 2)
#
#     savi = (nir - image_r) / (image_r + nir)
#     savi = np.expand_dims(savi, 2)
#
#     ccci = (nir - re) / (nir + re) * (nir - image_r) / (nir + image_r)
#     ccci = np.expand_dims(ccci, 2)
#
#     rescaled_P = np.expand_dims(rescaled_P, 2)
#
#     result = np.concatenate([rescaled_M, rescaled_P, ndwi, savi, evi, ccci, img_3], axis=2)
#
#     final_height = int(height / 4)
#     final_width = int(height / 4)
#
#     result = np.transpose(cv2.resize(result, (final_width, final_height), interpolation=cv2.INTER_CUBIC), (2, 0, 1))
#
#     # num_channels = 12
#     # for i in range(num_channels):
#     #     result[i, ...] -= train_11_mean()[i]
#     return result.astype(np.float16)
#
#
#
# def read_CCCI(image_id):
#     img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0))
#     img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id)).astype(np.float32) / 2047.0
#
#     height, width = img_p.shape
#
#     rescaled_M = cv2.resize(img_m.astype(np.float32) / 2047.0, (width, height), interpolation=cv2.INTER_LINEAR)
#
#     RE = rescaled_M[:, :, 5]
#     MIR = rescaled_M[:, :, 7]
#     R = rescaled_M[:, :, 0]
#
#     # canopy chloropyll content index
#     CCCI = (MIR - RE) / (MIR + RE) * (MIR - R) / (MIR + R)
#     return CCCI
#
#
# @jit
# def read_image_M(image_id):
#     img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0))
#     # img_a = np.transpose(tiff.imread("../data/sixteen_band/{}_A.tif".format(image_id)), (1, 2, 0))
#     img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0))
#     # img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id))#.astype(np.float32)
#
#     height_3 = img_3.shape[0]
#     width_3 = img_3.shape[1]
#
#     height_3_new = int(height_3 / 4)
#     width_3_new = int(width_3 / 4)
#
#     result = np.transpose(cv2.resize(img_m, (width_3_new, height_3_new), interpolation=cv2.INTER_CUBIC), (2, 0, 1)).astype(np.float32)
#
#     for i in range(result.shape[0]):
#         result[i, ...] /= train_M_max()[i]
#         result[i, ...] -= train_M_mean()[i]
#     return result.astype(np.float16)
#
#
# # @jit
# def make_prediction_1(model, X_train, size=(224, 224), num_channels=20, num_masks=10):
#
#     height = X_train.shape[1]
#     width = X_train.shape[2]
#
#     if height % size[0] == 0:
#         num_h_tiles = int(height / size[0])
#     else:
#         num_h_tiles = int(height / size[0]) + 1
#
#     if width % size[1] == 0:
#         num_w_tiles = int(width / size[1])
#     else:
#         num_w_tiles = int(width / size[1]) + 1
#
#     # print height, width, num_h_tiles, num_w_tiles
#
#     rounded_image = np.zeros((num_channels, num_h_tiles * size[0], num_w_tiles * size[1]))
#     predicted_mask = np.zeros((num_masks, num_h_tiles * size[0], num_w_tiles * size[1]))
#
#     #count_mask = np.zeros((num_h_tiles * size[0], num_w_tiles * size[1]))
#
#     rounded_image[:, :height, :width] = X_train
#
#     Xt = np.zeros((num_h_tiles * num_w_tiles, num_channels, size[0], size[1]))
#     # generate normal prediction
#     for w in range(num_w_tiles):
#         for h in range(num_h_tiles):
#             i = num_h_tiles * w + h
#             h_from = h * size[0]
#             h_to = (h + 1) * size[0]
#             w_from = w * size[1]
#             w_to = (w + 1) * size[1]
#             Xt[i] = rounded_image[:, h_from: h_to, w_from: w_to]
#
#     prediction = model.predict(Xt)
#
#     for w in range(num_w_tiles):
#         for h in range(num_h_tiles):
#             i = num_h_tiles * w + h
#             h_from = h * size[0]
#             h_to = (h + 1) * size[0]
#             w_from = w * size[1]
#             w_to = (w + 1) * size[1]
#
#             predicted_mask[:, h_from: h_to, w_from: w_to] = prediction[i]
#             # count_mask[h_from: h_to, w_from: w_to] += 1.0
#
#     # # generate w shifted prediction
#     # Xt = np.zeros((num_h_tiles * num_w_tiles, num_channels, size[0], size[1]))
#     #
#     # for w in range(num_w_tiles):
#     #     for h in range(num_h_tiles - 1):
#     #         i = (num_h_tiles - 1) * w + h
#     #         h_from = int(1/2 + h) * size[0]
#     #         h_to = (int(1 / 2 + h + 1) * size[0])
#     #         w_from = w * size[1]
#     #         w_to = (w + 1) * size[1]
#     #
#     #         Xt[i] = rounded_image[:, h_from: h_to, w_from: w_to]
#     #
#     # prediction = model.predict(Xt)
#     #
#     # for w in range(num_w_tiles):
#     #     for h in range(num_h_tiles - 1):
#     #         i = (num_h_tiles - 1) * w + h
#     #
#     #         h_from = int(1/2 + h) * size[0]
#     #         h_to = (int(1 / 2 + h + 1) * size[0])
#     #         w_from = w * size[1]
#     #         w_to = (w + 1) * size[1]
#     #
#     #         predicted_mask[:, h_from: h_to, w_from: w_to] += prediction[i]
#
#             # count_mask[h_from: h_to, w_from: w_to] += 1.0
#     # print count_mask[:height, :width].sum(), height * width
#
#     return predicted_mask[:, :height, :width] #/ count_mask[:height, :width]
#
#
@jit
def make_prediction_cropped(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=19, num_masks=10):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = X_train.shape[1]
    width = X_train.shape[2]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

    padded[:, :, width + shift:] = right

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i]

    return predicted_mask[:, :height, :width]
#
#
#
# @jit
# def calculate_score_mask(y_true, y_pred):
#     """
#     Function calculate jaccard score for real vs mask
#
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     num_mask_channels = y_true.shape[0]
#
#     result = np.ones(num_mask_channels)
#
#     for mask_channel in range(num_mask_channels):
#         intersection = np.dot(y_true[mask_channel, ...].flatten(), y_pred[mask_channel, ...].flatten())
#         _sum = y_true[mask_channel, ...].sum() + y_pred[mask_channel, ...].sum()
#         if _sum - intersection != 0:
#             result[mask_channel] = intersection / (_sum - intersection)
#     return result
#
#
# def calculate_score_from_model(model, image_id, threashold=0.5, num_mask_channels=10):
#     """
#     makes prediction with model and compares with real mask
#
#     :param model:
#     :param image_id:
#     :return:
#     """
#
#     predicted_mask = (make_prediction_1(model, image_id, num_masks=num_mask_channels) > threashold).astype(int)
#
#     height = predicted_mask.shape[1]
#     width = predicted_mask.shape[2]
#
#     mask = generate_mask(image_id, height, width)
#
#     return calculate_score_mask(mask, predicted_mask)
#
#
# def calculate_score_polygons(model, image_id, threashold=0.5, num_mask_channels=10):
#     """
#     makes prediction with model and compares with real mask
#
#     :param model:
#     :param image_id:
#     :return:
#     """
#
#     predicted_mask = (make_prediction_1(model, image_id, num_masks=num_mask_channels) > threashold).astype(int)
#     polygons = mask2polygons(predicted_mask, image_id)
#
#     height = predicted_mask.shape[1]
#     width = predicted_mask.shape[2]
#
#     predicted_mask = polygons2mask(height, width, polygons, image_id)
#
#     mask = generate_mask(image_id, height, width)
#
#     return calculate_score_mask(mask, predicted_mask)
#
#
# def mask2polygons(mask, image_id):
#     """
#     :param mask:
#     :return: list of the type: [Polygons_class_1, Polygons_class_2] in the final type scaling
#     """
#
#     num_mask_channels = mask.shape[0]
#     height = mask.shape[1]
#     width = mask.shape[2]
#
#     x_max = gs.loc[gs['ImageId'] == image_id, 'Xmax'].values[0]
#     y_min = gs.loc[gs['ImageId'] == image_id, 'Ymin'].values[0]
#
#     x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)
#
#     result = []
#
#     for mask_channel in range(num_mask_channels):
#         polygons = mask2polygons_layer(mask[mask_channel])
#
#         polygons = shapely.affinity.scale(polygons, xfact=1.0/x_scaler, yfact=1.0/y_scaler, origin=(0, 0, 0))
# #
# #         if not polygons.is_valid:
# #             polygons = polygons.buffer(0)
# #             # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
# #             # need to keep it a Multi throughout
# #             polygons = shapely.geometry.MultiPolygon([polygons])
# #
#         result += [polygons]
#
#     return result
# #
# #
# # def polygons2mask(polygons, height, width, image_id):
# #     xymax = _get_xmax_ymin(gs, image_id)
# #
# #     mask = np.zeros((len(polygons), height, width))
# #
# #     for i, p in enumerate(polygons):
# #         # polygon_list = _get_polygon_list(train_wkt, image_id, class_type)
# #         polygon_list = wkt_loads(p)
# #         if polygon_list.length == 0:
# #             continue
# #         contours = _get_and_convert_contours(polygon_list, (height, width), xymax)
# #         mask[i] = _plot_mask_from_contours((height, width), contours, 1)
# #     return mask
# #
# #
#
#
# def calculate_polygon_match(image_id, num_mask_channels=10):
#     """
#     calculates jaccard index between before poly and after poly
#
#     Ideally should be 1
#
#     :param model:
#     :param image_id:
#     :return:
#     """
#
#     image = read_image_3(image_id)
#
#     height = image.shape[1]
#     width = image.shape[2]
#
#     mask_before = generate_mask(image_id, height, width, num_mask_channels=num_mask_channels)
#
#     polygons = mask2polygons(mask_before, image_id) #here polygons is list [Multipolygon, Multipolygon, ...] in the scaled format
#
#     predicted_mask = polygons2mask(height, width, polygons, image_id)
#
#     return calculate_score_mask(mask_before, predicted_mask)
#
#
# if __name__ == '__main__':
#     temp = []
    # model = read_model('32_1000_2017-02-01-21-05')
    # for image_id in tqdm(train_wkt['ImageId'].unique()):
    #     temp += [calculate_score_polygons(model, image_id, threashold=0.5, num_mask_channels=10)]

    # for image_id in train_wkt['ImageId'].unique():
    #     print calculate_polygon_match(image_id)
