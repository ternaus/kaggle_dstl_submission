"""
Script that caches train data for future training
"""

from __future__ import division

import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np


data_path = '../data'

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))


def cache_train_16():
    print('num_train_images =', train_wkt['ImageId'].nunique())

    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]

    min_train_height = train_shapes['height'].min()
    min_train_width = train_shapes['width'].min()

    num_train = train_shapes.shape[0]

    image_rows = min_train_height
    image_cols = min_train_width

    num_channels = 16

    num_mask_channels = 10

    f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0
    for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
        image = extra_functions.read_image_16(image_id)
        _, height, width = image.shape

        imgs[i] = image[:, :min_train_height, :min_train_width]
        imgs_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train_16()
