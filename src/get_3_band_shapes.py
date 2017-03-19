"""
Script that scans 3 band tiff files and creates csv file with columns:
image_id, width, height
"""
from __future__ import division

import tifffile as tiff
import os
from tqdm import tqdm
import pandas as pd

data_path = '../data'

three_band_path = os.path.join(data_path, 'three_band')

file_names = []
widths_3 = []
heights_3 = []


for file_name in tqdm(sorted(os.listdir(three_band_path))):
    # TODO: crashes if there anything except tiff files in folder (for ex, QGIS creates a lot of aux files)
    image_id = file_name.split('.')[0]
    image_3 = tiff.imread(os.path.join(three_band_path, file_name))

    file_names += [file_name]
    _, height_3, width_3 = image_3.shape

    widths_3 += [width_3]
    heights_3 += [height_3]

df = pd.DataFrame({'file_name': file_names, 'width': widths_3, 'height': heights_3})

df['image_id'] = df['file_name'].apply(lambda x: x.split('.')[0])

df.to_csv(os.path.join(data_path, '3_shapes.csv'), index=False)
