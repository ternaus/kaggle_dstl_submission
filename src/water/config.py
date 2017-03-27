import pandas as pd
import numpy as np
# TODO: cleanup this file, as it is almost unused in current solution (only for water and only small number
# of variables actually used)

N_Cls = 10
inDir = '../../data'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv('sample_submission.csv')
smooth = 1e-12

scene_ids = ['6010', '6020', '6030', '6040', '6050', '6060',
             '6070', '6080', '6090', '6100', '6110', '6120',
             '6130', '6140', '6150', '6160', '6170', '6180']

train_ids = sorted(DF.ImageId.unique())

# Give short names, sensible colors and zorders to object types
CLASSES = {
        0 : 'Bldg',
        1 : 'Struct',
        2 : 'Road',
        3 : 'Track',
        4 : 'Trees',
        5 : 'Crops',
        6 : 'Fast H20',
        7 : 'Slow H20',
        8 : 'Truck',
        9 : 'Car',
        10 : 'Background',
        }
COLORS = {
        1 : '#FFFDFD',
        2 : '#FFFDFD',
        3 : '#FFFDFD',
        4 : '0.00',
        5 : '#FFFDFD',
        6 : '#FFFDFD',
        7 : '#FFFDFD',
        8 : '#FFFDFD',
        9 : '#FFFDFD',
        10: '#FFFDFD',
        }
ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }

# channels = 'all' 'three' other m p
channels = 'm'
channels_count = 8
weighting = 'random'

ISZ = 80
ISZ_mult = 3360 // ISZ # about 3350 = ISZ*ISZ_mult
batch_size = 128  # for bigger patch size may be even lower
learning_rate = 1e-4
total_runs = 100
CURRENT_CLASS = 3
val_patches = 2048
train_patches = 1024
aug = True
aug_a_lot = False
num_epochs = 2
load_weights = True
use_jaccard_loss = False
class_weights = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
loss_weights = np.array([0.1, 0.1, 0.3, 9, 0.1, 0.1, 0.1, 0.1, 0.001, 0.001])

model_type = 'unet'
# model_type = 'fc-densenet'
# model_type = 'resnet-38'
# model_type = 'get_another_net'
# model_type = 'two_head'








weights_path = 'weights/{}_{}_{}_{}_{}_last'.format(model_type, CLASSES[CURRENT_CLASS], channels, channels_count, weighting)
# weights_path = 'weights/unet_10_last9_0.32'
trs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
