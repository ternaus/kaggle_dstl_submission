"""
Merge predictions
"""

import pandas as pd

submissions = {0: 'temp_building.csv',
               1: 'temp_structures_4.csv',
               2: 'temp_road_4.csv',
               3: 'temp_track_100_0.3_geom.csv',
               4: 'temp_trees_2.csv',
               5: 'temp_crops_4.csv',
               6: 'temp_water_fast.csv',
               7: 'temp_water_slow.csv'
               }

for class_index in submissions:
    submissions[class_index] = pd.read_csv(submissions[class_index])

sample_submission = pd.read_csv('../data/sample_submission.csv')

sample_submission['MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'

for class_type in submissions.keys():
    class_sub = submissions[class_type]

    index = sample_submission['ClassType'] == class_type + 1
    sample_submission.loc[index, 'MultipolygonWKT'] = class_sub.loc[index, 'MultipolygonWKT'].values

sample_submission.to_csv('joined.csv', index=False)
