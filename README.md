# kaggle_dstl_submission

Data structure:
```
data / theree_band / *
     / sixteen_band / *
    grid_sizes.csv
    train_wkt_v4.csv
```
    
    
# Prepare data for training:
1. Run `python get_3_band_shapes.py`
2. Run `cache_train.py`

# Train model
`python unet_track.py`

# Create preditcion
`python make_prediction_cropped_track.py`
