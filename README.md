# Winning Model Documentation
Name: Vladimir Iglovikov

Location: San-Francisco, United States

Email: iglovikov@gmail.com

Name: Sergey Mushinskiy

Location: Angarsk, Russia

Email: cepera.ang@gmail.com	

Competition: Dstl Satellite Imagery Feature Detection


# Prerequisites
To train final models you will need the following:

- OS: Ubuntu 16.04 (although code was successfully ran on Windows 10 too)
- Required hardware: 
    - Any decent modern computer with x86-64 CPU, 
    - Fair amount of RAM (we had about 32Gb and 128Gb in our boxes, however, not all memory was used) 
    - Powerful GPU: we used Nvidia Titan X with 12Gb of RAM and Nvidia GeForce 
GTX 1080 with 8Gb of RAM.

### Main software for training neural networks:
- Python 2.7 (preferable and fully tested) or Python 3.5
- Keras 1.2.2
- Theano 0.9.0rc1

### Utility packages for geometry and image manipulation and other helper functions:
- h5py
- matplotlib
- numba
- numpy
- pandas
- rasterio
- Shapely
- scikit_image
- tifffile
- OpenCV 
- tqdm

1. Install required OS and Python
2. Install packages with `pip install -r requirements.txt`
3. Create following directory structure:
- Data structure:
```
data / theree_band / *
     / sixteen_band / *
    grid_sizes.csv
    train_wkt_v4.csv
```
- Source code
```
src / water / *
    / *.py

```
    
# Prepare data for training:
1. Run `python get_3_band_shapes.py`
2. Run `cache_train.py`

# Train models
Each class in our solution has separate neural network, so it requires running of
several distinct models one by one (or in parallel if there are enough computing resources)

`python unet_buidings.py`

`python unet_structures.py`

`python unet_road.py`

`python unet_track.py`

`python unet_trees.py`

`python unet_crops.py`

For water predictions we used different method and it can be created by running 
`python make_water.py` inside `water` directory.

After training finishes (it may require quite a long time depending on hardware used, 
in our case it was about 7 hours for each stage (50 epochs)) trained weights and model architectures
are saved in `cache` directory and can be used by prediction scripts (see the next section).

# Create predictions
To create predictions run every make_prediction_cropped_*.py file in `src` dir. 
It could take considerable amount of time to generate all predictions as there are 
a lot of data in test and we use separate models for each class and use test time 
augmentation and cropping for the best model performance. On Titan X GPU each class 
took about 5 hours to get predictions.

When all predictions are done they should be merged in a single file for submit:
- Edit `merge_predictions.py` to include desired csv file for each class to be merged (example below):

```
submissions = {0: 'temp_building.csv',
               1: 'temp_structures.csv',
               2: 'temp_road_4.csv',
               3: 'temp_track_100_0.3_geom.csv',
               4: 'temp_trees_4.csv',
               5: 'temp_crops_4.csv',
               6: 'water_auto_indices_fast.csv',
               7: 'water_auto_indices_slow.csv'}
```
- Run `python merge_predictions.py`
- Done!


# Remarks
Please, keep in mind that this isn't a production ready code but a very specific 
solution for the particular competition created in short time frame and with a lot 
of other constrains (limited training data, scarce computing resources and a 
small number of attents to check for improvements). 

So, there are a lot of hardcoded magic numbers and strings and 
there may be some inconsistensies and differences between different models. Sometimes, it
was indentended to get more accurate predictions and there wasn't enough resources to 
check if changes improve score for other classes after they were introduced for some of them.
Sometimes, it just slipped from our attention. If you have any questions, don't hesitate to 
contact us: cepera.ang@gmail.com or iglovikov@gmail.com