## THIS REPO IS UNMAINTEINED AND MOSTLY REMAINS AS HISTORIC ARTIFACT.

### Please, take a look at versions of software we used to generate our result -- it is ancient now. Do not expect this code to run on Keras above version 1.2.2 and so on. It will not run out of the box, we wouldn't answer any issues about it, sorry. If you manage to run this code on newer versions -- please, feel free to open pull request, we will merge it for the public good. 

## Take a look at https://github.com/ternaus/TernausNet if you need more up-to-date segmentation solution.


# Winning Model Documentation
Name: Vladimir Iglovikov

LinkedIn: https://www.linkedin.com/in/iglovikov/

Location: San-Francisco, United States


Name: Sergey Mushinskiy

LinkedIn: https://www.linkedin.com/in/sergeymushinskiy/

Location: Angarsk, Russia

Competition: Dstl Satellite Imagery Feature Detection

Blog post: http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/

If you find this code useful for your publications, please consider citing

```
@article{DBLP:journals/corr/IglovikovMO17,
  author    = {Vladimir Iglovikov and
               Sergey Mushinskiy and
               Vladimir Osin},
  title     = {Satellite Imagery Feature Detection using Deep Convolutional Neural
               Network: A Kaggle Competition},  
  volume    = {abs/1706.06169},
  year      = {2017},  
  archivePrefix = {arXiv},
  eprint    = {1706.06169},     
}
```

# Prerequisites
To train final models you will need the following:

- OS: Ubuntu 16.04 (although code was successfully ran on Windows 10 too)
- Required hardware: 
    - Any decent modern computer with x86-64 CPU, 
    - Fair amount of RAM (we had about 32Gb and 128Gb in our boxes, however, not all memory was used) 
    - Powerful GPU: we used Nvidia Titan X (Pascal) with 12Gb of RAM and Nvidia GeForce GTX 1080 with 8Gb of RAM.

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
src / *.py

```
    
# Prepare data for training:
1. Run `python get_3_band_shapes.py`
2. Run `cache_train.py`

# Train models
Each class in our solution has separate neural network, so it requires running of several distinct models one by one (or in parallel if there are enough computing resources)

1. Run `python unet_buidings.py`
2. Run `python unet_structures.py`
3. Run `python unet_road.py`
4. Run `python unet_track.py`
5. Run `python unet_trees.py`
6. Run `python unet_crops.py`

For water predictions we used different method and it can be created by running:

7. Run `python fast_water.py`
8. Run `python slow_water.py`

After training finishes (it may require quite a long time depending on hardware used, in our case it was about 7 hours for each stage (50 epochs)) trained weights and model architectures are saved in `cache` directory and can be used by prediction scripts (see the next section).

# Create predictions
To create predictions run every make_prediction_cropped_*.py file in `src` dir. It could take considerable amount of time to generate all predictions as there are a lot of data in test and we use separate models for each class and use test time augmentation and cropping for the best model performance. On Titan X GPU each class took about 5 hours to get predictions.

1. Run `python make_prediction_cropped_buildings.py`
2. Run `python make_prediction_cropped_structures.py`
3. Run `python make_prediction_cropped_track.py`
4. Run `python make_prediction_cropped_road.py`
5. Run `python make_prediction_cropped_trees.py`
6. Run `python make_prediction_cropped_crops.py`

When all predictions are done they should be merged in a single file for submit:
- Run `python merge_predictions.py`


7. Run `python merge_predictions.py`
The previous step will create file `joined.csv` that just merges predictions per class into the unified format.

- Last step in the pipeline is to

8. Run `python post_processing.py joined.csv`

that will perform some cleaning of the overlapping classes (remove predictions of the slow water from fast water, all other predictions from buildings, etc)

- Done!


# Remarks
Please, keep in mind that this isn't a production ready code but a very specific solution for the particular competition created in short time frame and with a lot of other constrains (limited training data, scarce computing resources and a small number of attents to check for improvements). 

So, there are a lot of hardcoded magic numbers and strings and there may be some inconsistensies and differences between different models. Sometimes, it was indentended to get more accurate predictions and there wasn't enough resources to check if changes improve score for other classes after they were introduced for some of them. Sometimes, it just slipped from our attention. 

Also, inherent stochasticity of neural networks training on many different levels (random initialization of weights, random cropping of patches into minibatch and so on) makes it impossible to reproduce exact submission from scratch. We went extra mile and reimplemented solution and training procedure from scratch as much as possible in the last two weeks after competition final. We've got up to 20% extra performance for some classes with abundant training data like buildings, tracks and so on. However, some classes proven more difficult to reliably reproduce because of lack of training data and small amount of time. Such classes show high variance of results between epochs. For competition we used our best performing combinations of epoch/model for those classes, which may not be exactly the same as trained for fixed number of epochs (as in this particular code). However, we believe that our model is equally capable to segment any classes, given enough data and/or clear definitions what exactly consists of each class (it wasn't clear how segmentation was performed in the first place for some classes, like road/tracks). 
