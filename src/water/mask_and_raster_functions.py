import numpy as np
import cv2
import shapely.wkt
from config import *
from shapely.wkt import loads as wkt_loads
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
from rasterio import features
import rasterio


def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == (cType)].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        # print('In _plot_mask_from_contours type is:', img_mask.dtype)
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    # print('In _plot_mask_from_contours type is:', img_mask.dtype)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    # if class_type in [1]:
    #     polygon_list = polygon_list.buffer(2e-5)
    #     polygon_list = MultiPolygon([polygon_list])
    # if class_type in [3, 4]:
    #     polygon_list = polygon_list.buffer(2e-5)
    #     polygon_list = MultiPolygon([polygon_list])
    # if class_type in [9, 10]:
    #     polygon_list = polygon_list.buffer(5e-5)
    #     polygon_list = MultiPolygon([polygon_list])
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    # print('In generate_mask_for_image_and_class type is:', mask.dtype, 'shape:', mask.shape)

    return mask



def mask_from_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def mask_to_polygons(mask, epsilon=1, min_area=1., engine='opencv', buffer_amount=0.001):
    #    print('Mask toi polygon')
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    if engine == 'opencv':
        # TODO check this shit. Тут добавил >=0.5 чтобы обойти кривые маски в файле (без бинаризации)
        image, contours, hierarchy = cv2.findContours(
            ((mask >= 0.5) * 255).astype(np.uint8),
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        # create approximate contours to have reasonable submission size
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                           for cnt in contours]
        if not contours:
            return MultiPolygon()
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
                poly = Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area])
                all_polygons.append(poly)
                # approximating polygons might have created invalid ones, fix them
    else:
        all_polygons = []
        for shape, value in features.shapes(mask.astype(np.int16),
                                            mask=(mask == 1),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = MultiPolygon(all_polygons)
    if True:  # not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(buffer_amount)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_from_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min