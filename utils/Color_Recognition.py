import os, webcolors, scipy
import scipy.cluster
from tqdm import tqdm
from collections import Counter
import numpy as np
from PIL import Image

def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_rgb_triplet():
    cars = os.listdir('./GUI_output/cropped_cars')
    colours = dict()
    for car in tqdm(cars, position=0):
        if not (str(car.split("_")[0]) in colours):
            colours[car.split("_")[0]] = []
        image = np.array(Image.open('./GUI_output/cropped_cars/%s' % car))
        shape = image.shape
        image = image.reshape(np.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(image, 5)
        vecs, dist = scipy.cluster.vq.vq(image, codes)  # assign codes
        counts, bins = np.histogram(vecs, len(codes))  # count occurrences

        index_max = np.argmax(counts)  # find most frequent
        peak = codes[index_max]
        colours[car.split("_")[0]].append(get_colour_name(peak))
    return colours

def color_recognition():
    car_colours = get_rgb_triplet()
    for car in car_colours:
        car_colours[car] = Counter(car_colours[car]).most_common(1)[0][0]
    return car_colours