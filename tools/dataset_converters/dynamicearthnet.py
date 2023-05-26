import os
import os.path as osp
from datetime import datetime
import shutil
from tqdm import tqdm
import csv
import mmcv
import rasterio
import numpy as np
import mmengine
import glob

# read file label train
train_txt = '../train.txt'
train_txt = [line.rstrip().split(' ') for line in tuple(open(train_txt, "r"))]
files, labels, year_months = list(zip(*train_txt))

dates = '01'
reference_date = "2018-01-01"
reference_date = datetime(*map(int, reference_date.split("-")))


planet, day = [], []
for i, year_month in enumerate(year_months):
    curr_date = year_month + '-' + dates
    planet.append(os.path.join(files[i][28:], curr_date + '.tif'))
    day.append((datetime(int(str(curr_date)[:4]), int(
        str(curr_date[5:7])), int(str(curr_date)[8:])) - reference_date).days)
# planet_day = list(zip(*[iter(planet)] * len('1'), *[iter(day)] * len('1')))

data_labels = list(zip(iter(planet), iter(labels)))


def copy_file(source_path, destination_path):
    try:
        shutil.copy2(source_path, destination_path)
    except FileNotFoundError:
        return


history_copy = [['source', 'destination']]

for data, label in tqdm(data_labels):
    destination_file_name = data[1:].replace('/', '_')

    source_file = '../planet'+data
    destination_file = '../data_monthly/planet/'+destination_file_name
    if not os.path.exists(destination_file):
        history_copy.append([source_file, destination_file])
        copy_file(source_file, destination_file)

    source_file = '..'+label
    destination_file = '../data_monthly/labels/'+destination_file_name
    if not os.path.exists(destination_file):
        history_copy.append([source_file, destination_file])
        copy_file(source_file, destination_file)


def save_list_of_lists_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print("CSV file saved successfully!")


save_list_of_lists_to_csv(history_copy, 'history_copy.csv')

# convert tif to png


def get_files_matching_pattern(directory, pattern):
    file_list = glob.glob(directory + pattern)
    return file_list


directory = "/home/Hung_Data/HungData/mmseg_data/Datasets/DynamicEarthNet/data_monthly/planet/"
pattern = "*.tif"

files_planets = get_files_matching_pattern(directory, pattern)
files_labels = [file.replace('/planet/', '/labels/') for file in files_planets]

# convert label
file_fails = []
for file in tqdm(files_labels):
    try:
        label = rasterio.open(file)
        label = label.read()
        mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

        for i in range(6):
            if sum(sum(label[i, :])) == 0:
                continue
            if i == 6:
                mask[label[i, :, :] == 255] = -1
            else:
                mask[label[i, :, :] == 255] = i
        mmcv.imwrite(mask, file[:-3]+'png')

    except:
        file_fails.append(file)

print(len(file_fails))

# convert data


def undo_normalize_scale_3(im):
    mean = [1042.59240722656, 915.618408203125, 671.260559082031]
    std = [957.958435058593, 715.548767089843, 596.943908691406]
    im = im * std + mean
    array_min, array_max = im.min(), im.max()
    im = (im - array_min) / (array_max - array_min)
    im *= 255.0
    return im.astype(np.uint8)


for file in tqdm(files_planets):
    if not os.path.exists(file[:-3]+'png'):
        try:
            img = rasterio.open(file)
            red = img.read(3)
            green = img.read(2)
            blue = img.read(1)
            #nir = img.read(4)

            image = np.dstack((red, green, blue))
            image = undo_normalize_scale_3(image)

            mmcv.imwrite(image, file[:-3]+'png')
        except:
            pass


# split train/val set
path_data = '../data_monthly/labels/'

filename_list = [osp.splitext(filename)[0]
                 for filename in mmengine.scandir(path_data, suffix='.png')]
# len(filename_list)

results = []
for file in filename_list:
    abs_file = '_'.join(file.split('_')[:2])
    results.append(abs_file)
abs_files = list(set(results))

data_root = '../data_monthly/'
split_dir = 'splits'

# select first 4/5 as train set
train = []
val = []
for file in filename_list:
    abs_file = '_'.join(file.split('_')[:2])
    if abs_file in abs_files[:43]:
        train.append(file)
    else:
        val.append(file)

with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    f.writelines(line + '\n' for line in train)
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    f.writelines(line + '\n' for line in val)
