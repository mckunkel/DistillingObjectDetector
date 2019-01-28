import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

import os
from tqdm import tqdm
import constants as c
# the folder from 256_ObjectCategories.tar file
train_dir = c.dir_train

# a folder where resized and split data will be stored
data_dir = ''
# Load the saved .csv of constant train-val split
# print(os.path.exists('train_metadata.csv'))
# print(os.path.dirname(os.path.realpath(__file__)))
train = pd.read_csv('train_metadata.csv')
val = pd.read_csv('val_metadata.csv')
# print(train.head())
# print(val.head())

if not os.path.isdir(data_dir + 'train_no_resizing'):
    os.mkdir(data_dir + 'train_no_resizing')
for i in range(1, 257 + 1):
    if not os.path.isdir(data_dir + 'train_no_resizing/' + str(i)):
        os.mkdir(data_dir + 'train_no_resizing/' + str(i))

if not os.path.isdir(data_dir + 'val_no_resizing'):
    os.mkdir(data_dir + 'val_no_resizing')
for i in range(1, 257 + 1):
    if not os.path.isdir(data_dir + 'val_no_resizing/' + str(i)):
        os.mkdir(data_dir + 'val_no_resizing/' + str(i))

val_size = len(val)

# RGB images
for i, row in tqdm(val.loc[val.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

# grayscale images
for i, row in tqdm(val.loc[val.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)

    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

train_size = len(train)

# RGB images
for i, row in tqdm(train.loc[train.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # save
    save_path = os.path.join(data_dir, 'train_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

# grayscale images
for i, row in tqdm(train.loc[train.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)

    # save
    save_path = os.path.join(data_dir, 'train_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')