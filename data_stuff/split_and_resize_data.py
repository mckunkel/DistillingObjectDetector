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
print(os.path.exists('train_metadata.csv'))
print(os.path.dirname(os.path.realpath(__file__)))
train = pd.read_csv('train_metadata.csv')
val = pd.read_csv('val_metadata.csv')
# print(train.head())
# print(val.head())

if not os.path.isdir(data_dir + 'train'):
    os.mkdir(data_dir + 'train')
for i in range(1, 257 + 1):
    if not os.path.isdir(data_dir + 'train/' + str(i)):
        os.mkdir(data_dir + 'train/' + str(i))

if not os.path.isdir(data_dir + 'val'):
    os.mkdir(data_dir + 'val')
for i in range(1, 257 + 1):
    if not os.path.isdir(data_dir + 'val/' + str(i)):
        os.mkdir(data_dir + 'val/' + str(i))


val_transform = transforms.Compose([
    transforms.Scale(299, Image.LANCZOS),
    transforms.CenterCrop(299)
])

val_size = len(val)

# resize RGB images
for i, row in tqdm(val.loc[val.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # transform it
    image = val_transform(image)

    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

# resize grayscale images
for i, row in tqdm(val.loc[val.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # transform it
    image = val_transform(image)

    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)

    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

enhancers = {
    0: lambda image, f: ImageEnhance.Color(image).enhance(f),
    1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
    2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
    3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
}

factors = {
    0: lambda: np.random.uniform(0.4, 1.6),
    1: lambda: np.random.uniform(0.8, 1.2),
    2: lambda: np.random.uniform(0.8, 1.2),
    3: lambda: np.random.uniform(0.4, 1.6)
}

# randomly enhance images in random order
def enhance(image):
    order = [0, 1, 2, 3]
    np.random.shuffle(order)
    for i in order:
        f = factors[i]()
        image = enhancers[i](image, f)
    return image

train_transform_rare = transforms.Compose([
    transforms.Scale(384, Image.LANCZOS),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(enhance)
])

train_transform = transforms.Compose([
    transforms.Scale(384, Image.LANCZOS),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
])

# number of images in each category
class_counts = dict(train.category_name.value_counts())
np.save('class_counts.npy', class_counts)

# sample with replacement 100 images from each category
train = train.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=100, replace=True))
train.reset_index(drop=True, inplace=True)
train_size = len(train)

# resize RGB images
for i, row in tqdm(train.loc[train.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # transform it
    if class_counts[row.category_name] < 100:
        image = train_transform_rare(image)
    else:
        image = train_transform(image)

    # save
    new_image_name = str(i) + '_' + row.img_name
    save_path = os.path.join(data_dir, 'train', str(row.category_number), new_image_name)
    image.save(save_path, 'jpeg')

# resize grayscale images
for i, row in tqdm(train.loc[train.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)

    # transform it
    if class_counts[row.category_name] < 100:
        image = train_transform_rare(image)
    else:
        image = train_transform(image)

    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)

    # save
    new_image_name = str(i) + '_' + row.img_name
    save_path = os.path.join(data_dir, 'train', str(row.category_number), new_image_name)
    image.save(save_path, 'jpeg')