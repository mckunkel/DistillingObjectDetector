import numpy as np
from tqdm import tqdm
import sys
sys.path.append('utils/')

# use non standard flow_from_directory
from image_preprocessing_ver1 import ImageDataGenerator
# it outputs not only x_batch and y_batch but also image names

from keras.models import Model
from xception import Xception, preprocess_input

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    data_dir + 'train',
    target_size=(299, 299),
    batch_size=64, shuffle=False
)

val_generator = data_generator.flow_from_directory(
    data_dir + 'val',
    target_size=(299, 299),
    batch_size=64, shuffle=False
)