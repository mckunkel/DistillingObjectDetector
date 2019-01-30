import numpy as np
import sys
sys.path.append('utils/')

import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# use non standard flow_from_directory
from utils.image_preprocessing_v2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception

from keras.models import Model
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K

from models.squeezenet import SqueezeNet, preprocess_input

import matplotlib.pyplot as plt
import  constants as c

data_dir = c.data_dir



train_logits = np.load(data_dir + 'train_logits.npy')[()]
val_logits = np.load(data_dir + 'val_logits.npy')[()]

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

# note: i'm also passing dicts of logits
train_generator = data_generator.flow_from_directory(
    data_dir + 'train', train_logits,
    target_size=(299, 299),
    batch_size=64
)

val_generator = data_generator.flow_from_directory(
    data_dir + 'val', val_logits,
    target_size=(299, 299),
    batch_size=64
)

temperature = 5.0

model = SqueezeNet(weight_decay=1e-4, image_size=299)

# remove softmax
model.layers.pop()

# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)

# softed probabilities
logits_T = Lambda(lambda x: x/temperature)(logits)
probabilities_T = Activation('softmax')(logits_T)

output = concatenate([probabilities, probabilities_T])
model = Model(model.input, output)


def knowledge_distillation_loss(y_true, y_pred, lambda_const):
    # split in
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :256], y_true[:, 256:]

    # convert logits to soft targets
    y_soft = K.softmax(logits / temperature)

    # split in
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :256], y_pred[:, 256:]

    return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)*temperature

def accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return categorical_accuracy(y_true, y_pred)

def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return top_k_categorical_accuracy(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return logloss(y_true, y_pred)



# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):
    logits = y_true[:, 256:]
    y_soft = K.softmax(logits/temperature)
    y_pred_soft = y_pred[:, 256:]
    return logloss(y_soft, y_pred_soft)

lambda_const = 0.2

model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
    metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
)

model.fit_generator(
    train_generator,
    steps_per_epoch=40, epochs=30, verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)

# metric plots
plt.plot(model.history.history['categorical_crossentropy'], label='train')
plt.plot(model.history.history['val_categorical_crossentropy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('logloss')
plt.savefig('student_logloss_vs_epoch.png')

plt.plot(model.history.history['acc'], label='train')
plt.plot(model.history.history['val_acc'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('student_accuracy_vs_epoch.png')

plt.plot(model.history.history['top_5_accuracy'], label='train')
plt.plot(model.history.history['val_top_5_accuracy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('top5_accuracy')
plt.savefig('student_top5_accuracy_vs_epoch.png')

val_generator_no_shuffle = data_generator.flow_from_directory(
    data_dir + 'val', val_logits,
    target_size=(299, 299),
    batch_size=64, shuffle=False
)

print(model.evaluate_generator(val_generator_no_shuffle, 80))