import sys
import numpy as np
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

from models.minixception import miniXception, preprocess_input

import matplotlib.pyplot as plt

import constants as c

data_dir = c.data_dir

train_logits = np.load(data_dir + 'train_logits.npy')[()]
val_logits = np.load(data_dir + 'val_logits.npy')[()]

data_generator = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.001,
    channel_shift_range=0.1,
    fill_mode='reflect',
    data_format='channels_last',
    preprocessing_function=preprocess_input

    # data_format='channels_last',
    # preprocessing_function=preprocess_input
)

# note: i'm also passing dicts of logits
train_generator = data_generator.flow_from_directory(
    data_dir + 'train_no_resizing', train_logits,
    target_size=(299, 299),
    batch_size=16
)

val_generator = data_generator.flow_from_directory(
    data_dir + 'val_no_resizing', val_logits,
    target_size=(299, 299),
    batch_size=8
)

def distill(temperature = 5.0, lambda_const = 0.07, num_residuals = 0):
    print('############# Temperature #############')
    print('#############     {} #############'.format(temperature))
    print('########################################')
    print('############# lambda_const #############')
    print('#############     {}  #############' .format(lambda_const))
    print('########################################')
    print('############# num_residuals #############')
    print('#############     {}  #############' .format(num_residuals))
    print('########################################')
    model = miniXception(weight_decay=1e-5, num_residuals=num_residuals)
    # remove softmax
    model.layers.pop()

    # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation('softmax')(logits)

    # softed probabilities
    logits_T = Lambda(lambda x: x / temperature)(logits)
    probabilities_T = Activation('softmax')(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model.input, output)

    # now model outputs 512 dimensional vectors

    # custom loss function
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

        return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    # For testing use usual output probabilities (without temperature)

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
        y_soft = K.softmax(logits / temperature)
        y_pred_soft = y_pred[:, 256:]
        return logloss(y_soft, y_pred_soft)

    # Train student model

    model.compile(
        optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
        metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=400, epochs=30, verbose=1,
        callbacks=[
            ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, min_delta=0.007),
            EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01)
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
        data_dir + 'val_no_resizing', val_logits,
        target_size=(299, 299),
        batch_size=16, shuffle=False
    )
    model.save_weights('miniXception_weights.hdf5')
    print(model.evaluate_generator(val_generator_no_shuffle, 80))


if __name__ == '__main__':
    temperature = float(sys.argv[1])
    lambda_const = float(sys.argv[2])
    num_residuals = int(sys.argv[3])
    distill(temperature, lambda_const,num_residuals)