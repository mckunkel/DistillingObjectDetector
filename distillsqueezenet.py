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

from models.squeezenet import SqueezeNet, preprocess_input

import matplotlib.pyplot as plt
import constants as c
from utils.knowledge_distallion_loss_fn import knowledge_distillation_loss as distill_fn
import utils.metric_functions as mf

data_dir = c.data_dir



train_logits = np.load(data_dir + 'train_logits.npy')[()]
val_logits = np.load(data_dir + 'val_logits.npy')[()]

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)
data_generator2 = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)
# note: i'm also passing dicts of logits
train_generator = data_generator.flow_from_directory(
    data_dir + 'train', train_logits,
    target_size=(299, 299),
    batch_size=64
)

val_generator = data_generator2.flow_from_directory(
    data_dir + 'val', val_logits,
    target_size=(299, 299),
    batch_size=64
)

def distill(temperature=5.0, lambda_const=0.07):
    model = SqueezeNet(weight_decay=1e-4, image_size=299)

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

    #lambda_const = 0.2

    model.compile(
        optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
        loss=lambda y_true, y_pred: distill_fn(y_true, y_pred, lambda_const),
        metrics=[mf.accuracy, mf.top_5_accuracy, mf.categorical_crossentropy, mf.soft_logloss(temperature=temperature)]
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

    plt.plot(model.history.history['accuracy'], label='train')
    plt.plot(model.history.history['val_accuracy'], label='val')
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

    # serialize model to JSON
    model_json = model.to_json()
    with open("distilledSqueezeNet_model_T_{}_lambda_{}.json".format(temperature, lambda_const), "w") as json_file:
        json_file.write(model_json)
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("distilledSqueezeNet_model_T_{}_lambda_{}.yaml".format(temperature, lambda_const), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("distilledSqueezeNet_model_T_{}_lambda_{}.h5".format(temperature, lambda_const))

    print("Saved model to disk")
    print(model.evaluate_generator(val_generator_no_shuffle, 80))

if __name__ == '__main__':
    _temperature = float(sys.argv[1])
    _lambda_const = float(sys.argv[2])
    distill(_temperature, _lambda_const)

