import argparse
import numpy as np

from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# use non standard flow_from_directory
from utils.image_preprocessing_v2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception

from keras.models import Model
from keras.models import load_model
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras import backend as K

from models.squeezenet import SqueezeNet, preprocess_input
import constants as c
from utils.knowledge_distallion_loss_fn import knowledge_distillation_loss as distill_fn
from utils import metric_functions as mf
from utils.plot_utils import plot_utils as plt_uts
from utils.history_utils import history_utils as hist_uts
from utils.save_utils import save_utils as save_uts


def distill(temperature=5.0, lambda_const=0.07, model_path='', save_name=''):

    print('############# Temperature #############')
    print('#############     {} #############'.format(temperature))
    print('########################################')
    print('############# lambda_const #############')
    print('#############     {}  #############' .format(lambda_const))
    print('########################################')

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

    input_model = load_model(model_path)
    model2 = load_model(model_path)
    for layer in model2.layers:
        layer.name = layer.name + str("_2")

    #model = load_model('models/student_squeezenet.hdf5')
    print(input_model.summary())


    # remove softmax
    input_model.layers.pop()
    print('###################')
    print(input_model.summary())

    # usual probabilities
    logits = input_model.layers[-1].output
    probabilities = Activation('softmax')(logits)
    #probabilities.name = probabilities.name +  str("original")

    # softed probabilities
    logits_T = Lambda(lambda x: x / temperature)(logits)
    probabilities_T = Activation('softmax')(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model2.input, output)
    print(model.summary())
    #lambda_const = 0.2

    # logloss with only soft probabilities and targets
    def soft_logloss(y_true, y_pred):
        logits = y_true[:, 256:]
        y_soft = K.softmax(logits / temperature)
        y_pred_soft = y_pred[:, 256:]
        return logloss(y_soft, y_pred_soft)

    model.compile(
        optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True),
        loss=lambda y_true, y_pred: distill_fn(y_true, y_pred, lambda_const, temperature),
        metrics=[mf.accuracy, mf.top_5_accuracy, mf.categorical_crossentropy, soft_logloss]
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=40, epochs=40, verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, min_delta=0.007)
        ],
        validation_data=val_generator, validation_steps=80, workers=4
    )

    plt_uts(model,save_name,temperature, lambda_const)

    hist_uts(model,save_name,temperature, lambda_const)

    save_uts(model,save_name,temperature, lambda_const)


    val_generator_no_shuffle = data_generator.flow_from_directory(
        data_dir + 'val', val_logits,
        target_size=(299, 299),
        batch_size=64, shuffle=False
    )

    print(model.evaluate_generator(val_generator_no_shuffle, 80))


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--temperature", type=float, required=True,
                    help="temperature for the softmax function")
    ap.add_argument("-l", "--lambda", type=float, required=True,
                    help="lambda parameter for scaling the hard targets")
    ap.add_argument("-m", "--student_model", type=str, required=True,
                    help="string name of hdf5 file of student model")
    ap.add_argument("-s", "--save_name", type=str, required=True,
                    help="string name of save name of trained student")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss/accuracy plot")
    ap.add_argument("-r", "--retrain", type=bool, default=False,
                    help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    print('Here is what was the input args["temperature"] = {},  args["model"]= {}'.format(args["temperature"], args["student_model"]))
    distill(args["temperature"], args["lambda"], args["student_model"], args["save_name"])
