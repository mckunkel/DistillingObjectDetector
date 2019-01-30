from keras import backend as K
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.losses import categorical_crossentropy as logloss


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
def soft_logloss(y_true, y_pred, temperature):
    logits = y_true[:, 256:]
    y_soft = K.softmax(logits / temperature)
    y_pred_soft = y_pred[:, 256:]
    return logloss(y_soft, y_pred_soft)