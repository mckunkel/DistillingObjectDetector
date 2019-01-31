import matplotlib.pyplot as plt
import numpy as np


data_dir = '/Volumes/MacStorage/InSight/Analysis/SqueezeNet/'
d1 = np.load(data_dir+'student_squeezenet_categorical_crossentropy_T_5_lambda_0.2.npy')
d2 = np.load(data_dir+'student_squeezenet_val_categorical_crossentropy_T_5_lambda_0.2.npy')


plt.plot(d1, label='train')
plt.plot(d2, label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('logloss')
plt.show()
