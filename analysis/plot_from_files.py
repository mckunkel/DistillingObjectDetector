import matplotlib.pyplot as plt
import numpy as np

data_dir = '/Volumes/MacStorage/InSight/Analysis/SqueezeNet/npyData/'

temps = [2.5, 5, 10, 15]
lamdas = [0.02, 0.2, 0.5, 1]

list3 = [(x, y) for x in temps for y in lamdas]

for temperature, lambda_constant in list3:
    d1 = np.load(data_dir + 'student_squeezenet_val_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    d2 = np.load(data_dir + 'student_squeezenet_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    print('#############################################')
    print('########## Temperature = {}  lambda constant = {} ##########'.format(temperature,lambda_constant))
    print(d2[-1],' accuracy')
    print(d1[-1],' validation accuracy')
    print()

    plt.plot(d1, label='val')
    plt.plot(d2, label='train')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('logloss')
    plt.show()





