from distillsqueezenet import distill as ds
from distill_model import distill as dsXcept

def run(x):
    if x == 'squeeze':
        distill_squeeze()
    elif x == 'xception':
        distill_Xception()

def distill_squeeze():
    temps = [2.5, 5, 10, 15]
    lamdas = [0.02, 0.2, 0.5, 1]

    list3 = [(x, y) for x in temps for y in lamdas]

    for temperature, lambda_constant in list3:
        print('################ Temperature ################')
        print('################     {}      ################'.format(temperature))

        print('################ Lambda Constant ################')
        print('################     {}      ################'.format(lambda_constant))

        ds(temperature, lambda_constant)

def distill_Xception():
    temps = [2.5, 5, 10, 15]
    lamdas = [0.02, 0.2, 0.5, 1]
    residuals = range(7)
    list3 = [(x, y, z) for x in temps for y in lamdas for z in residuals]
    for temperature, lambda_constant, residual in list3:
        dsXcept(temperature, lambda_constant, residual)
