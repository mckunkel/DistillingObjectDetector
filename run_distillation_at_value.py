from distillsqueezenet import distill as ds

temps=[2.5, 5, 10, 15]
lamdas=[0.02, 0.2, 0.5, 1]

list3=[(x,y) for x in temps for y in lamdas]

for temperature,lambda_constant in list3:
    print('################ Temperature ################')
    print('################     {}      ################'.format(temperature))

    print('################ Lambda Constant ################')
    print('################     {}      ################'.format(lambda_constant))

    ds(temperature, lambda_constant)
