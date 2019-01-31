import matplotlib.pyplot as plt


def plot_utils(model, student_name,temperature, lambda_const, num_residuals=0):
    #Lets make some save strings first, some models will not depend on num_residuals
    if num_residuals == 0:
        logloss_vs_epoch='student_{}_logloss_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature, lambda_const)
        accuracy_vs_epoch='student_{}_accuracy_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature, lambda_const)
        top5_accuracy_vs_epoch = 'student_{}_top5_accuracy_vs_epoch_T_{}_lambda_{}.png'.format(student_name, temperature,lambda_const)

    else:
        logloss_vs_epoch='student_{}_logloss_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)
        accuracy_vs_epoch='student_{}_accuracy_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)
        top5_accuracy_vs_epoch='student_{}_top5_accuracy_vs_epoch_T_{}_lambda_{}_numResiduals_{}.png'.format(student_name, temperature, lambda_const, num_residuals)

    # metric plots
    x = plt.plot(model.history.history['categorical_crossentropy'], label='train')
    x.plot(model.history.history['val_categorical_crossentropy'], label='val')
    x.legend()
    x.xlabel('epoch')
    x.ylabel('logloss')
    x.savefig(logloss_vs_epoch)

    y = plt.plot(model.history.history['accuracy'], label='train')
    y.plot(model.history.history['val_accuracy'], label='val')
    y.legend()
    y.xlabel('epoch')
    y.ylabel('accuracy')
    y.savefig(accuracy_vs_epoch)

    z = plt.plot(model.history.history['top_5_accuracy'], label='train')
    z.plot(model.history.history['val_top_5_accuracy'], label='val')
    z.legend()
    z.xlabel('epoch')
    z.ylabel('top5_accuracy')
    z.savefig(top5_accuracy_vs_epoch)