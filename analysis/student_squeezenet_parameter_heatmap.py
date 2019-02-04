import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

data_dir = '/Volumes/MacStorage/InSight/Analysis/SqueezeNet/npyData/'


temps = [2.5, 5, 10, 15]
#lamdas = [0.02, 0.2, 0.5, 1]
lamdas = [1, 0.5, 0.2, 0.02]

acc_List = []
val_accList = []

top_5_accList=[]
top_5_val_accList=[]

logloss_List = []
val_loglossList = []

list3 = [(x, y) for x in temps for y in lamdas]

for temperature, lambda_constant in list3:
    d1 = np.load(data_dir + 'student_squeezenet_val_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    d2 = np.load(data_dir + 'student_squeezenet_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    acc_List.append(round(d2[-1], 3))
    val_accList.append(round(d1[-1], 3))

    #Top 5 accuracy
    d3 = np.load(data_dir + 'student_squeezenet_val_top_5_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    d4 = np.load(data_dir + 'student_squeezenet_top_5_accuracy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    top_5_accList.append(round(d4[-1], 3))
    top_5_val_accList.append(round(d3[-1], 3))

    #logloss
    d5 = np.load(data_dir + 'student_squeezenet_val_categorical_crossentropy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    d6 = np.load(data_dir + 'student_squeezenet_categorical_crossentropy_T_{}_lambda_{}.npy'.format(temperature,lambda_constant))
    logloss_List.append(round(d6[-1], 3))
    val_loglossList.append(round(d5[-1], 3))

acc=np.array(acc_List).reshape(4,4).T
val_acc=np.array(val_accList).reshape(4,4).T

top5=np.array(top_5_accList).reshape(4,4).T
val_top5=np.array(top_5_val_accList).reshape(4,4).T

logloss=np.array(logloss_List).reshape(4,4).T
val_logloss=np.array(val_loglossList).reshape(4,4).T

print(acc)
# print(val_acc)
# print(top5)
# print(val_top5)
# print(logloss)
# print(val_logloss)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()

im = ax.imshow(acc,cmap='Spectral')
im2 = ax2.imshow(val_acc,cmap='Spectral')

im3 = ax3.imshow(top5,cmap='Spectral')
im4 = ax4.imshow(val_top5,cmap='Spectral')

im5 = ax5.imshow(logloss,cmap='Spectral')
im6 = ax6.imshow(val_logloss,cmap='Spectral')

# We want to show all ticks...
ax.set_xticks(np.arange(len(temps)))
ax.set_yticks(np.arange(len(lamdas)))
ax2.set_xticks(np.arange(len(temps)))
ax2.set_yticks(np.arange(len(lamdas)))
ax3.set_xticks(np.arange(len(temps)))
ax3.set_yticks(np.arange(len(lamdas)))
ax4.set_xticks(np.arange(len(temps)))
ax4.set_yticks(np.arange(len(lamdas)))
ax5.set_xticks(np.arange(len(temps)))
ax5.set_yticks(np.arange(len(lamdas)))
ax6.set_xticks(np.arange(len(temps)))
ax6.set_yticks(np.arange(len(lamdas)))
# ... and label them with the respective list entries
ax.set_xticklabels(temps)
ax.set_yticklabels(lamdas)
ax2.set_xticklabels(temps)
ax2.set_yticklabels(lamdas)
ax3.set_xticklabels(temps)
ax3.set_yticklabels(lamdas)
ax4.set_xticklabels(temps)
ax4.set_yticklabels(lamdas)
ax5.set_xticklabels(temps)
ax5.set_yticklabels(lamdas)
ax6.set_xticklabels(temps)
ax6.set_yticklabels(lamdas)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax6.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(lamdas)):
    for j in range(len(temps)):
        text = ax.text(j, i, acc[i, j],
                       ha="center", va="center", color="w")
        text2 = ax2.text(j, i, val_acc[i, j],
                       ha="center", va="center", color="w")
        text3 = ax3.text(j, i, top5[i, j],
                       ha="center", va="center", color="w")
        text4 = ax4.text(j, i, val_top5[i, j],
                       ha="center", va="center", color="w")
        text5 = ax5.text(j, i, logloss[i, j],
                       ha="center", va="center", color="w")
        text6 = ax6.text(j, i, val_logloss[i, j],
                       ha="center", va="center", color="w")
ax.set_title("Training Accuracy of Student model")
ax.set_xlabel("Temperature")
ax.set_ylabel("λ")
ax2.set_title("Validation Accuracy of Student model")
ax2.set_xlabel("Temperature")
ax2.set_ylabel("λ")

ax3.set_title("Training Top-5 Accuracy of Student model")
ax3.set_xlabel("Temperature")
ax3.set_ylabel("λ")
ax4.set_title("Validation Top-5 Accuracy of Student model")
ax4.set_xlabel("Temperature")
ax4.set_ylabel("λ")

ax5.set_title("Training LogLoss of Student model")
ax5.set_xlabel("Temperature")
ax5.set_ylabel("λ")
ax6.set_title("Validation LogLoss of Student model")
ax6.set_xlabel("Temperature")
ax6.set_ylabel("λ")

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
fig.savefig('DistilledSqueezenetTrainingAcc.jpg', bbox_inches = 'tight', pad_inches = 0.025)
fig2.savefig('DistilledSqueezenetValAcc.jpg', bbox_inches = 'tight', pad_inches = 0.02)
fig3.savefig('DistilledSqueezenetTrainingTop5Acc.jpg', bbox_inches = 'tight', pad_inches = 0.02)
fig4.savefig('DistilledSqueezenetValTop5Acc.jpg', bbox_inches = 'tight', pad_inches = 0.02)
fig5.savefig('DistilledSqueezenetTrainingLogloss.jpg', bbox_inches = 'tight', pad_inches = 0.02)
fig6.savefig('DistilledSqueezenetValLogLoss.jpg', bbox_inches = 'tight', pad_inches = 0.02)

#plt.show()

