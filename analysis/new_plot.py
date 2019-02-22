import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from matplotlib.patches import Ellipse



fig, ax = plt.subplots()
ax.set_yticks([0.2,0.4,0.6,0.8])
ax.set_xticks([0.8,2.5,5,10,20,30])
plt.grid(True)

make_axes_area_auto_adjustable(ax)


el = Ellipse((21.386024, 0.835), width=1.5, height=2/30, angle=0)  # in data coordinates!
el2 = Ellipse((0.853824, 0.672), width=1.5, height=2/30, angle=0, color = 'g')  # in data coordinates!

ax.add_artist(el2)
ax.add_artist(el)
ax.set_xlabel('Number of Parameters [million]')
ax.set_ylabel('Top-1 Accuracy [%]')

fig.tight_layout()
fig.savefig('acc_vs_Nparameters.jpg', bbox_inches = 'tight', pad_inches = 0.025) #

fig2, ax2 = plt.subplots()
ax2.set_yticks([67.2,69.6,83.5])
ax2.set_xticks([0.8,21.3,30])
ax2.set_ylim(60,90)
plt.grid(True)

make_axes_area_auto_adjustable(ax2)

el = Ellipse((21.386024, 83.5), width=1.0, height=100/75, angle=0)  # in data coordinates!
el2 = Ellipse((0.853824, 67.2), width=1.0, height=100/75, angle=0, color = 'g')  # in data coordinates!
el2new = Ellipse((0.853824, 69.6), width=1.0, height=100/75, angle=0, color = 'blue')  # in data coordinates!

ax2.add_artist(el2)
ax2.add_artist(el)
ax2.add_artist(el2new)

ax2.set_xlabel('Number of Parameters [million]')
ax2.set_ylabel('Top-1 Accuracy [%]')

fig2.tight_layout()
fig2.savefig('acc_vs_Nparameters_newV2.jpg', bbox_inches = 'tight', pad_inches = 0.025) #

plt.show()
