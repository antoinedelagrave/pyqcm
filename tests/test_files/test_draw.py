import pyqcm
from model_graphene_bath import model

import pyqcm.draw_operator as draw
import matplotlib.pyplot as plt


draw.draw_operator('t', show_neighbors=True, show_values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_t.pdf')
plt.cla()

draw.draw_cluster_operator(model.clus[0], 'tb1', show_values=True, show_labels=True, plt_ax=plt.gca())
plt.savefig('draw_tb1.pdf')


