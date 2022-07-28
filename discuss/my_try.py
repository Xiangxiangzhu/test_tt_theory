import pickle
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from test__ import get_standard_plot, scale

file_name = '100False_save.pkl'
with open(file_name, 'rb') as fo:
    tt_result = pickle.load(fo)

file_name = '100True_save.pkl'
with open(file_name, 'rb') as fo:
    bl_result = pickle.load(fo)

fig, ax = plt.subplots(figsize=(24 / scale, 15 / scale), dpi=100 * scale)
get_standard_plot(ax, tt_result)
get_standard_plot(ax, bl_result, agent_name="base_line_result", color="#7FB3D5")
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5), framealpha=0.6, fancybox=True, prop={'size': 30 / scale})
plt.show()

temp_tr = []
temp_br = []
##########################################################
for tr, br in zip(tt_result, bl_result):
    fig, ax = plt.subplots(figsize=(24 / scale, 15 / scale), dpi=100 * scale)
    get_standard_plot(ax, tt_result)
    get_standard_plot(ax, bl_result, agent_name="base_line_result", color="#7FB3D5")
    ax.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5), framealpha=0.6, fancybox=True, prop={'size': 30 / scale})
    # plt.show()
    plt.plot(tr, color="#800000", marker='o')
    plt.plot(br, color="#7FB3D5", marker='o')
    plt.plot(temp_tr, color="#800000", linestyle='--')
    plt.plot(temp_br, color="#7FB3D5", linestyle='--')
    temp_br = br
    temp_tr = tr
    plt.show()
