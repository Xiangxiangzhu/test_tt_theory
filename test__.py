import pickle
from test_theory_plot import get_mean_and_standard_deviation_difference_results
import matplotlib.pyplot as plt

scale = 2


def get_standard_plot(ax, result, scale=1, agent_name="this_result", color="#800000"):
    mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference_results(result)
    x_vals = list(range(len(mean_results)))

    ax.plot(x_vals, mean_results, label=agent_name, color=color)
    ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
    ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)

    # plt.show()


if __name__ == '__main__':
    file_name = '200False_save.pkl'
    with open(file_name, 'rb') as fo:
        tt_result = pickle.load(fo)

    file_name = '200True_save.pkl'
    with open(file_name, 'rb') as fo:
        bl_result = pickle.load(fo)

    fig, ax = plt.subplots(figsize=(24 / scale, 15 / scale), dpi=100 * scale)
    get_standard_plot(ax, tt_result)
    get_standard_plot(ax, bl_result, agent_name="base_line_result", color="#7FB3D5")
    ax.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5), framealpha=0.6, fancybox=True, prop={'size': 30 / scale})
    plt.show()
