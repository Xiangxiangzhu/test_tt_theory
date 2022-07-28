import numpy as np
import matplotlib.pyplot as plt


def get_mean_and_standard_deviation_difference_results(results):
    """From a list of lists of agent results it extracts the mean results and the mean results plus or minus
     some multiple of the standard deviation"""

    standard_deviation_results = 1.0

    def get_results_at_a_time_step(results, timestep):
        results_at_a_time_step = [result[timestep] for result in results]
        return results_at_a_time_step

    def get_standard_deviation_at_time_step(results, timestep):
        results_at_a_time_step = [result[timestep] for result in results]
        return np.std(results_at_a_time_step)

    mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
    mean_minus_x_std = [
        mean_val - standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
        for
        timestep, mean_val in enumerate(mean_results)]
    mean_plus_x_std = [
        mean_val + standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
        for
        timestep, mean_val in enumerate(mean_results)]
    return mean_minus_x_std, mean_results, mean_plus_x_std


def get_standard_plot(result, scale=2, agent_name="this_result", color="#800000"):
    fig, ax = plt.subplots(figsize=(24 / scale, 15 / scale), dpi=100 * scale)
    mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference_results(result)
    x_vals = list(range(len(mean_results)))

    ax.plot(x_vals, mean_results, label=agent_name, color=color)
    ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
    ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)

    ax.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5), framealpha=0.6, fancybox=True, prop={'size': 30 / scale})
    # plt.show()





if __name__ == '__main__':
    list_to_test = [[j + i for j in range(20)] for i in range(4)]
    get_standard_plot(list_to_test)
    pass
