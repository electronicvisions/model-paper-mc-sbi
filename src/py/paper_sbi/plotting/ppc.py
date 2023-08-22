from typing import Optional, Tuple, Dict, Sequence, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.extract import extract_observation


def plot_ppc_result(axs: Union[plt.Axes, np.ndarray],
                    observations: Sequence[np.ndarray],
                    target: pd.DataFrame) -> None:
    '''
    Plot the result of a predictive posterior check (PPC).

    Plot the mean deviation from the target observation and the standard
    deviation of the given observations. The deviations are scaled by the
    standard deviations of the target observation.

    :param axs: Axes in which to plot the results. Need to have the same
        dimension as the observation.
    :param observations: Arrays with observations.
    :param target_df: Target observation.
    :param observation: Type of observation to compare.
    '''
    norm_data = []
    for obs in observations:
        norm_data.append((obs - target.mean(0)) / target.std(0))

    if target.ndim == 1:
        plot_mean_and_std(axs, norm_data)
        return

    # iterate over dimensions of observations
    for curr_n, ax in enumerate(axs.flatten()):
        curr_data = [data[:, curr_n] for data in norm_data]
        plot_mean_and_std(ax, curr_data)


def plot_ppc_amplitudes(axs: np.ndarray, posterior_dfs: Sequence[pd.DataFrame],
                        target_df: pd.DataFrame):
    '''
    Plot the result of a predictive posterior check (PPC) where recorded
    amplitudes are compared.

    Plot the mean deviation from the target amplitudes and the standard
    deviation of the given amplitudes in `posterior_dfs`. The deviations
    are scaled by the standard deviations of the target amplitudes.

    :param axs: Axes in which to plot the results. Needs to have the same size
        as the number of recorded amplitudes. And is assumed to be arranged
        in a grid.
    :param posterior_dfs: DataFrames with recorded amplitudes.
    :param target_df: DataFrame from which the target observation can be
        extracted.
    '''
    observations = []
    for posterior_df in posterior_dfs:
        amplitudes = posterior_df['amplitudes']
        observations.append(extract_observation(
            amplitudes[~np.any(amplitudes.isna(), axis=1)],
            Observation.AMPLITUDES))
    target = extract_observation(target_df, Observation.AMPLITUDES)

    plot_ppc_result(axs, observations, target)

    # styling
    _add_observable_annotation(axs)

    for ax in axs[:-1, :].flatten():
        ax.axis('off')
    for ax in axs[-1, :]:
        ax.set_facecolor('none')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([-2, 0, 2])

    # mark target and sigma
    for ax in axs.flatten():
        ax.axvline(0, c='k', ls='-', lw=0.3, alpha=0.3, zorder=0)
        for sigma in [1, 2]:
            ax.axvline(-sigma, color='k', alpha=0.3, lw=0.3, ls=':', zorder=0)
            ax.axvline(sigma, color='k', alpha=0.3, lw=0.3, ls=':', zorder=0)

    ax.figure.supxlabel(r'Deviation from Target / $\sigma^*$', c='0.3', size=6)


def plot_mean_and_std(axis: plt.Axes,
                      datas: Sequence[np.ndarray],
                      x_lim: Tuple[float, float] = (-3, 3),
                      kwargs_outlier: Optional[Dict] = None) -> None:
    '''
    Mark mean and standard deviation of the given data with lines and bars.

    Draw a vertical line at the mean of the data and horizontal bar which
    represents the standard deviation of the data. The bar and line are colored
    according to the color cycle of the axis.

    :param ax: Axis in which to plot the data.
    :param datas: Arrays of data for which to plot the mean and std.
    :param x_lim: Restrict the x-axis to these limits and mark horizontal
        bars which exceed this limit.
    :param kwargs_outlier: Keyword arguments passed to text object with the
        arrows that mark bars which exceed the axis limit.
    '''
    bar_height = 4  # choose arbitrary number -> ratio stays the same
    line_height = 2 * bar_height
    # leave `bar_height` space between lines
    distance = line_height + bar_height
    for n_curr, data in enumerate(datas):
        y_center = -distance * n_curr  # start at top
        plot_mean_and_std_single(axis, data, y_center,
                                 bar_height=bar_height,
                                 line_height=line_height,
                                 x_min_max=x_lim,
                                 kwargs_outlier=kwargs_outlier)
    axis.set_xlim(x_lim)
    axis.set_ylim(-distance * (len(datas) - 0.5), distance * 0.5)


def plot_mean_and_std_single(ax: plt.Axes, data: np.array,
                             y_center: int = 0, *, bar_height: int = 4,
                             line_height: Optional[int] = None,
                             x_min_max: Optional[Tuple[int, int]] = None,
                             kwargs_outlier: Optional[Dict] = None) -> None:
    '''
    Mark mean with a vertical line and std with a horizontal bar.

    Draw a vertical line at the mean of the data and horizontal bar which
    represents the standard deviation of the data. The bar and line are colored
    according to the color cycle of the axis.

    :param ax: Axis in which to plot the data.
    :param data: Data for which to plot the mean and std.
    :param y_center: y value in data coordinates where the center of the
        bar/line are plotted.
    :param bar_height: Height of the horizontal bar.
    :param line_height: Height of the vertical line.
    :param x_min_max: Mark horizontal bars which exceed this range with arrows.
        This is useful if the limits of the x-axis do not include the whole
        data.
    :param kwargs_outlier: Keyword arguments passed to text object with the
        arrows that mark the outliers.
    '''
    color = _get_color(ax)

    if line_height is None:
        line_height = bar_height * 2

    data_mean = np.mean(data)
    data_std = np.std(data)

    # draw 1 sigma intervals
    x_rect = data_mean - data_std
    y_rect = y_center - bar_height / 2
    rect = Rectangle((x_rect, y_rect), 2 * data_std, bar_height,
                     edgecolor='none', facecolor=color,
                     alpha=0.3)
    ax.add_patch(rect)
    ax.vlines(data_mean, y_center - line_height / 2,
              y_center + line_height / 2, color=color)

    if x_min_max is None:
        return

    default_outlier_kwargs = {'c': 'w', 'size': 5, 'alpha': 0.9}
    if kwargs_outlier is not None:
        default_outlier_kwargs.update(kwargs_outlier)

    if data_mean - data_std < x_min_max[0]:
        ax.text(x_min_max[0], y_center, r'$\leftarrow$',
                ha='left', va='center', **default_outlier_kwargs)

    if data_mean + data_std > x_min_max[1]:
        ax.text(x_min_max[1], y_center, r'$\rightarrow$',
                ha='right', va='center', **default_outlier_kwargs)


def _add_observable_annotation(axs: plt.Axes, **kwargs) -> None:
    '''
    Annotate the observation in the axes.

    Annotate the name of the observation in the given axis. The axes are
    annotated like a matrix. The keyword arguments are passed to
    :meth:`plt.Axes.text`.

    :param axs: Axes to annotate.
    '''
    default_kwargs = {'size': 6, 'c': '0.3'}
    if kwargs is not None:
        default_kwargs.update(kwargs)
    for row, axs_row in enumerate(axs):
        for col, ax in enumerate(axs_row):
            ax.text(0.5, 1.04, rf'$h_{{{row}{col}}}$',
                    transform=ax.transAxes,
                    ha='center', va='bottom', **default_kwargs)


def _get_color(ax: plt.Axes) -> str:
    '''
    Get the next color in the color cycle of the given axis.

    This advances the color cycle by one.

    :param ax: Axis for which to get the next color.
    :returns: Next color in the color cycle.
    '''
    line = ax.plot([])[0]
    color = line.get_color()
    line.remove()
    return color
