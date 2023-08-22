'''
Illustrate the results of a grid search.
'''
from typing import Tuple, Sequence, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

from model_hw_mc_attenuation.plotting.grid_search import \
    create_obs_dataframe, plot_heat_map, plot_contour_lines
from model_hw_mc_attenuation import Observation

from paper_sbi.plotting.helper import Parameter, mark_points


def plot_grid_search(figsize: Tuple[float, float],
                     data: pd.DataFrame,
                     marker_points: Sequence[Parameter],
                     param_names: Sequence[str], *,
                     levels: Union[int, Sequence[float]] = 4,
                     label_locations: Optional[Sequence] = None
                     ) -> plt.Figure:
    '''
    Plot the measured length constant in a heat map.

    :param figsize: Size of the figure (width, height).
    :param data: Results of a two-dimensional grid search.
    :param marker_points: Points in the parameter space which should be marked
        with increasing numbers.
    :param param_names: Names of the parameters. Used to label the axes.
    :param levels: Levels of the plotted contour lines.
    :param label_locations: Positions where labels of contour plots should be
        placed.
    :returns: Figure with a heat map of the measured length constant.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.25, 'right': 0.93,
                                        'top': 0.83, 'bottom': 0.19})

    length_constants = create_obs_dataframe(data, Observation.LENGTH_CONSTANT)
    im_plot = plot_heat_map(ax, length_constants)

    color_bar = ax.figure.colorbar(im_plot, ax=ax)
    color_bar.set_label(r'Decay Constant $\tau$ / comp')

    color_bar.locator = ticker.MaxNLocator(integer=True)
    color_bar.update_ticks()

    contour = plot_contour_lines(ax, length_constants, smooth_sigma=3,
                                 levels=levels)
    if label_locations is None:
        ax.clabel(contour, inline=True)
    else:
        ax.clabel(contour, inline=True, manual=label_locations)

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    mark_points(ax, marker_points)

    return fig


def plot_grid_search_diff(figsize: Tuple[float, float],
                          data: pd.DataFrame,
                          marker_points: Sequence[Parameter],
                          param_names: Sequence[str], *,
                          levels: Union[int, Sequence[float]] = 4,
                          target_length_constant: float,
                          **kwargs
                          ) -> plt.Figure:
    '''
    Plot the measured length constant in a heat map.

    Keyword arguments are passed `plot_heat_map()`.

    :param figsize: Size of the figure (width, height).
    :param data: Results of a two-dimensional grid search.
    :param marker_points: Points in the parameter space which should be marked
        with increasing numbers.
    :param param_names: Names of the parameters. Used to label the axes.
    :param levels: Levels of the plotted contour lines.
    :returns: Figure with a heat map of the measured length constant.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.2, 'right': 0.9,
                                        'top': 0.83, 'bottom': 0.19})

    length_constants = create_obs_dataframe(data, Observation.LENGTH_CONSTANT)
    difference = np.abs(length_constants - target_length_constant)
    default_kwargs = {}
    default_kwargs.update(**kwargs)
    im_plot = plot_heat_map(ax, difference, **default_kwargs)

    color_bar = ax.figure.colorbar(im_plot, ax=ax)
    color_bar.set_label(r'Difference to Target $\tau - \tau^*$ / comp')

    contour = plot_contour_lines(ax, difference, smooth_sigma=3,
                                 levels=levels)
    ax.clabel(contour, inline=True)

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    mark_points(ax, marker_points)

    return fig
