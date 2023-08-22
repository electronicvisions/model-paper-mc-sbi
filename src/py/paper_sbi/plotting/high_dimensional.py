'''
Illustrate the results for a high-dimensional parameter space.
'''
import math
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation

from paper_sbi.plotting.expected_coverage import plot_expected_coverage
from paper_sbi.plotting.ppc import plot_ppc_amplitudes
from paper_sbi.plotting.helper import formatted_parameter_names, \
    formatted_parameter_names_bss, replace_latex, latex_enabled, \
    add_legend_with_patches, get_figure_width, DataSingleObservation

from paramopt.plotting.correlation import plot_correlation_matrix, \
    set_xlabels, set_ylabels
from paramopt.plotting.marginals import plot_marginals


def create_high_dim_plots(data_amp_first: DataSingleObservation,
                          data_amp_all: DataSingleObservation,
                          target_df: pd.DataFrame) -> None:
    '''
    Create figures which display the results obtained for a high-dimensional
    parameter space.

    :param data_amp_first: Data for experiments with amplitudes_first as an
        observation.
    :param data_amp_all: Data for experiments with all amplitudes as an
        observation.
    :param target_df: DataFrame from which the target observation can be
        extracted.
    '''
    posterior_dfs = [data_amp_first.posterior_samples,
                     data_amp_all.posterior_samples]
    fileformat = 'pgf' if latex_enabled() else 'svg'
    double_width = get_figure_width('double')
    height = 2

    # 1d marginals
    figure = plot_marginal_dist(
        (double_width * 0.4, height), posterior_dfs,
        original_parameters=target_df.attrs['parameters'])
    figure.savefig(f'1d_marginals.{fileformat}')
    plt.close()

    # Correlations
    figure = plot_correlations((double_width * 0.6, height), posterior_dfs)
    figure.savefig(f'correlations_md.{fileformat}')
    plt.close()

    # PPC
    figure = plot_ppc((double_width / 2, height), posterior_dfs, target_df)
    figure.savefig(f'observations_md.{fileformat}')
    plt.close()

    # Expected Coverage
    titles = [r'First comp. $\myvec{F}$', r'All comp. $\mymat{H}$']
    if not latex_enabled():
        titles = [replace_latex(title) for title in titles]
    figure = plot_expected_coverage(
        (double_width / 2, 2),
        data_ensemble=[data_amp_first.coverage_ensemble,
                       data_amp_all.coverage_ensemble],
        data_single=[data_amp_first.coverage_single,
                     data_amp_all.coverage_single],
        titles=titles)
    figure.savefig(f'expected_coverage_md.{fileformat}')
    plt.close()


def plot_marginal_dist(figsize: Tuple[int, int],
                       posterior_dfs: Sequence[pd.DataFrame],
                       original_parameters: np.ndarray
                       ) -> plt.Figure:
    '''
    Plot the 1d-marginals of the provided posterior samples.

    The one-dimensional distribution of each individual parameter is plotted
    in a separate axis in the figure.

    :param figsize: Size of the figure (width, height).
    :param posterior_dfs: Data Frames with posterior samples for which to plot
        the one-dimensional marginals. The first DataFrame is assumed to use
        the amplitudes in the first compartment as a target, the second
        DataFrame is assumed to use all amplitudes as a target.
    :param original_parameters: Parameters used to record the observation on
        which the posteriors where conditioned.
    :returns: Figure with the one-dimensional marginals.
    '''
    assert posterior_dfs[0].attrs['observation'] == \
        Observation.AMPLITUDES_FIRST.name
    assert posterior_dfs[1].attrs['observation'] == Observation.AMPLITUDES.name

    parameters = posterior_dfs[0]['parameters'].columns.to_list()
    n_columns = math.ceil(len(parameters) / 2)
    fig, axs = plt.subplots(2, n_columns,
                            figsize=figsize,
                            sharey=True,
                            gridspec_kw={'left': 0.01, 'right': 0.97,
                                         'top': 0.99, 'bottom': 0.18,
                                         'hspace': 1, 'wspace': 0.2})
    # For consistency with previous figures (2d parameter space), advance color
    # cycle by one
    for ax in axs.flatten():
        _advance_style_cycle(ax)

    plot_marginals(axs, posterior_dfs, original_parameters=original_parameters)

    # styling
    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
    axs[-1, -1].axis('off')  # No data in this axes due to odd number of params
    axs[0, 0].set_ylim(bottom=0)

    x_labels = formatted_parameter_names_bss(posterior_dfs[0].attrs['length'])
    labels = [r'First comp. $\myvec{F}$', r'All comp. $\mymat{H}$']
    title = r'$\textbf{Observable}$'

    if not latex_enabled():
        x_labels = replace_latex(x_labels)
        labels = [replace_latex(label) for label in labels]
        title = replace_latex(title)

    for label, ax in zip(x_labels, axs.flatten()):
        ax.set_xlabel(label)

    add_legend_with_patches(axs[-1, -1], labels,
                            [axs[0, 0].lines[-2].get_color(),
                             axs[0, 0].lines[-1].get_color()],
                            title=title,
                            loc='center')
    return fig


def plot_ppc(figsize: Tuple[float, float],
             posterior_dfs: Sequence[pd.DataFrame],
             target_df: pd.DataFrame) -> plt.Figure:
    '''
    Plot the mean distance to the target observation for the observations
    of the given posterior samples.

    We use this plot to illustrate the results of a posterior predictive check
    (PPC).

    :param figsize: Size of the figure (width, height).
    :param posterior_dfs: Data Frames with posterior samples and their recorded
        amplitudes. The first DataFrame is assumed to use the amplitudes in the
        first compartment as a target, the second DataFrame is assumed to use
        all amplitudes as a target.
    :param target_df: DataFrame from which the target observation can be
        extracted.
    :returns: Figure with the results of posterior predictive check.
    '''
    assert posterior_dfs[0].attrs['observation'] == \
        Observation.AMPLITUDES_FIRST.name
    assert posterior_dfs[1].attrs['observation'] == Observation.AMPLITUDES.name

    length = posterior_dfs[0].attrs['length']
    fig, axs = plt.subplots(length, length, figsize=figsize,
                            sharex=True, sharey=True,
                            gridspec_kw={'left': 0.02, 'right': 0.98,
                                         'top': 0.95, 'bottom': 0.17,
                                         'hspace': 0.6, 'wspace': 0.1})
    # For consistency with previous figures (2d parameter space), advance color
    # cycle by one
    for ax in axs.flatten():
        _advance_style_cycle(ax)

    plot_ppc_amplitudes(axs, posterior_dfs, target_df)

    return fig


def plot_correlations(figsize: Tuple[float, float],
                      posterior_dfs: Sequence[pd.DataFrame]) -> plt.Figure:
    '''
    Plot the correlation between samples drawn from the posterior.

    :param figsize: Size of the figure (width, height).
    :param posterior_dfs: Data Frames with posterior samples.
        The first DataFrame is assumed to use the amplitudes in the
        first compartment as a target, the second DataFrame is assumed to use
        all amplitudes as a target.
    :returns: Figure with the correlation matrices of both sets of posterior
        samples.
    '''
    assert posterior_dfs[0].attrs['observation'] == \
        Observation.AMPLITUDES_FIRST.name
    assert posterior_dfs[1].attrs['observation'] == Observation.AMPLITUDES.name

    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={'left': 0.1, 'right': 0.99,
                                         'top': 0.86, 'bottom': 0.11,
                                         'hspace': 0.1, 'wspace': 0.1})

    parameter_names = formatted_parameter_names(
        posterior_dfs[0].attrs['length'])
    if not latex_enabled():
        parameter_names = replace_latex(parameter_names)

    mappables = []
    for ax, data in zip(axs, posterior_dfs):
        mappables.append(plot_correlation_matrix(ax, data['parameters']))
    set_xlabels(axs[0], parameter_names)
    set_ylabels(axs[0], parameter_names)
    set_xlabels(axs[1], parameter_names)

    color_bar = fig.colorbar(mappables[0], ax=axs, pad=0.02)
    color_bar.set_label(r'Pearson Correlation')
    color_bar.locator = ticker.MaxNLocator(4)
    color_bar.update_ticks()

    titles = [r'First comp. $\myvec{F}$', r'All comp. $\mymat{H}$']

    if not latex_enabled():
        titles = [replace_latex(title) for title in titles]

    for ax, title in zip(axs, titles):
        ax.set_xlabel(title)

    return fig


##############################################################################
# Helper
##############################################################################
def _advance_style_cycle(ax: plt.Axes) -> None:
    '''
    Advance the style cycle of the given axis.

    This is archived by plotting a line and then removing it.

    :param ax: Axis for which to advance the style
    '''
    line = ax.plot([])[0]
    line.remove()
