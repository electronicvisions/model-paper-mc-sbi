'''
Evaluation of the experiments with a two-dimensional parameter space.
'''
from copy import copy, deepcopy
import logging
from typing import List, Tuple, Sequence, Optional, Dict, Union

import neo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import quantities as pq
import scipy

from model_hw_mc_attenuation.plotting.grid_search import \
    create_obs_dataframe, plot_heat_map, plot_contour_lines
from model_hw_mc_attenuation import Observation, extract_psp_heights
from model_hw_mc_attenuation.extract import extract_observation

from paper_sbi.plotting.helper import get_figure_width, COMPARTMENT_COLORS, \
    add_scalebar, formatted_parameter_names_bss, replace_latex, \
    latex_enabled, add_legend_with_patches

from paramopt.plotting.posterior import plot_posterior, label_low_high
from paramopt.plotting.density import plot_2d_scatter
from paramopt.plotting.pairplot import pairplot, create_axes_grid


Parameter = Tuple[float, float]


def log_results(posterior_dfs: Dict[Observation, pd.DataFrame],
                target_df: pd.DataFrame) -> None:
    '''
    Log the results for experiments executed with a two-dimensional parameter
    space.

    This logs the target length constant, the result of posterior predictive
    check of the length constant as well as the correlation between parameters.

    :param posterior_dfs: DataFrames with parameters and observations.
    :param target_df: DataFrame from which the target was extracted.
    '''

    log_target(target_df)
    log_ppc_result(posterior_dfs[Observation.LENGTH_CONSTANT])
    log_correlations(posterior_dfs)


def log_target(target_df: pd.DataFrame) -> None:
    '''
    Log the target length constant and its standard deviation.

    :param target_df: DataFrame from which the target was extracted.
    '''
    obs = extract_observation(target_df, Observation.LENGTH_CONSTANT)
    logging.info('Target length_constant %.2f +- %.2f', obs.mean(), obs.std())


def log_ppc_result(posterior_df: pd.DataFrame) -> None:
    '''
    Log the mean measured length constant of predictive posterior samples and
    its standard deviation.

    :param posterior_df: DataFrames with posterior samples and recorded
        observations.
    '''
    obs = extract_observation(posterior_df['amplitudes'],
                              Observation.LENGTH_CONSTANT)
    logging.info('PPC length_constant %.2f +- %.2f', obs.mean(), obs.std())


def log_correlations(posterior_dfs: Dict[Observation, pd.DataFrame]) -> None:
    '''
    Log the Pearson correlation coefficient of the first two parameters.

    :param posterior_dfs: DataFrames with parameters for which to log the
        correlation.
    '''
    for observation, samples in posterior_dfs.items():
        corr = scipy.stats.pearsonr(samples['parameters'].values[:, 0],
                                    samples['parameters'].values[:, 1])
        logging.info('Correlation %s: %s', observation, corr)


###############################################################################
# Plotting
###############################################################################
def create_two_dim_plots(*,
                         grid_search_df: pd.DataFrame,
                         example_traces: Sequence[neo.Block],
                         posterior,
                         posterior_dfs: Sequence[pd.DataFrame],
                         original_parameters: Parameter,
                         v_per_madc: float):
    '''
    Create all figures which are used to illustrate the results obtained with
    a two-dimensional parameter space.

    This includes the results of a grid search, the posterior probability when
    observing the length constant, posterior samples (for the observation of
    the length constant and the PSP amplitudes in the first compartment) and
    example traces for different points in the parameter space.

    :param grid_search_df: Results of a two-dimensional grid search.
    :param example_traces: Blocks with example traces to plot.
    :param posterior: Posterior for which to plot the probability distribution.
    :param posterior_dfs: DataFrames with samples drawn from the posterior.
        The first DataFrame is assumed to use the length constant as a target,
        the second DataFrame is assumed to use the amplitudes in the first
        compartment as a target.
    :param original_parameters: Initial parameters used to record the
        observation on which the posteriors are conditioned.
    :param v_per_madc: Characterization of the MADC. How much volt a single
        bit in an MADC measurement corresponds to.
    '''
    fileformat = 'pgf' if latex_enabled() else 'svg'
    figure_width = get_figure_width('double') / 3
    heights_2d = 2
    height_traces = 0.4

    trace_params = [block.annotations['parameters'][[0, -1]] for block
                    in example_traces]

    param_names = formatted_parameter_names_bss()
    if not latex_enabled():
        param_names = replace_latex(param_names)

    # Grid Search
    figure = plot_grid_search((figure_width, heights_2d), grid_search_df,
                              trace_params, param_names,
                              levels=[0.8, 1.17, 1.6, 2],
                              label_locations=[(350, 900),
                                               (500, 800),
                                               (700, 550),
                                               (650, 200)])
    figure.savefig(f'grid_search.{fileformat}')
    plt.close()

    # Posterior
    figure = plot_posterior_prob((figure_width, heights_2d), posterior,
                                 trace_params, param_names)
    figure.savefig(f'posterior.{fileformat}')
    plt.close()

    # Samples
    figure = plot_posterior_samples((figure_width, heights_2d),
                                    posterior_dfs,
                                    original_parameters,
                                    trace_params,
                                    param_names)
    figure.savefig(f'posterior_samples.{fileformat}')
    plt.close()

    # Traces
    example_traces = [_rescale_times(block, pq.us) for block in example_traces]
    traces_amp = [_scale_by_psp_amplitude(block) for block in example_traces]
    figure = plot_traces((figure_width, height_traces), traces_amp)
    figure.savefig(f'traces_scaled.{fileformat}')
    plt.close()

    traces_v = [_scale(block, v_per_madc * pq.V) for block in example_traces]
    figure = plot_traces((figure_width, height_traces), traces_v)
    figure.savefig(f'traces.{fileformat}')
    plt.close()


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
                           gridspec_kw={'left': 0.2, 'right': 0.9,
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


def plot_posterior_prob(figsize: Tuple[float, float],
                        posterior,
                        marker_points: Sequence[Parameter],
                        param_names: Sequence[str]) -> plt.Figure:
    '''
    Plot the posterior probability in a heat map.

    :param figsize: Size of the figure (width, height).
    :param posterior: Posterior for which to plot the probability distribution.
    :param marker_points: Points in the parameter space which should be marked
        with increasing numbers.
    :param param_names: Names of the parameters. Used to label the axes.
    :returns: Figure with a heat map of the posterior probability.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.23, 'right': 0.93,
                                        'top': 0.83, 'bottom': 0.19})

    c_mesh = plot_posterior(ax, posterior, rasterized=True)

    color_bar = fig.colorbar(c_mesh, ax=ax)
    color_bar.set_label(r'Posterior $p(\theta \mid x)$')
    label_low_high(color_bar, low=0.1, high=0.9)

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    mark_points(ax, marker_points)

    return fig


def plot_posterior_samples(figsize: Tuple[float, float],
                           posterior_dfs: Sequence[pd.DataFrame],
                           original_parameters: Parameter,
                           marker_points: Sequence[Parameter],
                           param_names: Sequence[str]) -> plt.Figure:
    '''
    Plot the posterior samples in a two-dimensional scatter plot.

    :param figsize: Size of the figure (width, height).
    :param posterior_dfs: DataFrames with samples drawn from the posterior.
        The first DataFrame is assumed to use the length constant as a target,
        the second DataFrame is assumed to use the amplitudes in the first
        compartment as a target.
    :param original_parameters: Initial parameters used to record the
        observation on which the posteriors are conditioned.
    :param marker_points: Points in the parameter space which should be marked
        with increasing numbers.
    :param param_names: Names of the parameters. Used to label the axes.
    :returns: Figure with scatter plot of the posterior samples.
    '''
    assert posterior_dfs[0].attrs['observation'] == \
        Observation.LENGTH_CONSTANT.name
    assert posterior_dfs[1].attrs['observation'] == \
        Observation.AMPLITUDES_FIRST.name

    fig = plt.figure(figsize=figsize)
    base_grid = fig.add_gridspec(1, left=0.25, right=0.99, top=0.99,
                                 bottom=0.19)
    axs = create_axes_grid(base_grid[0], 2)

    for samples in posterior_dfs:
        pairplot(axs, samples['parameters'].values[::2],
                 labels=param_names,
                 plot_2d_dist=plot_2d_scatter,
                 limits=samples.attrs['limits'],
                 kwargs_2d={'s': 3, 'alpha': 0.3, 'ec': 'none'},
                 target_params=original_parameters)

    mark_points(axs[1, 0], marker_points)

    labels = [r'Decay $\tau$', r'Height $\myvec{F}$']
    title = r'$\textbf{Observable}$'
    if not latex_enabled():
        labels = [replace_latex(label) for label in labels]
        title = replace_latex(title)
    add_legend_with_patches(axs[-1, 0], labels, ['C0', 'C1'],
                            title=title,
                            loc='upper right')
    return fig


def plot_traces(figsize: Tuple[float, float],
                blocks: Sequence[neo.Block]) -> plt.Figure:
    '''
    Plot the response to an input in the first compartment for the given
    traces.


    :param figsize: Size of the figure (width, height).
    :param blocks: Blocks with reposes to plot.
    :returns: Figure with an axes for each response. In each axes the response
        to an input in the first compartment is plotted.
    '''
    fig, axes = _create_traces_figure(figsize, blocks)
    _add_scalebar(axes[-1],
                  blocks[-1].segments[-1].irregularlysampledsignals[0])
    return fig


###############################################################################
# Helper
###############################################################################
def subtract_baseline(signal: neo.IrregularlySampledSignal
                      ) -> neo.IrregularlySampledSignal:
    '''
    Subtract the baseline from the given signal.

    :param signal: Signal from which to subtract the baseline.
    :returns: New signal with baseline subtracted. The baseline is the mean
        signal from the start to the first input spike.
    '''
    baseline = signal.time_slice(signal.t_start,
                                 signal.annotations['input_spikes'][0]).mean()
    return signal - baseline


def trim_around_first_input(signal: neo.IrregularlySampledSignal,
                            margin: Tuple[float, float] = (0.05, 0.3)
                            ) -> neo.IrregularlySampledSignal:
    '''
    Cut signal around first input spike.

    :param signal: Signals to cut.
    :param margin: Margin around first input spike to which the trace is
        cut. This value is given in percent of the mean time difference
        between input spikes.
    :returns: Trace cut to region around first input spike.
    '''
    spike_times = signal.annotations['input_spikes']
    isi = np.diff(spike_times).mean()

    t_start = spike_times[0] - isi * margin[0]
    t_stop = spike_times[0] + isi * margin[1]
    return signal.time_slice(t_start, t_stop)


def annotate_circled(ax: plt.Axes, point: Tuple[float, float], text: str,
                     **kwargs) -> None:
    '''
    Add an annotation with a circle around it.

    Keyword arguments are forwarded to :meth:`plt.Axes.annotate`.

    :param ax: Axis to which the annotation is added.
    :param point:  x and y coordinate where the annotation is added.
    :param text: Text of the annotation.
    '''
    default_kwargs = {'ha': 'center', 'va': 'center',
                      'bbox': {'boxstyle': 'circle, pad=0.3', 'fc': 'w',
                               'ec': 'k', 'lw': 0.8},
                      'size': 6}
    default_kwargs.update(kwargs)
    ax.annotate(text, xy=point, **default_kwargs)


def mark_points(ax: plt.Axes, points: Sequence[Parameter]) -> None:
    '''
    Mark the given points with increasing numbers.

    :param ax: Axis in which to mark the points.
    :param point: Points at which annotations with increasing numbers are
        added.
    '''
    for n_point, point in enumerate(points):
        annotate_circled(ax, point, str(n_point))


def _create_traces_figure(figsize: Tuple[float, float],
                          blocks: Sequence[neo.Block],
                          ) -> Tuple[plt.Figure, List[plt.Axes]]:
    '''
    Plot the response to an input in the first compartment for the given
    traces.


    :param figsize: Size of the figure (width, height).
    :param blocks: Blocks with reposes to plot.
    :returns: Figure with an axes for each response. In each axes the response
        to an input in the first compartment is plotted. Also return a list
        of all axes.
    '''
    fig = plt.figure(figsize=figsize)
    traces_grid = fig.add_gridspec(1, len(blocks), wspace=0.05,
                                   left=0.03, right=0.89, bottom=0.35,
                                   top=0.98)
    ax = None  # share axes later
    axes = []
    for n_trace, block in enumerate(blocks):
        traces = block.segments[-1].irregularlysampledsignals

        ax = fig.add_subplot(traces_grid[n_trace], sharex=ax, sharey=ax)
        ax.set_prop_cycle(color=COMPARTMENT_COLORS)

        traces = [trim_around_first_input(sig) for sig in traces]
        traces = [subtract_baseline(sig) for sig in traces]

        step_size = 30
        for signal in traces:
            ax.plot(signal.times[::step_size], signal[::step_size],
                    label=signal.annotations['compartment'])
        annotate_circled(ax, (0.5, -0.25), str(n_trace),
                         xycoords='axes fraction')
        ax.axis('off')
        axes.append(ax)

    return fig, axes


def _add_scalebar(ax: plt.Axes, signal: Sequence[neo.Block]) -> None:
    '''
    Add a scalebar to the given axes.

    :param ax: Axis to which a scalebar should be added.
    :param signal: Irregularly sampled for which to add the scalebar. The
        units of the x and y axes are extracted from it.
    '''
    font_style = {'c': '0.3', 'size': 6}
    ylabel_args = copy(font_style)

    # ylabel
    if signal.units == pq.dimensionless:
        ylabel_args.update(text=r'$h_{00}$')
        scale = 1
        ylabel = str(scale)
    else:
        scale = np.diff(ax.get_ylim())
        # round down to next "nice" number
        factor = 10**np.floor(np.log10(scale))
        scale = float(scale // factor * factor)
        ylabel = rf'$\SI{{{scale}}}{{\V}}$' if latex_enabled() else f'{scale}V'

    # xlabel
    x_unit = signal.times.dimensionality.string
    xlabel = rf'$\SI{{20}}{{\{x_unit}}}$' if latex_enabled() else f'20{x_unit}'

    kwargs = {'loc': 'lower right', 'borderpad': 0, 'pad': 0,
              'bbox_to_anchor': (0.98, 0.02),
              'bbox_transform': ax.figure.transFigure}

    add_scalebar(ax,
                 x_label=xlabel,
                 y_label=ylabel,
                 args_xlabel=font_style,
                 args_ylabel=ylabel_args,
                 args_scale={'c': '0.3'},
                 margin=(scale / 10, 5), **kwargs)


def _scale(block: neo.Block, factor: pq.Quantity) -> neo.Block:
    '''
    Scale all IrregularlySampledSignal in the block by the factor.

    :param blocks: Block with IrregularlySampledSignals to scale.
    :param factor: Factor by which each signal is multiplied.
    :returns: Block with scaled signals.
    '''
    scaled_block = deepcopy(block)
    signals = scaled_block.segments[-1].irregularlysampledsignals
    scaled_signals = [sig * factor for sig in signals]
    scaled_block.segments[-1].irregularlysampledsignals = scaled_signals

    return scaled_block


def _rescale_times(block: neo.Block, unit: pq.Quantity) -> neo.Block:
    '''
    Rescale the times of all IrregularlySampledSignal in the block.

    :param blocks: Block with IrregularlySampledSignals to rescale.
    :param factor: Unit by which to rescale the times.
    :returns: Block with times of signals rescaled.
    '''
    scaled_block = deepcopy(block)
    for sig in scaled_block.segments[-1].irregularlysampledsignals:
        sig.times = sig.times.rescale(unit)
    return scaled_block


def _scale_by_psp_amplitude(block: neo.Block) -> neo.Block:
    '''
    Scale all IrregularlySampledSignal in the block by the amplitude of the
    first input.

    :param blocks: Block with IrregularlySampledSignals to scale.
    :returns: Block with signals scaled by the PSP amplitude of the first
        input.
    '''
    signals = block.segments[-1].irregularlysampledsignals
    scale = 1 / extract_psp_heights(signals)[0, 0]
    return _scale(block, scale)
