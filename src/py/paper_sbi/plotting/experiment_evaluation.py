'''
Illustrate what quantities are used as observations and how they are extracted
from the recorded membrane traces.
'''
from typing import Tuple, Sequence

import neo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory
import quantities as pq

from paper_sbi.plotting.grid_search import plot_grid_search
from paper_sbi.plotting.helper import get_figure_width, COMPARTMENT_COLORS, \
    add_scalebar, latex_enabled, formatted_parameter_names_bss, replace_latex


def create_evaluation_plots(example_traces: Sequence[neo.Block],
                            grid_search_df: pd.DataFrame,
                            v_per_madc: float) -> None:
    '''
    Create plot which illustrate how the experiments are evaluated and
        illustiate the results of the grid search.

    :param example_traces: Blocks with example traces to plot. The first block
        is used to vizualize the evaluation method.
    :param traces: Sequence of membrane traces, one for each compartment.
        Used to illustrate how PSP amplitudes are extracted.
    :param v_per_madc: Value of a single MADC bit in volts.
    :param grid_search_df: Results of a two-dimensional grid search.
    '''
    fileformat = 'pdf' if latex_enabled() else 'svg'
    width_double = get_figure_width('double')

    # Evaluation
    figure = plot_psp_evaluation(
        (width_double * 0.7, 1.5),
        example_traces[0].segments[-1].irregularlysampledsignals,
        v_per_madc)
    figure.savefig(f'psp_evaluation.{fileformat}')
    plt.close()

    # Grid search
    trace_params = [block.annotations['parameters'][[0, -1]] for block
                    in example_traces]

    param_names = formatted_parameter_names_bss()
    if not latex_enabled():
        param_names = replace_latex(param_names)
    figure = plot_grid_search((width_double * 0.3, 1.9), grid_search_df,
                              trace_params, param_names,
                              levels=[0.8, 1.17, 1.6, 2],
                              label_locations=[(350, 900),
                                               (500, 800),
                                               (700, 550),
                                               (650, 200)])
    figure.savefig(f'grid_search.{fileformat}')
    plt.close()


def plot_psp_evaluation(figsize: Tuple[float, float],
                        traces: Sequence[neo.IrregularlySampledSignal],
                        v_per_madc: float) -> plt.Figure:
    '''
    Plot the given membrane traces and mark the amplitudes for the different
    inputs.

    :param figsize: Size of the figure (width, height).
    :param traces: Sequence of membrane traces, one for each compartment.
    :param v_per_madc: Value of a single MADC bit in volts.
    :returns: Figure with the membrane traces. The amplitudes in the different
        traces are marked and annotated.
    '''
    spike_times = traces[0].annotations['input_spikes']
    fig, axs = plt.subplots(len(traces), sharex=True, sharey=True,
                            figsize=figsize,
                            gridspec_kw={'left': 0.1, 'right': 0.96,
                                         'top': 0.93, 'bottom': 0.005,
                                         'hspace': 0.2})

    # scale traces and decrease sample density
    traces = [trace[::30] * v_per_madc for trace in traces]

    # work in us
    for trace in traces:
        trace.times = trace.times.rescale(pq.us)
    t_start = spike_times[0].rescale(pq.us) - 20 * pq.us

    # plot traces
    for n_trace, trace in enumerate(traces):
        ax = axs[n_trace]
        ax.plot(trace.times, trace, c=COMPARTMENT_COLORS[n_trace])

    # mark amplitudes
    for n_trace, (ax, trace) in enumerate(zip(axs, traces)):
        annotate_amplitudes(ax, trace, annotation_prefix=f'$h_{{{n_trace}')

    # remove axes
    for ax in axs.flatten():
        ax.axis('off')

    # plot input spikes
    for n_spike, spike in enumerate(spike_times):
        spike = spike.rescale(pq.us)
        plot_vline(spike, axs[0], axs[-1], color=COMPARTMENT_COLORS[n_spike],
                   lw=0.6, zorder=0)

    # set limits
    axs[0].set_xlim(t_start, traces[0].t_stop)

    # add scalebar
    font_style = {'c': '0.3', 'size': 6}
    margin = (0.05, 5)

    kwargs = {'loc': 'lower right', 'borderpad': 0, 'pad': 0,
              'bbox_to_anchor': (0.999, 0.05),
              'bbox_transform': ax.figure.transFigure}

    add_scalebar(ax,
                 x_label=r'$\SI{20}{\us}$' if latex_enabled() else '20us',
                 y_label=r'$\SI{0.5}{\V}$' if latex_enabled() else '0.5V',
                 args_xlabel=font_style,
                 args_ylabel=font_style,
                 args_scale={'c': '0.3'},
                 margin=margin, **kwargs)
    return fig


###############################################################################
# Helper
###############################################################################
def annotate_amplitudes(ax: plt.Axes,
                        trace: neo.IrregularlySampledSignal,
                        annotation_prefix: str = '$a_{',
                        annotation_suffix: str = '}$',
                        **kwargs) -> None:
    '''
    Annotate the extracted amplitudes in the given trace.

    Keyword arguments are passed to :meth:`plt.Axes.annotate`.

    :param ax: Axes in which to plot the annotations.
    :param trace: Trace for which to annotate the amplitudes for different
        inputs. Needs to have the annotation 'input_spikes'.
    :param annotation_prefix: String in the annotation just before the number
        of the current input for which the PSP amplitude is extracted.
    :param annotation_suffix: String in the annotation just after the number
        of the current input for which the PSP amplitude is extracted.
    '''
    spike_times = trace.annotations['input_spikes'].rescale(pq.ms)

    start_stop = np.concatenate([
        trace.t_start.rescale(pq.ms)[np.newaxis],
        spike_times[:-1] + np.diff(spike_times) / 2,
        trace.t_stop.rescale(pq.ms)[np.newaxis]]) * pq.ms

    for n_psp, (start, stop) in enumerate(zip(start_stop[:-1],
                                              start_stop[1:])):
        cut_trace = trace.time_slice(start, stop)
        max_index = cut_trace.argmax()
        max_height = float(cut_trace[max_index])
        max_time = cut_trace.times[max_index]
        ax.scatter(max_time, max_height, fc='k', edgecolor='none', s=5,
                   zorder=20)

        default_kwargs = {'size': 6,
                          'xytext': (0.1, 0.1), 'textcoords': 'offset points',
                          'ha': 'left', 'va': 'bottom',
                          'alpha': 0.6}
        default_kwargs.update(**kwargs)
        ax.annotate(f'{annotation_prefix}{n_psp}{annotation_suffix}',
                    xy=(max_time, max_height), **default_kwargs)


def plot_vline(x_value: float, ax_0: plt.Axes, ax_1: plt.Axes, **kwargs):
    '''
    Plot a vertical line from the top of the first axes to the bottom
    of the second axes.

    Keyword arguments are passed to :class:`ConnectionPatch`. The
    :class:`ConnectionPatch` artist is added to ax_0.

    :param x_value: Value in data coordinates where to plot the line.
    :param ax_0: Axes where the line ends at the top.
    :param ax_1: Axes where the line starts at the bottom.
    '''
    tform0 = blended_transform_factory(ax_0.transData, ax_0.transAxes)
    tform1 = blended_transform_factory(ax_1.transData, ax_1.transAxes)
    con = ConnectionPatch(xyA=(x_value, 1), coordsA=tform0,
                          xyB=(x_value, 0), coordsB=tform1,
                          **kwargs)
    ax_0.figure.add_artist(con)
