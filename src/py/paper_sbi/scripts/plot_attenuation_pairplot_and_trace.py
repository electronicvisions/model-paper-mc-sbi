#!/usr/bin/env python3
from typing import Optional, Callable, List
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import neo
import quantities as pq
import numpy as np

from model_hw_mc_attenuation import AttenuationExperiment
from model_hw_mc_attenuation.extract import get_experiment

from paramopt.plotting.density import plot_2d_enumerate, plot_1d_empty, \
    plot_2d_hist
from paramopt.plotting.pairplot import pairplot, create_axes_grid


def plot_trace_attenuation(ax: plt.Axes, block: neo.Block, step_size: int = 6,
                           **kwargs) -> None:
    '''
    Plot the membrane traces of different compartments.

    Keyword arguments are passed to :meth:`plt.Axes.plot`.

    :param ax: Axis in which to plot the membrane traces.
    :param block: :class:`neo.Block` with the recorded membrane traces saved
        as :class:`neo.IrregularlySampledSignal`.
    :param step_size: Do not plot every recorded sample but only every
        step_size one.
    '''
    spike_times = block.annotations['spike_times']
    isi = np.diff(spike_times).mean()

    t_start = spike_times[0] - isi / 20
    t_stop = spike_times[0] + isi / 3

    for signal in block.segments[-1].irregularlysampledsignals:
        cut_sig = signal.time_slice(t_start, t_stop)
        time = (cut_sig.times - t_start.rescale(pq.ms)) \
            * 1000
        norm_voltage = cut_sig - np.mean(cut_sig.magnitude[:100])
        ax.plot(time[::step_size], norm_voltage[::step_size],
                label=signal.annotations['compartment'],
                **kwargs)


def plot_posterior_and_enumerate(subplot: plt.SubplotSpec,
                                 posterior_samples_df: pd.DataFrame,
                                 enumerate_samples: np.ndarray) -> None:
    '''
    Create pairplots of the posterior samples and mark the given samples.

    :param subplot: Space in which to create the axes for the pairplots of the
        posterior samples.
    :param posterior_samples: DataFrame with samples drawn from the posterior.
    :param enumerate_samples: Array with the parameters of the samples to
        enumerate.
    '''

    axes = create_axes_grid(subplot,
                            posterior_samples_df['parameters'].shape[-1])

    pairplot(axes, posterior_samples_df['parameters'].values,
             labels=posterior_samples_df['parameters'].columns,
             plot_2d_dist=plot_2d_hist,
             limits=posterior_samples_df.attrs['limits'],
             kwargs_2d={'cmap': plt.cm.get_cmap('Greys')})

    pairplot(axes, enumerate_samples, plot_1d_dist=plot_1d_empty,
             plot_2d_dist=plot_2d_enumerate)


def plot_traces(subplot: plt.SubplotSpec,
                experiment: AttenuationExperiment,
                parameters: List[np.ndarray],
                plot_trace: Callable) -> List[plt.Axes]:

    '''
    Plot membrane traces for an attenuation experiment configured with the
    given parameters.

    For each parameter set a axis is created and the recorded membrane traces
    are plotted in it.

    :param subplot: Space in which to create the axes for the membrane traces.
    :param experiment: Attenuation experiment used to record the membrane
        traces.
    :param parameters: Parameters at which the attenuation experiment is
        configured.
    :param plot_trace: Function used to plot the membrane trace of a single
        experiment.
    :returns: List of created axes.
    '''

    n_plots = len(parameters)
    cols = min(math.ceil(np.sqrt(n_plots)), 3)
    rows = math.ceil(n_plots / cols)
    sub_grid = subplot.subgridspec(rows, cols)

    fig = subplot.get_gridspec().figure

    axes = []
    ax = None
    for n_sample, sample in enumerate(parameters):
        ax = fig.figure.add_subplot(
            sub_grid[math.floor(n_sample / cols), n_sample % cols],
            sharex=ax, sharey=ax)
        plot_trace(ax, experiment.record_data(sample))
        axes.append(ax)
    return axes


def plot_pairplot_and_trace(samples_posterior: pd.DataFrame,
                            experiment: AttenuationExperiment,
                            plot_trace: Callable,
                            original_parameters: Optional[np.ndarray] = None,
                            n_samples: int = 9) -> plt.Figure:
    '''
    Plot the distribution of the posterior samples and the membrane traces of
    random samples from them.

    Display the posterior sample distribution in form of pairplots, draw random
    samples from these samples and display the membrane traces for these random
    samples.

    :param posterior_samples: DataFrame with samples drawn from the posterior.
    :param experiment: Attenuation experiment used to record the membrane
        traces.
    :param plot_trace: Function used to plot the membrane trace of a single
        experiment.
    :param original_parameters: If provided mark the original parameters on
        which the posterior was conditioned and plot the membrane trace for
        these parameters.
    :param n_samples: Number of random samples to draw from the posterior.
    :returns: The created figure.
    '''
    # Random samples
    rand_idx = np.random.choice(samples_posterior.shape[0],
                                size=n_samples,
                                replace=False)
    samples = samples_posterior['parameters'].values[rand_idx, :]
    axis_annotations = [str(n_sample) for n_sample, _ in enumerate(samples)]

    # Original parameters
    if original_parameters is not None:
        # Parameters might be set globally -> restrict to two values
        if samples.shape[1] == 2:
            original_parameters = original_parameters[[0, -1]]
        samples = np.vstack([samples, original_parameters])
        axis_annotations.append(f'{n_samples} (Target)')

    # Plotting
    fig = plt.figure(figsize=np.array([1, 2]) * 15)
    base_grid = fig.add_gridspec(2)

    plot_posterior_and_enumerate(base_grid[0], samples_posterior, samples)
    axes = plot_traces(base_grid[1], experiment, samples, plot_trace)

    for ax, annotation in zip(axes, axis_annotations):
        text = AnchoredText(annotation, prop=dict(size=15), frameon=False,
                            loc='upper right')
        ax.add_artist(text)

    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Draw samples from the posterior, mark them '
                    'in the pairplot and plot the membrane potentials '
                    'recorded for these samples.')
    parser.add_argument('posterior_samples_file',
                        type=str,
                        help='Path to pickled DataFrame with samples from the '
                             'posterior.')
    parser.add_argument('-n_samples',
                        type=int,
                        default=9,
                        help='Number of random samples/trace to plot.')
    parser.add_argument('--plot_original_parameter',
                        action='store_true',
                        help='Extract the original parameter used to record '
                             'the target and plot the membrane potential for '
                             'this parameter.')
    args = parser.parse_args()

    pos_samples_df = pd.read_pickle(args.posterior_samples_file)

    target_df = pos_samples_df.attrs['target_file']
    attenuation_exp = get_experiment(target_df)

    orig_parameters = None
    if args.plot_original_parameter:
        orig_parameters = target_df.attrs['parameters']

    figure = plot_pairplot_and_trace(pos_samples_df,
                                     attenuation_exp,
                                     plot_trace_attenuation,
                                     original_parameters=orig_parameters,
                                     n_samples=args.n_samples)
    figure.savefig('pairplot_and_traces.png')
