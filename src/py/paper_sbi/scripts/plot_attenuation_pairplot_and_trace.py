#!/usr/bin/env python3
from typing import Optional, Callable, List, Sequence
from functools import partial
import math
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import neo
import numpy as np

from model_hw_mc_attenuation.extract import get_experiment

from paramopt.plotting.density import plot_2d_enumerate, plot_1d_empty, \
    plot_2d_hist
from paramopt.plotting.pairplot import pairplot, create_axes_grid


def plot_trace_attenuation(
        subplot: plt.SubplotSpec, parameters: np.ndarray,
        recording_function: Callable,
        prev_axs: Optional[Sequence[plt.Axes]] = None,
        step_size: int = 6, **kwargs) -> List[plt.Axes]:
    '''
    Plot the membrane traces of different compartments.

    Keyword arguments are passed to :meth:`plt.Axes.plot`.

    :param subplot: SubplotSpec in which the data will be plotted.
    :param parameters: Parameters for which to plot the observation.
    :param recording_function: Function which takes parameters as an input and
        returns an observation.
    :param prev_axs: Axes previously created by calls to this function. These
        axes can be used to share x and y scales.
        returns an observation.
    :param step_size: Do not plot every recorded sample but only every
        step_size one.
    :returns: List of created axes.
    '''
    prev_ax = None if prev_axs is None else prev_axs[0]

    block = recording_function(parameters)
    ax = subplot.get_gridspec().figure.add_subplot(subplot, sharex=prev_ax,
                                                   sharey=prev_ax)

    spike_times = block.annotations['spike_times']
    isi = np.diff(spike_times).mean()

    t_start = spike_times[0] - isi / 20
    t_stop = spike_times[0] + isi / 3

    def cut_and_norm_signal(signal: neo.IrregularlySampledSignal,
                            ) -> neo.IrregularlySampledSignal:
        cut_sig = signal.time_slice(t_start, t_stop)
        cut_sig.time_shift(-cut_sig.t_start)
        return cut_sig - np.mean(cut_sig.magnitude[:100])

    for signal in block.segments[-1].irregularlysampledsignals:
        signal = cut_and_norm_signal(signal)
        ax.plot(signal.times.rescale(pq.us)[::step_size],
                signal[::step_size],
                label=signal.annotations['compartment'],
                **kwargs)
    return [ax]


def plot_posterior_and_enumerate(subplot: plt.SubplotSpec,
                                 posterior_samples_df: pd.DataFrame,
                                 samples: np.ndarray) -> None:
    '''
    Create pairplots of the posterior samples and mark the given samples.

    :param subplot: Space in which to create the axes for the pairplots of the
        posterior samples.
    :param posterior_samples: DataFrame with samples drawn from the posterior.
    :param samples: Array with the parameters of the samples to
        enumerate.
    '''

    axes = create_axes_grid(subplot,
                            posterior_samples_df['parameters'].shape[-1])

    pairplot(axes, posterior_samples_df['parameters'].values,
             labels=posterior_samples_df['parameters'].columns,
             plot_2d_dist=plot_2d_hist,
             limits=posterior_samples_df.attrs['limits'])

    pairplot(axes, samples, plot_1d_dist=plot_1d_empty,
             plot_2d_dist=plot_2d_enumerate,
             limits=posterior_samples_df.attrs['limits'])


def plot_traces(subplot: plt.SubplotSpec,
                samples: List[np.ndarray],
                plot_trace: Callable) -> List[plt.Axes]:

    '''
    Plot observations for the given samples in a grid.

    :param subplot: Space in which to create the grid for the membrane traces.
    :param samples: Parameters at which the attenuation experiment is
        configured.
    :param plot_trace: Function used to plot the observation of a single
        parameter.
    :returns: List of created axes.
    '''

    n_plots = len(samples)
    cols = min(math.ceil(np.sqrt(n_plots)), 3)
    rows = math.ceil(n_plots / cols)
    sub_grid = subplot.subgridspec(rows, cols)

    axes = []
    ax = None
    for grid_cell, sample in zip(sub_grid, samples):
        ax = plot_trace(grid_cell, sample, prev_axs=ax)
        axes.append(ax)
    return axes


def get_random_samples(samples_posterior: pd.DataFrame,
                       n_samples: int = 9) -> np.ndarray:
    '''
    Draw random samples from the posterior samples.

    :param posterior_samples: DataFrame with samples drawn from the posterior.
    :param n_samples: Number of random samples to draw from the posterior.
    :returns: List of random parameters sampled from the posterior samples.
    '''
    rand_idx = np.random.choice(samples_posterior.shape[0],
                                size=n_samples,
                                replace=False)
    return samples_posterior['parameters'].values[rand_idx, :]


def plot_pairplot_and_trace(samples_posterior: pd.DataFrame,
                            samples: np.ndarray,
                            plot_trace: Callable,
                            annotations: Optional[List[str]] = None
                            ) -> plt.Figure:
    '''
    Plot the distribution of the posterior samples and the membrane traces of
    selected samples from them.

    Display the posterior sample distribution in form of pairplots and mark
    selected samples in them. For each selected sample display the experiment
    observation.

    :param posterior_samples: DataFrame with samples drawn from the posterior.
    :param samples: Selected samples to mark in posterior distribution and
        for which to display the observation.
    :param plot_trace: Function used to plot the membrane trace of a single
        experiment.
    :param annotations: Annotations for the different samples.
    :returns: The created figure.
    '''
    if annotations is None:
        annotations = [str(n_sample) for n_sample, _ in enumerate(samples)]

    fig = plt.figure(figsize=np.array([1, 2]) * 15)
    grid = fig.add_gridspec(2)

    plot_posterior_and_enumerate(grid[0], samples_posterior, samples)
    axes = plot_traces(grid[1], samples, plot_trace)

    for ax, annotation in zip(axes, annotations):
        ax[0].set_title(annotation, loc='left')

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
    target_df = pd.read_pickle(pos_samples_df.attrs['target_file'])

    # Get random samples
    params = get_random_samples(pos_samples_df, args.n_samples)
    param_annotation = [str(n_sample) for n_sample, _ in enumerate(params)]
    if args.plot_original_parameter:
        original_parameters = target_df.attrs['parameters']
        # Parameters might be set globally -> restrict to two values
        if params.shape[1] == 2:
            original_parameters = original_parameters[[0, -1]]
        params = np.vstack([params, original_parameters])
        param_annotation.append(f'{len(params) - 1} (Target)')

    experiment = get_experiment(target_df)
    plotting_func = partial(plot_trace_attenuation,
                            recording_function=experiment.record_data)

    figure = plot_pairplot_and_trace(pos_samples_df,
                                     params,
                                     plotting_func,
                                     annotations=param_annotation)
    figure.savefig('pairplot_and_traces.png')
