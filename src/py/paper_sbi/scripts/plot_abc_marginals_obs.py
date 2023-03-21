#!/usr/bin/env python3
import numbers
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_hw_mc_genetic.attenuation import Observation
from model_hw_mc_genetic.attenuation.helper import extract_observation

from paramopt.plotting.density import plot_1d_density
from paramopt.helper import get_identical_attr


def _create_figure(observation_size: int, chain_length: int
                   ) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    '''
    Create a figure and axes depending on the size of the observation.

    :param observation_size: Size of the observation.
    :param chain_length: Length of the compartmet chain.
    :returns: Tuple of created figure and created axis/axes.
    '''
    # Create figure
    if observation_size % chain_length == 0:
        n_columns = chain_length
        n_rows = observation_size // n_columns
    else:
        n_rows = 1
        n_columns = observation_size

    return plt.subplots(n_rows, n_columns,
                        figsize=np.array([n_columns, n_rows]) * 4,
                        sharey=True, sharex=True, tight_layout=True,
                        squeeze=False)


def main(posterior_dfs: List[pd.DataFrame],
         observation: Observation,
         labels: Optional[List[str]] = None) -> plt.Figure:
    '''
    Plot the marginal distributions of the given observations.

    :param posterior_dfs: DataFrames with samples drawn from the posterior.
        And recorded observations of these samples.
    :param observation: Kind of observation to plot.
    :param labels: Labels for the different posterior samples/observations.
    :returns: Generated figure.
    '''

    if labels is None:
        labels = np.arange(len(posterior_dfs))

    target_df = pd.read_pickle(get_identical_attr(posterior_dfs,
                                                  'target_file'))
    target_amplitudes = target_df.values.mean(0)
    targets = extract_observation(target_df, observation, target_amplitudes
                                  ).mean(0)
    if isinstance(targets, numbers.Number):
        targets = np.array([targets])

    fig, axs = _create_figure(targets.size, posterior_dfs[0].attrs['length'])

    # Plot distribution of observations
    for label, posterior_df in zip(labels, posterior_dfs):
        # Extract samples with observations
        amplitudes = posterior_df['amplitudes']
        amplitudes = amplitudes[~np.any(np.isnan(amplitudes), axis=1)]
        data = extract_observation(amplitudes, observation, target_amplitudes)

        if data.ndim == 1:
            data = data[:, np.newaxis]

        for ax, obs, target in zip(axs.flatten(), data.T, targets):
            plot_1d_density(ax, obs - target, label=label)
            ax.axvline(0, c='k', alpha=0.5, ls='-')
            ax.set_title(f' Target: {target:.0f}')

    axs.flatten()[-1].legend()

    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot the marginal distribution of the recorded '
                    'observations.')
    parser.add_argument('posterior_files',
                        type=str,
                        nargs='+',
                        help='Path to pickled DataFrames with samples drawn '
                             'from the posterior. And recorded observations '
                             'of these samples.')
    parser.add_argument("-observation",
                        help="Determines what kind of observation is "
                             "extracted from the attenuation experiment and "
                             "then plotted.",
                        type=str,
                        default=Observation.AMPLITUDES.name.lower(),
                        choices=[obs.name.lower() for obs in Observation])
    parser.add_argument('-labels',
                        nargs='+',
                        type=str,
                        help='Label for each file with posterior samples.')
    args = parser.parse_args()

    posterior_samples = []
    for posterior_file in args.posterior_files:
        posterior_samples.append(pd.read_pickle(posterior_file))

    figure = main(posterior_samples, Observation[args.observation.upper()],
                  args.labels)
    figure.savefig('abc_marginals_obs.svg')
