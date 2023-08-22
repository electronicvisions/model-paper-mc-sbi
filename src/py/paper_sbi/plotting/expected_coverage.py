'''
Calculate and plot the expected coverage.
'''
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sbi.inference.posteriors.base_posterior import NeuralPosterior

from paper_sbi.scripts.attenuation_draw_sbc_samples import extract_arrays

from paramopt.sbi.sbc import expected_coverage


@dataclass
class ExpectedCoverageData:
    '''
    Data needed to calculate the expected coverage.

    :ivar prior_samples: Prior samples and observations which were used to
        draw samples from the posterior.
    :ivar sbc_samples: Samples drawn from the posterior (conditioned on the
        observations mentioned above).
    :ivar posterior: Posterior for which to determine the expected coverage.
        The sbc_samples are assumed to be drawn from this posterior.
    '''
    prior_samples: pd.DataFrame
    sbc_samples: pd.DataFrame
    posterior: NeuralPosterior


def plot_expected_coverage(
        figsize: Tuple[float, float],
        data_ensemble: Sequence[ExpectedCoverageData],
        data_single: Sequence[Sequence[ExpectedCoverageData]],
        titles: Optional[Sequence[str]] = None,
) -> plt.Figure:
    '''
    Plot expected coverage of different ensembles.

    Each ensemble is plotted in a separate axes.

    :param figsize: Size of the figure (width, height).
    :param data_ensemble: List of data needed to calculate the expected
        coverage of the different ensemble posterior.
    :param data_single: Each item in the list contains a list of data which is
        needed to calculate the expected coverage of the posteriors which make
        up one ensemble.
    :param titles: Titles for the different axes.
    '''

    fig, axs = plt.subplots(1, len(data_ensemble),
                            figsize=figsize,
                            gridspec_kw={'left': 0.14, 'right': 0.99,
                                         'top': 0.9, 'bottom': 0.17,
                                         'wspace': 0.1})

    for ax, ensemble, single in zip(axs, data_ensemble, data_single):
        _plot_expected_coverage(ax, ensemble, single)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1], [0, 0.5, 1])

    axs[1].tick_params(which='both', labelleft=False)
    axs[1].get_yaxis().set_ticks([])

    if titles is not None:
        for ax, title in zip(axs, titles):
            ax.set_title(title)

    axs[0].set_ylabel('Expected Coverage')
    fig.supxlabel(r'Confidence level $1 - \alpha$', c='0.3', size=6)
    return fig


def _plot_expected_coverage(ax: plt.Axes,
                            data_ensemble: ExpectedCoverageData,
                            data_single: Sequence[ExpectedCoverageData]):
    '''
    Plot expected coverage of ensemble and all posteriors which make up the
        ensemble.

    :param ax: Axes to plot the data in.
    :param data_ensemble: Data needed to calculate the expected coverage of
        the ensemble posterior.
    :param data_single: Data needed to calculate the expected coverage of the
        posteriors which make up the ensemble.
    '''
    alphas = np.linspace(0, 1, 50)

    # Plot target
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, c='C0', ls=':')

    # Plot single
    for data in data_single:
        original_parameters, observations, sbc_samples = extract_arrays(
            sbc_df=data.sbc_samples, samples_df=data.prior_samples)
        cov = expected_coverage(original_parameters,
                                observations,
                                samples=sbc_samples,
                                posterior=data.posterior,
                                alphas=alphas)
        ax.plot(1 - alphas, cov, alpha=0.2, c='k')

    # Plot ensemble
    original_parameters, observations, sbc_samples = extract_arrays(
        sbc_df=data_ensemble.sbc_samples,
        samples_df=data_ensemble.prior_samples)
    cov = expected_coverage(original_parameters,
                            observations,
                            samples=sbc_samples,
                            posterior=data_ensemble.posterior,
                            alphas=alphas)
    ax.plot(1 - alphas, cov, c='k')
