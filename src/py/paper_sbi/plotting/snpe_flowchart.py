'''
Plotting functions which create the icons used in the SNPE flowchart.
'''
from typing import Tuple, Sequence

import neo
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import torch

from paper_sbi.plotting.helper import latex_enabled

from paramopt.plotting.posterior import plot_posterior


def create_icons(posterior,
                 traces: Sequence[neo.IrregularlySampledSignal]) -> None:
    '''
    Create figures used in the SNPE flow chart.

    :param posterior: Posterior to plot.
    :param traces: Traces to plot. The first trace is used to illustrate
        the target trace.
    '''
    fileformat = 'pdf' if latex_enabled() else 'svg'
    figure_width = 0.6
    figure_height = 0.6

    # Target trace
    figure = plot_traces((figure_width, 0.2), traces[:1])
    figure.savefig(f'sbi_target_trace_icon.{fileformat}')
    plt.close()

    # Sampled traces
    figure = plot_traces((figure_width, figure_height), traces)
    figure.savefig(f'sbi_sample_trace_icon.{fileformat}')
    plt.close()

    # Prior
    limits = np.array(
        [posterior.prior.support.base_constraint.lower_bound.numpy(),
         posterior.prior.support.base_constraint.upper_bound.numpy()]).T
    figure = plot_prior_icon((figure_width, figure_height), limits)
    figure.savefig(f'sbi_prior_icon.{fileformat}')
    plt.close()

    # Posterior
    figure = plot_posterior_icon((figure_width, figure_height), posterior)
    figure.savefig(f'sbi_posterior_icon.{fileformat}')
    plt.close()


def plot_traces(figsize: Tuple[float, float],
                traces: Sequence[neo.IrregularlySampledSignal]) -> plt.Figure:
    '''
    Plot the given target traces.

    Traces are offset to the top right and faded out.

    :param figsize: Size of the figure (width, height).
    :param traces: Traces to plot.
    :returns: Figure with plotted trace.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.01, 'right': 0.99,
                                        'top': 0.99, 'bottom': 0.01})
    for n_trace, sig in enumerate(traces):
        sig = sig - int(sig[:100].mean())
        sig += n_trace * 100
        sig = sig.time_shift(n_trace * 60 * pq.us)
        alpha = 1 - 0.25 * n_trace
        sig = sig[::50]
        ax.plot(sig.times, sig, color='k', alpha=alpha, lw=1)
        ax.axis('off')

    return fig


def plot_prior_icon(figsize: Tuple[float, float],
                    limits: np.ndarray) -> plt.Figure:
    '''
    Create a density plot of a multivariate Gaussian which is used to
    illustrate a prior distribution.

    :param figsize: Size of the figure (width, height).
    :param limits: Limits in which the distribution should be plotted.
    :returns: Figure with a multivariate Gaussian centered in the given
        limits.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.23, 'right': 0.99,
                                        'top': 0.99, 'bottom': 0.23})

    limits = torch.tensor(limits).float()
    loc = limits.mean(axis=1)
    cov = torch.eye(2) * limits.diff(axis=1) * 20

    dist = torch.distributions.MultivariateNormal(loc, cov)
    plot_posterior(ax, dist, rasterized=True, n_points=100,
                   limits=limits)
    _style_axis(ax)

    return fig


def plot_posterior_icon(figsize: Tuple[float, float], posterior) -> plt.Figure:
    '''
    Create a density plot of the given posterior distribution.

    :param figsize: Size of the figure (width, height).
    :param posterior: Posterior for which to display the probability
        distribution.
    :returns: Figure with a heat map of the posterior probability distribution.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.23, 'right': 0.99,
                                        'top': 0.99, 'bottom': 0.23})
    plot_posterior(ax, posterior, rasterized=True, n_points=100)
    _style_axis(ax)
    return fig


###############################################################################
# Helper
###############################################################################
def _style_axis(ax: plt.Axes) -> None:
    '''
    Style axis of probability distributions (prior/posterior).

    Remove ticks and spines; add x- and y-labels.
    :param ax: Axis to style.
    '''
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\theta_0$', labelpad=1)
    ax.set_ylabel(r'$\theta_1$', labelpad=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
