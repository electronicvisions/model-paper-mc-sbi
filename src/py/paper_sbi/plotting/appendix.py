'''
Plotting functions which created the plots used in the appendix.
'''
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import Tuple, Sequence, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import neo
import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation, exponential_decay, \
    fit_exponential, extract_psp_heights
from model_hw_mc_attenuation.bss import integration_bounds
from model_hw_mc_attenuation.extract import extract_observation

from paper_sbi.plotting.helper import latex_enabled, \
    formatted_parameter_names_bss, replace_latex, get_figure_width

from paramopt.plotting.posterior import plot_posterior, label_low_high
from paramopt.plotting.density import plot_2d_scatter


###############################################################################
# Data
###############################################################################
@dataclass
class HyperparameterInfo:
    '''
    Track information related to the hyperparameter search of the neural
    density estimator (NDE).
    The approximation was executed several times with the same parameters.
    Approximations with the same parameters were grouped. This information here
    represents information about these groups.

    :ivar indices: Experiment indices of different groups.
    :ivar n_transforms: Number of transformations used in group.
    :ivar n_hidden: Number of hidden neurons used in group.
    :ivar n_parameters: Number of model parameters of the estimator.
    '''
    indices: Sequence[Sequence[int]]
    n_transforms: Sequence[int]
    n_hidden: Sequence[int]
    n_parameters: Sequence[int]


@dataclass
class AmortizedPosteriorPlot:
    '''
    Combine all information needed to create the plot with an amortized
    posterior.

    :ivar posterior: Amortized posterior from which samples can be drawn for
        different observations.
    :ivar targets: DataFrames from which different targets are extracted.
    '''
    posterior: Any
    targets: Sequence[pd.DataFrame]


@dataclass
class PosteriorEvolutionPlot:
    '''
    Combine all information needed to create the plot which illustrates the
    evolution of two different posteriors.

    :ivar posteriors_good: Posteriors which evolve to a good approximation.
    :ivar posteriors_bad: Posteriors which evolve to a bad approximation.
    :ivar target_parameters: Parameters used to generate the target
        observation.
    '''
    posteriors_good: Sequence
    posteriors_bad: Sequence
    target_parameters: np.ndarray


@dataclass
class HyperparameterPlot:
    '''
    Combine all information needed to create the plot with illustrates the
    influence of hyperparameters of the neural density estimator on the
    approximation success.

    :ivar deviations: Mean deviation from the target observation. Different
        columns represent different trials, different rows different parameter
        configurations.
    :ivar info: Information about the different hyperparameter configurations.
        This info can be generated with :func:`get_group_infos`.
    '''
    deviations: np.ndarray
    info: HyperparameterInfo


###############################################################################
# Evaluation
###############################################################################
def log_median_distance(posterior,
                        targets: Sequence[pd.DataFrame],
                        n_samples: int = 10000):
    '''
    Calculate the mean distance of samples drawn from the posterior to the
    given targets.

    From the provided target DataFrames a target observation is extracted and
    samples are drawn from the amortized posterior given these targets. The
    distance between these samples and the parameters used to create the
    target observations is calculated.
    For each target the initial parameters and the median Euclidean distance
    between these parameters and the posterior samples is logged.

    :param posterior: Amortized posterior.
    :param targets: DataFrames from which targets can be extracted.
    :param n_samples: Number of samples to draw.
    '''
    for target in targets:
        distances = distance_of_posterior_samples(posterior, target,
                                                  Observation.AMPLITUDES_FIRST,
                                                  n_samples=n_samples)
        target_params = target.attrs['parameters'][[0, -1]]

        logging.info('Target parameter: %s, Median distance: %f',
                     target_params, np.median(distances))


###############################################################################
# Plotting
###############################################################################
def create_appendix_plots(
        *,
        amortized_plot: AmortizedPosteriorPlot,
        hyperparameter_plot: HyperparameterPlot,
        posterior_evolution_plot: PosteriorEvolutionPlot,
        ppc_deviations: np.ndarray,
        example_blocks: Sequence[neo.IrregularlySampledSignal],
        madc_slope: float):
    '''
    Create plots used in the appendix.

    :param amortized_plot: Data needed for the plot which displays samples
        drawn from an amortized posterior for different target observations.
    :param hyperparameter_plot: Data needed for the plot which illustrates
        how the approximation changes for different hyperparameters of the
        neural density estimator.
    :param posterior_evolution_plot: Data needed for the plot which illustrates
        how a good and bad approximation evolves over different rounds.
    :param ppc_deviation: Mean deviation from the target observation. Different
        columns represent different rounds, different rows different trials
        of the approximation. Used to illustrate how the mean deviation
        changes over rounds.
    :param example_blocks: Traces from which the PSP amplitudes are extracted
        and an exponential is fitted to.
    :param madc_slope: Characterization of the MADC. How much volt a single
        bit in an MADC measurement corresponds to.
    '''
    fileformat = 'pgf' if latex_enabled() else 'svg'
    figure_width = get_figure_width('single')

    # Amortized Posterior
    figure = plot_amortized_samples((figure_width, 2),
                                    amortized_plot.posterior,
                                    amortized_plot.targets)
    figure.savefig(f'posterior_samples_amortized.{fileformat}')
    plt.close()

    # Posterior evolution
    figure = plot_posterior_evolution(
        (figure_width, 1.5),
        posterior_evolution_plot.posteriors_good,
        posterior_evolution_plot.posteriors_bad,
        posterior_evolution_plot.target_parameters)
    figure.savefig(f'posterior_evolution.{fileformat}')
    plt.close()

    # PPC evolution
    figure = plot_ppc_evolution((figure_width, 1.5), ppc_deviations)
    figure.savefig(f'ppc_vs_sim.{fileformat}')
    plt.close()

    # Hyperparameter
    figure = plot_hyperparameter((figure_width, 2.5),
                                 hyperparameter_plot.deviations,
                                 hyperparameter_plot.info)
    figure.savefig(f'ppc_vs_trans.{fileformat}')
    plt.close()

    # Exponential Fits
    figure = plot_example_fits((figure_width, 1.8),
                               example_blocks, madc_slope)
    figure.savefig(f'exponential_fit.{fileformat}')
    plt.close()


def plot_posterior_evolution(figsize: Tuple[float, float],
                             posteriors_good: Sequence,
                             posteriors_bad: Sequence,
                             target_parameters: Tuple[float, float]
                             ) -> plt.Figure:
    '''
    Plot the posterior distribution for different rounds in the approximation.

    :param figsize: Size of the created figure (width, height).
    :param posterior_good: Posteriors which evolve to a good approximation.
    :param posterior_bad: Posteriors which evolve to a bad approximation.
    :param target_parameters: Parameters used to generate the target
        observation.
    :returns: Figure with the good evolution at the top and the bad evolution
        at the bottom.
    '''
    indices = range(3)
    fig, axs = plt.subplots(2, len(indices),
                            figsize=figsize,
                            sharex=True, sharey=True,
                            gridspec_kw={'left': 0.1, 'right': 0.995,
                                         'top': 0.87, 'bottom': 0.17,
                                         'wspace': 0.1, 'hspace': 0.3})

    meshes = _plot_posteriors(axs,
                              [posteriors_good[idx] for idx in indices],
                              [posteriors_bad[idx] for idx in indices])

    for ax in axs.flatten():
        ax.scatter(target_parameters[0], target_parameters[-1],
                   ec='w', lw=0.1, fc='r', s=7, marker='.')

    # hide ticks and labels
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for n_round, ax in zip(indices, axs[0]):
        ax.set_title(f'Round {n_round}')

    args_annotation = {'xy': (1.1, 0.5), 'xycoords': 'axes fraction',
                       'ha': 'left', 'va': 'center', 'ma': 'center'}
    axs[0, -1].annotate('good\napprox.', **args_annotation)
    axs[1, -1].annotate('poor\napprox.', **args_annotation)

    param_names = formatted_parameter_names_bss()
    if not latex_enabled():
        param_names = replace_latex(param_names)
    fig.supxlabel(param_names[0])
    fig.supylabel(param_names[1])

    color_bar = fig.colorbar(meshes[0], ax=axs, pad=0.18)
    color_bar.set_label(r'Posterior $p(\theta \mid x)$')
    label_low_high(color_bar)

    return fig


def plot_amortized_samples(figsize: Tuple[float, float],
                           posterior,
                           targets: Sequence[pd.DataFrame],
                           n_samples: int = 500) -> plt.Figure:
    '''
    Plot samples drawn from an amortized posterior for different target
    observations.

    :param figsize: Size of the figure (width, height).
    :param posterior: Amortized posterior from which samples can be drawn for
        different observations.
    :param targets: DataFrames from which different targets are extracted.
    :param n_samples: Numbers of samples to draw from the posterior for each
        target.
    :returns: Figure with samples drawn for different targets.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           gridspec_kw={'left': 0.3, 'right': 0.8,
                                        'top': 0.99, 'bottom': 0.17})

    for target_df in targets:
        target_obs = extract_observation(target_df,
                                         Observation.AMPLITUDES_FIRST).mean(0)
        samples = posterior.sample((n_samples,),
                                   x=target_obs).numpy().reshape(n_samples, -1)

        plot_2d_scatter(ax, samples[:, 0], samples[:, 1], c='C1', s=3,
                        alpha=0.3, ec='none')

        target_params = target_df.attrs['parameters']
        ax.scatter(target_params[0], target_params[-1], fc='k', marker='X',
                   s=20, ec='w')

    param_names = formatted_parameter_names_bss()
    if not latex_enabled():
        param_names = replace_latex(param_names)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])

    limits = np.array(
        [posterior.prior.support.base_constraint.lower_bound.numpy(),
         posterior.prior.support.base_constraint.upper_bound.numpy()]).T
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    return fig


def plot_ppc_evolution(figsize: Tuple[float, float],
                       deviations: np.ndarray) -> plt.Figure:
    '''
    Plot the result of a posterior predictive check.

    The figure illustrate the mean distance from the target for different
    rounds.

    :param figsize: Size of the figure (width, height).
    :param deviation: Mean deviation from the target observation. Different
        columns represent different rounds, different rows different trials
        of the approximation.
    :returns: Figure which shows how the mean deviation develops over rounds.
    '''
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.plot(deviations.T, linestyle='-', marker='', color='k', alpha=0.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean distance to\ntarget $E$')
    ax.xaxis.set_tick_params(which='minor', bottom=False)

    return fig


def plot_hyperparameter(figsize: Tuple[float, float],
                        deviations: np.ndarray,
                        info: HyperparameterInfo) -> plt.Figure:
    '''
    Plot mean distance to target for different hyperparameters of the neural
    density estimator.

    :param figsize: Size of the figure (width, height).
    :param deviation: Mean deviation from the target observation. Different
        columns represent different trials, different rows different parameter
        configurations.
    :param info: Information about the different hyperparameter configurations.
        This info can be generated with :func:`get_group_infos`.
    :returns: Figure which shows how the mean deviation varies between
        different hyperparameter configurations.
    '''
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for n_group, deviations_group in enumerate(deviations):
        ax.scatter(np.full_like(deviations_group, n_group, dtype=int),
                   deviations_group,
                   c=f'C{info.n_transforms[n_group]}',
                   alpha=0.6, s=15, lw=0)

    ax.set_xticks(range(len(deviations)),
                  list(zip(info.n_transforms, info.n_hidden)),
                  rotation=45, ha="right")
    ax.set_ylabel('Mean distance to target $E$')
    ax.set_xlabel('(transformations, hidden units)')
    ax.minorticks_off()

    ax_params = ax.twiny()
    ax_params.set_xlim(ax.get_xlim())
    ax_params.set_xticks(range(len(deviations)), info.n_parameters,
                         rotation=45, ha="left")
    ax_params.set_xlabel('Number of parameters')
    ax_params.tick_params(axis='x', which='minor', bottom=False)
    ax_params.minorticks_off()

    return fig


def plot_example_fits(figsize: Tuple[float, float],
                      traces: Sequence[neo.Block],
                      v_per_madc: float) -> plt.Figure:
    '''
    Fit exponential to the PSP amplitudes in the given traces and display them.

    The exponential are fitted to the response to an input in the first
    compartment.

    :param figsize: Size of the figure (width, height).
    :param traces: Traces from which the PSP amplitudes are extracted and an
        exponential is fitted to.
    :param v_per_madc: Characterization of the MADC. How much volt a single
        bit in an MADC measurement corresponds to.
    :returns: Figure which the extracted PSP amplitudes and exponentials fit
        to them.
    '''
    fig, ax = plt.subplots(figsize=figsize,
                           tight_layout=True)
    for n_trace, block in enumerate(traces):
        sigs = block.segments[-1].irregularlysampledsignals
        heights = extract_psp_heights(sigs)[:, 0]
        compartments = np.arange(len(heights))

        x_fit = np.linspace(compartments[0], compartments[-1], 100)
        assert block.annotations['experiment'] == 'attenuation_bss'
        y_fit = exponential_decay(
            x_fit, *fit_exponential(heights, bounds=integration_bounds))

        line = ax.plot(x_fit, y_fit * v_per_madc, alpha=0.5)[0]
        ax.scatter(compartments, heights * v_per_madc, c=line.get_color(),
                   label=str(n_trace), s=3, zorder=2.1)

    title = r'$\textbf{Location}$'
    ax.legend(title=title if latex_enabled() else replace_latex(title))

    ax.set_xticks(compartments)

    ax.set_xlabel('Compartment (height)')
    ax.set_ylabel(r'$\text{PSP Height}\,/\,\si{V}$' if latex_enabled()
                  else 'PSP Height / V')

    ax.set_xlim(-0.1, compartments[-1] + 0.1)
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.set_xticklabels(['0\n($h_{00}$)', '1\n($h_{01}$)',
                        '2\n($h_{02}$)', '3\n($h_{03}$)'])
    return fig


###############################################################################
# Helper
###############################################################################
def distance_of_posterior_samples(posterior,
                                  amplitudes: np.ndarray,
                                  observation: Observation,
                                  n_samples: int = 10000) -> np.ndarray:
    '''
    Calculate the distance of posterior samples from the initial parameters.

    :param posterior: Posterior from which samples are drawn. Needs to be
        amortized, i.e. samples can be drawn for different target observations.
    :param amplitudes: Measured PSP amplitudes form which a target observation
        is extracted.
    :param observation: Observation to extract from the amplitudes. Needs to
        match the type of observation for which the posterior was trained.
    :param n_samples: Number of samples to draw from the posterior.
    :returns: Euclidean distance between the samples drawn from the posterior
        and the initial parameters (parameters used to record the target
        amplitudes).
    '''
    target = extract_observation(amplitudes, observation).mean(0)
    target_params = amplitudes.attrs['parameters']

    samples = posterior.sample((n_samples,), x=target).numpy().reshape(
        n_samples, -1)
    if samples.shape[1] == 2:
        # global parameters
        target_params = target_params[[0, -1]]

    return np.linalg.norm(samples - target_params, axis=-1)


def extract_deviations_dataframe(results_folder: Path,
                                 indices: Sequence[Sequence[int]]
                                 ) -> np.ndarray:
    '''
    Extract the mean distance to the target observation for the given
    experiments.

    The experiments are given by the indices. Each experiment should
    have a folder named after its index in `results_folder`. In this folder a
    file `abc_samples.pkl` with samples drawn during the approximation and a
    file `posterior_samples_{last_idx}.pkl` with samples drawn from the last
    approximated posterior (and amplitudes recorded with these parameters)
    should be located.

    :param results_folder: Folder which contains folders for the different
        experiments.
    :param indices: Indices of the different experiments. They are assumed to
        be organized in groups. Compare :func:`get_group_infos`.
    :returns: Mean distance of the observations from the posterior samples to
        the target observation. Different rows represent different groups.
        Different columns different experiments within the same group.
    '''
    deviations = []
    for exp_indices in indices:
        deviations_group = []
        for n_exp in exp_indices:
            folder = results_folder.joinpath(str(n_exp))
            n_last = pd.read_pickle(
                folder.joinpath('abc_samples.pkl'))['round'].max()

            samples = pd.read_pickle(folder.joinpath(
                f'posterior_samples_{n_last}.pkl'))
            deviation = np.mean(get_distance_to_target_obs(
                samples['amplitudes'], samples.attrs['target'],
                Observation[samples.attrs['observation']]))
            deviations_group.append(deviation)
        deviations.append(deviations_group)
    return np.array(deviations)


def get_group_infos(overview: pd.DataFrame, results_folder: Path
                    ) -> HyperparameterInfo:
    '''
    Group the experiments in the given DataFrame and gather group information.

    The experiments are grouped by the `n_transforms` and `n_hidden`.

    :param overview: DataFrame with the different experiments. Needs to have
        the columns `n_transforms` and `n_hidden`. The index of the DataFrame
        is assumed to be the index of the experiment. Each experiment should
        have a folder named after its index in `results_folder`. In this
        folder a file `posteriors.pkl` with the approximated posterior is
        assumed.
    :param results_folder: Folder which contains folders for the different
        experiments.
    :returns: Information about the different groups.
    '''
    columns = ['n_transforms', 'n_hidden']
    groups = overview.groupby(columns)

    # extract number of model parameters from first experiment in group
    n_params = []
    for idxs in list(groups.indices.values()):
        with open(results_folder.joinpath(str(idxs[0]), 'posteriors.pkl'),
                  'rb') as handle:
            pos = pickle.load(handle)[0]
        n_params.append(
            sum(p.numel() for p in pos.posterior_estimator.parameters()))

    keys = np.array(list(groups.groups.keys()))
    return HyperparameterInfo(indices=list(groups.indices.values()),
                              n_transforms=keys[:, 0].tolist(),
                              n_hidden=keys[:, 1].tolist(),
                              n_parameters=n_params)


def extract_deviation_from_files(folders: Sequence[Path],
                                 posterior_files: Sequence[str]
                                 ) -> np.ndarray:
    '''
    Extract the mean deviation from the target from the given files.

    The files are found by iterating through all folders and all posterior
    file names.

    :param folders: Folders in which to search for posterior samples and
        their observation.
    :param posterior_files: Name of the files with posterior samples and
        observations.
    :returns: Mean deviation from the different posterior observations to
        the target observation. Different folders are in different rows.
        Different files in different columns.
    '''
    deviations = []
    for folder in folders:
        deviations_folder = []
        for posterior_file in posterior_files:
            samples = pd.read_pickle(folder.joinpath(posterior_file))
            deviation = np.mean(get_distance_to_target_obs(
                samples['amplitudes'], samples.attrs['target'],
                Observation[samples.attrs['observation']]))
            deviations_folder.append(deviation)
        deviations.append(deviations_folder)

    return np.array(deviations)


def get_distance_to_target_obs(amplitudes: pd.DataFrame,
                               target: np.ndarray,
                               observation: Observation) -> np.ndarray:
    '''
    Calculate Euclidean distance between the target observation and
    observations measured for the given amplitudes.

    :param amplitudes: DataFrame with amplitudes measured from an attenuation
        experiment.
    :param target: Target.
    :param observation: Observation to extract from `amplitudes`.
    :returns: Euclidean distance between the target and the observations
        extracted from amplitudes.
    '''
    measured = extract_observation(amplitudes, observation)
    return np.linalg.norm(measured - target, axis=-1)


def _plot_posteriors(axs: np.ndarray,
                     posteriors_top: Sequence,
                     posteriors_bot: Sequence
                     ) -> Sequence[mpl.collections.QuadMesh]:
    '''
    Plot the given posteriors in the axes grid.

    The color values of the different posterior plots are adjusted to all have
    the same range.

    :param axs: Axes in which to plot the posteriors. The axes should be
        organized in two rows and the number of columns should fit the number
        of posteriors to plot.
    :param posteriors_top: Posteriors displayed at the top.
    :param posteriors_bot: Posteriors displayed at the bottom.
    :returns: Artists created by :meth:`plt.Axes.pcolormesh`.
    '''
    assert len(posteriors_top) == len(posteriors_bot)
    assert len(posteriors_top) == len(axs[0])

    meshes = []
    for ax_col, good_pos, bad_pos in zip(axs.T, posteriors_top,
                                         posteriors_bot):
        meshes.append(plot_posterior(ax_col[0], good_pos, rasterized=True))
        meshes.append(plot_posterior(ax_col[1], bad_pos, rasterized=True))

    # set z-scales to same values
    limits = [mesh.get_clim() for mesh in meshes]
    norm = Normalize(np.min(limits), np.max(limits))
    for mesh in meshes:
        mesh.set_norm(norm)

    return meshes
