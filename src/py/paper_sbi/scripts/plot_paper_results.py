#!/usr/bin/env python3
'''
Create all plots and log all data which are used in the publication
"Simulation-based Inference for Model Parameterization on Analog Neuromorphic
Hardware".
'''
from dataclasses import dataclass
from typing import Sequence
import json
import logging
from pathlib import Path
import pickle
import re
import yaml

import neo
import numpy as np
import pandas as pd

from paper_sbi.plotting.expected_coverage import ExpectedCoverageData
from paper_sbi.plotting.helper import apply_custom_styling, \
    DataSingleObservation
from paper_sbi.plotting.snpe_flowchart import create_icons
from paper_sbi.plotting.appendix import \
    create_appendix_plots, extract_deviation_from_files, \
    extract_deviations_dataframe, get_group_infos, log_median_distance, \
    AmortizedPosteriorPlot, PosteriorEvolutionPlot, HyperparameterPlot
from paper_sbi.plotting.experiment_evaluation import create_evaluation_plots
from paper_sbi.plotting.two_dimensional import create_two_dim_plots, \
    log_results
from paper_sbi.plotting.high_dimensional import create_high_dim_plots
from paper_sbi.plotting.arbor_simulations import create_simulation_plots


@dataclass
class ExperimentLocator:
    '''
    Save path to results folder and list of experiments which were used to
    build the ensemble.

    :ivar folder: Relative pasth to results folder.
    :ivar indices: Experiment indices which were used to build ensemble.
    '''
    folder: str
    indices: Sequence[int]


@dataclass
class PPCEvolution:
    '''
    Save path to results folder with data needed to display the PPC evolution.

    :ivar folder: Relative pasth to results folder.
    :ivar indices: Indices of experiments used for evolution plot
    :ivar posterior_good: Id of experiment with good posterior approximation.
    :ivar posterior_bad: Id of experiment with bad posterior approximation.
    '''
    folder: str
    indices: Sequence[int]
    posterior_good: int
    posterior_bad: int


# Data locations (relative to base folder)
GRIDSEARCH = 'attenuation_grid_search.pkl'
GRIDSEARCH_ARBOR = 'arbor_simulations/attenuation_grid_search.pkl'
EXAMPLE_TRACES = 'example_traces'
EXAMPLE_TRACES_ARBOR = 'arbor_simulations/example_traces'
MADC_CALIB = 'madc_calibration_result.json'
TARGETS = 'targets'
TARGET = 'targets/attenuation_variations_511_511.pkl'
TARGET_ARBOR = 'arbor_simulations/attenuation_variations.pkl'

LENGTH_2D = ExperimentLocator(
    '2d_parameter_space/nde_length_constant/results', range(0, 5))
AMPLITUDES_2D = ExperimentLocator(
    '2d_parameter_space/nde_amplitudes_first/results', range(80, 85))
AMP_FIRST_HD = ExperimentLocator(
    '7d_parameter_space/observations/results', range(0, 5))
AMP_ALL_HD = ExperimentLocator(
    '7d_parameter_space/observations/results', range(10, 15))
LENGTH_2D_ARBOR = ExperimentLocator(
    'arbor_simulations/length_constant/results', range(0, 5))
AMPLITUDES_2D_ARBOR = ExperimentLocator(
    'arbor_simulations/amplitudes_first/results', range(0, 5))

NDE_HYPERPARAMETER = '2d_parameter_space/nde_amplitudes_first'
PPC_EVOLUTION = PPCEvolution(
    '2d_parameter_space/sim_amplitudes_first/results', range(10), 2, 1)


def _get_last_posterior_samples(folder: Path) -> pd.DataFrame:
    candidate_files = [file.absolute() for file in folder.glob(
        'posterior_samples_[0-9]*.pkl')]
    idx_of_highest_posterior = np.argmax(
        [re.findall('posterior_samples_([0-9]*).pkl', str(candidate))[0] for
         candidate in candidate_files])
    samples_file = candidate_files[idx_of_highest_posterior]
    return pd.read_pickle(samples_file)


def _get_coverage_data(folder: Path) -> ExpectedCoverageData:
    with open(folder.joinpath('posteriors.pkl'), 'rb') as file_handle:
        posterior = pickle.load(file_handle)[0]
    prior_samples = pd.read_pickle(folder.joinpath('prior_samples.pkl'))
    sbc_samples = pd.read_pickle(folder.joinpath('sbc_samples.pkl'))
    return ExpectedCoverageData(prior_samples, sbc_samples, posterior)


def _get_experiment_data(folder: Path, indices: Sequence[int]
                         ) -> DataSingleObservation:
    '''
    Get all data relevant for plotting.

    :param folder: Path to results folder.
    :param indices: Indices of experiments which were used for the ensemble.
    '''

    ensemble_folder = folder.joinpath('ensembles',
                                      f'exp_{indices[0]}_to_{indices[-1]}')
    with open(ensemble_folder.joinpath('posteriors.pkl'), 'rb') as file_handle:
        posteriors = pickle.load(file_handle)
    posterior_samples = _get_last_posterior_samples(ensemble_folder)

    single_folders = [folder.joinpath(str(exp)) for exp in indices]

    # Coverage is not needed for all experiments -> set to NOne if not found
    try:
        coverage_ensemble = _get_coverage_data(ensemble_folder)
    except FileNotFoundError:
        coverage_ensemble = None
    try:
        coverage_single = [_get_coverage_data(fol) for fol in single_folders]
    except FileNotFoundError:
        coverage_single = None

    return DataSingleObservation(posteriors=posteriors,
                                 posterior_samples=posterior_samples,
                                 coverage_ensemble=coverage_ensemble,
                                 coverage_single=coverage_single)


def _get_parameter_space(spec_file: Path) -> pd.DataFrame:
    '''
    Extract the parmaeter space from an experiment specification.

    :param spec_file: Path to yaml file with experiment specification.
    :returns: Parameter space.
    '''
    with open(spec_file, 'r', encoding='utf-8') as file_handle:
        spec = yaml.safe_load(file_handle)
    space_dict = spec['parameter_space']
    data_frame = pd.DataFrame(
        space_dict['data'],
        columns=pd.MultiIndex.from_tuples(space_dict['columns']))
    return data_frame['parameters']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Create all plots presented in the publication "
                    "\"Simulation-based Inference for Model Parameterization  "
                    "on Analog Neuromorphic Hardware\".")
    parser.add_argument("experiment_folder",
                        help="Folder in which the experiment results are "
                             "located. The data can be downloaded from "
                             "heiData once the submission process is "
                             "finished.",
                        type=str)
    parser.add_argument('--plot_appendix',
                        help="Plot the figures presented in the appendix.",
                        action='store_true')
    parser.add_argument('--use_latex',
                        help="Render text in figures with latex.",
                        action='store_true')
    args = parser.parse_args()
    experiment_folder = Path(args.experiment_folder)

    logging.basicConfig(level=logging.INFO)
    apply_custom_styling(args.use_latex)

    # Load data
    data_2d_length = _get_experiment_data(
        experiment_folder.joinpath(LENGTH_2D.folder), LENGTH_2D.indices)
    data_2d_amplitudes = _get_experiment_data(
        experiment_folder.joinpath(AMPLITUDES_2D.folder),
        AMPLITUDES_2D.indices)
    data_hd_amp_first = _get_experiment_data(
        experiment_folder.joinpath(AMP_FIRST_HD.folder), AMP_FIRST_HD.indices)
    data_hd_amp_all = _get_experiment_data(
        experiment_folder.joinpath(AMP_ALL_HD.folder), AMP_ALL_HD.indices)

    grid_search_df = pd.read_pickle(experiment_folder.joinpath(GRIDSEARCH))
    target_df = pd.read_pickle(experiment_folder.joinpath(TARGET))
    original_parameters = target_df.attrs['parameters']

    # Membrane traces
    traces_path = experiment_folder.joinpath(EXAMPLE_TRACES).glob('*.pkl')
    blocks = np.array([neo.PickleIO(path.resolve()).read_block() for
                       path in traces_path])
    blocks = sorted(blocks,
                    key=lambda block: block.annotations['parameters'][0])
    example_traces = [block.segments[-1].irregularlysampledsignals[0] for
                      block in blocks]

    # Targets
    target_files = list(experiment_folder.joinpath(TARGETS).glob('*.pkl'))
    amortized_targets = [pd.read_pickle(path) for path in target_files]

    # calibration
    with open(experiment_folder.joinpath(MADC_CALIB),
              encoding='utf-8') as madc_file:
        madc_calib = json.load(madc_file)

    # SNPE flowchart
    create_icons(data_2d_length.posteriors[-1], example_traces)

    # Experiment evaluation
    create_evaluation_plots(blocks,
                            grid_search_df,
                            madc_calib['slope'])

    # Two-dimensional parameter space
    create_two_dim_plots(grid_search_df=grid_search_df,
                         example_traces=blocks,
                         data_length_constant=data_2d_length,
                         data_amplitudes=data_2d_amplitudes,
                         original_parameters=original_parameters[[0, -1]],
                         v_per_madc=madc_calib['slope'])
    log_results(data_2d_length.posterior_samples,
                data_2d_amplitudes.posterior_samples,
                target_df)

    # High-dimensional parameter space
    create_high_dim_plots(data_amp_first=data_hd_amp_first,
                          data_amp_all=data_hd_amp_all,
                          target_df=target_df)

    # Appendix
    if args.plot_appendix:
        ppc_folder = experiment_folder.joinpath(PPC_EVOLUTION.folder)
        folders = [ppc_folder.joinpath(str(i)) for i in PPC_EVOLUTION.indices]
        posterior_files = [f'posterior_samples_{i}.pkl' for i in range(10)]
        ppc_deviations = extract_deviation_from_files(folders, posterior_files)
        with open(ppc_folder.joinpath(str(PPC_EVOLUTION.posterior_good),
                                      'posteriors.pkl'), 'rb') as handle:
            posteriors_good = pickle.load(handle)
        with open(ppc_folder.joinpath(str(PPC_EVOLUTION.posterior_bad),
                                      'posteriors.pkl'), 'rb') as handle:
            posteriors_bad = pickle.load(handle)

        transforms_folder = experiment_folder.joinpath(NDE_HYPERPARAMETER)
        overview_df = _get_parameter_space(
            transforms_folder.joinpath('experiment_specification.yaml'))
        info_trans = get_group_infos(overview_df,
                                     transforms_folder.joinpath('results'))

        amortized_plot = AmortizedPosteriorPlot(
            posterior=data_2d_amplitudes.posteriors[0],
            targets=amortized_targets)
        hyperparameter_plot = HyperparameterPlot(
            deviations=extract_deviations_dataframe(
                transforms_folder.joinpath('results'), info_trans.indices),
            info=info_trans)
        posterior_evo_plot = PosteriorEvolutionPlot(
            posteriors_good=posteriors_good, posteriors_bad=posteriors_bad,
            target_parameters=original_parameters)

        create_appendix_plots(amortized_plot=amortized_plot,
                              hyperparameter_plot=hyperparameter_plot,
                              posterior_evolution_plot=posterior_evo_plot,
                              ppc_deviations=ppc_deviations,
                              data_hd_amp_first=data_hd_amp_first,
                              data_hd_amp_all=data_hd_amp_all,
                              target_df=target_df,
                              example_blocks=blocks,
                              madc_slope=madc_calib['slope'])

        log_median_distance(data_2d_amplitudes.posteriors[0],
                            amortized_targets)

        # Arbor Simulations
        grid_search_df_sim = pd.read_pickle(
            experiment_folder.joinpath(GRIDSEARCH_ARBOR))
        data_2d_length_arbor = _get_experiment_data(
            experiment_folder.joinpath(LENGTH_2D_ARBOR.folder),
            LENGTH_2D_ARBOR.indices)
        data_2d_amplitudes_arbor = _get_experiment_data(
            experiment_folder.joinpath(AMPLITUDES_2D_ARBOR.folder),
            AMPLITUDES_2D_ARBOR.indices)

        # target
        target_df_sim = pd.read_pickle(experiment_folder.joinpath(
            TARGET_ARBOR))
        original_parameters_sim = target_df_sim.attrs['parameters']

        # membrane traces
        traces_path_sim = experiment_folder.joinpath(
            EXAMPLE_TRACES_ARBOR).glob('*.pkl')
        blocks_sim = np.array([neo.PickleIO(path.resolve()).read_block() for
                               path in traces_path_sim])
        blocks_sim = sorted(
            blocks_sim,
            key=lambda block: block.annotations['parameters'][0])

        create_simulation_plots(
            grid_search_df=grid_search_df_sim,
            example_traces=blocks_sim,
            data_length_constant=data_2d_length_arbor,
            data_amplitudes=data_2d_amplitudes_arbor,
            original_parameters=original_parameters_sim[[0, -1]])
        logging.info('Simulation Results: ')
        log_results(data_2d_length_arbor.posterior_samples,
                    data_2d_amplitudes_arbor.posterior_samples,
                    target_df_sim)
