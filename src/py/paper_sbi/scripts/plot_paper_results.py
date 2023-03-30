#!/usr/bin/env python3
'''
Create all plots and log all data which are used in the publication
"Simulation-based Inference for Model Parameterization on Analog Neuromorphic
Hardware".
'''
import json
import logging
from pathlib import Path
import pickle

import neo
import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation

from paper_sbi.plotting.helper import apply_custom_styling
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

    # Specify where results are located
    length_experiment = experiment_folder.joinpath(
        '2d_transfroms_length', 'results', '0')
    heights_experiment = experiment_folder.joinpath(
        '2d_transfroms_amplitudes_first', 'results', '80')
    hd_folder = experiment_folder.joinpath('hd_targets', 'results')

    grid_search_df = pd.read_pickle(
        experiment_folder.joinpath('attenuation_grid_search.pkl'))

    # samples drawn from posterior
    pos_samples = {}
    pos_samples[Observation.LENGTH_CONSTANT] = pd.read_pickle(
        length_experiment.joinpath('posterior_samples_2.pkl'))
    pos_samples[Observation.AMPLITUDES_FIRST] = pd.read_pickle(
        heights_experiment.joinpath('posterior_samples_10.pkl'))

    hd_posterior_dfs = \
        [pd.read_pickle(hd_folder.joinpath(n_exp, 'posterior_samples_1.pkl'))
         for n_exp in ['0', '1']]

    target_df = pd.read_pickle(experiment_folder.joinpath(
        'targets', 'attenuation_variations_511_511.pkl'))
    original_parameters = target_df.attrs['parameters']

    # Membrane traces
    traces_path = experiment_folder.joinpath('example_traces').glob('*.pkl')
    blocks = np.array([neo.PickleIO(path.resolve()).read_block() for
                       path in traces_path])
    blocks = sorted(blocks,
                    key=lambda block: block.annotations['parameters'][0])
    example_traces = [block.segments[-1].irregularlysampledsignals[0] for
                      block in blocks]

    # Targets
    target_files = list(experiment_folder.joinpath('targets').glob('*.pkl'))
    amortized_targets = [pd.read_pickle(path) for path in target_files]

    # Posteriors
    with open(length_experiment.joinpath('posteriors.pkl'), 'rb') as handle:
        posterior_length = pickle.load(handle)[-1]

    with open(heights_experiment.joinpath('posteriors.pkl'), 'rb') as handle:
        amortized_posterior = pickle.load(handle)[0]

    # calibration
    with open(experiment_folder.joinpath('madc_calibration_result.json'),
              encoding='utf-8') as madc_file:
        madc_calib = json.load(madc_file)

    # SNPE flowchart
    create_icons(posterior_length, example_traces)

    # Experiment evaluation
    create_evaluation_plots(blocks[0].segments[-1].irregularlysampledsignals,
                            madc_calib['slope'])

    # Two-dimensional parameter space
    create_two_dim_plots(grid_search_df=grid_search_df,
                         example_traces=blocks,
                         posterior=posterior_length,
                         posterior_dfs=list(pos_samples.values()),
                         original_parameters=original_parameters[[0, -1]],
                         v_per_madc=madc_calib['slope'])
    log_results(pos_samples, target_df)

    # High-dimensional parameter space
    create_high_dim_plots(hd_posterior_dfs, target_df)

    # Appendix
    if args.plot_appendix:
        results_folder = experiment_folder.joinpath('2d_sim_amplitudes_first',
                                                    'results')

        folders = [results_folder.joinpath(str(i)) for i in range(10)]
        posterior_files = [f'posterior_samples_{i}.pkl' for i in range(10)]
        ppc_deviations = extract_deviation_from_files(folders, posterior_files)
        with open(results_folder.joinpath('5', 'posteriors.pkl'),
                  'rb') as handle:
            posteriors_good = pickle.load(handle)
        with open(results_folder.joinpath('1', 'posteriors.pkl'),
                  'rb') as handle:
            posteriors_bad = pickle.load(handle)

        transforms_folder = experiment_folder.joinpath(
            '2d_transfroms_amplitudes_first')
        overview_df = pd.read_pickle(
            transforms_folder.joinpath('parameter_space.pkl'))
        info_trans = get_group_infos(overview_df,
                                     transforms_folder.joinpath('results'))

        amortized_plot = AmortizedPosteriorPlot(posterior=amortized_posterior,
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
                              example_blocks=blocks,
                              madc_slope=madc_calib['slope'])

        log_median_distance(amortized_posterior, amortized_targets)

        sim_folder = experiment_folder.joinpath('arbor_simulations')
        grid_search_df_sim = pd.read_pickle(
            sim_folder.joinpath('attenuation_grid_search.pkl'))

        # Simulations
        # samples drawn from posterior
        pos_samples_sim = {}
        pos_samples_sim[Observation.LENGTH_CONSTANT] = pd.read_pickle(
            sim_folder.joinpath('2d_length', 'posterior_samples_2.pkl'))
        pos_samples_sim[Observation.AMPLITUDES_FIRST] = pd.read_pickle(
            sim_folder.joinpath('2d_amplitudes_first',
                                'posterior_samples_10.pkl'))

        # target
        target_df_sim = pd.read_pickle(sim_folder.joinpath(
            'attenuation_variations.pkl'))
        original_parameters_sim = target_df_sim.attrs['parameters']

        # posterior
        with open(sim_folder.joinpath('2d_length',
                                      'posteriors.pkl'), 'rb') as handle:
            posterior_sim = pickle.load(handle)[-1]

        # membrane traces
        traces_path_sim = sim_folder.joinpath('example_traces').glob('*.pkl')
        blocks_sim = np.array([neo.PickleIO(path.resolve()).read_block() for
                               path in traces_path_sim])
        blocks_sim = sorted(
            blocks_sim,
            key=lambda block: block.annotations['parameters'][0])

        create_simulation_plots(
            grid_search_df=grid_search_df_sim,
            example_traces=blocks_sim,
            posterior=posterior_sim,
            posterior_dfs=list(pos_samples_sim.values()),
            original_parameters=original_parameters_sim[[0, -1]])
        logging.info('Simulation Results: ')
        log_results(pos_samples_sim, target_df_sim)
