#!/usr/bin/env python3
from itertools import product
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.utils import posterior_nn


from model_hw_mc_attenuation.helper import get_license_and_chip
from model_hw_mc_attenuation import Observation, fit_length_constant, \
    AttenuationExperiment
from model_hw_mc_attenuation.extract import extract_observation, \
    get_experiment, get_bounds

from paramopt.sbi import Algorithm, perform_sequential_estimation, \
    perform_mcabc


def get_evaluation_function(experiment: AttenuationExperiment,
                            observation: Observation,
                            bounds: Optional[Tuple] = None) -> Callable:
    '''
    Define a function which extract the given observation from the given
    experiment.

    The returned function takes the parameterization of the experiment as an
    input and returns an observation

    :param experiment: Experiment used to record the observation.
    :param observation: Type of observation to return by the function.
    :param bounds: Bounds for exponential fit.
    :returns: Function which executes an experiment and returns the given
        observation.
    '''

    if observation == Observation.AMPLITUDES:
        def func_amplitudes(params: torch.Tensor) -> np.ndarray:
            return experiment.measure_response(np.asarray(params)).flatten()
        return func_amplitudes

    if observation == Observation.AMPLITUDES_FIRST:
        def func_amplitudes_single(params: torch.Tensor) -> np.ndarray:
            return experiment.measure_response(np.asarray(params))[:, 0]
        return func_amplitudes_single

    if observation == Observation.LENGTH_CONSTANT:
        def func_length_constant(params: torch.Tensor) -> np.ndarray:
            data = experiment.measure_response(np.asarray(params))
            return np.array([fit_length_constant(data[:, 0], bounds=bounds)])
        return func_length_constant

    raise ValueError(f'The observation "{observation}" is not supported.')


def _get_epsilon(chain_length: int, observation: Observation,
                 target: np.ndarray) -> float:
    '''
    Determine epsilon for rejection based SBI algorithms.

    :param chain_length: Length of the chain with which the experiment is
        performed.
    :param observation: Type of observation extracted form the experiment.
    :param target: Target observation used by the SBI algorithm.
    :returns: Function which executes an experiment and returns the given
        observation.
    '''
    if observation == Observation.AMPLITUDES:
        return chain_length**2 * 3
    if observation == Observation.AMPLITUDES_FIRST:
        return chain_length * 3
    if observation == Observation.LENGTH_CONSTANT:
        return target * 0.05

    raise ValueError(f'The observation "{observation}" is not supported.')


def _get_column_names(parameter_names: List[str],
                      observation: Observation,
                      chain_length: int) -> pd.MultiIndex:
    '''
    Determine the column names of the DataFrame with the samples drawn during
    SBI.

    :param parameter_names: Names of the parameters configured during the
        experiment.
    :param observation: Type of observation which will be logged in the
        DataFrame.
    :returns: Column names for a DataFrame.
    '''
    col_index = pd.MultiIndex.from_product([['parameters'], parameter_names])

    if observation == Observation.AMPLITUDES:
        cols = [f"A_{i}{j}" for i, j
                in product(range(chain_length), range(chain_length))]
        col_index = col_index.append(pd.MultiIndex.from_product(
            [[observation.name.lower()], cols]))
    elif observation == Observation.AMPLITUDES_FIRST:
        cols = [f'A_{i}0' for i in range(chain_length)]
        col_index = col_index.append(pd.MultiIndex.from_product(
            [[observation.name.lower()], cols]))
    elif observation == Observation.LENGTH_CONSTANT:
        col_index = col_index.append(
            pd.MultiIndex.from_tuples([(observation.name.lower(), '')]))

    return col_index.append(pd.MultiIndex.from_tuples([('round', '')]))


def main(target_data: pd.DataFrame,
         observation: Observation,
         algorithm: Algorithm, *,
         simulations: Optional[List[int]] = None,
         density_estimator: Union[str, Callable] = 'maf',
         global_parameters: bool = False) -> Tuple[np.ndarray, List]:
    '''
    Approximate the posterior distribution conditioned on the given target.

    This function supports sequential neural density estimation algorithms as
    well as Monte-Carlo ABC.


    :param target_data: DataFrame from which the target is extracted on which
        the posterior is condition on.
    :param observation: Type of observation to extract from the target data.
    :param algorithm: Approximation algorithm.
    :param simulations: List of simulations in each round.
    :param density_estimator: Neural density estimator for the SNPE algorithm.
    :param global_parameters: Configure the leak and inter-compartment
        conductance globally, i.e. use the same values for all compartments.
    :returns: Tuple of array with sample information and List of posteriors for
        each round. The array contains the proposal samples drawn in each
        round, the observations and the corresponding round. The list of
        posteriors is empty in case of Monte-Carlo ABC.
    '''
    attenuation_exp = get_experiment(target_data)
    limits = attenuation_exp.default_limits
    param_names = attenuation_exp.parameter_names(global_parameters)
    target_obs = extract_observation(target_data, observation).mean(0)

    evaluation_function = get_evaluation_function(
        attenuation_exp, observation, bounds=get_bounds(target_data))

    if global_parameters:
        limits = limits[[0, -1]]

    # Prepare inference
    prior = utils.BoxUniform(low=limits[:, 0].flatten(),
                             high=limits[:, 1].flatten())

    if algorithm == Algorithm.MCABC:
        if simulations is not None and len(simulations) > 1:
            raise RuntimeError('MCABC only supports single round inference.')

        log = perform_mcabc(
            prior, evaluation_function, target_obs,
            eps=_get_epsilon(attenuation_exp.length, observation, target_obs),
            simulations=simulations[0])
        # Only a single round -> log round 0
        log = np.hstack([log, np.full((len(log), 1), 0)])
        posteriors = []
    else:
        log, posteriors = perform_sequential_estimation(
            algorithm, prior, evaluation_function, target_obs,
            simulations=simulations,
            density_estimator=density_estimator)

    data = pd.DataFrame(log,
                        columns=_get_column_names(param_names, observation,
                                                  attenuation_exp.length))
    data = data.astype({('round', ''): 'int32'})

    data.attrs['limits'] = limits
    data.attrs['length'] = attenuation_exp.length
    data.attrs['observation'] = observation.name
    data.attrs['algorithm'] = algorithm.name
    data.attrs['target'] = target_obs
    data.attrs['chip_id'] = get_license_and_chip()
    data.attrs['experiment'] = target_data.attrs['experiment']

    return data, posteriors


if __name__ == '__main__':
    import argparse
    import pickle
    import yaml

    parser = argparse.ArgumentParser(
        description="Perform an SBI algorithm to approximate a posterior."
                    "The posterior will be restricted to a given target "
                    "observation if several rounds of a sequential algorithm "
                    "are executed or a rejection based algorithm is used.")
    parser.add_argument("configuration",
                        help="Path to YAML file with configuration.",
                        type=str)
    args = parser.parse_args()
    with open(args.configuration, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    target_df = pd.read_pickle(config['target'])

    n_simulations = [config['sbi']['n_sim_first']] + \
        [config['sbi']['n_sim_rest']] * (config['sbi']['n_rounds'] - 1)

    # Density estimator network
    density_nn = posterior_nn(model=config['nde']['model'],
                              z_score_x=config['nde']['z_score'],
                              num_transforms=config['nde']['n_transforms'],
                              hidden_features=config['nde']['n_hidden'])

    df, approx_pos = main(target_df,
                          Observation[config['observation'].upper()],
                          Algorithm[config['sbi']['algorithm']],
                          simulations=n_simulations,
                          density_estimator=density_nn,
                          global_parameters=config['global_parameters'])
    df.attrs['target_file'] = str(Path(config['target']).resolve())

    df.to_pickle('sbi_samples.pkl')
    if len(approx_pos) > 0:
        with open(r"posteriors.pkl", "wb") as output_file:
            pickle.dump(approx_pos, output_file)
