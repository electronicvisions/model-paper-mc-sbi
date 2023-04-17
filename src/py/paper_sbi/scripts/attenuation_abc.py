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

from paramopt.abc import Algorithm, perform_sequential_estimation, \
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
    Determine epsilon for rejection based ABC algorithms.

    :param chain_length: Length of the chain with which the experiment is
        performed.
    :param observation: Type of observation extracted form the experiment.
    :param target: Target observation used by the ABC algorithm.
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
    ABC.

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

    parser = argparse.ArgumentParser(
        description="Perform an ABC algorithm to approximate a posterior."
                    "The posterior will be restricted to a given target "
                    "observation if several rounds of a sequential algorithm "
                    "are executed or a rejection based algorithm is used.")
    parser.add_argument("target",
                        help="Path to pickled DataFrame which contains "
                             "amplitudes of an attenuation experiment. The "
                             "mean over different runs will be used as a "
                             "target.",
                        type=str)
    parser.add_argument("-observation",
                        help="Determines what kind of observation is "
                             "extracted from the attenuation experiment and "
                             "the provided target",
                        type=str,
                        default=Observation.LENGTH_CONSTANT.name.lower(),
                        choices=[Observation.LENGTH_CONSTANT.name.lower(),
                                 Observation.AMPLITUDES.name.lower(),
                                 Observation.AMPLITUDES_FIRST.name.lower()])

    parser.add_argument("-algorithm",
                        help="Algorithm used to approximate a posterior.",
                        type=str,
                        default=Algorithm.SNPE.name,
                        choices=[alg.name for alg in Algorithm])
    parser.add_argument("-n_sim_first",
                        help="Number of simulations in first approximation "
                             "round.",
                        type=int,
                        default=500)
    parser.add_argument("-n_sim_rest",
                        help="Number of simulations in remaining "
                             "approximation rounds.",
                        type=int,
                        default=200)
    parser.add_argument("-n_rounds",
                        help="Number of approximation rounds. Needs to be '1' "
                             "for non-sequential algorithms.",
                        type=int,
                        default=2)

    parser.add_argument("-neural_estimator",
                        help="Type of neural density estimator.",
                        type=str,
                        choices=['maf', 'nsf', 'mdn'],
                        default='maf')
    parser.add_argument("-n_transforms",
                        help="Number of transforms for flow-based neural "
                             "density estimators.",
                        type=int,
                        default=5)
    parser.add_argument("-n_hidden",
                        help="Number of features in hidden layers of neural "
                             "density estimators.",
                        type=int,
                        default=50)
    parser.add_argument("-z_score",
                        help="How the observations are z-scored.",
                        type=str,
                        choices=['none', 'independent', 'structured'],
                        default='independent')

    parser.add_argument("-seed",
                        help="Random seed for pytorch and numpy.",
                        type=int)
    parser.add_argument('--global_parameters',
                        help="Use the same leak conductance in all "
                             "compartments and the same conductance between "
                             "compartments for all connections between "
                             "compartments.",
                        action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    target_df = pd.read_pickle(args.target)

    n_simulations = [args.n_sim_first] + \
        [args.n_sim_rest] * (args.n_rounds - 1)

    # Density estimator network
    density_nn = posterior_nn(model=args.neural_estimator,
                              z_score_x=args.z_score,
                              num_transforms=args.n_transforms,
                              hidden_features=args.n_hidden)

    df, approx_pos = main(target_df,
                          Observation[args.observation.upper()],
                          Algorithm[args.algorithm],
                          simulations=n_simulations,
                          density_estimator=density_nn,
                          global_parameters=args.global_parameters)
    df.attrs['target_file'] = str(Path(args.target).resolve())

    df.to_pickle('abc_samples.pkl')
    if len(approx_pos) > 0:
        with open(r"posteriors.pkl", "wb") as output_file:
            pickle.dump(approx_pos, output_file)
