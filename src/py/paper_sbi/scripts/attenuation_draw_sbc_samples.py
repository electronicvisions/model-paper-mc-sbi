#!/usr/bin/env python3
from typing import Tuple
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.extract import extract_observation

from paramopt.abc.sbc import sbc_data


def extract_arrays(sbc_df: pd.DataFrame,
                   samples_df: pd.DataFrame
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Extract arrays with the original parameters, the observations and
    posterior samples from data frames.

    :param sbc_df: Data frame with samples drawn for SBC.
    :param samples_Df: Datam frame which was used to extract observations for
        the SBC samples.
    :return: Original parameters, observations and posterior samples.
    '''
    groups = sbc_df.groupby("obs_id")
    obs_ids = list(groups.groups.keys())

    sbc_samples = []
    for _, group in groups:
        sbc_samples.append(group['parameters'].values)
    sbc_samples = np.array(sbc_samples)

    observations = extract_observation(
        samples_df['amplitudes'].iloc[obs_ids],
        Observation[sbc_df.attrs['observation'].upper()])

    return samples_df['parameters'].iloc[obs_ids].values, \
        observations, sbc_samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Draw samples from the provided posterior, conditioned "
                    "on different observations. The samples are saved in a "
                    "DataFrame.")
    parser.add_argument("posterior_file",
                        type=str,
                        help="Path to pickled file with list of posteriors. "
                             "Samples are drawn from the first posterior. "
                             "This posterior has to be amortized.")
    parser.add_argument("samples_file",
                        type=str,
                        help="File with samples drawn from the prior and "
                             "the measured amplitudes.")
    parser.add_argument("observation",
                        help="Type of observation on which the posterior is "
                             "conditioned. Has to match with supplied "
                             "posterior.",
                        type=str,
                        choices=[Observation.LENGTH_CONSTANT.name.lower(),
                                 Observation.AMPLITUDES.name.lower(),
                                 Observation.AMPLITUDES_FIRST.name.lower()])
    parser.add_argument("-n_sbc_samples",
                        type=int,
                        help="Number of samples/observations for which sbc  "
                             "samples are drawn. If not provided, "
                             "all observations in the given data frame are "
                             "considered.")
    parser.add_argument("-n_samples",
                        type=int,
                        default=10000,
                        help="Number of samples to draw from each posterior.")
    parser.add_argument("-out_file",
                        type=str,
                        default='sbc_samples.pkl',
                        help="Location where the result is saved.")
    args = parser.parse_args()

    with open(args.posterior_file, 'rb') as handle:
        approx_posterior = pickle.load(handle)[0]
    prior_samples = pd.read_pickle(args.samples_file)

    data = sbc_data(approx_posterior,
                    extract_observation(prior_samples['amplitudes'],
                                        Observation[args.observation.upper()]),
                    num_sbc_samples=args.n_sbc_samples,
                    num_samples=args.n_samples,
                    parameter_names=prior_samples['parameters'].columns)
    data.attrs['observation'] = args.observation
    data.attrs['posterior_file'] = Path(args.posterior_file)
    data.attrs['prior_samples'] = Path(args.samples_file)
    data.to_pickle(args.out_file)
