#!/usr/bin/env python3
from pathlib import Path
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from model_hw_mc_attenuation import AttenuationExperiment
from model_hw_mc_attenuation.extract import get_experiment


def add_observables(posterior_samples: pd.DataFrame,
                    experiment: AttenuationExperiment,
                    max_simulations: Optional[int] = None) -> pd.DataFrame:
    '''
    Perform an attenuation experiment with parameters of the posterior samples.

    Configure an attenuation experiment as originally used to approximate the
    posterior distribution and extract the PSP amplitudes in each compartment
    from the emulation/simulation. The results are added to a copy of the input
    DataFrame and returned.

    :param posterior_samples: DataFrame with parameters to emulate/simulate.
    :param experiment: Experiment used to record observations.
    :param max_simulations: Maximum number of emulations/simulations to run.
        If not provided all possible simulations are run.
    :returns: Copy of the original DataFrame with the recorded amplitudes.
    '''
    length = experiment.length

    # see if observable already exists. If not add columns
    try:
        results = posterior_samples.loc[:, 'amplitudes'].values
        return_df = posterior_samples.copy()
    except KeyError:
        # add new columns
        n_amplitudes = length**2
        results = np.full((len(posterior_samples), n_amplitudes), np.nan)

        cols = [f"A_{i}{j}" for i, j in product(range(length), range(length))]
        columns = pd.MultiIndex.from_product([['amplitudes'], cols])
        return_df = pd.merge(posterior_samples,
                             pd.DataFrame(results, columns=columns),
                             right_index=True, left_index=True)
        return_df.attrs = posterior_samples.attrs

    # Measure all rows where at least one amplitude is missing
    idx_to_measure = np.any(np.isnan(results), axis=1).nonzero()[0]

    if (max_simulations is not None) and \
            (max_simulations < len(idx_to_measure)):
        idx_to_measure = np.random.choice(idx_to_measure,
                                          size=max_simulations,
                                          replace=False)

    for idx in idx_to_measure:
        parameters = posterior_samples['parameters'].values[idx]
        results[idx] = experiment.measure_response(parameters).flatten()

    return_df['amplitudes'] = results

    return return_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Configure an attenuation experiment as originally used '
                    'to approximate the posterior distribution and extract '
                    'the PSP amplitudes in each compartment from the '
                    'emulation/simulation. The results are added to the input '
                    'DataFrame which contains the parameters to configure.')
    parser.add_argument("samples_file",
                        help="Path to pickled DataFrame with parameters "
                             "to emulate/simulate. The results are added "
                             "to this DataFrame.",
                        type=str)
    parser.add_argument("-target_path",
                        help="DataFrame from which the configuration of the "
                             "experiment can be extracted. This is typically "
                             "the data frame which contains the target for "
                             "sbi. If not provided it is tried to be infered "
                             "from the smaples file.",
                        type=str)
    parser.add_argument("-max_simulations",
                        help="Maximum number of emulations/simulations to "
                             "run. If not provided all possible simulations "
                             "are run.",
                        type=int)
    args = parser.parse_args()

    samples = pd.read_pickle(args.samples_file)

    # Create experiment
    if args.target_path is None:
        if 'target_file' not in samples.attrs:
            raise RuntimeError("Experiment configuration can not be "
                               "inferred. Please provide `target_path`.")
        target_df_path = samples.attrs['target_file']
    else:
        target_df_path = args.target_path
    target_df = pd.read_pickle(target_df_path)
    attenuation_exp = get_experiment(target_df)

    samples = add_observables(samples,
                              attenuation_exp,
                              args.max_simulations)
    samples.attrs['target_df'] = Path(target_df_path)
    samples.attrs['experiment'] = target_df.attrs['experiment']
    samples.attrs['length'] = target_df.attrs['length']
    samples.to_pickle(args.samples_file)
