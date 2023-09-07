#!/usr/bin/env python3
import pandas as pd
from sbi import utils

from model_hw_mc_attenuation.bss import default_conductance_limits
from model_hw_mc_attenuation import parameter_names

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Draw samples from a uniform prior.")
    parser.add_argument("n_samples",
                        help="Number of prior samples to draw.",
                        type=int)
    parser.add_argument("-length",
                        help="Length of the compartment chain. If not given, "
                             "the parameters are assumed to be set globally.",
                        type=int)
    parser.add_argument("-out_file",
                        type=str,
                        default='prior_samples.pkl',
                        help="Location where the result is saved.")
    args = parser.parse_args()

    limits = default_conductance_limits
    if args.length is not None:
        limits = limits.repeat(args.length, axis=0)[:-1]
    prior = utils.BoxUniform(low=limits[:, 0].flatten(),
                             high=limits[:, 1].flatten())

    samples = prior.sample((args.n_samples,))

    columns = pd.MultiIndex.from_product([['parameters'],
                                          parameter_names(args.length)])

    data = pd.DataFrame(samples, columns=columns)
    data.attrs['limits'] = limits
    data.to_pickle(args.out_file)
