#!/usr/bin/env python3
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import quantities as pq

from pynn_brainscales.brainscales2.helper import nightly_calib_path

from dlens_vx_v3 import sta, hal, halco

from model_hw_mc_genetic.helper import get_license_and_chip
from model_hw_mc_genetic.attenuation.bss import default_conductance_limits

from paper_sbi.helper import measure_time_constants, get_calibration_dumper, \
    extract_neuron_configs

from calix.multicomp.neuron_icc_bias import ICCMADCCalib


def enable_icc_div_mul(builder: sta.PlaybackProgramBuilder,
                       neuron_configs: hal.NeuronConfig,
                       div: bool, mul: bool) -> None:
    '''
    Enable division/multiplication for the inter-compartment conductance.

    Enabling and disabling is applied to the neuron configurations and also
    written to the builder.

    :param builder: Builder to which the neuron configurations are written.
    :param neuron_configs: Neuron configurations to change and write.
    :param div: Enable division.
    :param mul: Enable multiplication.
    '''
    for coord, neuron_config in \
            zip(halco.iter_all(halco.NeuronConfigOnDLS), neuron_configs):
        neuron_config.enable_divide_multicomp_conductance_bias = div
        neuron_config.enable_multiply_multicomp_conductance_bias = mul
        builder.write(coord, neuron_config)


def measure_tau_icc(parameter_range: Tuple[int, int, int],
                    calibration: Optional[str] = None) -> pd.DataFrame:
    '''
    Measure the inter-compartment time constant.

    :param calibration: Path to portable binary calibration. If not provided
        the latest nightly calibration is used.
    :param parameter_range: Inter-compartment conductance (in CapMem values).
        Provide as a tuple with the following values: (lower_bound,
        upper bound, num_steps).
    :returns: DataFame with the measured time constants and the parameters
        at which the time constants were measured.
    '''
    if calibration is None:
        calibration = Path(nightly_calib_path())
    else:
        calibration = Path(calibration)
    calib_dumper = get_calibration_dumper(calibration)

    # setup calibration routine
    calib_routine = ICCMADCCalib(
        target=1. * pq.us,
        neuron_configs=extract_neuron_configs(calib_dumper))

    parameters = np.linspace(*parameter_range)
    results = measure_time_constants(calib_routine, parameters,
                                     calib_dumper, enable_icc_div_mul)

    # save result in DataFrame
    measure_col = pd.MultiIndex.from_product(
        [['time_constants'], [f"n_{coord}" for coord in range(512)]])
    columns = pd.MultiIndex.from_tuples(
        [('parameters', 'g_icc')] + list(measure_col))

    data = pd.DataFrame(np.hstack([parameters[:, np.newaxis], results]),
                        columns=columns)
    data.attrs['calibration'] = str(calibration.resolve())
    data.attrs['chip_id'] = get_license_and_chip()
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Measure the time constant of inter-compartment '
                    'conductance on BSS-2.')
    parser.add_argument("-g_icc",
                        help="Inter-compartment conductance (in CapMem "
                             "values). Provide as a list with the following "
                             "values: (lower_bound, upper bound, num_steps). "
                             "The steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=int,
                        default=default_conductance_limits[1].tolist() + [3])
    parser.add_argument('-calibration',
                        type=str,
                        help='Path to portable binary calibration. If not '
                             'provided the latest nightly calibration is '
                             'used.')
    args = parser.parse_args()

    df = measure_tau_icc(args.g_icc, args.calibration)
    df.to_pickle('tau_icc.pkl')
