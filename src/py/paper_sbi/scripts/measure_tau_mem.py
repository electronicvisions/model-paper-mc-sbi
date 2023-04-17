#!/usr/bin/env python3
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import quantities as pq

from pynn_brainscales.brainscales2.helper import nightly_calib_path

from dlens_vx_v3 import sta, hal, halco

from model_hw_mc_attenuation.helper import get_license_and_chip
from model_hw_mc_attenuation.bss import default_conductance_limits

from paper_sbi.helper import measure_time_constants, get_calibration_dumper, \
    extract_neuron_configs

from calix.hagen.neuron_leak_bias import MembraneTimeConstCalibOffset


def enable_leak_div_mul(builder: sta.PlaybackProgramBuilder,
                        configs: hal.NeuronConfig,
                        div: bool, mul: bool) -> None:
    '''
    Enable division/multiplication for the leak conductance.

    Enabling and disabling is applied to the neuron configurations and also
    written to the builder.

    :param builder: Builder to which the neuron configurations are written.
    :param neuron_configs: Neuron configurations to change and write.
    :param div: Enable division.
    :param mul: Enable multiplication.
    '''
    for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS), configs):
        config.enable_leak_division = div
        config.enable_leak_multiplication = mul
        builder.write(coord, config)


def measure_tau_mem(parameter_range: Tuple[int, int, int],
                    calibration: Optional[str] = None) -> pd.DataFrame:
    '''
    Measure the membrane time constant.

    :param calibration: Path to portable binary calibration. If not provided
        the latest nightly calibration is used.
    :param parameter_range: Leak conductance (in CapMem values). Provide as a
        tuple with the following values: (lower_bound, upper bound, num_steps).
    :returns: DataFame with the measured time constants and the parameters
        at which the time constants were measured.
    '''
    if calibration is None:
        calibration = Path(nightly_calib_path())
    else:
        calibration = Path(calibration)
    calib_dumper = get_calibration_dumper(calibration)

    neuron_configs = extract_neuron_configs(calib_dumper)
    for neuron_config in neuron_configs:
        neuron_config.enable_threshold_comparator = False

    # setup calib_routine routine
    calib_routine = MembraneTimeConstCalibOffset(neuron_configs=neuron_configs,
                                                 target=10 * pq.us)
    calib_routine.adjust_bias_range = False

    parameters = np.linspace(*parameter_range)
    results = measure_time_constants(calib_routine, parameters,
                                     calib_dumper, enable_leak_div_mul)

    # save result in DataFrame
    measure_col = pd.MultiIndex.from_product(
        [['time_constants'], [f"n_{coord}" for coord in range(512)]])
    columns = pd.MultiIndex.from_tuples(
        [('parameters', 'g_leak')] + list(measure_col))

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
    parser.add_argument("-g_leak",
                        help="Leak conductance (in CapMem values). "
                             "Provide as a list with the following values: "
                             "(lower_bound, upper bound, num_steps). The "
                             "steps will be distributed evenly over the "
                             "parameter space.",
                        nargs=3,
                        type=int,
                        default=default_conductance_limits[0].tolist() + [3])
    parser.add_argument('-calibration',
                        type=str,
                        help='Path to portable binary calibration. If not '
                             'provided the latest nightly calibration is '
                             'used.')
    args = parser.parse_args()

    df = measure_tau_mem(args.g_leak, args.calibration)
    df.to_pickle('tau_mem.pkl')
