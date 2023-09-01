from typing import Callable, List, Optional
from pathlib import Path

import numpy as np

from pynn_brainscales.brainscales2.helper import nightly_calib_path

from dlens_vx_v3 import sta, hxcomm, halco, hal

from model_hw_mc_attenuation.helper import conductance_to_capmem

from calix.common import base
from calix.hagen import neuron_helpers


def extract_neuron_configs(config_dumper: sta.DumperDone
                           ) -> List[hal. NeuronConfig]:
    '''
    Extract neuron configs from a dumper.

    :param dumper: Dumper with configurations for all
        :class:`halco.AtomicNeuronOnDLS`.
    :returns: List of NeuronConfigs.
    '''
    cocos = dict(config_dumper.tolist())
    neuron_configs = []
    for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        atomic = cocos[neuron_coord]
        neuron_configs.append(atomic.asNeuronConfig())

    return neuron_configs


def get_calibration_dumper(path_to_calib: Optional[Path] = None
                           ) -> sta.DumperDone:
    '''
    Load the calibration in a dumper.

    :param path_to_calib: Path to portable binary calibration. If not provided
        the latest nightly calibration is used.
    :returns: Dumper with calibration applied.
    '''
    if path_to_calib is None:
        path_to_calib = Path(nightly_calib_path())

    with open(path_to_calib, 'rb') as fd:
        data = fd.read()
    dumper = sta.DumperDone()
    sta.from_portablebinary(dumper, data)

    return dumper


def measure_time_constants(calib_routine: base.Calib,
                           parameters: List[float],
                           calib_dumper: sta.DumperDone,
                           enable_mul_div_func: Callable) -> np.ndarray:
    '''
    Configure the given parameters and record the time constants.

    The given calibration routine is initialized and for each of the
    provided parameters the result is measured.

    :param calib_routine: Calibration routine.
    :param parameters: Parameters for which to measure the results.
    :param calib_dumper: Dumper of a calibration which should be applied at the
        beginning.
    :param enable_mul_div_func: Function which enables the
        multiplication/division of the measured conductance.
    :returns: The measured time constants for all neurons and parameters.
        Different parameters are in different rows, different neurons in
        different columns.
    '''

    with hxcomm.ManagedConnection() as connection:
        stateful_connection = base.StatefulConnection(connection)

        # Apply calibration (includes CADC calib)
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder, _ = neuron_helpers.configure_chip(builder)
        builder.builder.merge_back(sta.convert_to_builder(calib_dumper))
        base.run(stateful_connection, builder)

        calib_routine.prelude(stateful_connection)

        results = []

        for param in parameters:
            builder = base.WriteRecordingPlaybackProgramBuilder()

            capmem, div, mul = conductance_to_capmem(param)
            enable_mul_div_func(builder, calib_routine.neuron_configs,
                                div, mul)

            calib_routine.configure_parameters(builder, capmem)
            results.append(
                calib_routine.measure_results(stateful_connection, builder))

        return np.array(results)
