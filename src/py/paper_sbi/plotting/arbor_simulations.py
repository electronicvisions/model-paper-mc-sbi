'''
Evaluation of arbor simulations.
'''
from typing import Tuple, Sequence

import matplotlib.pyplot as plt
import neo
import pandas as pd
import quantities as pq

from paper_sbi.plotting.helper import get_figure_width, \
    formatted_parameter_names, replace_latex, latex_enabled, \
    DataSingleObservation
from paper_sbi.plotting.grid_search import plot_grid_search_diff
from paper_sbi.plotting.two_dimensional import _scale_by_psp_amplitude, \
    _scale, plot_traces, plot_posterior_prob, plot_posterior_samples


Parameter = Tuple[float, float]


###############################################################################
# Plotting
###############################################################################
def create_simulation_plots(*,
                            grid_search_df: pd.DataFrame,
                            example_traces: Sequence[neo.Block],
                            data_length_constant: DataSingleObservation,
                            data_amplitudes: DataSingleObservation,
                            original_parameters: Parameter):
    '''
    Create all figures which are used to illustrate the results obtained for
    simulations performed with arbor.

    This includes the results of a grid search, the posterior probability when
    observing the length constant, posterior samples (for the observation of
    the length constant and the PSP amplitudes in the first compartment) and
    example traces for different points in the parameter space.

    :param grid_search_df: Results of a two-dimensional grid search.
    :param example_traces: Blocks with example traces to plot.
    :param data_length_constant: Data for experiments with length constant
        as an observation.
    :param data_amplitudes: Data for experiments with amplitudes as an
        observation.
    :param original_parameters: Initial parameters used to record the
        observation on which the posteriors are conditioned.
    '''
    fileformat = 'pgf' if latex_enabled() else 'svg'
    figure_width = get_figure_width('double') / 3
    heights_2d = 2
    height_traces = 0.4

    trace_params = [block.annotations['parameters'][[0, -1]] for block
                    in example_traces]
    param_names = _get_parameter_names()

    # Grid Search
    target = data_length_constant.posterior_samples.attrs['target']
    figure = plot_grid_search_diff((figure_width, heights_2d), grid_search_df,
                                   trace_params, param_names,
                                   levels=[0.5, 1],
                                   target_length_constant=target)
    figure.savefig(f'grid_search_arbor.{fileformat}')
    plt.close()

    # Posterior
    figure = plot_posterior_prob((figure_width, heights_2d),
                                 data_length_constant.posteriors[-1],
                                 trace_params, param_names)
    figure.savefig(f'posterior_arbor.{fileformat}')
    plt.close()

    # Samples
    figure = plot_posterior_samples(
        (figure_width, heights_2d),
        samples_length=data_length_constant.posterior_samples,
        samples_amplitudes=data_amplitudes.posterior_samples,
        original_parameters=original_parameters,
        marker_points=[],  # points would cover samples
        param_names=param_names)
    figure.savefig(f'posterior_samples_arbor.{fileformat}')
    plt.close()

    # Traces
    traces_amp = [_scale_by_psp_amplitude(block) for block in example_traces]
    figure = plot_traces((figure_width, height_traces), traces_amp)
    figure.savefig(f'traces_arbor_scaled.{fileformat}')
    plt.close()

    traces_v = [_scale(block, 1e-3 * pq.V) for block in example_traces]
    figure = plot_traces((figure_width, height_traces), traces_v)
    figure.savefig(f'traces_arbor.{fileformat}')
    plt.close()


###############################################################################
# Helper
###############################################################################
def _get_parameter_names():
    '''
    Get the names of the two-dimensional parameters inclusive units.
    '''
    if latex_enabled():
        units = (r'\si{\siemens\per\square\cm}', r'\si{\siemens\per\cm}')
        return formatted_parameter_names(units=units)

    units = ('S/cm^2', 'S/cm')
    return replace_latex(formatted_parameter_names(units=units))
