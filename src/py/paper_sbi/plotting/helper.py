'''
Helper functions for styling and annotating plots.
'''
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Sequence, Dict
import re
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import Patch

from sbi.inference.posteriors.base_posterior import NeuralPosterior

from paper_sbi.plotting.expected_coverage import ExpectedCoverageData


Parameter = Tuple[float, float]


# Colors for the different compartments in an attenuation experiment with four
# compartments
COMPARTMENT_COLORS = ["#66a61e", "#e6ab02", "#a6761d", "#666666"]


@dataclass
class DataSingleObservation:
    '''
    Data of a single observation type.

    :ivar posterior: Posterior distribution.
    :ivar posterior_samples: Samples drawn from the posterior.
    :ivar coverage_ensemble: Data needed to calculate the expected coverage.
    :ivar coverage_single: Data needed to calculate the expected coverage of
        the posteriors which make up the ensemble.
    '''
    posteriors: List[NeuralPosterior]
    posterior_samples: pd.DataFrame
    coverage_ensemble: ExpectedCoverageData
    coverage_single: List[ExpectedCoverageData]


def latex_enabled():
    '''
    Test if custom LaTex commands are defined in the preamble for matplotlib.

    These custom command can be added to the preamble with
    :func:`apply_custom_styling`.

    :returns: Whether custom commands are set in
        `plt.rcParams['pgf.preamble']`.
    '''
    return 'myvec' in plt.rcParams['pgf.preamble']


def apply_custom_styling(use_latex: bool = False):
    '''
    Set default style of matplotlib plots.

    Set default font sizes, colors and other styles.

    :param use_latex: Whether matplotlib should use the 'pgf' engine to render
        text. When enabled custom LaTex commands are defined in the preamble
        and several packages such as `siunitx` are loaded.
    '''
    if use_latex:
        mpl.use("pgf")
        mpl.rcParams['pgf.texsystem'] = "pdflatex"
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['pgf.rcfonts'] = False
        mpl.rcParams['pgf.preamble'] = \
            "\n".join([r"\usepackage[utf8]{inputenc}",
                       r"\usepackage{amsmath}",
                       r"\usepackage{amssymb}",
                       r"\usepackage[detect-all]{siunitx}",
                       r"\usepackage{bm}",
                       r"\newcommand{\myvec}[1]{\bm{#1}}",
                       r"\newcommand{\mymat}[1]{\bm{#1}}",
                       ])
    # colorblind friendly
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#59a14f",
                                                        "#006ba4",
                                                        "#ff800e",
                                                        "#c85200",
                                                        "#ababab",
                                                        "#5f9ed1",
                                                        "#595959",
                                                        "#898989",
                                                        "#a2c8ec"])

    mpl.rcParams['font.size'] = 7

    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['axes.labelsize'] = 6
    mpl.rcParams['axes.labelcolor'] = '0.3'
    mpl.rcParams['axes.edgecolor'] = '0.3'
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.grid.which'] = 'major'
    mpl.rcParams['grid.color'] = 'k'
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.6
    mpl.rcParams['grid.alpha'] = 0.3

    mpl.rcParams['lines.linewidth'] = 1

    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['xtick.minor.width'] = 0.2
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.color'] = '0.3'
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['ytick.minor.width'] = 0.2
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['ytick.color'] = '0.3'

    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['legend.edgecolor'] = '1'
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.shadow'] = False
    mpl.rcParams['legend.labelspacing'] = 0.2
    mpl.rcParams['legend.handlelength'] = 0.8
    mpl.rcParams['legend.handleheight'] = 0.5
    mpl.rcParams['legend.handletextpad'] = 0.3
    mpl.rcParams['legend.borderpad'] = 0
    mpl.rcParams['legend.fontsize'] = mpl.rcParams['axes.labelsize']


def get_figure_width(columns: str = 'double') -> float:
    '''
    Return figure width in inches for PNAS publications.

    :param columns: Size of the image in columns; 'single', 'onehalf' or
        'double'.
    :returns: Figure width in inches.
    '''

    if columns == 'single':
        return 3.42
    if columns == 'onehalf':
        return 4.5
    if columns == 'double':
        return 7

    raise ValueError('Desired width not available.')


def add_scalebar(ax: plt.Axes, *,
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None,
                 margin: Tuple[float, float] = (1, 1),
                 args_scale: Optional[Dict] = None,
                 args_xlabel: Optional[Dict] = None,
                 args_ylabel: Optional[Dict] = None,
                 **kwargs):
    '''
    Add a scalebar to the given axis.

    Depending on which labels are provided, a horizontal, vertical or l-shaped
    scalebar is created and placed in a `AnchoredOffsetbox` which is returned.
    The length of the scalebars is extracted from the labels.
    Keyword arguments are passed to the `AnchoredOffsetbox`.

    :param ax: Axis for which to create a scalebar.
    :param x_label: Label for the x-scale. The size of the scale is extracted
        from the text, i.e. a single number is extracted.
    :param y_label: Label for the y-scale. The size of the scale is extracted
        from the text, i.e. a single number is extracted.
    :param margin: Margin around the labels in data-coordinates. The first
        entry represents the margin of the x-label, the second the margin of
        the y-label.
    :param args_scale: Keyword arguments provided to :class:`plt.Line2D` patch
        which represents the scalebar.
    :param args_xlabel: Keyword arguments provided to :class:`plt.Text` which
        contains the x-label.
    :param args_ylabel: Keyword arguments provided to :class:`plt.Text` which
        contains the y-label.
    :return: AnchoredOffsetbox which contains the scalebar and the labels.
    '''

    # extract numbers from labels
    number_regex = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
    x_scale = 0 if x_label is None \
        else float(re.findall(number_regex, x_label)[0])
    y_scale = 0 if y_label is None \
        else float(re.findall(number_regex, y_label)[0])

    # create scale
    scalebar = AuxTransformBox(ax.transData)

    def_args = {'lw': 1, 'c': 'k'}
    def_args.update(args_scale if args_scale is not None else {})
    scale = plt.Line2D((0, x_scale, x_scale), (0, 0, y_scale), **def_args)
    scalebar.add_artist(scale)

    # add labels
    if x_label is not None:
        def_args = {'x': x_scale / 2, 'y': -margin[0], 'text': x_label,
                    'ha': 'center', 'va': 'top'}
        def_args.update(args_xlabel if args_xlabel is not None else {})
        scalebar.add_artist(plt.Text(**def_args))

    if y_label is not None:
        def_args = {'x': x_scale + margin[1], 'y': y_scale / 2,
                    'text': y_label, 'rotation': 90, 'ha': 'left',
                    'va': 'center'}
        def_args.update(args_ylabel if args_ylabel is not None else {})
        scalebar.add_artist(plt.Text(**def_args))

    # place in Offsetbox
    def_args = {'loc': 'upper left', 'child': scalebar, 'frameon': False}
    def_args.update(kwargs)
    box = AnchoredOffsetbox(**def_args)

    ax.add_artist(box)
    return box


def formatted_parameter_names(
        length: Optional[int] = None,
        units: Optional[Tuple[str, str]] = None) -> List[str]:
    '''
    Parameter names of an attenuation experiment whith formatting which can be
    rendered by TeX/matplotlib.

    :param length: Length of the compartment chain. If not supplied the base
        names are returned.
    :param units: Units of the leak and axial conductance.
    :returns: List of parameters (leak and inter-compartment conductance).
    '''
    base_names = [r'g_\text{leak}', r'g_\text{axial}']

    # add style to units if supplied else set empty strings for program flow
    units = ('', '') if units is None else [rf'\,/\,{unit}' for unit in units]

    if length is None:
        return [f'${base_name}{unit}$' for base_name, unit in zip(base_names,
                                                                  units)]
    comps = np.arange(length)

    leak_names = [f'${base_names[0]}^{i}{units[0]}$' for i in comps]
    icc_names = [rf'${base_names[1]}^{{{pre} \leftrightarrow {post}}}'
                 f'{units[1]}$' for pre, post in zip(comps[:-1], comps[1:])]

    return leak_names + icc_names


formatted_parameter_names_bss = partial(formatted_parameter_names,
                                        units=[r'\mathrm{DAC}'] * 2)


def replace_latex(strings: Union[str, Sequence[str]]
                  ) -> Union[str, List[str]]:
    '''
    Replace commands which can not be rendered by matplotlibs math engine.

    :param strings: String or Sequence of strings in which to replace the
        commands.
    :returns: String with latex/custom commands replaced such that it can be
        rendered by matplotlib.
    '''
    if isinstance(strings, str):
        match_replacement = [(r'\myvec', r'\mathbf'),
                             (r'\mymat', r'\mathbf'),
                             (r'\textbf', r'\mathbf'),
                             (r'\text', r'\mathrm')]
        for match, replacement in match_replacement:
            strings = strings.replace(match, replacement)
        return strings

    if not np.all([isinstance(string, str) for string in strings]):
        raise ValueError('Please provide a str or a sequence of strings as '
                         'input.')
    return [replace_latex(string) for string in strings]


def add_legend_with_patches(parent: Union[plt.Axes, plt.Figure],
                            labels: List[str], colors: List,
                            **kwargs) -> None:
    '''
    Add a legend with patches of the given colors to the given axes/figure.

    Keyword arguments are passed to :meth:`plt.Axes.legend`.

    :param parent: Axes/figure to which the legend is added.
    :param labels: Labels for the entries in the legend.
    :param colors: Colors of the patches.
    '''
    legend_elements = [Patch(facecolor=color, ec='none', label=label)
                       for color, label in zip(colors, labels)]
    parent.legend(handles=legend_elements, **kwargs)


def mark_points(ax: plt.Axes, points: Sequence[Parameter]) -> None:
    '''
    Mark the given points with increasing numbers.

    :param ax: Axis in which to mark the points.
    :param point: Points at which annotations with increasing numbers are
        added.
    '''
    for n_point, point in enumerate(points):
        annotate_circled(ax, point, str(n_point))


def annotate_circled(ax: plt.Axes, point: Tuple[float, float], text: str,
                     **kwargs) -> None:
    '''
    Add an annotation with a circle around it.

    Keyword arguments are forwarded to :meth:`plt.Axes.annotate`.

    :param ax: Axis to which the annotation is added.
    :param point:  x and y coordinate where the annotation is added.
    :param text: Text of the annotation.
    '''
    default_kwargs = {'ha': 'center', 'va': 'center',
                      'bbox': {'boxstyle': 'circle, pad=0.3', 'fc': 'w',
                               'ec': 'k', 'lw': 0.8},
                      'size': 6}
    default_kwargs.update(kwargs)
    ax.annotate(text, xy=point, **default_kwargs)
