"""
Utils (utilities)

This is a collection of utilities that are used throughout the python
reliability library. Functions have been placed here as to declutter the
dropdown lists of your IDE and to provide a common resource across multiple
modules. It is not expected that users will be using any utils directly.

Included functions are:

- ALT_MLE_optimization - performs optimization for the ALT_Fitters
- ALT_fitters_input_checking - performs input checking for the ALT_Fitters
- ALT_least_squares - least squares estimation for ALT_Fitters
- ALT_prob_plot - probability plotting for ALT_Fitters
- LS_optimization - least squares optimization for Fitters
- MLE_optimization - maximum likelihood estimation optimization for Fitters
- anderson_darling - calculated the anderson darling (AD) goodness of fit statistic
- axes_transforms - Custom scale functions used in Probability_plotting
- clean_CI_arrays - cleans the CI arrays of nan and illegal values
- colorprint - prints to the console in color, bold, italic, and underline
- distribution_confidence_intervals - calculates and plots the confidence intervals for the distributions
- fill_no_autoscale - creates a shaded region without adding it to the global list of objects to consider when autoscale is calculated
- fitters_input_checking - error checking and default values for all the fitters
- generate_X_array - generates the X values for all distributions
- get_axes_limits - gets the current axes limits
- least_squares - provides parameter estimates for distributions using the method of least squares. Used extensively by Fitters.
- life_stress_plot - generates the life stress plot for ALT_Fitters
- line_no_autoscale - creates a line without adding it to the global list of objects to consider when autoscale is calculated
- linear_regression - given x and y data it will return slope and intercept of line of best fit. Includes options to specify slope or intercept.
- make_fitted_dist_params_for_ALT_probplots - creates a class structure for the ALT probability plots to give to Probability_plotting
- no_reverse - corrects for reversals in confidence intervals
- probability_plot_xylims - sets the x and y limits on probability plots
- probability_plot_xyticks - sets the x and y ticks on probability plots
- removeNaNs - removes nan
- restore_axes_limits - restores the axes limits based on values from get_axes_limits()
- round_and_string - applies different rounding rules and converts to string
- show_figure_from_object - Re-shows a figure from an axes or figure handle even after the figure has been closed.
- transform_spaced - Creates linearly spaced array (in transform space) based on a specified transform. This is like np.logspace but it can make an array that is weibull spaced, normal spaced, etc.
- validate_CI_params - checks that the confidence intervals have all the right parameters to be generated
- write_df_to_xlsx - converts a dataframe to an xlsx file
- xy_transform - provides conversions between spatial (-inf,inf) and axes coordinates (0,1).
- zeroise_below_gamma - sets all y values to zero when x < gamma. Used when the HF and CHF equations are specified
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.axes import _axes
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import ticker, colors
from autograd import jacobian as jac
from autograd_gamma import gammainccinv as agammainccinv
from autograd_gamma import gammaincc as agammaincc
from autograd import value_and_grad
import autograd.numpy as anp
from scipy.special import gammainc, betainc, erf
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from numpy.linalg import LinAlgError
import warnings
import os
import pandas as pd

warnings.filterwarnings(
    action="ignore", category=OptimizeWarning
)  # ignores the optimize warning that curve_fit sometimes outputs when there are 3 data points to fit a 3P curve
warnings.filterwarnings(
    action="ignore", category=RuntimeWarning
)  # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required


def round_and_string(number, decimals=5, integer_floats_to_ints=True, large_scientific_limit=1e9, small_scientific_limit=1e-4):
    """
    This function is used to round a number to a specified number of decimals and convert to string.
    It is used heavily in the formatting of the parameter titles within reliability.Distributions

    The rules applied are slightly different than rounding to a number of significant figures as it keeps more
    preceeding zeros. Special formatting rules are applied when:
        abs(number) <= small_scientific_limit
        abs(number) >= large_scientific_limit
        number = 0

    Parameters
    ----------
    number : float
        The number to be rounded
    decimals : int
        The number of decimals (not including preceeding zeros) that are to be
        in the output
    integer_floats_to_ints : bool, optional
        Default is True. Removes trailing zeros from floats if there are no
        significant decimals (eg. 12.0 becomes 12).
    large_scientific_limit : int, float, optional
        The limit above which to keep numbers formatted in scientific notation. Default is 1e9.
    small_scientific_limit : int, float, optional
        The limit below which to keep numbers formatted in scientific notation. Default is 1e-4.

    Returns
    -------
    out : string
        The number formatted and converted to string.

    Notes
    -----
    Examples (with decimals = 5):
        original: -1e+20  // new: -1e+20
        original: -100000000.0  // new: -100000000
        original: -10.0  // new: -10
        original: -5.4857485e-05  // new: -5.48575e-05
        original: -2.54875415e-16  // new: -2.54875e-16
        original: 0.0  // new: 0
        original: 0  // new: 0
        original: 2.54875415e-16  // new: 2.54875e-16
        original: 5.4857485e-05  // new: 5.48575e-05
        original: 10.0  // new: 10
        original: 100000000.0  // new: 100000000
        original: 1e+20  // new: 1e+20
    """
    if decimals <1:
        decimals = 0

    if np.isfinite(number):  # check the input is not NaN
        decimal = number % 1
        if number == 0:
            if integer_floats_to_ints is True and decimal == 0:
                number = int(number)
            out = number
        elif abs(number) >= large_scientific_limit or abs(number) <= small_scientific_limit:
            if decimal != 0:
                # special formatting is only applied when the number is a float in the ranges specified above
                out = str('{:0.' + str(decimals) + 'e}').format(number)
                excess_zeros = "".join(['0'] * decimals)
                out = out.replace("."+excess_zeros+"e","e")
            else:
                out = number
                if integer_floats_to_ints is True and decimal == 0:
                    out = int(out)
        else:
            out = round(number, decimals)
            if integer_floats_to_ints is True and decimal == 0:
                out = int(out)
    else: # NaN
        out = number
    return str(out)

def transform_spaced(
    transform,
    y_lower=1e-8,
    y_upper=1 - 1e-8,
    num=1000,
    alpha=None,
    beta=None,
):
    """
    Creates linearly spaced array based on a specified transform.

    This is similar to np.linspace or np.logspace but is designed for weibull
    space, exponential space, normal space, gamma space, loglogistic space,
    gumbel space and beta space.

    It is useful if the points generated are going to be plotted on axes that
    are scaled using the same transform and need to look equally spaced in the
    transform space.

    Parameters
    ----------
    transform : str
        The transform name. Must be either weibull, exponential, normal, gamma,
        gumbel, loglogistic, or beta.
    y_upper : float, optional
        The lower bound (must be within the bounds 0 to 1). Default is 1e-8
    y_lower : float, optional
        The upper bound (must be within the bounds 0 to 1). Default is 1-1e-8
    num : int, optional
        The number of values in the array. Default is 1000.
    alpha : int, float, optional
        The alpha value of the beta distribution. Only used if the transform is
        beta
    beta : int, float, optional
        The beta value of the beta or gamma distribution. Only used if the
        transform is beta or gamma

    Returns
    -------
    transformed_array : array
        transform spaced array. This appears linearly spaced when plotted in
        transform space.

    Notes
    -----
    Note that lognormal is the same as normal, since the x-axis is what is
    transformed in lognormal, not the y-axis.
    """
    np.seterr("ignore")  # this is required due to an error in scipy.stats
    if y_lower > y_upper:
        y_lower, y_upper = y_upper, y_lower
    if y_lower <= 0 or y_upper >= 1:
        raise ValueError("y_lower and y_upper must be within the range 0 to 1")
    if num <= 2:
        raise ValueError("num must be greater than 2")
    if transform in ["normal", "Normal", "norm", "Norm"]:
        fwd = lambda x: ss.norm.ppf(x)
        inv = lambda x: ss.norm.cdf(x)
    elif transform in ["gumbel", "Gumbel", "gbl", "gum", "Gum", "Gbl"]:
        fwd = lambda x: ss.gumbel_l.ppf(x)
        inv = lambda x: ss.gumbel_l.cdf(x)
    elif transform in ["weibull", "Weibull", "weib", "Weib", "wbl"]:
        fwd = lambda x: np.log(-np.log(1 - x))
        inv = lambda x: 1 - np.exp(-np.exp(x))
    elif transform in ["loglogistic", "Loglogistic", "LL", "ll", "loglog"]:
        fwd = lambda x: np.log(1 / x - 1)
        inv = lambda x: 1 / (np.exp(x) + 1)
    elif transform in ["exponential", "Exponential", "expon", "Expon", "exp", "Exp"]:
        fwd = lambda x: ss.expon.ppf(x)
        inv = lambda x: ss.expon.cdf(x)
    elif transform in ["gamma", "Gamma", "gam", "Gam"]:
        if beta is None:
            raise ValueError("beta must be specified to use the gamma transform")
        else:
            fwd = lambda x: ss.gamma.ppf(x, a=beta)
            inv = lambda x: ss.gamma.cdf(x, a=beta)
    elif transform in ["beta", "Beta"]:
        if alpha is None or beta is None:
            raise ValueError(
                "alpha and beta must be specified to use the beta transform"
            )
        else:
            fwd = lambda x: ss.beta.ppf(x, a=beta, b=alpha)
            inv = lambda x: ss.beta.cdf(x, a=beta, b=alpha)
    elif transform in [
        "lognormal",
        "Lognormal",
        "LN",
        "ln",
        "lognorm",
        "Lognorm",
    ]:  # the transform is the same, it's just the xscale that is ln for lognormal
        raise ValueError(
            "the Lognormal transform is the same as the normal transform. Specify normal and try again"
        )
    else:
        raise ValueError(
            "transform must be either exponential, normal, weibull, loglogistic, gamma, or beta"
        )

    # find the value of the bounds in tranform space
    upper = fwd(y_upper)
    lower = fwd(y_lower)
    # generate the array in transform space
    arr = np.linspace(lower, upper, num)
    # convert the array back from transform space
    transform_array = inv(arr)
    return transform_array


class axes_transforms:
    """
    Custom scale functions used in Probability_plotting
    Each of these functions is either a forward or inverse transform.

    There are no parameters for this class, only a collection of subfunctions
    which can be called individually to perform the transforms.
    """

    @staticmethod
    def weibull_forward(F):
        return np.log(-np.log(1 - F))

    @staticmethod
    def weibull_inverse(R):
        return 1 - np.exp(-np.exp(R))

    @staticmethod
    def loglogistic_forward(F):
        return np.log(1 / (1 - F) - 1)

    @staticmethod
    def loglogistic_inverse(R):
        return 1 - 1 / (np.exp(R) + 1)

    @staticmethod
    def exponential_forward(F):
        return ss.expon.ppf(F)

    @staticmethod
    def exponential_inverse(R):
        return ss.expon.cdf(R)

    @staticmethod
    def normal_forward(F):
        return ss.norm.ppf(F)

    @staticmethod
    def normal_inverse(R):
        return ss.norm.cdf(R)

    @staticmethod
    def gumbel_forward(F):
        return ss.gumbel_l.ppf(F)

    @staticmethod
    def gumbel_inverse(R):
        return ss.gumbel_l.cdf(R)

    @staticmethod
    def gamma_forward(F, beta):
        return ss.gamma.ppf(F, a=beta)

    @staticmethod
    def gamma_inverse(R, beta):
        return ss.gamma.cdf(R, a=beta)

    @staticmethod
    def beta_forward(F, alpha, beta):
        return ss.beta.ppf(F, a=alpha, b=beta)

    @staticmethod
    def beta_inverse(R, alpha, beta):
        return ss.beta.cdf(R, a=alpha, b=beta)


def get_axes_limits():
    """
    This function works in a pair with restore_axes_limits
    This function gets the previous xlim and ylim and also checks whether there
    was a previous plot (based on whether the default 0,1 axes had been
    changed).

    It returns a list of items that are used by restore_axes_limits after the
    plot has been performed.

    Parameters
    ----------
    None
        The plot properties are extracted automatically for analysis

    Returns
    -------
    output : list
        A list of [xlims, ylims, use_prev_lims]. These values are used by
        restore_axes_limits to determine how the axes limits need to be
        changed after plotting.
    """
    xlims = plt.xlim(auto=None)  # get previous xlim
    ylims = plt.ylim(auto=None)  # get previous ylim
    if xlims == (0, 1) and ylims == (
        0,
        1,
    ):  # this checks if there was a previous plot. If the lims were 0,1 and 0,1 then there probably wasn't
        use_prev_lims = False
    else:
        use_prev_lims = True
    out = [xlims, ylims, use_prev_lims]
    return out


def restore_axes_limits(limits, dist, func, X, Y, xvals=None, xmin=None, xmax=None):
    """
    This function works in a pair with get_axes_limits. Using the values
    producted by get_axes_limits which are [xlims, ylims, use_prev_lims], this
    function will determine how to change the axes limits to meet the style
    requirements of the library.

    Parameters
    ----------
    limits : list
        A list of [xlims, ylims, use_prev_lims] created by get_axes_limits
    dist : object
        The distribution object which the axes limits are influenced by.
    X : array, list
        The x-values of the plot
    Y : array, list
        The y-values of the plot
    xvals : array, list, optional
        The plot xvals if specified. May be None if not specified.
    xmin : int, float, optional
        The plot xmin if specified. May be None if not specified.
    xmax : int, float, optional
        The plot xmax if specified. May be None if not specified.

    Returns
    -------
    None
        This function will scale the plot but it does not return anything

    Notes
    -----
    No scaling will be done if the axes are not linear due to errors that result
    from log and function scaled axes when a limit of 0 is used. This means that
    this function is not able to be applied to the probability plots are they
    have non-linear scaled axes.
    """
    xlims = limits[0]
    ylims = limits[1]
    use_prev_lims = limits[2]

    ################## XLIMS ########################
    # obtain the xlims as if we did not consider prev limits
    if xvals is None:
        if xmin is None:
            if dist.name in [
                "Weibull",
                "Gamma",
                "Loglogistic",
                "Exponential",
                "Lognormal",
            ]:
                if dist.gamma == 0:
                    xlim_lower = 0
                else:
                    diff = dist.quantile(0.999) - dist.quantile(0.001)
                    xlim_lower = max(0, dist.quantile(0.001) - diff * 0.1)
            elif dist.name in ["Normal", "Gumbel"]:
                xlim_lower = dist.quantile(0.001)
            elif dist.name == "Beta":
                xlim_lower = 0
            elif dist.name in ["Mixture", "Competing risks"]:
                # DSZI not required here as limits are same as base distribution
                xlim_lower = min(X)
            else:
                raise ValueError("Unrecognised distribution name")
        else:
            xlim_lower = xmin

        if xmax is None:
            if dist.name == "Beta":
                xlim_upper = 1
            else:
                xlim_upper = dist.quantile(0.999)
        else:
            xlim_upper = xmax

        if xlim_lower > xlim_upper:
            xlim_lower, xlim_upper = (
                xlim_upper,
                xlim_lower,
            )  # switch them if xmin and xmax were given in the wrong order
    else:  # if the xlims have been specified then these are the limits to be used
        xlim_lower = min(xvals)
        xlim_upper = max(xvals)

    # determine what to set the xlims based on whether to use_prev_lims
    if use_prev_lims == False:
        xlim_LOWER = xlim_lower
        xlim_UPPER = xlim_upper
    else:  # need to consider previous axes limits
        xlim_LOWER = min(xlim_lower, xlims[0])
        xlim_UPPER = max(xlim_upper, xlims[1])

    if plt.gca().get_xscale() == "linear" and len(X) > 1:
        plt.xlim(xlim_LOWER, xlim_UPPER, auto=None)

    ################## YLIMS ########################

    top_spacing = 1.1  # the amount of space between the max value and the upper axis limit. 1.1 means the axis lies 10% above the max value
    if func in ["pdf", "PDF"]:
        if not np.isfinite(dist._pdf0) and not np.isfinite(
            Y[-1]
        ):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif (
            not np.isfinite(Y[-1]) or dist._pdf0 == np.inf or dist._pdf0 > 10
        ):  # asymptote on the right or on the left
            ylim_upper = max(Y)
        else:  # an increasing pdf. Not asymptote
            ylim_upper = max(Y) * top_spacing
    elif func in ["cdf", "CDF", "SF", "sf"]:
        ylim_upper = top_spacing
    elif func in ["hf", "HF"]:
        if not np.isfinite(dist._hf0) and not np.isfinite(
            Y[-1]
        ):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif (
            not np.isfinite(Y[-1]) or dist._hf0 == np.inf or dist._hf0 > 10
        ):  # asymptote of the right or on the left
            ylim_upper = max(Y)
        elif max(Y) > Y[-1]:  # a peaked hf
            ylim_upper = max(Y) * top_spacing
        else:  # an increasing hf. Not an asymptote
            if len(np.where(X >= plt.xlim()[1])[0]) == 0:
                idx = len(X) - 1  # this is for the mixture model and CR model
            else:
                idx = np.where(X >= plt.xlim()[1])[0][0]
            ylim_upper = Y[idx] * top_spacing
    elif func in ["chf", "CHF"]:
        if len(np.where(X >= xlim_upper)[0]) == 0:
            idx = len(X) - 1  # this is for the mixture model and CR model
        else:
            idx = np.where(X >= xlim_upper)[0][
                0
            ]  # index of the chf where it is equal to b95
        if np.isfinite(Y[idx]):
            ylim_upper = Y[idx] * top_spacing
        else:
            ylim_upper = Y[idx - 1] * top_spacing
    else:
        raise ValueError("func is invalid")
    ylim_lower = 0

    # determine what to set the ylims based on whether to use_prev_lims
    if use_prev_lims == False:
        ylim_LOWER = ylim_lower
        ylim_UPPER = ylim_upper
    else:  # need to consider previous axes limits
        ylim_LOWER = min(ylim_lower, ylims[0])
        ylim_UPPER = max(ylim_upper, ylims[1])

    if plt.gca().get_yscale() == "linear" and len(Y) > 1:
        if ylim_LOWER != ylim_UPPER and np.isfinite(ylim_UPPER):
            plt.ylim(ylim_LOWER, ylim_UPPER, auto=None)
        else:
            plt.ylim(bottom=ylim_LOWER, auto=None)


def generate_X_array(dist, xvals=None, xmin=None, xmax=None):
    """
    Generates the array of X values for each of the PDf, CDF, SF, HF, CHF
    functions within reliability.Distributions

    This is done with a variety of cases in order to ensure that for regions of
    high gradient (particularly asymptotes to inf) the points are more
    concentrated. This ensures that the line always looks as smooth as possible
    using only 200 data points.

    Parameters
    ----------
    dist : object
        The distribution object
    xvals : array, list, optional
        The xvals for the plot if specified
    xmin : array, list, optional
        The xmin for the plot if specified
    xmax : array, list, optional
        The xmax for the plot if specified

    Returns
    -------
    X : array
        The X array that was generated.
    """

    # obtain the xvals array
    points = 200  # the number of points to use when generating the X array
    points_right = 25  # the number of points given to the area above QU. The total points is still equal to 'points' so the area below QU receives 'points - points_right'
    QL = dist.quantile(0.0001)  # quantile lower
    QU = dist.quantile(0.99)  # quantile upper
    if xvals is not None:
        X = xvals
        if type(X) in [float, int, np.float64]:
            if X < 0 and dist.name not in ["Normal", "Gumbel"]:
                raise ValueError("the value given for xvals is less than 0")
            if X > 1 and dist.name == "Beta":
                raise ValueError(
                    "the value given for xvals is greater than 1. The beta distribution is bounded between 0 and 1."
                )
            X = np.array([X])
        elif type(X) is list:
            X = np.array(X)
        elif type(X) is np.ndarray:
            pass
        else:
            raise ValueError(
                "unexpected type in xvals. Must be int, float, list, or array"
            )
        if (
            type(X) is np.ndarray
            and min(X) < 0
            and dist.name not in ["Normal", "Gumbel"]
        ):
            raise ValueError("xvals was found to contain values below 0")
        if type(X) is np.ndarray and max(X) > 1 and dist.name == "Beta":
            raise ValueError(
                "xvals was found to contain values above 1. The beta distribution is bounded between 0 and 1."
            )
    else:
        if dist.name in ["Weibull", "Lognormal", "Loglogistic", "Exponential", "Gamma"]:
            if xmin is None:
                xmin = 0
            if xmin < 0:
                raise ValueError(
                    "xmin must be greater than or equal to 0 for all distributions except the Normal and Gumbel distributions"
                )
            if xmax is None:
                xmax = dist.quantile(0.9999)
            if xmin > xmax:
                xmin, xmax = (
                    xmax,
                    xmin,
                )  # switch them if they are given in the wrong order
            if (
                (xmin < QL and xmax < QL)
                or (xmin >= QL and xmax <= QU)
                or (xmin > QU and xmax > QU)
            ):
                X = np.linspace(xmin, xmax, points)
            elif xmin < QL and xmax > QL and xmax < QU:
                if dist.gamma == 0:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                    else:  # pdf is asymptotic to inf at x=0
                        X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
                else:  # gamma > 0
                    if dist._pdf0 == 0:
                        X = np.hstack(
                            [xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)]
                        )
                    else:  # pdf is asymptotic to inf at x=0
                        detail = (
                            np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2)
                            + dist.gamma
                        )
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail])
            elif xmin > QL and xmin < QU and xmax > QU:
                if dist._pdf0 == 0:
                    X = np.hstack(
                        [
                            np.linspace(xmin, QU, points - points_right),
                            np.linspace(QU, xmax, points_right),
                        ]
                    )
                else:  # pdf is asymptotic to inf at x=0
                    try:
                        detail = (
                            np.geomspace(
                                xmin - dist.gamma,
                                QU - dist.gamma,
                                points - points_right,
                            )
                            + dist.gamma
                        )
                        right = (
                            np.geomspace(
                                QU - dist.gamma, xmax - dist.gamma, points_right
                            )
                            + dist.gamma
                        )
                    except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                        detail = np.linspace(xmin, QU, points - points_right)
                        right = np.linspace(QU, xmax, points_right)
                    X = np.hstack([detail, right])
            else:  # xmin < QL and xmax > QU
                if dist.gamma == 0:
                    if dist._pdf0 == 0:
                        X = np.hstack(
                            [
                                xmin,
                                np.linspace(QL, QU, points - (points_right + 1)),
                                np.geomspace(QU, xmax, points_right),
                            ]
                        )
                    else:  # pdf is asymptotic to inf at x=0
                        try:
                            X = np.hstack(
                                [
                                    xmin,
                                    np.geomspace(QL, QU, points - (points_right + 1)),
                                    np.geomspace(QU, xmax, points_right),
                                ]
                            )
                        except ValueError:  # occurs for very low shape params causing QL to be zero
                            X = np.hstack(
                                [
                                    xmin,
                                    np.linspace(QL, QU, points - (points_right + 1)),
                                    np.geomspace(QU, xmax, points_right),
                                ]
                            )
                else:  # gamma > 0
                    if dist._pdf0 == 0:
                        X = np.hstack(
                            [
                                xmin,
                                dist.gamma - 1e-8,
                                np.linspace(QL, QU, points - (points_right + 2)),
                                np.geomspace(
                                    QU - dist.gamma, xmax - dist.gamma, points_right
                                )
                                + dist.gamma,
                            ]
                        )
                    else:  # pdf is asymptotic to inf at x=0
                        try:
                            detail = (
                                np.geomspace(
                                    QL - dist.gamma,
                                    QU - dist.gamma,
                                    points - (points_right + 2),
                                )
                                + dist.gamma
                            )
                            right = (
                                np.geomspace(
                                    QU - dist.gamma, xmax - dist.gamma, points_right
                                )
                                + dist.gamma
                            )
                        except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                            detail = np.linspace(QL, QU, points - (points_right + 2))
                            right = np.linspace(QU, xmax, points_right)
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail, right])
        elif dist.name in ["Normal", "Gumbel"]:
            if xmin is None:
                xmin = dist.quantile(0.0001)
            if xmax is None:
                xmax = dist.quantile(0.9999)
            if xmin > xmax:
                xmin, xmax = (
                    xmax,
                    xmin,
                )  # switch them if they are given in the wrong order
            if xmin <= 0 or xmin > dist.quantile(0.0001):
                X = np.linspace(xmin, xmax, points)
            else:
                X = np.hstack(
                    [0, np.linspace(xmin, xmax, points - 1)]
                )  # this ensures that the distribution is at least plotted from 0 if its xmin is above 0
        elif dist.name == "Beta":
            if xmin is None:
                xmin = 0
            if xmax is None:
                xmax = 1
            if xmax > 1:
                raise ValueError(
                    "xmax must be less than or equal to 1 for the beta distribution"
                )
            if xmin > xmax:
                xmin, xmax = (
                    xmax,
                    xmin,
                )  # switch them if they are given in the wrong order
            X = np.linspace(xmin, xmax, points)
        else:
            raise ValueError("Unrecognised distribution name")
    return X


def no_reverse(x, CI_type, plot_type):
    """
    This is used to convert an array that decreases and then increases into an
    array that decreases then is constant at its minimum.

    The always decreasing rule will apply unless CI_type = 'time' and
    plot_type = 'CHF'

    This function is used to provide a correction to the confidence intervals
    which mathematically are correct but practically should never decrease.

    Parameters
    ----------
    x : array, list
        The array or list to which the no_reverse correction is applied
    CI_type : str
        Must be either 'time' or 'reliability'
    plot_type : str
        Must be either 'CDF', 'SF', or 'CHF'

    Returns
    -------
    x : array
        A corrected form of the input x that obeys the always decreasing rule
        (or the always increasing rule in the case of CI_type = 'time' and
        plot_type = 'CHF').
    """
    if type(x) not in [np.ndarray, list]:
        raise ValueError("x must be a list or array")
    if len(x) < 2:
        raise ValueError("x must be a list or array with length greater than 1")
    if CI_type == "time" and plot_type == "CHF":
        decreasing = False
    else:
        decreasing = True

    x = np.copy(np.asarray(x))
    if all(np.isfinite(x)):
        # it will not work if there are any nans
        if decreasing is True:
            idxmin = np.where(x == min(x))[0][0]
            if idxmin < len(x) - 1:
                x[idxmin::] = min(x)
        elif decreasing is False:
            idxmax = np.where(x == max(x))[0][0]
            if idxmax < len(x) - 1:
                x[idxmax::] = max(x)
        else:
            return ValueError("The parameter 'decreasing' must be True or False")
    return x


def zeroise_below_gamma(X, Y, gamma):
    """
    This will make all Y values 0 for the corresponding X values being below
    gamma (the threshold parameter for Weibull, Exponential, Gamma, Loglogistic,
    and Lognormal).

    Used by HF and CHF which need to be zeroized if the gamma shifted form of
    the equation is used.

    Parameters
    ----------
    X : array, list
        The x values of the distribution. These areused to determine whis Y
        values to zeroize.
    Y : array, list
        The y-values of the distribution
    gamma : float, int
        The gamma parameter. This is the point at which Y values corresponding
        to X values below gamma will be zeroized.

    Returns
    -------
    Y : array
        The zeroized Y array
    """
    if gamma > 0:
        if len(np.where(X > gamma)[0]) == 0:
            Y[0::] = 0  # zeroize everything if there is no X values above gamma
        else:
            Y[0 : (np.where(X > gamma)[0][0])] = 0  # zeroize below X=gamma
    return Y


def xy_transform(value, direction="forward", axis="x"):
    """
    This function converts between data values and axes coordinates (based on
    xlim() or ylim()).

    If direction is forward the returned value will always be between 0 and 1
    provided the value is on the plot.

    If direction is reverse the input should be between 0 and 1 and the returned
    value will be the data value based on the current plot lims

    Parameters
    ----------
    value: int, float, list, array
        The value/s to be transformed
    direction : str, optional
        Must be "forward" or "inverse". Default is "forward"
    axis : str, optional
        Must be "x" or "y". Default is "x".

    Returns
    -------
    transformed_values : float, array
        The transformed values. This will be a float if the input was int or
        float, or an array if the input was list or array.

    """
    if direction not in ["reverse", "inverse", "inv", "rev", "forward", "fwd"]:
        raise ValueError('direction must be "forward" or "reverse"')
    if axis not in ["X", "x", "Y", "y"]:
        raise ValueError("axis must be x or y. Default is x")

    ax = plt.gca()
    if direction in ["reverse", "inverse", "inv", "rev"]:
        if type(value) in [int, float, np.float64]:
            if axis == "x":
                # x transform
                transformed_values = ax.transData.inverted().transform(
                    (ax.transAxes.transform((value, 0.5))[0], 0.5)
                )[0]
            else:
                # y transform
                transformed_values = ax.transData.inverted().transform(
                    (1, ax.transAxes.transform((1, value))[1])
                )[1]
        elif type(value) in [list, np.ndarray]:
            transformed_values = []
            for item in value:
                if axis == "x":
                    transformed_values.append(
                        ax.transData.inverted().transform(
                            (ax.transAxes.transform((item, 0.5))[0], 0.5)
                        )[0]
                    )  # x transform
                else:
                    transformed_values.append(
                        ax.transData.inverted().transform(
                            (1, ax.transAxes.transform((1, item))[1])
                        )[1]
                    )  # y transform
        else:
            raise ValueError("type of value is not recognized")
    else:  # direction is forward
        if type(value) in [int, float, np.float64]:
            if axis == "x":
                transformed_values = ax.transAxes.inverted().transform(
                    ax.transData.transform((value, 0.5))
                )[
                    0
                ]  # x transform
            else:
                transformed_values = ax.transAxes.inverted().transform(
                    ax.transData.transform((1, value))
                )[
                    1
                ]  # y transform
        elif type(value) in [list, np.ndarray]:
            transformed_values = []
            for item in value:
                if axis == "x":
                    transformed_values.append(
                        ax.transAxes.inverted().transform(
                            ax.transData.transform((item, 0.5))
                        )[0]
                    )  # x transform
                else:
                    transformed_values.append(
                        ax.transAxes.inverted().transform(
                            ax.transData.transform((1, value))
                        )[1]
                    )  # y transform
        else:
            raise ValueError("type of value is not recognized")
    return transformed_values


def probability_plot_xylims(
    x, y, dist, spacing=0.1, gamma_beta=None, beta_alpha=None, beta_beta=None
):
    """
    This function finds what the x and y limits of probability plots should be
    and sets these limits. This is similar to autoscaling, but the rules here
    are different to the matplotlib defaults.
    It is used extensively by the functions within the probability_plotting
    module to achieve the plotting style used within this library.

    Parameters
    ----------
    x : list, array
        The x-values from the plot
    y : list, array
        The y-values from the plot
    dist : str
        Must be either "weibull", "lognormal", "loglogistic", "normal", "gamma",
        "exponential", "beta", or "gumbel".
    spacing : float
        The spacing between the points and the edge of the plot. Default is 0.1
        for 10% spacing.
    gamma_beta : float, int, optional
        The beta parameter from the gamma distribution. Only required if dist =
        "gamma".
    beta_alpha : float, int, optional
        The alpha parameter from the beta distribution. Only required if dist =
        "beta".
    beta_beta : float, int, optional
        The beta parameter from the beta distribution. Only required if dist =
        "beta".

    Returns
    -------
    None
        There are no outputs from this function. It will set the xlim() and
        ylim() of the probability plot automatically.
    """

    # remove inf
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    # x limits
    if dist in ["weibull", "lognormal", "loglogistic"]:
        min_x_log = np.log10(min_x)
        max_x_log = np.log10(max_x)
        dx_log = max_x_log - min_x_log
        xlim_lower = 10 ** (min_x_log - dx_log * spacing)
        xlim_upper = 10 ** (max_x_log + dx_log * spacing)
        if xlim_lower == xlim_upper:
            xlim_lower = 10 ** (np.log10(xlim_lower) - 10 * spacing)
            xlim_upper = 10 ** (np.log10(xlim_upper) + 10 * spacing)
    elif dist in ["normal", "gamma", "exponential", "beta", "gumbel"]:
        dx = max_x - min_x
        xlim_lower = min_x - dx * spacing
        xlim_upper = max_x + dx * spacing
        if xlim_lower == xlim_upper:
            xlim_lower = 0
            xlim_upper = xlim_upper * 2
    else:
        raise ValueError("dist is unrecognised")
    if xlim_lower < 0 and dist not in ["normal", "gumbel"]:
        xlim_lower = 0
    # set xlims
    plt.xlim(xlim_lower, xlim_upper)

    # y limits
    if dist == "weibull":
        min_y_tfm = axes_transforms.weibull_forward(min_y)
        max_y_tfm = axes_transforms.weibull_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.weibull_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.weibull_inverse(max_y_tfm + dy_tfm * spacing)
    if dist == "exponential":
        min_y_tfm = axes_transforms.exponential_forward(min_y)
        max_y_tfm = axes_transforms.exponential_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.exponential_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.exponential_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gamma":
        min_y_tfm = axes_transforms.gamma_forward(min_y, gamma_beta)
        max_y_tfm = axes_transforms.gamma_forward(max_y, gamma_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gamma_inverse(
            min_y_tfm - dy_tfm * spacing, gamma_beta
        )
        ylim_upper = axes_transforms.gamma_inverse(
            max_y_tfm + dy_tfm * spacing, gamma_beta
        )
    elif dist in ["normal", "lognormal"]:
        min_y_tfm = axes_transforms.normal_forward(min_y)
        max_y_tfm = axes_transforms.normal_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.normal_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.normal_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gumbel":
        min_y_tfm = axes_transforms.gumbel_forward(min_y)
        max_y_tfm = axes_transforms.gumbel_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gumbel_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.gumbel_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "beta":
        min_y_tfm = axes_transforms.beta_forward(min_y, beta_alpha, beta_beta)
        max_y_tfm = axes_transforms.beta_forward(max_y, beta_alpha, beta_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.beta_inverse(
            min_y_tfm - dy_tfm * spacing, beta_alpha, beta_beta
        )
        ylim_upper = axes_transforms.beta_inverse(
            max_y_tfm + dy_tfm * spacing, beta_alpha, beta_beta
        )
    elif dist == "loglogistic":
        min_y_tfm = axes_transforms.loglogistic_forward(min_y)
        max_y_tfm = axes_transforms.loglogistic_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.loglogistic_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.loglogistic_inverse(max_y_tfm + dy_tfm * spacing)
    if ylim_upper == ylim_lower:
        dx = min(1 - ylim_upper, ylim_upper - 1)
        ylim_upper = ylim_upper - spacing * dx
        ylim_lower = ylim_lower + spacing * dx

    # correction for the case where ylims are is 0 or 1
    if ylim_lower == 0:
        if min_y > 0:
            ylim_lower = min_y
        else:
            ylim_lower = 0.00001
    if ylim_upper == 1:
        if max_y < 1:
            ylim_upper = max_y
        else:
            ylim_upper = 0.99999
    # set ylims
    plt.ylim(ylim_lower, ylim_upper)


def probability_plot_xyticks(yticks=None):
    """
    This function sets the x and y ticks for probability plots.

    X ticks are selected using either MaxNLocator or LogLocator. X ticks are
    formatted using a custom formatter.

    Y ticks are specified with FixedLocator due to their irregular spacing.
    Minor y ticks use MaxNLocator. Y ticks are formatted using a custom Percent
    Formatter that handles decimals better than the default.

    This function is used by all the probability plots.

    Within this function are several sub functions that are called internally.

    Parameters
    ----------
    yticks : list, array
        The yticks to use. If unspecified, the default yticks are [0.0001, 0.001,
        0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.999999].

    Returns
    -------
    None
        This function will set the ticks but it does not return anything.
    """

    def get_tick_locations(major_or_minor, in_lims=True, axis="x"):
        """
        returns the major or minor tick locations for the current axis.

        Parameters
        ----------
        major_or_minor : str
            Specifies which ticks to get. Must be either "major" or
            "minor".
        in_lims : bool, optional
            If in_lims=True then it will only return the ticks that are within
            the current xlim() or ylim(). Default is True.
        axis : str, optional
            Specifies which axis to get the ticks for. Must be "x" or "y".
            Default is "x".

        Returns
        -------
        locations : array
            The tick locations
        """
        if axis == "x":
            AXIS = ax.xaxis
            L = xlower
            U = xupper
        elif axis == "y":
            AXIS = ax.yaxis
            L = ylower
            U = yupper
        else:
            raise ValueError("axis must be x or y. Default is x")

        if major_or_minor == "major":
            all_locations = AXIS.get_major_locator().tick_values(L, U)
        elif major_or_minor == "minor":
            all_locations = AXIS.get_minor_locator().tick_values(L, U)
        else:
            raise ValueError('major_or_minor must be "major" or "minor"')
        if in_lims is True:
            locations = []
            for item in all_locations:
                if item >= L and item <= U:
                    locations.append(item)
        else:
            locations = all_locations
        return locations

    def customFormatter(value, _):
        """
        Provides custom string formatting that is used for the xticks

        Parameters
        ----------
        value : int, float
            The value to be formatted

        Returns
        -------
        label : str
            The formatted string
        """
        if value == 0:
            label = "0"
        elif (
            abs(value) >= 10000 or abs(value) <= 0.0001
        ):  # small numbers and big numbers are formatted with scientific notation
            if value < 0:
                sign = "-"
                value *= -1
            else:
                sign = ""
            exponent = int(np.floor(np.log10(value)))
            multiplier = value / (10**exponent)
            if multiplier % 1 < 0.0000001:
                multiplier = int(multiplier)
            if multiplier == 1:
                label = str((r"$%s%s^{%d}$") % (sign, 10, exponent))
            else:
                label = str((r"$%s%g\times%s^{%d}$") % (sign, multiplier, 10, exponent))
        else:  # numbers between 0.0001 and 10000 are formatted without scientific notation
            label = str("{0:g}".format(value))
        return label

    def customPercentFormatter(value, _):
        """
        Provides custom percent string formatting that is used for the yticks
        This function is slightly different than matplotlib's PercentFormatter
        as it does not force a particular number of decimals. ie. in this
        function 99.00 becomes 99 while 99.99 still displays as such.

        Parameters
        ----------
        value : int, float
            The value to be formatted

        Returns
        -------
        label : str
            The formatted string
        """
        value100 = value * 100
        value100dec = round(
            value100 % 1, 8
        )  # this breaks down after 8 decimal places due to python's auto rounding. Not likely to be an issue as we're rarely dealing with this many decimals
        if value100dec == 0:
            value100dec = int(value100dec)
        value100whole = int(value100 - value100dec)
        combined = value100dec + value100whole
        label = str(str(combined) + str("%"))
        return label

    def get_edge_distances():
        """
        Finds the sum of the distance (in axes coords (0 to 1)) of the distances
        from the edge ticks to the edges

        Parameters
        ----------
        None

        Returns
        -------
        distances : float
            The edge distances
        """
        xtick_locations = get_tick_locations("major", axis="x")
        if len(xtick_locations)<2:
            return 1 # this deals with very small numbers which cause one or no major ticks. It will trigger the get_edge_distances() > 0.5 on line 1249. This resolves issue #55 on github
        else:
            left_tick_distance = xy_transform(
                xtick_locations[0], direction="forward", axis="x"
            ) - xy_transform(xlower, direction="forward", axis="x")
            right_tick_distance = xy_transform(
                xupper, direction="forward", axis="x"
            ) - xy_transform(xtick_locations[-1], direction="forward", axis="x")
            return left_tick_distance + right_tick_distance

    ################# xticks
    MaxNLocator = ticker.MaxNLocator(nbins=10, min_n_ticks=2, steps=[1, 2, 5, 10])
    LogLocator = ticker.LogLocator()
    ax = plt.gca()
    xlower, xupper = plt.xlim()
    if (
        xlower <= 0
    ):  # can't use log scale if 0 (gamma) or negative numbers (normal and gumbel)
        loc_x = MaxNLocator
    elif xupper < 0.1:  # very small positive values
        loc_x = ticker.LogLocator()
    elif (
        xupper < 1000 or np.log10(xupper) - np.log10(xlower) < 1.5
    ):  # not too big and not too small OR it may be big but not too spread out
        loc_x = MaxNLocator
    else:  # it is really big (>1000) and spread out
        loc_x = ticker.LogLocator()
    ax.xaxis.set_major_locator(loc_x)  # apply the tick locator
    # do not apply a minor locator. It is never as good as the default

    if (
        get_edge_distances() > 0.5
    ):  # 0.5 means 50% of the axis is without ticks on either side. Above this is considered unacceptable. This has a weakness where there's only 1 tick it will return 0. Changing 0 to 1 can make things too crowded
        # find which locator is better
        ax.xaxis.set_major_locator(MaxNLocator)
        edges_maxNLocator = get_edge_distances()
        ax.xaxis.set_major_locator(LogLocator)
        edges_LogLocator = get_edge_distances()
        if edges_LogLocator < edges_maxNLocator:
            ax.xaxis.set_major_locator(LogLocator)  # apply a new locator
        else:
            ax.xaxis.set_major_locator(MaxNLocator)  # apply a new locator
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(customFormatter)
    )  # the custom formatter is always applied to the major ticks

    num_major_x_ticks_shown = len(get_tick_locations("major", axis="x"))
    num_minor_x_xticks_shown = len(get_tick_locations("minor", axis="x"))
    if max(abs(xlower), abs(xupper)) < 1000 and min(abs(xlower), abs(xupper)) > 0.001:
        max_minor_ticks = 15
    else:
        max_minor_ticks = 10
    if num_major_x_ticks_shown < 2 and num_minor_x_xticks_shown <= max_minor_ticks:
        ax.xaxis.set_minor_formatter(
            ticker.FuncFormatter(customFormatter)
        )  # if there are less than 2 major ticks within the plotting limits then the minor ticks should be labeled. Only do this if there aren't too many minor ticks

    ################# yticks
    if yticks is None:
        yticks = [
            0.0001,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9999,
            0.999999,
        ]
    loc_y = ticker.FixedLocator(yticks)
    loc_y_minor = ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10])
    ax.yaxis.set_major_locator(loc_y)  # sets the tick spacing
    ax.yaxis.set_minor_locator(loc_y_minor)  # sets the tick spacing
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.format_coord = lambda x, y: "x={:g}, y={:.1%}".format(
        x, y
    )  # sets the formatting of the axes coordinates in the bottom right of the figure. Without this the FuncFormatter raw strings make it into the axes coords and don't look good.


def anderson_darling(fitted_cdf, empirical_cdf):
    """
    Calculates the Anderson-Darling goodness of fit statistic.
    These formulas are based on the method used in MINITAB which gives an
    adjusted form of the original AD statistic described on Wikipedia.

    Parameters
    ----------
    fitted_cdf : list, array
        The fitted CDF values at the data points
    empirical_cdf  : list, array
        The empirical (rank adjustment) CDF values at the data points

    Returns
    -------
    AD : float
        The anderson darling (adjusted) test statistic.
    """
    if type(fitted_cdf) != np.ndarray:
        fitted_cdf = [fitted_cdf]  # required when there is only 1 failure
    Z = np.sort(np.asarray(fitted_cdf))
    Zi = np.hstack([Z, 1 - 1e-12])
    Zi_1 = (np.hstack([0, Zi]))[0:-1]  # Z_i-1
    FnZi = np.sort(np.asarray(empirical_cdf))
    FnZi_1 = np.hstack([0, FnZi])  # Fn(Z_i-1)
    lnZi = np.log(Zi)
    lnZi_1 = np.hstack([0, lnZi])[0:-1]

    A = -Zi - np.log(1 - Zi) + Zi_1 + np.log(1 - Zi_1)
    B = 2 * np.log(1 - Zi) * FnZi_1 - 2 * np.log(1 - Zi_1) * FnZi_1
    C = (
        lnZi * FnZi_1**2
        - np.log(1 - Zi) * FnZi_1**2
        - lnZi_1 * FnZi_1**2
        + np.log(1 - Zi_1) * FnZi_1**2
    )
    n = len(fitted_cdf)
    AD = n * ((A + B + C).sum())
    return AD


def colorprint(
    string,
    text_color=None,
    background_color=None,
    bold=False,
    underline=False,
    italic=False,
):
    """
    Provides easy access to color printing in the console.

    This function is used to print warnings in red text, but it can also do a
    lot more.

    Parameters
    ----------
    string
    text_color : str, None, optional
        Must be either grey, red, green, yellow, blue, pink, or turquoise. Use
        None to leave the color as white. Default is None.
    background_color : str, None, optional
        Must be either grey, red, green, yellow, blue, pink, or turquoise. Use
        None to leave the color as the transparent. Default is None.
    bold : bool, optional
        Default is False.
    underline : bool, optional
        Default is False.
    italic : bool, optional
        Default is False.

    Returns
    -------
    None
        The output is printed to the console.

    Notes
    -----
    Some flexibility in color names is allowed. eg. red and r will both give red.

    As there is only one string argument, if you have multiple strings to print,
    you must first combine them using str(string_1,string_2,...).
    """
    text_colors = {
        "grey": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "pink": "\033[95m",
        "turquoise": "\033[96m",
        None: "\033[39m",
    }

    background_colors = {
        "grey": "\033[100m",
        "red": "\033[101m",
        "green": "\033[102m",
        "yellow": "\033[103m",
        "blue": "\033[104m",
        "pink": "\033[105m",
        "turquoise": "\033[106m",
        None: "\033[49m",
    }

    if bold is True:
        BOLD = "\033[1m"
    else:
        BOLD = "\033[21m"

    if underline is True:
        UNDERLINE = "\033[4m"
    else:
        UNDERLINE = "\033[24m"

    if italic is True:
        ITALIC = "\033[3m"
    else:
        ITALIC = "\033[23m"

    if type(text_color) not in [str, np.str_, type(None)]:
        raise ValueError("text_color must be a string")
    elif text_color is None:
        pass
    elif text_color.upper() in ["GREY", "GRAY", "GR"]:
        text_color = "grey"
    elif text_color.upper() in ["RED", "R"]:
        text_color = "red"
    elif text_color.upper() in ["GREEN", "G"]:
        text_color = "green"
    elif text_color.upper() in ["YELLOW", "Y"]:
        text_color = "yellow"
    elif text_color.upper() in ["BLUE", "B", "DARKBLUE", "DARK BLUE"]:
        text_color = "blue"
    elif text_color.upper() in ["PINK", "P", "PURPLE"]:
        text_color = "pink"
    elif text_color.upper() in [
        "TURQUOISE",
        "TURQ",
        "T",
        "CYAN",
        "C",
        "LIGHTBLUE",
        "LIGHT BLUE",
        "LB",
    ]:
        text_color = "turquoise"
    else:
        raise ValueError(
            "Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise."
        )

    if type(background_color) not in [str, np.str_, type(None)]:
        raise ValueError("background_color must be a string")
    if background_color is None:
        pass
    elif background_color.upper() in ["GREY", "GRAY", "GR"]:
        background_color = "grey"
    elif background_color.upper() in ["RED", "R"]:
        background_color = "red"
    elif background_color.upper() in ["GREEN", "G"]:
        background_color = "green"
    elif background_color.upper() in ["YELLOW", "Y"]:
        background_color = "yellow"
    elif background_color.upper() in ["BLUE", "B", "DARKBLUE", "DARK BLUE"]:
        background_color = "blue"
    elif background_color.upper() in ["PINK", "P", "PURPLE"]:
        background_color = "pink"
    elif background_color.upper() in [
        "TURQUOISE",
        "TURQ",
        "T",
        "CYAN",
        "C",
        "LIGHTBLUE",
        "LIGHT BLUE",
        "LB",
    ]:
        background_color = "turquoise"
    else:
        raise ValueError(
            "Unknown text_color. Options are grey, red, green, yellow, blue, pink, turquoise."
        )

    print(
        BOLD
        + ITALIC
        + UNDERLINE
        + background_colors[background_color]
        + text_colors[text_color]
        + string
        + "\033[0m"
    )


class fitters_input_checking:
    """
    This function performs error checking and some basic default operations for
    all the inputs given to each of the fitters.

    Parameters
    ----------
    dist : str
        Must be one of "Everything", "Weibull_2P", "Weibull_3P", "Gamma_2P",
        "Gamma_3P", "Exponential_1P", "Exponential_2P", "Gumbel_2P",
        "Normal_2P", "Lognormal_2P", "Lognormal_3P", "Loglogistic_2P",
        "Loglogistic_3P", "Beta_2P", "Weibull_Mixture", "Weibull_CR",
        "Weibull_DSZI", "Weibull_DS", "Weibull_ZI".
    failures : array, list
        The failure data
    right_censored : array, list, optional
        The right censored data
    method : str, optional
        Must be either "MLE","LS","RRX", or "RRY". Some flexibility in input is
        tolerated. eg "LS", "LEAST SQUARES", "LSQ", "NLRR", "NLLS" will all be
        recogsised as "LS". Default is MLE
    optimizer : str, optional
        Must be one of "TNC", "L-BFGS-B", "nelder-mead", "powell", "best".
        Default is None which will result in each being tried until one
        succeeds. For more detail see the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default is 0.95 for 95%
        confidence interval (2 sided).
    quantiles : array, list, bool, optional
        An array or list of the quantiles to calculate. If True then the
        default array will be used. Default array is [0.01, 0.05, 0.1, 0.2, 0.25,
        0.5, 0.75, 0.8, 0.9, 0.95, 0.99].
        If False then no quantiles will be calculated. Default is False.
    force_beta : float, int, optional
        Used to force beta for the Weibull_2P distribution. Default is None
        which will not force beta.
    force_sigma : float, int, optional
        Used to force sigma for the Normal_2P and Lognormal_2P distributions.
        Default is None which will not force sigma.
    CI_type : str, optional
        Must be either "time" or "reliability". Default is None which results in
        "time" being used (controlled in Fitters). Some flexibility is strings
        is allowed. eg. "r", "R", "rel", "REL", "reliability", "RELIABILITY"
        will all be recognized as "reliability".

    Returns
    -------
    failures : array
        The failure times
    right_censored : array
        The right censored times. This will be an empty array if the input was
        None.
    CI : float
        The confidence interval (between 0 and 1)
    method : str, None
        This will return "MLE", "LS", "RRX", "RRY" or None.
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None.
    quantiles : array, None
        The quantiles or None.
    force_beta : float, None
        The beta parameter to be forced in Weibull_2P
    force_sigma : float, None
            The sigma parameter to be forced in Normal_2P, or Lognormal_2P
    CI_type : str, None
        "time", "reliability", 'None' or None

    Notes
    -----
    For full detail on what is checked and the errors produced, you should read
    the source code.

    Some returns are None if the input is None. How None affects the behavior
    is governed by other functions such as the individual fitters and other
    Utils.
    """

    def __init__(
        self,
        dist,
        failures,
        right_censored=None,
        method=None,
        optimizer=None,
        CI=0.95,
        quantiles=False,
        force_beta=None,
        force_sigma=None,
        CI_type=None,
    ):

        if dist not in [
            "Everything",
            "Weibull_2P",
            "Weibull_3P",
            "Gamma_2P",
            "Gamma_3P",
            "Exponential_1P",
            "Exponential_2P",
            "Gumbel_2P",
            "Normal_2P",
            "Lognormal_2P",
            "Lognormal_3P",
            "Loglogistic_2P",
            "Loglogistic_3P",
            "Beta_2P",
            "Weibull_Mixture",
            "Weibull_CR",
            "Weibull_DSZI",
            "Weibull_DS",
            "Weibull_ZI",
        ]:
            raise ValueError(
                "incorrect dist specified. Use the correct name. eg. Weibull_2P"
            )

        # fill right_censored with empty list if not specified
        if right_censored is None:
            right_censored = []

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            raise ValueError("failures must be a list or array of failure data")
        if type(right_censored) not in [list, np.ndarray]:
            raise ValueError(
                "right_censored must be a list or array of right censored failure data"
            )
        failures = np.asarray(failures).astype(float)
        right_censored = np.asarray(right_censored).astype(float)

        # check failures and right_censored are in the right range for the distribution
        if dist not in ["Normal_2P", "Gumbel_2P"]:
            # raise an error for values below zero
            all_data = np.hstack([failures, right_censored])
            if dist == "Beta_2P" and (min(all_data) < 0 or max(all_data) > 1):
                raise ValueError(
                    "All failure and censoring times for the beta distribution must be between 0 and 1."
                )
            elif min(all_data) < 0:
                raise ValueError(
                    "All failure and censoring times must be greater than zero."
                )
            # remove zeros and issue a warning. These are impossible since the pdf should be 0 at t=0. Leaving them in causes an error.
            rc0 = right_censored
            f0 = failures
            right_censored = rc0[rc0 != 0]
            failures = f0[f0 != 0]
            if len(failures) != len(f0):
                if dist == "Everything":
                    colorprint(
                        "WARNING: failures contained zeros. These have been removed to enable fitting of all distributions. Consider using Fit_Weibull_ZI or Fit_Weibull_DSZI if you need to include the zero inflation in the models.",
                        text_color="red",
                    )
                else:
                    colorprint(
                        str(
                            "WARNING: failures contained zeros. These have been removed to enable fitting of the "
                            + dist
                            + " distribution. Consider using Fit_Weibull_ZI or Fit_Weibull_DSZI if you need to include the zero inflation in the model."
                        ),
                        text_color="red",
                    )

            if len(right_censored) != len(rc0):
                if dist == "Everything":
                    colorprint(
                        "WARNING: right_censored contained zeros. These have been removed to enable fitting of all distributions.",
                        text_color="red",
                    )
                else:
                    colorprint(
                        str(
                            "WARNING: right_censored contained zeros. These have been removed to enable fitting of the "
                            + dist
                            + " distribution."
                        ),
                        text_color="red",
                    )
            if dist == "Beta_2P":
                rc1 = right_censored
                f1 = failures
                right_censored = rc1[rc1 != 1]
                failures = f1[f1 != 1]
                if len(failures) != len(f1):
                    colorprint(
                        "WARNING: failures contained ones. These have been removed to enable fitting of the Beta_2P distribution.",
                        text_color="red",
                    )
                if len(right_censored) != len(rc1):
                    colorprint(
                        "WARNING: right_censored contained ones. These have been removed to enable fitting of the Beta_2P distribution.",
                        text_color="red",
                    )

        # type and value checking for CI
        if type(CI) not in [float, np.float64]:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        # error checking for optimizer
        if optimizer is not None:
            if type(optimizer) is not str:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                )
            if optimizer.upper() == "TNC":
                optimizer = "TNC"
            elif optimizer.upper() == "POWELL":
                optimizer = "powell"
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizer = "L-BFGS-B"
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD", "NM"]:
                optimizer = "nelder-mead"
            elif optimizer.upper() in ["ALL", "BEST"]:
                optimizer = "best"
            else:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                )

        # error checking for method
        if method is not None:
            if type(method) is not str:
                raise ValueError(
                    'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).'
                )
            if method.upper() == "RRX":
                method = "RRX"
            elif method.upper() == "RRY":
                method = "RRY"
            elif method.upper() in ["LS", "LEAST SQUARES", "LSQ", "NLRR", "NLLS"]:
                method = "LS"
            elif method.upper() in [
                "MLE",
                "ML",
                "MAXIMUM LIKELIHOOD ESTIMATION",
                "MAXIMUM LIKELIHOOD",
                "MAX LIKELIHOOD",
            ]:
                method = "MLE"
            else:
                raise ValueError(
                    'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).'
                )

        # quantiles error checking
        if type(quantiles) in [str, bool]:
            if quantiles in ["auto", True, "default", "on"]:
                # quantiles to be used as the defaults in the table of quantiles #
                quantiles = np.array(
                    [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
                )
        elif quantiles is not None:
            if type(quantiles) not in [list, np.ndarray]:
                raise ValueError("quantiles must be a list or array")
            quantiles = np.asarray(quantiles)
            if max(quantiles) >= 1 or min(quantiles) <= 0:
                raise ValueError("quantiles must be between 0 and 1")

        # force_beta and force_sigma error checking
        if force_beta is not None:
            if force_beta <= 0:
                raise ValueError("force_beta must be greater than 0.")
            if type(force_beta) == int:
                force_beta = float(
                    force_beta
                )  # autograd needs floats. crashes with ints
        if force_sigma is not None:
            if force_sigma <= 0:
                raise ValueError("force_sigma must be greater than 0.")
            if type(force_sigma) == int:
                force_sigma = float(
                    force_sigma
                )  # autograd needs floats. crashes with ints

        # minimum number of failures checking
        if dist in ["Weibull_3P", "Gamma_3P", "Lognormal_3P", "Loglogistic_3P"]:
            min_failures = 3
        elif dist in [
            "Weibull_2P",
            "Gamma_2P",
            "Normal_2P",
            "Lognormal_2P",
            "Gumbel_2P",
            "Loglogistic_2P",
            "Beta_2P",
            "Exponential_2P",
            "Everything",
            "Weibull_ZI",
            "Weibull_DS",
            "Weibull_DSZI",
        ]:
            if force_sigma is None and force_beta is None:
                min_failures = 2
            else:
                min_failures = 1
        elif dist == "Exponential_1P":
            min_failures = 1
        elif dist in ["Weibull_Mixture", "Weibull_CR"]:
            min_failures = 4

        number_of_unique_failures = len(
            np.unique(failures)
        )  # failures need to be unique. ie. [4,4] counts as 1 distinct failure
        if number_of_unique_failures < min_failures:
            if force_beta is not None:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution with force_beta specified is "
                        + str(min_failures)
                        + "."
                    )
                )
            elif force_sigma is not None:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution with force_sigma specified is "
                        + str(min_failures)
                        + "."
                    )
                )
            elif dist == "Everything":
                raise ValueError(
                    "The minimum number of distinct failures required to fit everything is "
                    + str(min_failures)
                    + "."
                )
            else:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution is "
                        + str(min_failures)
                        + "."
                    )
                )

        # error checking for CI_type
        if type(CI_type) not in [str, type(None)]:
            raise ValueError('CI_type must be "time", "reliability", or "none"')
        if CI_type is not None:
            if CI_type.upper() in ["T", "TIME"]:
                CI_type = "time"
            elif CI_type.upper() in ["R","REL","RELIABILITY"]:
                CI_type = "reliability"
            elif CI_type.upper() in ["NONE","OFF"]:
                CI_type = 'none'
            else:
                raise ValueError('CI_type must be "time", "reliability", or "none"')

        # return everything
        self.failures = failures
        self.right_censored = right_censored
        self.CI = CI
        self.method = method
        self.optimizer = optimizer
        self.quantiles = quantiles
        self.force_beta = force_beta
        self.force_sigma = force_sigma
        self.CI_type = CI_type


class ALT_fitters_input_checking:
    """
    This function performs error checking and some basic default operations for
    all the inputs given to each of the ALT_fitters.

    Parameters
    ----------
    dist : str
        Must be one of "Exponential", "Weibull", "Lognormal", "Normal",
        "Everything".
    life_stress_model : str
        Must be one of "Exponential", "Eyring", "Power","Dual_Exponential",
        "Power_Exponential", "Dual_Power", "Everything".
    failures : array, list
        The failure data
    failure_stress_1 : array, list
        The stresses corresponding to the failure data
    failure_stress_2 : array, list, optional
        The second stresses corresponding to the failure data. Only required for
        dual stress models. Default is None.
    right_censored : array, list, optional
        The right censored data. Default is None.
    right_censored_stress_1 : array, list, optional
        The stresses corresponding to the right censored data. Default is None.
    right_censored_stress_2 : array, list, optional
        The second stresses corresponding to the right censored data. Only
        required for dual stress models. Default is None.
    CI : float, optional
        The confidence interval (between 0 and 1). Default is 0.95 for 95%
        confidence interval (two sided).
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None. Default is None.
    use_level_stress : float, int, list, array, optional
        The use level stress. Must be float or int for single stress models.
        Must be array or list [stress_1, stress_2] for dual stress models.
        Default is None.

    Returns
    -------
    failures : array
        The failure times
    failure_stress_1 : array
        The failure stresses
    failure_stress_2 : array
        The second failure stresses. This will be an empty array if the input
        was None.
    right_censored : array
        The right censored times. This will be an empty array if the input was
        None.
    right_censored_stress_1 : array
        The right censored failure stresses. This will be an empty array if the
        input was None.
    right_censored_stress_2 : array
        The right censored second failure stresses. This will be an empty array
        if the input was None.
    CI : float
        The confidence interval (between 0 and 1)
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None.
    use_level_stress : float, array, None
        The use level stress. This will be a float for single stress models, or
        an array for dual stress models. This will be None if the input was
        None.
    failure_groups : array
        An array of arrays. This is the failure data grouped by failure
        stresses.
    right_censored_groups : array
        An array of arrays. This is the right censored data grouped by right
        censored stresses.
    stresses_for_groups : array
        An array of arrays. These are the stresses for each of the groups.

    Notes
    -----
    For full detail on what is checked and the errors produced, you should read
    the source code.

    Some returns are None if the input is None. How None affects the behavior
    is governed by other functions such as the individual ALT fitters and other
    Utils.
    """

    def __init__(
        self,
        dist,
        life_stress_model,
        failures,
        failure_stress_1,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        CI=0.95,
        use_level_stress=None,
        optimizer=None,
    ):

        if dist not in ["Exponential", "Weibull", "Lognormal", "Normal", "Everything"]:
            raise ValueError(
                "dist must be one of Exponential, Weibull, Lognormal, Normal, Everything."
            )
        if life_stress_model not in [
            "Exponential",
            "Eyring",
            "Power",
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
            "Everything",
        ]:
            raise ValueError(
                "life_stess_model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power, Everything."
            )

        if life_stress_model == "Everything":
            if failure_stress_2 is not None:
                is_dual_stress = True
                min_failures_reqd = 4
            else:
                is_dual_stress = False
                min_failures_reqd = 3
        elif life_stress_model in [
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
        ]:
            is_dual_stress = True
            min_failures_reqd = 4
        else:
            is_dual_stress = False
            min_failures_reqd = 3

        # failure checks
        if is_dual_stress is True and (
            failure_stress_1 is None or failure_stress_2 is None
        ):
            raise ValueError(
                "failure_stress_1 and failure_stress_2 must be provided for dual stress models."
            )
        if is_dual_stress is False:
            if failure_stress_1 is None:
                raise ValueError("failure_stress_1 must be provided")
            if failure_stress_2 is not None:
                colorprint(
                    str(
                        "WARNING: failure_stress_2 is not being used as "
                        + life_stress_model
                        + " is a single stress model."
                    ),
                    text_color="red",
                )
            failure_stress_2 = []

        # right_censored checks
        if right_censored is None:
            if right_censored_stress_1 is not None:
                colorprint(
                    "WARNING: right_censored_stress_1 is not being used as right_censored was not provided.",
                    text_color="red",
                )
            if right_censored_stress_2 is not None:
                colorprint(
                    "WARNING: right_censored_stress_2 is not being used as right_censored was not provided.",
                    text_color="red",
                )
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        else:
            if is_dual_stress is True and (
                right_censored_stress_1 is None or right_censored_stress_2 is None
            ):
                raise ValueError(
                    "right_censored_stress_1 and right_censored_stress_2 must be provided for dual stress models."
                )
            if is_dual_stress is False:
                if right_censored_stress_1 is None:
                    raise ValueError("right_censored_stress_1 must be provided")
                if right_censored_stress_2 is not None:
                    colorprint(
                        str(
                            "WARNING: right_censored_stress_2 is not being used as "
                            + life_stress_model
                            + " is a single stress model."
                        ),
                        text_color="red",
                    )
                right_censored_stress_2 = []

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            raise ValueError("failures must be a list or array of failure data")
        if type(failure_stress_1) not in [list, np.ndarray]:
            raise ValueError(
                "failure_stress_1 must be a list or array of failure stress data"
            )
        if type(failure_stress_2) not in [list, np.ndarray]:
            raise ValueError(
                "failure_stress_2 must be a list or array of failure stress data"
            )

        if type(right_censored) not in [list, np.ndarray]:
            raise ValueError(
                "right_censored must be a list or array of right censored failure data"
            )
        if type(right_censored_stress_1) not in [list, np.ndarray]:
            raise ValueError(
                "right_censored_stress_1 must be a list or array of right censored failure stress data"
            )
        if type(right_censored_stress_2) not in [list, np.ndarray]:
            raise ValueError(
                "right_censored_stress_2 must be a list or array of right censored failure stress data"
            )

        failures = np.asarray(failures).astype(float)
        failure_stress_1 = np.asarray(failure_stress_1).astype(float)
        failure_stress_2 = np.asarray(failure_stress_2).astype(float)
        right_censored = np.asarray(right_censored).astype(float)
        right_censored_stress_1 = np.asarray(right_censored_stress_1).astype(float)
        right_censored_stress_2 = np.asarray(right_censored_stress_2).astype(float)

        # check that list lengths match
        if is_dual_stress is False:
            if len(failures) != len(failure_stress_1):
                raise ValueError(
                    "failures must have the same number of elements as failure_stress_1"
                )
            if len(right_censored) != len(right_censored_stress_1):
                raise ValueError(
                    "right_censored must have the same number of elements as right_censored_stress_1"
                )
        else:
            if len(failures) != len(failure_stress_1) or len(failures) != len(
                failure_stress_2
            ):
                raise ValueError(
                    "failures must have the same number of elements as failure_stress_1 and failure_stress_2"
                )
            if len(right_censored) != len(right_censored_stress_1) or len(
                right_censored
            ) != len(right_censored_stress_2):
                raise ValueError(
                    "right_censored must have the same number of elements as right_censored_stress_1 and right_censored_stress_2"
                )

        # raise an error for values <= 0. Not even the Normal Distribution is allowed to have failures at negative life.
        if min(np.hstack([failures, right_censored])) <= 0:
            raise ValueError(
                "All failure and right censored values must be greater than zero."
            )

        # type and value checking for CI
        if type(CI) not in [float, np.float64]:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
            )

        # error checking for optimizer
        if optimizer is not None:
            if type(optimizer) is not str:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                )
            if optimizer.upper() == "TNC":
                optimizer = "TNC"
            elif optimizer.upper() == "POWELL":
                optimizer = "powell"
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizer = "L-BFGS-B"
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD", "NM"]:
                optimizer = "nelder-mead"
            elif optimizer.upper() in ["ALL", "BEST"]:
                optimizer = "best"
            else:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                )

        # check the number of unique stresses
        unique_stresses_1 = np.unique(failure_stress_1)
        if len(unique_stresses_1) < 2:
            raise ValueError("failure_stress_1 must have at least 2 unique stresses.")
        if is_dual_stress is True:
            unique_stresses_2 = np.unique(failure_stress_2)
            if len(unique_stresses_2) < 2:
                raise ValueError(
                    "failure_stress_2 must have at least 2 unique stresses when using a dual stress model."
                )

        # group the failures into their failure_stresses and then check there are enough to fit the model
        if is_dual_stress is False:
            failure_df_ungrouped = pd.DataFrame(
                data={"failures": failures, "failure_stress_1": failure_stress_1},
                columns=["failures", "failure_stress_1"],
            )
            failure_groups = []
            unique_failure_stresses = []
            for key, items in failure_df_ungrouped.groupby(["failure_stress_1"]):
                key = key[0]
                values = list(items.iloc[:, 0].values)
                failure_groups.append(values)
                unique_failure_stresses.append(key)
            # Check that there are enough failures to fit the model.
            # This does not mean 2 failures at each stress.
            # All we need is as many failures as there are parameters in the model.
            total_unique_failures = 0
            for i, failure_group in enumerate(failure_groups):
                total_unique_failures += len(np.unique(failure_group))
            if total_unique_failures < min_failures_reqd:
                if life_stress_model == "Everything":
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for all ALT models to be fitted."
                        )
                    )
                else:
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for the "
                            + dist
                            + "-"
                            + life_stress_model
                            + " model to be fitted."
                        )
                    )

            if len(right_censored) > 0:
                right_censored_df_ungrouped = pd.DataFrame(
                    data={
                        "right_censored": right_censored,
                        "right_censored_stress_1": right_censored_stress_1,
                    },
                    columns=["right_censored", "right_censored_stress_1"],
                )
                right_censored_groups = []
                unique_right_censored_stresses = []
                for key, items in right_censored_df_ungrouped.groupby(
                    ["right_censored_stress_1"]
                ):
                    key = key[0]
                    values = list(items.iloc[:, 0].values)
                    right_censored_groups.append(values)
                    unique_right_censored_stresses.append(key)
                    # if key not in unique_failure_stresses:
                    #     raise ValueError(
                    #         str(
                    #             "The right censored stress "
                    #             + str(key)
                    #             + " does not appear in failure stresses."
                    #         )
                    #     )

                # add in empty lists for stresses which appear in failure_stress_1 but not in right_censored_stress_1
                for i, stress in enumerate(unique_failure_stresses):
                    if stress not in unique_right_censored_stresses:
                        right_censored_groups.insert(i, [])

            else:
                right_censored_groups = None
        else:  # This is for dual stress cases
            # concatenate the stresses to deal with them as a pair
            failure_stress_pairs = []
            for i in range(len(failure_stress_1)):
                failure_stress_pairs.append(
                    str(failure_stress_1[i]) + "_" + str(failure_stress_2[i])
                )

            failure_df_ungrouped = pd.DataFrame(
                data={
                    "failures": failures,
                    "failure_stress_pairs": failure_stress_pairs,
                },
                columns=["failures", "failure_stress_pairs"],
            )
            failure_groups = []
            unique_failure_stresses_str = []
            for key, items in failure_df_ungrouped.groupby(["failure_stress_pairs"]):
                key = key[0]
                values = list(items.iloc[:, 0].values)
                failure_groups.append(values)
                unique_failure_stresses_str.append(key)
            # Check that there are enough failures to fit the model.
            # This does not mean 2 failures at each stress.
            # All we need is as many failures as there are parameters in the model.
            total_unique_failures = 0
            for i, failure_group in enumerate(failure_groups):
                total_unique_failures += len(np.unique(failure_group))
            if total_unique_failures < min_failures_reqd:
                if life_stress_model == "Everything":
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for all ALT models to be fitted."
                        )
                    )
                else:
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for the "
                            + dist
                            + "-"
                            + life_stress_model
                            + "model to be fitted."
                        )
                    )

            # unpack the concatenated string for dual stresses ==> ['10.0_1000.0','20.0_2000.0','5.0_500.0'] should be [[10.0,1000.0],[20.0,2000.0],[5.0,500.0]]
            unique_failure_stresses = []
            for item in unique_failure_stresses_str:
                stress_pair = [float(x) for x in list(item.split("_"))]
                unique_failure_stresses.append(stress_pair)

            if len(right_censored) > 0:
                # concatenate the right censored stresses to deal with them as a pair
                right_censored_stress_pairs = []
                for i in range(len(right_censored_stress_1)):
                    right_censored_stress_pairs.append(
                        str(right_censored_stress_1[i])
                        + "_"
                        + str(right_censored_stress_2[i])
                    )

                right_censored_df_ungrouped = pd.DataFrame(
                    data={
                        "right_censored": right_censored,
                        "right_censored_stress_pairs": right_censored_stress_pairs,
                    },
                    columns=["right_censored", "right_censored_stress_pairs"],
                )
                right_censored_groups = []
                unique_right_censored_stresses_str = []
                for key, items in right_censored_df_ungrouped.groupby(
                    ["right_censored_stress_pairs"]
                ):
                    key = key[0]
                    values = list(items.iloc[:, 0].values)
                    right_censored_groups.append(values)
                    unique_right_censored_stresses_str.append(key)
                    if key not in unique_failure_stresses_str:
                        raise ValueError(
                            str(
                                "The right censored stress pair "
                                + str([float(x) for x in list(key.split("_"))])
                                + " does not appear in failure stresses."
                            )
                        )

                # add in empty lists for stresses which appear in failure_stress but not in right_censored_stress
                for i, stress in enumerate(unique_failure_stresses_str):
                    if stress not in unique_right_censored_stresses_str:
                        right_censored_groups.insert(i, [])
            else:
                right_censored_groups = None

        # check that use level stress is the correct type
        if is_dual_stress is False and use_level_stress is not None:
            if type(use_level_stress) in [list, tuple, np.ndarray, str, bool, dict]:
                raise ValueError("use_level_stress must be a number")
            use_level_stress = float(use_level_stress)
        if is_dual_stress is True and use_level_stress is not None:
            if type(use_level_stress) not in [list, np.ndarray]:
                raise ValueError(
                    "use_level_stress must be an array or list of the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                )
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be an array or list of length 2 with the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                )
            use_level_stress = np.asarray(use_level_stress)

        # return everything
        self.failures = failures
        self.failure_stress_1 = failure_stress_1
        self.failure_stress_2 = failure_stress_2
        self.right_censored = right_censored
        self.right_censored_stress_1 = right_censored_stress_1
        self.right_censored_stress_2 = right_censored_stress_2
        self.CI = CI
        self.optimizer = optimizer
        self.use_level_stress = use_level_stress
        self.failure_groups = failure_groups[::-1]
        if right_censored_groups is None:
            self.right_censored_groups = right_censored_groups
        else:
            self.right_censored_groups = right_censored_groups[::-1]
        self.stresses_for_groups = unique_failure_stresses[::-1]


def validate_CI_params(*args):
    """
    Returns False if any of the args is None or Nan, else returns True.
    This function is different to using all() because it performs the checks
    using np.isfinite(arg).

    Parameters
    ----------
    *args : bool
        Any number of boolean arguments

    Returns
    -------
    is_valid : bool
        False if any of the args is None or Nan else returns True.
    """
    is_valid = True
    for arg in args:
        if arg is None or np.isfinite(arg) is np.False_:
            is_valid = False
    return is_valid


def clean_CI_arrays(xlower, xupper, ylower, yupper, plot_type="CDF", x=None, q=None):
    """
    This function cleans the CI arrays of nans and numbers <= 0 and also removes
    numbers >= 1 if plot_type is CDF or SF.

    Parameters
    ----------
    xlower : list, array
        The lower x array for the confidence interval
    xupper : list, array
        The upper x array for the confidence interval
    ylower : list, array
        The lower y array for the confidence interval
    yupper : list, array
        The upper y array for the confidence interval
    plot_type : str, optional
        Must be "CDF", "SF", "CHF". Default is "CDF"
    x : array, optional
        The x array for CI extraction
    q : array, optional
        The q array for CI extraction

    Returns
    -------
    xlower : array
        The "cleaned" lower x array for the confidence interval
    xupper : array
        The "cleaned" upper x array for the confidence interval
    ylower : array
        The "cleaned" lower y array for the confidence interval
    ylower : array
        The "cleaned" upper y array for the confidence interval

    Notes
    -----
    The returned arrays will all be the same length

    The cleaning is done by deleting values. If the cleaned arrays are < 2 items
    in length then an error will be triggered.
    """
    # format the input as arrays
    xlower = np.asarray(xlower)
    xupper = np.asarray(xupper)
    ylower = np.asarray(ylower)
    yupper = np.asarray(yupper)

    # create empty arrays to fill with cleaned values
    xlower_out = np.array([])
    xupper_out = np.array([])
    ylower_out = np.array([])
    yupper_out = np.array([])

    xlower_out2 = np.array([])
    xupper_out2 = np.array([])
    ylower_out2 = np.array([])
    yupper_out2 = np.array([])

    xlower_out3 = np.array([])
    xupper_out3 = np.array([])
    ylower_out3 = np.array([])
    yupper_out3 = np.array([])

    # remove nans in all arrays
    for i in np.arange(len(xlower)):
        if (
            np.isfinite(xlower[i])
            and np.isfinite(xupper[i])
            and np.isfinite(ylower[i])
            and np.isfinite(yupper[i])
        ):
            xlower_out = np.append(xlower_out, xlower[i])
            xupper_out = np.append(xupper_out, xupper[i])
            ylower_out = np.append(ylower_out, ylower[i])
            yupper_out = np.append(yupper_out, yupper[i])

    # remove values >= 1 for CDF and SF
    if plot_type.upper() in ["CDF", "SF"]:
        for i in np.arange(len(xlower_out)):
            if ylower_out[i] < 1 and yupper_out[i] < 1:
                xlower_out2 = np.append(xlower_out2, xlower_out[i])
                xupper_out2 = np.append(xupper_out2, xupper_out[i])
                ylower_out2 = np.append(ylower_out2, ylower_out[i])
                yupper_out2 = np.append(yupper_out2, yupper_out[i])
    else:  # do nothing
        xlower_out2 = xlower_out
        xupper_out2 = xupper_out
        ylower_out2 = ylower_out
        yupper_out2 = yupper_out

    # remove values <=0 for all cases
    tol = 1e-50  # tolerance for equivalent to zero. Accounts for precision error
    for i in np.arange(len(xlower_out2)):
        if ylower_out2[i] > tol and yupper_out2[i] > tol:
            xlower_out3 = np.append(xlower_out3, xlower_out2[i])
            xupper_out3 = np.append(xupper_out3, xupper_out2[i])
            ylower_out3 = np.append(ylower_out3, ylower_out2[i])
            yupper_out3 = np.append(yupper_out3, yupper_out2[i])

    # checks whether CI_x or CI_y was specified and resulted in values being deleted due to being illegal values. Raises a more detailed error for the user.
    if len(xlower_out3) != len(xlower) and x is not None:
        raise ValueError(
            "The confidence intervals for CI_x cannot be returned because they are NaN. This may occur when the SF=0. Try specifying CI_x values closer to the mean of the distribution."
        )
    if len(ylower_out3) != len(ylower) and q is not None:
        raise ValueError(
            "The confidence intervals for CI_y cannot be returned because they are NaN. This may occur when the CI_y is near 0 or 1. Try specifying CI_y values closer to 0.5."
        )

    # final error check for lengths matching and there still being at least 2 elements remaining
    if (
        len(xlower_out3) != len(xupper_out3)
        or len(xlower_out3) != len(yupper_out3)
        or len(xlower_out3) != len(ylower_out3)
        or len(xlower_out3) < 1
    ):
        colorprint(
            "ERROR in clean_CI_arrays: Confidence intervals could not be plotted due to the presence of too many NaNs in the arrays.",
            text_color="red",
        )

    return xlower_out3, xupper_out3, ylower_out3, yupper_out3


def fill_no_autoscale(xlower, xupper, ylower, yupper, **kwargs):
    """
    Creates a filled region (polygon) without adding it to the global list of
    autoscale objects. Use this function when you want to plot something but not
    have it considered when autoscale sets the range.

    Parameters
    ----------
    xlower : list, array
        The lower x array for the polygon.
    xupper : list, array
        The upper x array for the polygon.
    ylower : list, array
        The lower y array for the polygon.
    ylower : list, array
        The upper y array for the polygon.
    kwargs
        keyword arguments passed to the matplotlib PolyCollection

    Returns
    -------
    None
        The filled polygon will be added to the plot.
    """
    # generate the polygon
    xstack = np.hstack([xlower, xupper[::-1]])
    ystack = np.hstack([ylower, yupper[::-1]])

    # this corrects illegal yvalues in the probability plot
    if plt.gca().get_yscale() == "function":
        ystack[np.where(ystack >= 1)] = 0.9999999
        ystack[np.where(ystack <= 0)] = 0.0000001

    polygon = np.column_stack([xstack, ystack])
    # this is equivalent to fill as it makes a polygon
    col = PolyCollection([polygon], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def line_no_autoscale(x, y, **kwargs):
    """
    Creates a line without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when
    autoscale sets the range.

    Parameters
    ----------
    x : array, list
        The x values for the line
    y : array, list
        The y values for the line
    kwargs
        keyword arguments passed to the matplotlib LineCollection

    Returns
    -------
    None
        The line will be added to the plot.
    """
    # this is equivalent to plot as it makes a line
    line = np.column_stack([x, y])
    col = LineCollection([line], **kwargs)
    plt.gca().add_collection(col, autolim=False)


class distribution_confidence_intervals:
    """
    This class contains several subfunctions that provide all the confidence
    intervals for CDF, SF, CHF for each distribution for which it is
    implemented.

    The class has no parameters or returns as it is used primarily to create the
    confidence interval object which is used by the subfunctions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    @staticmethod
    def exponential_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Exponential distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF"
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None.
        x : array, list, optional
            The x-values to be calculated. Default is None.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if q is not None.
        t_upper :array
            The upper bounds on time. Only returned if q is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.Lambda_SE and self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For the Exponential distribution, the bounds on time and reliability are
        the same.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200

        # this section plots the confidence interval
        if (
            self.Lambda_SE is not None
            and self.Z is not None
            and (plot_CI is True or q is not None or x is not None)
        ):

            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)

                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title + "\n" + str(CI_100) + "% confidence bounds"
                )
                # add a line to the plot title to include the confidence bounds information
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
            Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))

            if x is not None:
                t = x - self.gamma
            else:
                t0 = self.quantile(0.00001) - self.gamma
                if t0 <= 0:
                    t0 = 0.0001
                t = np.geomspace(
                    t0,
                    self.quantile(0.99999) - self.gamma,
                    points,
                )

            # calculate the CIs using the formula for SF
            Y_lower = np.exp(-Lambda_lower * t)
            Y_upper = np.exp(-Lambda_upper * t)

            # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
            t, t, Y_lower, Y_upper = clean_CI_arrays(
                xlower=t,
                xupper=t,
                ylower=Y_lower,
                yupper=Y_upper,
                plot_type=func,
                q=q,
                x=x,
            )
            # artificially correct for any reversals
            if (x is None or q is None) and len(Y_lower) > 2 and len(Y_upper) > 2:
                Y_lower = no_reverse(Y_lower, CI_type=None, plot_type=func)
                Y_upper = no_reverse(Y_upper, CI_type=None, plot_type=func)

            if func == "CDF":
                yy_upper = 1 - Y_upper
                yy_lower = 1 - Y_lower
            elif func == "SF":
                yy_upper = Y_upper
                yy_lower = Y_lower
            elif func == "CHF":
                yy_upper = -np.log(Y_upper)  # same as -np.log(SF)
                yy_lower = -np.log(Y_lower)

            if plot_CI is True:
                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_lower,color='blue',marker='.')
                # plt.scatter(t + self.gamma, yy_upper, color='red', marker='.')

            if q is not None:
                t_lower = -np.log(q) / Lambda_upper + self.gamma
                t_upper = -np.log(q) / Lambda_lower + self.gamma
                return t_lower, t_upper
            elif x is not None:
                return Y_lower, Y_upper

    @staticmethod
    def weibull_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Weibull distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF"
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if
            CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if
            CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, alpha, beta):  # u = ln(-ln(R))
                return beta * (anp.log(t) - anp.log(alpha))  # weibull SF linearized

            def v(R, alpha, beta):  # v = ln(t)
                return (1 / beta) * anp.log(-anp.log(R)) + anp.log(
                    alpha
                )  # weibull SF rearranged for t

            du_da = jac(u, 1)  # derivative wrt alpha (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt alpha (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * du_da(v, self.alpha, self.beta)
                    * du_db(v, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * dv_da(u, self.alpha, self.beta)
                    * dv_db(u, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        Y = transform_spaced(
                            "weibull", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is ln(t)
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma  # transform back from ln(t)
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )

                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability ln(-ln(R))
                u_lower = (
                    u(t, self.alpha, self.beta) + Z * var_u(self, t) ** 0.5
                )  # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Weibull_2P across
                u_upper = u(t, self.alpha, self.beta) - Z * var_u(self, t) ** 0.5

                Y_lower = np.exp(-np.exp(u_lower))  # transform back from ln(-ln(R))
                Y_upper = np.exp(-np.exp(u_upper))

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def gamma_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Gamma distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.beta_SE, self.Cov_mu_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.

        if (
            validate_CI_params(self.mu_SE, self.beta_SE, self.Cov_mu_beta, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, beta):  # u = R
                return agammaincc(beta, t / anp.exp(mu))

            def v(R, mu, beta):  # v = ln(t)
                return anp.log(agammainccinv(beta, R)) + mu

            du_dm = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_dm = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_dm(v, self.mu, self.beta) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * du_dm(v, self.mu, self.beta)
                    * du_db(v, self.mu, self.beta)
                    * self.Cov_mu_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_dm(u, self.mu, self.beta) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * dv_dm(u, self.mu, self.beta)
                    * dv_db(u, self.mu, self.beta)
                    * self.Cov_mu_beta
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        if self.beta > 3:
                            Y = transform_spaced(
                                "gamma",
                                y_lower=1e-8,
                                y_upper=1 - 1e-8,
                                beta=self.beta,
                                num=points,
                            )
                        else:
                            Y = np.linspace(1e-8, 1 - 1e-8, points)

                # v is ln(t)
                v_lower = v(Y, self.mu, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.mu, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                if x is not None:
                    t = x - self.gamma
                else:
                    if self.gamma == 0:
                        t0 = 0.0001
                    else:
                        t0 = self.quantile(0.0000001)
                    t = np.linspace(
                        t0 - self.gamma,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability
                # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Gamma_2P across
                R = u(t, self.mu, self.beta)
                varR = var_u(self, t)
                R_lower = R / (R + (1 - R) * np.exp((Z * varR**0.5) / (R * (1 - R))))
                R_upper = R / (R + (1 - R) * np.exp((-Z * varR**0.5) / (R * (1 - R))))

                # transform back from u = R
                Y_lower = R_lower
                Y_upper = R_upper

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )

                    line_no_autoscale(
                        x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def normal_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Normal distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = phiinv(R)
                return (mu - t) / sigma  # normal SF linearlized

            def v(R, mu, sigma):  # v = t
                return mu - sigma * ss.norm.ppf(R)

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * dv_da(u, self.mu, self.sigma)
                    * dv_db(u, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        Y = transform_spaced(
                            "normal", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is t
                t_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                t_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                if x is not None:
                    t = x
                else:
                    t = np.linspace(
                        self.quantile(0.00001), self.quantile(0.99999), points
                    )

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t,
                        xupper=t,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t, yy_upper, color='red')
                    # plt.scatter(t, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def lognormal_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Lognormal distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )

            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = phiinv(R)
                return (mu - np.log(t)) / sigma  # lognormal SF linearlized

            def v(R, mu, sigma):  # v = ln(t)
                return mu - sigma * ss.norm.ppf(R)

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * dv_da(u, self.mu, self.sigma)
                    * dv_db(u, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            if CI_type == "time":
                # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        Y = transform_spaced(
                            "normal", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is ln(t)
                v_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif CI_type == "reliability":
                # Confidence bounds on Reliability (in terms of time)
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t+ self.gamma, yy_upper, color='red')
                    # plt.scatter(t+ self.gamma, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def loglogistic_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Loglogistic distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, alpha, beta):  # u = ln(1/R - 1)
                return beta * (anp.log(t) - anp.log(alpha))  # loglogistic SF linearized

            def v(R, alpha, beta):  # v = ln(t)
                return (1 / beta) * anp.log(1 / R - 1) + anp.log(
                    alpha
                )  # loglogistic SF rearranged for t

            du_da = jac(u, 1)  # derivative wrt alpha (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt alpha (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * du_da(v, self.alpha, self.beta)
                    * du_db(v, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2
                    * dv_da(u, self.alpha, self.beta)
                    * dv_db(u, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            if CI_type == "time":  # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        Y = transform_spaced(
                            "loglogistic", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is ln(t)
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma  # transform back from ln(t)
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability ln(1/R - 1)
                u_lower = (
                    u(t, self.alpha, self.beta) + Z * var_u(self, t) ** 0.5
                )  # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Weibull_2P across
                u_upper = u(t, self.alpha, self.beta) - Z * var_u(self, t) ** 0.5

                Y_lower = 1 / (np.exp(u_lower) + 1)  # transform back from ln(1/R - 1)
                Y_upper = 1 / (np.exp(u_upper) + 1)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def gumbel_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF of the
        Gumbel distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z)
            is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "q must be a list or array of quantiles. Default is None"
                )
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError(
                    "x must be a list or array of x-values. Default is None"
                )
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(
                    text_title
                    + "\n"
                    + str(CI_100)
                    + "% confidence bounds on "
                    + CI_type
                )
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = ln(-ln(R))
                return (t - mu) / sigma  # gumbel SF linearlized

            def v(R, mu, sigma):  # v = t
                return mu + sigma * anp.log(-anp.log(R))  # Gumbel SF rearranged for t

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2
                    * dv_da(u, self.mu, self.sigma)
                    * dv_db(u, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            if CI_type == "time":  # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    if q is not None:
                        Y = q
                    else:
                        Y = transform_spaced(
                            "gumbel", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is t
                t_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                t_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                if x is not None:
                    t = x
                else:
                    t = np.linspace(
                        self.quantile(0.00001), self.quantile(0.99999), points
                    )

                # u is reliability u = ln(-ln(R))
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = np.exp(-np.exp(u_lower))  # transform back from ln(-ln(R))
                Y_upper = np.exp(-np.exp(u_upper))

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t,
                        xupper=t,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t, y=yy_lower, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t, y=yy_upper, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t, yy_upper, color='red')
                    # plt.scatter(t, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper


def linear_regression(
    x,
    y,
    slope=None,
    x_intercept=None,
    y_intercept=None,
    RRX_or_RRY="RRX",
    show_plot=False,
    **kwargs
):
    """
    This function provides the linear algebra solution to find line of best fit
    passing through points (x,y). Options to specify slope or intercept enable
    these parameters to be forced.

    Rank regression can be on X (RRX) or Y (RRY). Default is RRX.
    Note that slope depends on RRX_or_RRY. If you use RRY then slope is dy/dx
    but if you use RRX then slope is dx/dy.

    Parameters
    ----------
    x : array, list
        The x values
    y : array, list
        The y values
    slope : float, int, optional
        Used to force the slope. Default is None.
    x_intercept : float, int, optional
        Used to force the x-intercept. Default is None. Only used for RRY.
    y_intercept : float, int, optional
        Used to force the y-intercept. Default is None. Only used for RRX.
    RRX_or_RRY : str, optional
        Must be "RRY" or "RRX". Default is "RRY".
    show_plot : bool, optional
        If True, a plot of the line and points will be generated. Use plt.show()
        to show it.
    kwargs
        Keyword arguments for the plot that are passed to matplotlib for the
        line.

    Returns
    -------
    slope : float
        The slope of the line.
    intercept : float
        The intercept (x or y depending on RRX_or_RRY) of the line.

    Notes
    -----
    The equation of a line used here is Y = slope * X + intercept. This is the
    RRY form. For RRX it can be rearranged to X = (Y - intercept)/slope

    For more information on linear regression, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y are different lengths")
    if RRX_or_RRY not in ["RRX", "RRY"]:
        raise ValueError('RRX_or_RRY must be either "RRX" or "RRY". Default is "RRY".')
    if x_intercept is not None and RRX_or_RRY == "RRY":
        raise ValueError("RRY must use y_intercept not x_intercept")
    if y_intercept is not None and RRX_or_RRY == "RRX":
        raise ValueError("RRX must use x_intercept not y_intercept")
    if slope is not None and (x_intercept is not None or y_intercept is not None):
        raise ValueError("You can not specify both slope and intercept")

    if RRX_or_RRY == "RRY":
        if y_intercept is not None:  # only the slope must be found
            min_pts = 1
            xx = np.array([x]).T
            yy = (y - y_intercept).T
        elif slope is not None:  # only the intercept must be found
            min_pts = 1
            xx = np.array([np.ones_like(x)]).T
            yy = (y - slope * x).T
        else:  # both slope and intercept must be found
            min_pts = 2
            xx = np.array([x, np.ones_like(x)]).T
            yy = y.T
    else:  # RRX
        if x_intercept is not None:  # only the slope must be found
            min_pts = 1
            yy = np.array([y]).T
            xx = (x - x_intercept).T
        elif slope is not None:  # only the intercept must be found
            min_pts = 1
            yy = np.array([np.ones_like(y)]).T
            xx = (x - slope * y).T
        else:  # both slope and intercept must be found
            min_pts = 2
            yy = np.array([y, np.ones_like(y)]).T
            xx = x.T

    if len(x) < min_pts:
        if slope is not None:
            err_str = str(
                "A minimum of 1 point is required to fit the line when the slope is specified."
            )
        elif x_intercept is not None and y_intercept is not None:
            err_str = str(
                "A minimum of 1 point is required to fit the line when the intercept is specified."
            )
        else:
            err_str = str(
                "A minimum of 2 points are required to fit the line when slope or intercept are not specified."
            )
        raise ValueError(err_str)

    if RRX_or_RRY == "RRY":
        try:
            solution = (
                np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)
            )  # linear regression formula for RRY
        except LinAlgError:
            raise RuntimeError(
                "An error has occurred when attempting to find the initial guess using least squares estimation.\nThis error is caused by a non-invertable matrix.\nThis can occur when there are only two very similar failure times like 10 and 10.000001.\nThere is no solution to this error, other than to use failure times that are more unique."
            )
        if y_intercept is not None:
            m = solution[0]
            c = y_intercept
        elif slope is not None:
            m = slope
            c = solution[0]
        else:
            m = solution[0]
            c = solution[1]
    else:  # RRX
        try:
            solution = (
                np.linalg.inv(yy.T.dot(yy)).dot(yy.T).dot(xx)
            )  # linear regression formula for RRX
        except LinAlgError:
            raise RuntimeError(
                "An error has occurred when attempting to find the initial guess using least squares estimation.\nThis error is caused by a non-invertable matrix.\nThis can occur when there are only two very similar failure times like 10 and 10.000001.\nThere is no solution to this error, other than to use failure times that are more unique."
            )
        if x_intercept is not None:
            m_x = solution[0]
            m = 1 / m_x
            c = -x_intercept / m_x
        elif slope is not None:
            m = 1 / slope
            c_x = solution[0]
            c = -c_x / slope
        else:
            m_x = solution[0]
            c_x = solution[1]
            m = 1 / m_x
            c = -c_x / m_x

    if show_plot is True:
        plt.scatter(x, y, marker=".", color="k")
        delta_x = max(x) - min(x)
        delta_y = max(y) - min(y)
        xvals = np.linspace(min(x) - delta_x, max(x) + delta_x, 10)
        yvals = m * xvals + c
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str(
                "y="
                + round_and_string(m, 2)
                + ".x + "
                + round_and_string(c, 2)
            )
        plt.plot(xvals, yvals, label=label, **kwargs)
        plt.xlim(min(x) - delta_x * 0.2, max(x) + delta_x * 0.2)
        plt.ylim(min(y) - delta_y * 0.2, max(y) + delta_y * 0.2)
    return m, c


def least_squares(dist, failures, right_censored, method="RRX", force_shape=None):
    """
    Uses least squares or non-linear least squares estimation to fit the
    parameters of the distribution to the plotting positions.

    Plotting positions are based on failures and right_censored so while least
    squares estimation does not consider the right_censored data in the same way
    as MLE, the plotting positions do. This means that right censored data are
    not ignored by least squares estimation, but the way the values are treated
    differs between least squares and MLE.

    The output of this method may be used as the initial guess for the MLE
    method.

    Parameters
    ----------
    dist : object
        Thew distribution object
    failures : array, list
        The failure data
    right_censored : array, list
        The right censored data. If there is no data then this should be an
        empty list.
    method : str, optional
        Must be "RRX" or "RRY". Default is RRX.
    force_shape : float, int, optional
        Used to force the shape (beta or sigma) parameter. Default is None which
        will not force the slope.

    Returns
    -------
    model_parameters : list
        The model's parameters in a list. eg. for "Weibull_2P" it will return
        [alpha,beta]. For "Weibull_3P" it will return [alpha,beta,gamma].

    Notes
    -----
    For more information on least squares estimation, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.

    For cases where the CDF is not linearizable (eg. Weibull_3P), this function
    uses non-linear least squares (NLLS) which uses scipy's curve_fit to find
    the parameters. This may sometimes fail as curve_fit is an optimization
    routine that needs an initial guess provided by this function.
    """

    if min(failures) <= 0 and dist not in ["Normal_2P", "Gumbel_2P"]:
        raise ValueError(
            "failures contains zeros or negative values which are only suitable when dist is Normal_2P or Gumbel_2P"
        )
    if max(failures) >= 1 and dist == "Beta_2P":
        raise ValueError(
            "failures contains values greater than or equal to one which is not allowed when dist is Beta_2P"
        )
    if force_shape is not None and dist not in [
        "Weibull_2P",
        "Normal_2P",
        "Lognormal_2P",
    ]:
        raise ValueError(
            "force_shape can only be applied to Weibull_2P, Normal_2P, and Lognormal_2P"
        )
    if method not in ["RRX", "RRY"]:
        raise ValueError('method must be either "RRX" or "RRY". Default is RRX.')

    from reliability.Probability_plotting import (
        plotting_positions,
    )  # this import needs to be here to prevent circular import if it is in the main module

    x, y = plotting_positions(failures=failures, right_censored=right_censored)
    x = np.array(x)
    y = np.array(y)
    gamma0 = (
        min(np.hstack([failures, right_censored])) - 0.001
    )  # initial guess for gamma when it is required for the 3P fitters
    if gamma0 < 0:
        gamma0 = 0

    if dist == "Weibull_2P":
        xlin = np.log(x)
        ylin = np.log(-np.log(1 - y))
        if force_shape is not None and method == "RRX":
            force_shape = 1 / force_shape  # only needs to be inverted for RRX not RRY
        slope, intercept = linear_regression(
            xlin, ylin, slope=force_shape, RRX_or_RRY=method
        )
        LS_beta = slope
        LS_alpha = np.exp(-intercept / LS_beta)
        guess = [LS_alpha, LS_beta]

    elif dist == "Weibull_3P":
        # Weibull_2P estimate to create the guess for Weibull_3P
        xlin = np.log(x - gamma0)
        ylin = np.log(-np.log(1 - y))
        slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
        LS_beta = slope
        LS_alpha = np.exp(-intercept / LS_beta)

        # NLLS for Weibull_3P
        def __weibull_3P_CDF(t, alpha, beta, gamma):
            return 1 - np.exp(-(((t - gamma) / alpha) ** beta))

        try:
            curve_fit_bounds = (
                [0, 0, 0],
                [1e20, 1000, gamma0],
            )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
            popt, _ = curve_fit(
                __weibull_3P_CDF,
                x,
                y,
                p0=[LS_alpha, LS_beta, gamma0],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma]
            NLLS_alpha = popt[0]
            NLLS_beta = popt[1]
            NLLS_gamma = popt[2]
            guess = [NLLS_alpha, NLLS_beta, NLLS_gamma]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Weibull_3P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [LS_alpha, LS_beta, gamma0]

    elif dist == "Exponential_1P":
        if method == "RRY":
            x_intercept = None
            y_intercept = 0
        elif method == "RRX":
            y_intercept = None
            x_intercept = 0

        ylin = -np.log(1 - y)
        slope, _ = linear_regression(
            x, ylin, x_intercept=x_intercept, y_intercept=y_intercept, RRX_or_RRY=method
        )  # equivalent to y = m.x
        LS_Lambda = slope
        guess = [LS_Lambda]

    elif dist == "Exponential_2P":
        # Exponential_1P estimate to create the guess for Exponential_2P
        # while it is mathematically possible to use ordinary least squares (y=mx+c) for this, the LS method does not allow bounds on gamma. This can result in gamma > min(data) which should be impossible and will cause other errors.
        xlin = x - gamma0
        ylin = -np.log(1 - y)
        slope, _ = linear_regression(xlin, ylin, x_intercept=0, RRX_or_RRY="RRX")
        LS_Lambda = slope
        # NLLS for Exponential_2P
        def __exponential_2P_CDF(t, Lambda, gamma):
            return 1 - np.exp(-Lambda * (t - gamma))

        try:
            curve_fit_bounds = (
                [0, 0],
                [1e20, gamma0],
            )  # ([Lambda_lower,gamma_lower],[Lambda_upper,gamma_upper])
            popt, _ = curve_fit(
                __exponential_2P_CDF,
                x,
                y,
                p0=[LS_Lambda, gamma0],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )
            NLLS_Lambda = popt[0]
            NLLS_gamma = popt[1]
            guess = [NLLS_Lambda, NLLS_gamma]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Exponential_2P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [LS_Lambda, gamma0]

    elif dist == "Normal_2P":
        ylin = ss.norm.ppf(y)
        if force_shape is not None and method == "RRY":
            force_shape = 1 / force_shape  # only needs to be inverted for RRY not RRX
        slope, intercept = linear_regression(
            x, ylin, slope=force_shape, RRX_or_RRY=method
        )
        LS_sigma = 1 / slope
        LS_mu = -intercept * LS_sigma
        guess = [LS_mu, LS_sigma]

    elif dist == "Gumbel_2P":
        ylin = np.log(-np.log(1 - y))
        slope, intercept = linear_regression(x, ylin, RRX_or_RRY=method)
        LS_sigma = 1 / slope
        LS_mu = -intercept * LS_sigma
        guess = [LS_mu, LS_sigma]

    elif dist == "Lognormal_2P":
        xlin = np.log(x)
        ylin = ss.norm.ppf(y)
        if force_shape is not None and method == "RRY":
            force_shape = 1 / force_shape  # only needs to be inverted for RRY not RRX
        slope, intercept = linear_regression(
            xlin, ylin, slope=force_shape, RRX_or_RRY=method
        )
        LS_sigma = 1 / slope
        LS_mu = -intercept * LS_sigma
        guess = [LS_mu, LS_sigma]

    elif dist == "Lognormal_3P":
        # uses least squares to fit a normal distribution to the log of the data and minimizes the correlation coefficient (1 - R^2)
        def __gamma_optimizer(gamma_guess, x, y):
            xlin = np.log(x - gamma_guess)
            ylin = ss.norm.ppf(y)
            _, _, r, _, _ = ss.linregress(xlin, ylin)
            return 1 - (r**2)

        # NLLS for Normal_2P which is used by Lognormal_3P by taking the log of the data. This is more accurate than doing it with Lognormal_3P.
        def __normal_2P_CDF(t, mu, sigma):
            return (1 + erf(((t - mu) / sigma) / 2**0.5)) / 2

        res = minimize(
            __gamma_optimizer, gamma0, args=(x, y), method="TNC", bounds=[([0, gamma0])]
        )  # this obtains gamma
        gamma = res.x[0]

        try:
            curve_fit_bounds = (
                [-1e20, 0],
                [1e20, 1000],
            )  # ([mu_lower,sigma_lower],[mu_upper,sigma_upper])
            popt, _ = curve_fit(
                __normal_2P_CDF,
                np.log(x - gamma),
                y,
                p0=[np.mean(np.log(x - gamma)), np.std(np.log(x - gamma))],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [mu,sigma].
            NLLS_mu = popt[0]
            NLLS_sigma = popt[1]
            guess = [NLLS_mu, NLLS_sigma, gamma]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Lognormal_3P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [np.mean(np.log(x - gamma)), np.std(np.log(x - gamma)), gamma]

    elif dist == "Loglogistic_2P":
        xlin = np.log(x)
        ylin = np.log(1 / y - 1)
        slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
        LS_beta = -slope
        LS_alpha = np.exp(intercept / LS_beta)
        guess = [LS_alpha, LS_beta]

    elif dist == "Loglogistic_3P":

        def __loglogistic_3P_CDF(t, alpha, beta, gamma):
            return 1 / (1 + ((t - gamma) / alpha) ** -beta)

        # Loglogistic_2P estimate to create the guess for Loglogistic_3P
        xlin = np.log(x - gamma0)
        ylin = np.log(1 / y - 1)
        slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
        LS_beta = -slope
        LS_alpha = np.exp(intercept / LS_beta)

        try:
            # Loglogistic_3P estimate
            curve_fit_bounds = (
                [0, 0, 0],
                [1e20, 1000, gamma0],
            )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
            popt, _ = curve_fit(
                __loglogistic_3P_CDF,
                x,
                y,
                p0=[LS_alpha, LS_beta, gamma0],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma].
            NLLS_alpha = popt[0]
            NLLS_beta = popt[1]
            NLLS_gamma = popt[2]
            guess = [NLLS_alpha, NLLS_beta, NLLS_gamma]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Loglogistic_3P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [LS_alpha, LS_beta, gamma0]

    elif dist == "Gamma_2P":

        def __gamma_2P_CDF(t, alpha, beta):
            return gammainc(beta, t / alpha)

        # Weibull_2P estimate which is converted to a Gamma_2P initial guess
        xlin = np.log(x)
        ylin = np.log(-np.log(1 - y))
        slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
        LS_beta = slope
        LS_alpha = np.exp(-intercept / LS_beta)

        # conversion of weibull parameters to gamma parameters. These values were found empirically and the relationship is only an approximate model
        beta_guess = abs(0.6932 * LS_beta**2 - 0.0908 * LS_beta + 0.2804)
        alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess**2 + 1.1119 * beta_guess))

        def __perform_curve_fit():  # separated out for repeated use
            curve_fit_bounds = (
                [0, 0],
                [1e20, 1000],
            )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
            popt, _ = curve_fit(
                __gamma_2P_CDF,
                x,
                y,
                p0=[alpha_guess, beta_guess],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
            return [popt[0], popt[1]]

        try:
            # Gamma_2P estimate
            guess = __perform_curve_fit()
        except (ValueError, LinAlgError, RuntimeError):
            try:
                guess = __perform_curve_fit()
                # We repeat the same attempt at a curve_fit because of a very strange event.
                # When Fit_Gamma_2P is run twice in a row, the second attempt fails if there was a probability plot generated for the first attempt.
                # This was unable to debugged since the curve_fit has identical inputs each run and the curve_fit should not interact with the probability plot in any way.
                # One possible cause of this error may relate to memory usage though this is not confirmed.
                # By simply repeating the attempted curve_fit one more time, it often will work perfectly on the second try.
                # If it fails the second try then we report the failure and return the initial guess.
            except (ValueError, LinAlgError, RuntimeError):
                colorprint(
                    "WARNING: Non-linear least squares for Gamma_2P failed. The result returned is an estimate that is likely to be incorrect.",
                    text_color="red",
                )
                guess = [alpha_guess, beta_guess]

    elif dist == "Gamma_3P":

        def __gamma_2P_CDF(t, alpha, beta):
            return gammainc(beta, t / alpha)

        def __gamma_3P_CDF(t, alpha, beta, gamma):
            return gammainc(beta, (t - gamma) / alpha)

        # Weibull_2P estimate which is converted to a Gamma_2P initial guess
        xlin = np.log(x - gamma0 * 0.98)
        ylin = np.log(-np.log(1 - y))
        slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
        LS_beta = slope
        LS_alpha = np.exp(-intercept / LS_beta)

        # conversion of weibull parameters to gamma parameters. These values were found empirically and the relationship is only an approximate model
        beta_guess = abs(0.6932 * LS_beta**2 - 0.0908 * LS_beta + 0.2804)
        alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess**2 + 1.1119 * beta_guess))

        def __perform_curve_fit_gamma_2P():  # separated out for repeated use
            curve_fit_bounds = (
                [0, 0],
                [1e20, 1000],
            )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
            popt, _ = curve_fit(
                __gamma_2P_CDF,
                x - gamma0 * 0.98,
                y,
                p0=[alpha_guess, beta_guess],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
            return [popt[0], popt[1]]

        def __perform_curve_fit_gamma_3P():  # separated out for repeated use
            curve_fit_bounds_3P = (
                [0, 0, 0],
                [1e20, 1000, gamma0],
            )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
            popt, _ = curve_fit(
                __gamma_3P_CDF,
                x,
                y,
                p0=[NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98],
                bounds=curve_fit_bounds_3P,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma]
            return [popt[0], popt[1], popt[2]]

        try:
            # Gamma_2P estimate to create the guess for Gamma_3P
            guess_2P = __perform_curve_fit_gamma_2P()
            NLLS_alpha_2P, NLLS_beta_2P = guess_2P[0], guess_2P[1]
            try:
                # Gamma_3P estimate
                guess = __perform_curve_fit_gamma_3P()
            except (ValueError, LinAlgError, RuntimeError):
                try:
                    # try gamma_3P a second time
                    guess = __perform_curve_fit_gamma_3P()
                except (ValueError, LinAlgError, RuntimeError):
                    colorprint(
                        "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_3P optimization. The result returned is an estimate that is likely to be incorrect.",
                        text_color="red",
                    )
                    guess = [NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98]
        except (ValueError, LinAlgError, RuntimeError):
            # We repeat the same attempt at a curve_fit because of a very strange event.
            # When Fit_Gamma_3P is run twice in a row, the second attempt fails if there was a probability plot generated for the first attempt.
            # This was unable to debugged since the curve_fit has identical inputs each run and the curve_fit should not interact with the probability plot in any way.
            # One possible cause of this error may relate to memory usage though this is not confirmed.
            # By simply repeating the attempted curve_fit one more time, it often will work perfectly on the second try.
            # If it fails the second try then we report the failure and return the initial guess.
            try:
                guess_2P = __perform_curve_fit_gamma_2P()
                NLLS_alpha_2P, NLLS_beta_2P = guess_2P[0], guess_2P[1]
                try:
                    # Gamma_3P estimate
                    guess = __perform_curve_fit_gamma_3P()
                except (ValueError, LinAlgError, RuntimeError):
                    try:
                        # try gamma_3P for a second time
                        guess = __perform_curve_fit_gamma_3P()
                    except (ValueError, LinAlgError, RuntimeError):
                        colorprint(
                            "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_3P optimization. The result returned is an estimate that is likely to be incorrect.",
                            text_color="red",
                        )
                        guess = [NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98]
            except (ValueError, LinAlgError, RuntimeError):
                colorprint(
                    "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_2P optimization. The result returned is an estimate that is likely to be incorrect.",
                    text_color="red",
                )
                guess = [alpha_guess, beta_guess, gamma0 * 0.98]

    elif dist == "Beta_2P":

        def __beta_2P_CDF(t, alpha, beta):
            return betainc(alpha, beta, t)

        try:
            curve_fit_bounds = (
                [0, 0],
                [100, 100],
            )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
            popt, _ = curve_fit(
                __beta_2P_CDF,
                x,
                y,
                p0=[2, 1],
                bounds=curve_fit_bounds,
                jac="3-point",
                method="trf",
                max_nfev=300 * len(failures),
            )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
            NLLS_alpha = popt[0]
            NLLS_beta = popt[1]
            guess = [NLLS_alpha, NLLS_beta]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Beta_2P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [2, 1]
    else:
        raise ValueError('Unknown dist. Use the correct name. eg. "Weibull_2P"')
    return guess


def ALT_least_squares(model, failures, stress_1_array, stress_2_array=None):
    """
    Uses least squares estimation to fit the parameters of the ALT stress-life
    model to the time to failure data.

    Unlike least_squares for probability distributions, this function does not
    use the plotting positions because it is working on the life-stress model
    L(S) and not the life distribution F(t), so it uses the failure data directly.

    This function therefore only fits the parameters of the model to the failure
    data and does not take into account the right censored data. Right censored
    data is only used when fitting the life-stress distribution (eg. "Weibull
    Eyring") which is done using MLE.

    The output of this function is used as the initial guess for the MLE
    method for the life-stress distribution.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power"
    failures : array, list
        The failure data
    stress_1_array : list, array
        The stresses corresponding to the failure data.
    stress_2_array : list, array, optional
        The second stresses corresponding to the failure data. Used only for
        dual-stress models. Default is None.

    Returns
    -------
    model_parameters : list
        The model's parameters in a list. This depends on the model.
        Exponential - [a,b], Eyring - [a,c], Power - [a,n],
        Dual_Exponential - [a,b,c], Power_Exponential - [a,c,n],
        Dual_Power - [c,m,n]

    Notes
    -----
    For more information on least squares estimation, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.
    For more information on fitting ALT models, see the `documentation <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_.

    For models with more than two parameters, linear algebra is equally valid,
    but in these cases it is not finding the line of best fit, it is instead
    finding the plane of best fit.
    """

    def non_invertable_handler(xx, yy, model):
        """
        This subfunction performs the linear algebra to find the solution.
        It also handles the occasional case of a non-invertable matrix.
        This function is separated out for repeated use.

        Parameters
        ----------
        xx : list, array
            The x data
        yy : list, array
            The y data
        model : str
            The model. Used only for printing the correct error string.

        Returns
        -------
        model_parameters : list
            The model parameters in a list. These are the linearized form and
            need to be converted back to find the actual model parameters.
        """
        try:
            # linear regression formula for RRY
            out = np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)
        except LinAlgError:
            try:
                # if the matrix is perfectly symmetrical, it becomes non-invertable
                # in this case we introduce a small amount of noise so it can be inverted
                # this noise doesn't affect the result much and it doesn't really matter since it is just a guess being passed to MLE for optimization
                noise = np.random.normal(loc=0, scale=0.01, size=(3, 3))
                out = np.linalg.inv(xx.T.dot(xx) + noise).dot(xx.T).dot(yy)
            except LinAlgError:
                colorprint(
                    "WARNING: Least squares estimates failed for " + model + " model.",
                    text_color="red",
                )
                out = [1, 2, 3]  # return a dummy solution for MLE to deal with
        return out

    L = np.asarray(failures)
    S1 = np.asarray(stress_1_array)
    if stress_2_array is not None:
        S2 = np.asarray(stress_2_array)
    else:
        S2 = None
    if model == "Exponential":
        m, c = linear_regression(x=1 / S1, y=np.log(L), RRX_or_RRY="RRY")
        output = [m, np.exp(c)]  # a,b
    elif model == "Eyring":
        m, c = linear_regression(x=1 / S1, y=np.log(L) + np.log(S1), RRX_or_RRY="RRY")
        output = [m, -c]  # a,c
    elif model == "Power":
        m, c = linear_regression(x=np.log(S1), y=np.log(L), RRX_or_RRY="RRY")
        output = [np.exp(c), m]  # a,n
    elif model == "Dual_Exponential":
        X = 1 / S1
        Y = 1 / S2
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        # linear regression formula for RRY
        solution = non_invertable_handler(xx, yy, model)
        output = [solution[1], solution[2], np.exp(solution[0])]  # a,b,c
    elif model == "Power_Exponential":
        X = 1 / S1
        Y = np.log(S2)
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        # linear regression formula for RRY
        solution = non_invertable_handler(xx, yy, model)
        output = [solution[1], np.exp(solution[0]), solution[2]]  # a,c,n
    elif model == "Dual_Power":
        X = np.log(S1)
        Y = np.log(S2)
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        solution = non_invertable_handler(xx, yy, model)
        output = [np.exp(solution[0]), solution[1], solution[2]]  # c,m,n
    else:
        raise ValueError(
            "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power."
        )
    return output


class LS_optimization:
    """
    This function is a control function for least squares regression and it is
    used by each of the Fitters. There is no actual "optimization" done here,
    with the exception of checking which method (RRX or RRY) gives the better
    solution.

    Parameters
    ----------
    func_name : str
        The function to be fitted. Eg. "Weibull_2P".
    LL_func : function
        The log-likelihood function from the fitter
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    method : str, optional
        Must be either "RRX", "RRY", "LS", or "NLLS". Default is "LS".
    force_shape : float, int, optional
        The shape parameter to be forced. Default is None which results in no
        forcing of the shape parameter.
    LL_func_force : function
        The log-likelihood function for when the shape parameter is forced. Only
        required if force_shape is not None.

    Returns
    -------
    guess : list
        The guess of the models parameters. The length of this list depends on
        the number of parameters in the model. The guess is obtained using
        Utils.least_squares
    method : str
        The method used. This will be either "RRX", "RRY" or "NLLS".

    Notes
    -----
    If method="LS" then both "RRX" and "RRY" will be tried and the best one will
    be returned.
    """

    def __init__(
        self,
        func_name,
        LL_func,
        failures,
        right_censored,
        method="LS",
        force_shape=None,
        LL_func_force=None,
    ):
        if method not in ["RRX", "RRY", "LS", "NLLS"]:
            raise ValueError(
                "method must be either RRX, RRY, LS, or NLLS. Default is LS"
            )
        if func_name in [
            "Weibull_3P",
            "Gamma_2P",
            "Gamma_3P",
            "Beta_2P",
            "Lognormal_3P",
            "Loglogistic_3P",
            "Exponential_2P",
        ]:
            guess = least_squares(
                dist=func_name, failures=failures, right_censored=right_censored
            )
            LS_method = "NLLS"
        elif method in ["RRX", "RRY"]:
            guess = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method=method,
                force_shape=force_shape,
            )
            LS_method = method
        else:  # LS
            # RRX
            guess_RRX = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_shape,
            )
            if force_shape is not None:
                loglik_RRX = -LL_func_force(
                    guess_RRX, failures, right_censored, force_shape
                )
            else:
                loglik_RRX = -LL_func(guess_RRX, failures, right_censored)
            # RRY
            guess_RRY = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_shape,
            )
            if force_shape is not None:
                loglik_RRY = -LL_func_force(
                    guess_RRY, failures, right_censored, force_shape
                )
            else:
                loglik_RRY = -LL_func(guess_RRY, failures, right_censored)
            # take the best one
            if abs(loglik_RRX) < abs(loglik_RRY):  # RRX is best
                LS_method = "RRX"
                guess = guess_RRX
            else:  # RRY is best
                LS_method = "RRY"
                guess = guess_RRY
        self.guess = guess
        self.method = LS_method


class MLE_optimization:
    """
    This function performs Maximum Likelihood Estimation (MLE) to find the
    optimal parameters of the probability distribution. This functions is used
    by each of the fitters.

    Parameters
    ----------
    func_name : str
        The function to be fitted. Eg. "Weibull_2P".
    LL_func : function
        The log-likelihood function from the fitter
    initial_guess : list, array
        The initial guess of the model parameters that is used by the optimizer.
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    optimizer : str, None
        This must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best",
        "all" or None. Fot detail on how these optimizers are used, please see
        the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    force_shape : float, int, optional
        The shape parameter to be forced. Default is None which results in no
        forcing of the shape parameter.
    LL_func_force : function
        The log-likelihood function for when the shape parameter is forced. Only
        required if force_shape is not None.

    Returns
    -------
    scale : float
        Only returned for Weibull_2P, Weibull_3P, Lognormal_2P, Lognormal_3P,
        Gamma_2P, Gamma_3P, Loglogistic_2P, Loglogistic_3P, Exponential_1P,
        Exponential_2P, Normal_2P, Beta_2P, and Gumbel_2P
    shape : float
        Only returned for Weibull_2P, Weibull_3P, Lognormal_2P, Lognormal_3P,
        Gamma_2P, Gamma_3P, Loglogistic_2P, Loglogistic_3P, Normal_2P, Beta_2P,
        and Gumbel_2P
    alpha : float
        Only returned for Weibull_DS, Weibull_ZI and Weibull_DSZI
    beta : float
        Only returned for Weibull_DS, Weibull_ZI and Weibull_DSZI
    gamma : float
        Only returned for Weibull_3P, Exponential_2P, Gamma_3P, Lognormal_3P,
        and Loglogistic_3P.
    DS : float
        Only returned for Weibull_DS and Weibull_DSZI
    ZI : float
        Only returned for Weibull_ZI and Weibull_DSZI
    alpha_1 : float
        Only returned for Weibull_mixture and Weibull_CR
    beta_1 : float
        Only returned for Weibull_mixture and Weibull_CR
    alpha_2 : float
        Only returned for Weibull_mixture and Weibull_CR
    beta_2 : float
        Only returned for Weibull_mixture and Weibull_CR
    proportion_1 : float
        Only returned for Weibull_mixture
    proportion_2 : float
        Only returned for Weibull_mixture
    success : bool
        Whether at least one optimizer succeeded. If False then the least
        squares result will be returned in place of the MLE result.
    optimizer : str, None
        The optimizer used. If MLE failed then None is returned as the
        optimizer.

    Notes
    -----
    Not all of the above returns are always returned. It depends on which model
    is being used.

    If the MLE method fails then the initial guess (from least squares) will be
    returned with a printed warning.
    """

    def __init__(
        self,
        func_name,
        LL_func,
        initial_guess,
        failures,
        right_censored,
        optimizer,
        force_shape=None,
        LL_func_force=None,
    ):
        # this sub-function does the actual optimization. It is called each time a new optimizer is tried
        def loglik_optimizer(
            LL_func,
            guess,
            failures,
            right_censored,
            bounds,
            optimizer,
            force_shape,
            LL_func_force,
            func_name,
        ):
            """
            This sub-function does the actual optimization. It is called each
            time a new optimizer is tried.

            Parameters
            ----------
            LL_func : function
                The log-likelihood function from the fitter
            guess : list, array
                The initial guess of the model parameters that is used by the optimizer.
            failures : list, array
                The failure data
            right_censored : list, array
                The right censored data. If there is no right censored data then this
                should be an empty array.
            bounds : list
                The bounds on the solution
            optimizer : str, None
                This must be either "TNC", "L-BFGS-B", "nelder-mead", or
                "powell".
            force_shape : float, int, optional
                The shape parameter to be forced. Default is None which results in no
                forcing of the shape parameter.
            LL_func_force : function
                The log-likelihood function for when the shape parameter is forced. Only
                required if force_shape is not None.
            func_name : str
                The function name. eg. "Weibull_2P"

            Returns
            -------
            success : bool
                Whether the optimizer was successful
            log_likelihood : float
                The log-likelihood of the solution
            model_parameters : array
                The model parameters of the solution

            Notes
            -----
            The returns are provided in a tuple of success, log_likelihood,
            model_parameters.
            """
            delta_LL = 1
            LL_array = [1000000]
            runs = 0

            if func_name in ["Weibull_ZI", "Weibull_DSZI"]:
                ZI = True
            else:
                ZI = False

            if ZI is True:  # Zero Inflated distribution (applies to ZI and DSZI)
                args = (failures[failures == 0], failures[failures > 0], right_censored)
            else:
                args = (failures, right_censored)

            if force_shape is None:
                while delta_LL > 0.001 and runs < 5:
                    # exits after LL convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=args,
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    guess = result.x  # update the guess each iteration
                    if ZI is True:
                        LL2 = 2 * LL_func(
                            guess,
                            failures[failures == 0],
                            failures[failures > 0],
                            right_censored,
                        )
                    else:
                        LL2 = 2 * LL_func(guess, failures, right_censored)
                    LL_array.append(np.abs(LL2))
                    delta_LL = abs(LL_array[-1] - LL_array[-2])
            else:  # this will only be run for Weibull_2P, Normal_2P, and Lognormal_2P so the guess is structured with this in mind
                bounds = [bounds[0]]
                guess = [guess[0]]
                while (
                    delta_LL > 0.001 and runs < 5
                ):  # exits after LL convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(LL_func_force),
                        guess,
                        args=(failures, right_censored, force_shape),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    guess = result.x
                    LL2 = 2 * LL_func_force(
                        guess, failures, right_censored, force_shape
                    )
                    LL_array.append(np.abs(LL2))
                    delta_LL = abs(LL_array[-1] - LL_array[-2])
                    guess = result.x  # update the guess each iteration
            return result.success, LL_array[-1], result.x

        # generate the bounds on the solution
        gamma0 = max(0, min(np.hstack([failures, right_censored])) - 0.0001)
        if func_name in ["Weibull_2P", "Gamma_2P", "Beta_2P", "Loglogistic_2P"]:
            bounds = [(0, None), (0, None)]
        elif func_name in ["Weibull_3P", "Gamma_3P", "Loglogistic_3P"]:
            bounds = [(0, None), (0, None), (0, gamma0)]
        elif func_name in ["Normal_2P", "Gumbel_2P", "Lognormal_2P"]:
            bounds = [(None, None), (0, None)]
        elif func_name == "Lognormal_3P":
            bounds = [(None, None), (0, None), (0, gamma0)]
        elif func_name == "Exponential_1P":
            bounds = [(0, None)]
        elif func_name == "Exponential_2P":
            bounds = [(0, None), (0, gamma0)]
        elif func_name == "Weibull_mixture":
            bounds = [
                (0.0001, None),
                (0.0001, None),
                (0.0001, None),
                (0.0001, None),
                (0.0001, 0.9999),
            ]
        elif func_name == "Weibull_CR":
            bounds = [(0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None)]
        elif func_name == "Weibull_DSZI":
            bounds = [(0.0001, None), (0.0001, None), (0.00001, 1), (0, 0.99999)]
        elif func_name == "Weibull_DS":
            bounds = [(0.0001, None), (0.0001, None), (0.00001, 1)]
        elif func_name == "Weibull_ZI":
            bounds = [(0.0001, None), (0.0001, None), (0, 0.99999)]
        else:
            raise ValueError(
                'func_name is not recognised. Use the correct name e.g. "Weibull_2P"'
            )

        # determine which optimizers to use
        stop_after_success = False
        if (
            optimizer is None
        ):  # default is to try in this order but stop after one succeeds
            optimizers_to_try = ["TNC", "L-BFGS-B", "nelder-mead", "powell"]
            stop_after_success = True
        else:
            if optimizer in [
                "best",
                "BEST",
                "all",
                "ALL",
            ]:  # try all of the bounded optimizers
                optimizers_to_try = ["TNC", "L-BFGS-B", "nelder-mead", "powell"]
            elif optimizer.upper() == "TNC":
                optimizers_to_try = ["TNC"]
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizers_to_try = ["L-BFGS-B"]
            elif optimizer.upper() == "POWELL":
                optimizers_to_try = ["powell"]
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD"]:
                optimizers_to_try = ["nelder-mead"]
            else:
                raise ValueError(
                    str(
                        str(optimizer)
                        + ' is not a valid optimizer. Please specify either "TNC", "L-BFGS-B", "nelder-mead", "powell" or "best".'
                    )
                )

        # use each of the optimizers specified
        ALL_successes = []
        ALL_loglik = []
        ALL_results = []
        ALL_opt_names = []
        optimizers_tried_str = "Optimizers tried:"
        for opt in optimizers_to_try:
            optim_results = loglik_optimizer(
                LL_func,
                initial_guess,
                failures,
                right_censored,
                bounds,
                opt,
                force_shape,
                LL_func_force,
                func_name,
            )
            ALL_successes.append(optim_results[0])
            ALL_loglik.append(optim_results[1])
            ALL_results.append(optim_results[2])
            ALL_opt_names.append(opt)
            optimizers_tried_str = optimizers_tried_str + " " + opt + ","
            if optim_results[0] is True and stop_after_success is True:
                break  # stops after it finds one that works
        optimizers_tried_str = optimizers_tried_str[0:-1]  # remove the last comma
        # extract the results
        if True not in ALL_successes:
            # everything failed, need to return the initial guess
            self.success = False
            self.optimizer = None
            if func_name == "Weibull_mixture":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_mixture. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha_1 = initial_guess[0]
                self.beta_1 = initial_guess[1]
                self.alpha_2 = initial_guess[2]
                self.beta_2 = initial_guess[3]
                self.proportion_1 = initial_guess[4]
                self.proportion_2 = 1 - initial_guess[4]
            elif func_name == "Weibull_CR":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_CR. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha_1 = initial_guess[0]
                self.beta_1 = initial_guess[1]
                self.alpha_2 = initial_guess[2]
                self.beta_2 = initial_guess[3]
            elif func_name == "Weibull_DSZI":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_DSZI. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.DS = initial_guess[2]
                self.ZI = initial_guess[3]
            elif func_name == "Weibull_DS":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_DS. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.DS = initial_guess[2]
            elif func_name == "Weibull_ZI":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_ZI. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.ZI = initial_guess[2]
            else:
                colorprint(
                    str(
                        "WARNING: MLE estimates failed for "
                        + func_name
                        + ". The least squares estimates have been returned. These results may not be as accurate as MLE. "
                        + optimizers_tried_str
                    ),
                    text_color="red",
                )
                if force_shape is None:
                    self.scale = initial_guess[0]  # alpha, mu, Lambda
                    if func_name not in ["Exponential_1P", "Exponential_2P"]:
                        self.shape = initial_guess[1]  # beta, sigma
                    else:
                        if func_name == "Exponential_2P":
                            self.gamma = initial_guess[1]  # gamma for Exponential_2P
                    if func_name in [
                        "Weibull_3P",
                        "Gamma_3P",
                        "Loglogistic_3P",
                        "Lognormal_3P",
                    ]:
                        # gamma for Weibull_3P, Gamma_3P, Loglogistic_3P, Lognormal_3P
                        self.gamma = initial_guess[2]
                # this will only be reached for Weibull_2P, Normal_2P and Lognormal_2P so the scale and shape extraction is fine for these
                else:
                    self.scale = initial_guess[0]
                    self.shape = force_shape
        else:
            # at least one optimizer succeeded. Need to drop the failed ones then get the best of the successes
            items = np.arange(0, len(ALL_successes))[::-1]

            for i in items:
                if ALL_successes[i] is not True:
                    ALL_successes.pop(i)
                    ALL_loglik.pop(i)
                    ALL_results.pop(i)
                    ALL_opt_names.pop(i)
            idx_best = ALL_loglik.index(min(ALL_loglik))
            params = ALL_results[idx_best]
            self.optimizer = ALL_opt_names[idx_best]
            self.success = True

            if func_name == "Weibull_mixture":
                self.alpha_1 = params[0]
                self.beta_1 = params[1]
                self.alpha_2 = params[2]
                self.beta_2 = params[3]
                self.proportion_1 = params[4]
                self.proportion_2 = 1 - params[4]
            elif func_name == "Weibull_CR":
                self.alpha_1 = params[0]
                self.beta_1 = params[1]
                self.alpha_2 = params[2]
                self.beta_2 = params[3]
            elif func_name == "Weibull_DSZI":
                self.alpha = params[0]
                self.beta = params[1]
                self.DS = params[2]
                self.ZI = params[3]
            elif func_name == "Weibull_DS":
                self.alpha = params[0]
                self.beta = params[1]
                self.DS = params[2]
            elif func_name == "Weibull_ZI":
                self.alpha = params[0]
                self.beta = params[1]
                self.ZI = params[2]
            else:
                if force_shape is None:
                    self.scale = params[0]  # alpha, mu, Lambda
                    if func_name not in ["Exponential_1P", "Exponential_2P"]:
                        self.shape = params[1]  # beta, sigma
                    else:
                        if func_name == "Exponential_2P":
                            self.gamma = params[1]  # gamma for Exponential_2P
                    if func_name in [
                        "Weibull_3P",
                        "Gamma_3P",
                        "Loglogistic_3P",
                        "Lognormal_3P",
                    ]:
                        self.gamma = params[
                            2
                        ]  # gamma for Weibull_3P, Gamma_3P, Loglogistic_3P, Lognormal_3P
                else:  # this will only be reached for Weibull_2P, Normal_2P and Lognormal_2P so the scale and shape extraction is fine for these
                    self.scale = params[0]
                    self.shape = force_shape


class ALT_MLE_optimization:
    """
    This performs the MLE method to find the parameters.
    If the optimizer is None then all bounded optimizers will be tried and the
    best result (lowest log-likelihood) will be returned. If the optimizer is
    specified then it will be used. If it fails then the initial guess will be
    returned with a warning.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal".
    LL_func : function
        The log-likelihood function from the fitter
    initial_guess : list, array
        The initial guess of the model parameters that is used by the
        optimizer.
    optimizer : str, None
        This must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best",
        "all" or None. Fot detail on how these optimizers are used, please see
        the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    failure_stress_1 : array, list
        The failure stresses.
    failure_stress_2 : array, list
        The failure second stresses. This is only used for daul stress
        models.
    right_censored_stress_1 : array, list
        The right censored stresses. If there is no right censored data
        then this should be an empty array.
    right_censored_stress_2 : array, list
        The right censored second stresses. If there is no right
        censored data then this should be an empty array. This is only
        used for daul stress models.

    Returns
    -------
    a : float
        Only returned for Exponential, Eyring, Power, Dual_exponential, and
        Power_Exponential
    b : float
        Only returned for Exponential and Dual_Exponential
    c : float
        Only returned for Eyring, Dual_Exponential, Power_Exponential and
        Dual_Power
    n : float
        Only returned for Power, Power_Exponential, and Dual_Power
    m : float
        Only returned for Dual_Power
    beta : float
        Only returned for Weibull models
    sigma : float
        Only returned for Normal and Lognormal models
    success : bool
        Whether at least one optimizer succeeded. If False then the least
        squares result will be returned in place of the MLE result.
    optimizer : str, None
        The optimizer used. If MLE failed then None is returned as the
        optimizer.

    Notes
    -----
    Not all of the above returns are always returned. It depends on which model
    is being used.

    If the MLE method fails then the initial guess (from least squares) will be
    returned with a printed warning.
    """

    def __init__(
        self,
        model,
        dist,
        LL_func,
        initial_guess,
        optimizer,
        failures,
        failure_stress_1,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
    ):
        def loglik_optimizer(
            initial_guess,
            dual_stress,
            LL_func,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
            bounds,
            optimizer,
        ):
            """
            This sub-function does the actual optimization. It is called each
            time a new optimizer is tried.

            Parameters
            ----------
            initial_guess : list, array
                The initial guess of the model parameters that is used by the
                optimizer.
            dual_stress : bool
                Whether this is a dual_stress model.
            LL_func : function
                The log-likelihood function from the fitter
            failures : list, array
                The failure data
            right_censored : list, array
                The right censored data. If there is no right censored data then this
                should be an empty array.
            failure_stress_1 : array, list
                The failure stresses.
            failure_stress_2 : array, list
                The failure second stresses. This is only used for daul stress
                models.
            right_censored_stress_1 : array, list
                The right censored stresses. If there is no right censored data
                then this should be an empty array.
            right_censored_stress_2 : array, list
                The right censored second stresses. If there is no right
                censored data then this should be an empty array. This is only
                used for daul stress models.
            bounds : list
                The bounds on the solution
            optimizer : str, None
                This must be either "TNC", "L-BFGS-B", "nelder-mead", or
                "powell".

            Returns
            -------
            success : bool
                Whether the optimizer was successful
            log_likelihood : float
                The log-likelihood of the solution
            model_parameters : array
                The model parameters of the solution

            Notes
            -----
            The returns are provided in a tuple of success, log_likelihood,
            model_parameters.
            """
            delta_LL = 1
            LL_array = [1000000]
            runs = 0
            guess = initial_guess  # set the current guess as the initial guess and then update the current guess each iteration
            while (
                delta_LL > 0.001 and runs < 5
            ):  # exits after BIC convergence or 5 iterations
                runs += 1
                # single stress model
                if dual_stress is False:
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=(
                            failures,
                            right_censored,
                            failure_stress_1,
                            right_censored_stress_1,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    LL2 = -LL_func(
                        result.x,
                        failures,
                        right_censored,
                        failure_stress_1,
                        right_censored_stress_1,
                    )
                else:
                    # dual stress model
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=(
                            failures,
                            right_censored,
                            failure_stress_1,
                            failure_stress_2,
                            right_censored_stress_1,
                            right_censored_stress_2,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    LL2 = -LL_func(
                        result.x,
                        failures,
                        right_censored,
                        failure_stress_1,
                        failure_stress_2,
                        right_censored_stress_1,
                        right_censored_stress_2,
                    )
                LL_array.append(np.abs(LL2))
                delta_LL = abs(LL_array[-1] - LL_array[-2])
                guess = result.x  # update the guess each iteration
            return result.success, LL_array[-1], result.x

        if model == "Exponential":
            bounds = [(None, None), (0, None), (0, None)]  # a, b, shape
            dual_stress = False
        elif model == "Eyring":
            bounds = [(None, None), (None, None), (0, None)]  # a, c, shape
            dual_stress = False
        elif model == "Power":
            bounds = [(0, None), (None, None), (0, None)]  # a, n, shape
            dual_stress = False
        elif model == "Dual_Exponential":
            bounds = [
                (None, None),
                (None, None),
                (0, None),
                (0, None),
            ]  # a, b, c, shape
            dual_stress = True
        elif model == "Power_Exponential":
            bounds = [
                (None, None),
                (0, None),
                (None, None),
                (0, None),
            ]  # a, c, n, shape
            dual_stress = True
        elif model == "Dual_Power":
            bounds = [
                (0, None),
                (None, None),
                (None, None),
                (0, None),
            ]  # c, m, n, shape
            dual_stress = True
        else:
            raise ValueError(
                "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power"
            )

        if dist not in ["Weibull", "Exponential", "Lognormal", "Normal"]:
            raise ValueError(
                "dist must be one of Weibull, Exponential, Lognormal, Normal."
            )

        # remove the last bound as Exponential does not need a bound for shape
        if dist == "Exponential":
            bounds = bounds[0:-1]

        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []

        # determine which optimizers to use
        stop_after_success = False
        if (
            optimizer is None
        ):  # default is to try in this order but stop after one succeeds
            optimizers_to_try = ["TNC", "L-BFGS-B", "nelder-mead", "powell"]
            stop_after_success = True
        else:
            if optimizer in [
                "best",
                "BEST",
                "all",
                "ALL",
            ]:  # try all of the bounded optimizers
                optimizers_to_try = ["TNC", "L-BFGS-B", "nelder-mead", "powell"]
            elif optimizer.upper() == "TNC":
                optimizers_to_try = ["TNC"]
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizers_to_try = ["L-BFGS-B"]
            elif optimizer.upper() == "POWELL":
                optimizers_to_try = ["powell"]
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD"]:
                optimizers_to_try = ["nelder-mead"]
            else:
                raise ValueError(
                    str(
                        str(optimizer)
                        + ' is not a valid optimizer. Please specify either "TNC", "L-BFGS-B", "nelder-mead", "powell" or "best".'
                    )
                )

        # use each of the optimizers specified
        ALL_successes = []
        ALL_loglik = []
        ALL_results = []
        ALL_opt_names = []
        optimizers_tried_str = "Optimizers tried:"
        for opt in optimizers_to_try:
            optim_results = loglik_optimizer(
                initial_guess,
                dual_stress,
                LL_func,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
                bounds,
                opt,
            )
            ALL_successes.append(optim_results[0])
            ALL_loglik.append(optim_results[1])
            ALL_results.append(optim_results[2])
            ALL_opt_names.append(opt)
            optimizers_tried_str = optimizers_tried_str + " " + opt + ","
            if optim_results[0] is True and stop_after_success is True:
                break  # stops after it finds one that works
        optimizers_tried_str = optimizers_tried_str[0:-1]  # remove the last comma

        # extract the results
        if True not in ALL_successes:
            # everything failed, need to return the initial guess
            self.success = False
            self.optimizer = None
            colorprint(
                str(
                    "WARNING: MLE estimates failed for "
                    + dist
                    + "_"
                    + model
                    + ". The least squares estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str
                ),
                text_color="red",
            )

            if model == "Exponential":
                self.a = initial_guess[0]
                self.b = initial_guess[1]
            elif model == "Eyring":
                self.a = initial_guess[0]
                self.c = initial_guess[1]
            elif model == "Power":
                self.a = initial_guess[0]
                self.n = initial_guess[1]
            elif model == "Dual_Exponential":
                self.a = initial_guess[0]
                self.b = initial_guess[1]
                self.c = initial_guess[2]
            elif model == "Power_Exponential":
                self.a = initial_guess[0]
                self.c = initial_guess[1]
                self.n = initial_guess[2]
            elif model == "Dual_Power":
                self.c = initial_guess[0]
                self.m = initial_guess[1]
                self.n = initial_guess[2]

            if dual_stress is False:
                if dist == "Weibull":
                    self.beta = initial_guess[2]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = initial_guess[2]
            else:
                if dist == "Weibull":
                    self.beta = initial_guess[3]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = initial_guess[3]
        else:
            # at least one optimizer succeeded. Need to drop the failed ones then get the best of the successes
            items = np.arange(0, len(ALL_successes))[::-1]
            for i in items:
                if ALL_successes[i] is not True:
                    ALL_successes.pop(i)
                    ALL_loglik.pop(i)
                    ALL_results.pop(i)
                    ALL_opt_names.pop(i)
            idx_best = ALL_loglik.index(min(ALL_loglik))
            params = ALL_results[idx_best]
            self.optimizer = ALL_opt_names[idx_best]
            self.success = True

            if model == "Exponential":
                self.a = params[0]
                self.b = params[1]
            elif model == "Eyring":
                self.a = params[0]
                self.c = params[1]
            elif model == "Power":
                self.a = params[0]
                self.n = params[1]
            elif model == "Dual_Exponential":
                self.a = params[0]
                self.b = params[1]
                self.c = params[2]
            elif model == "Power_Exponential":
                self.a = params[0]
                self.c = params[1]
                self.n = params[2]
            elif model == "Dual_Power":
                self.c = params[0]
                self.m = params[1]
                self.n = params[2]

            if dual_stress is False:
                if dist == "Weibull":
                    self.beta = params[2]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = params[2]
            else:
                if dist == "Weibull":
                    self.beta = params[3]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = params[3]


def write_df_to_xlsx(df, path, **kwargs):
    """
    Writes a dataframe to an xlsx file
    For use exclusively by the Convert_data module

    Parameters
    ----------
    df : dataframe
        The dataframe to be written
    path : str
        The file path to the xlsx file.

    Returns
    -------
    None
        Writing the dataframe is the only action from this function.

    Notes
    -----
    The path must include the full file path including the extension. It is also
    necessary to use r at the start to specify raw text. See the
    `documentation <https://reliability.readthedocs.io/en/latest/Converting%20data%20between%20different%20formats.html>`_ for an example.
    """
    # this section checks whether the file exists and reprompts the user based on their choices
    ready_to_write = False
    counter1 = 0
    counter2 = 0
    path_changed = False
    while ready_to_write is False:
        counter1 += 1
        counter2 += 1
        try:
            f = open(path)  # try to open the file to see if it exists
            f.close()
            if counter1 == 1:
                colorprint(
                    "WARNING: the specified output file already exists",
                    text_color="red",
                )
            if counter2 == 1:
                choice = input("Do you want to overwrite the existing file (Y/N): ")
            else:
                choice = "N"  # subsequent loops can only be entered if the user did not want to overwrite the file
            if choice.upper() == "N":
                X = os.path.split(path)
                Y = X[1].split(".")
                Z = str(
                    Y[0] + "(new)" + "." + Y[1]
                )  # auto renaming will keep adding (new) to the filename if it already exists
                path = str(X[0] + "\\" + Z)
                path_changed = True
            elif choice.upper() == "Y":
                ready_to_write = True
            else:
                print("Invalid choice. Please specify Y or N")
                counter2 = 0
        except IOError:  # file does not exist
            ready_to_write = True
    if path_changed is True:
        print("Your output file has been renamed to:", path)
    # this section does the writing
    keys = kwargs.keys()
    if "excel_writer" in keys:
        colorprint(
            "WARNING: excel_writer has been overridden by path. Please only use path to specify the file path for the xlsx file to write.",
            text_color="red",
        )
        kwargs.pop("excel_writer")
    if "index" in keys:
        write_index = kwargs.pop("index")
    else:
        write_index = False
    df.to_excel(path, index=write_index, **kwargs)


def removeNaNs(X):
    """
    Removes NaNs from a list or array.

    Parameters
    ----------
    X : array, list
        The array or list to be processed.

    Returns
    -------
    output : list, array
        A list or array of the same type as the input with the NaNs removed.

    Notes
    -----
    This is better than simply using "x = x[numpy.logical_not(numpy.isnan(x))]"
    as numpy crashes for str and bool.
    """
    if type(X) == np.ndarray:
        X = list(X)
        arr_out = True
    else:
        arr_out = False
    out = []
    for i in X:
        if type(i) in [str, bool, np.str_]:
            if i != "nan":
                out.append(i)
        elif np.logical_not(np.isnan(i)):  # this only works for numbers
            out.append(i)
    if arr_out is True:
        out = np.asarray(out)
    return out


class make_fitted_dist_params_for_ALT_probplots:
    """
    This function creates a class structure for the ALT probability plots to
    give to Probability_plotting.

    Parameters
    ----------
    dist : str
        The distribution. Must be either "Weibull", "Lognormal", "Normal", or
        "Exponential".
    params : list, array
        The parameters of the model. Must be 2 elements for Weibull, Lognormal,
        and Normal, and must be 1 element for Exponential.

    Returns
    -------
    alpha : float
        Only returned for Weibull
    beta : float
        Only returned for Weibull
    gamma : int
        This will always be 0. Only returned for Weibull, Lognormal, and
        Exponential.
    alpha_SE : None
        Only returned for Weibull
    beta_SE : None
        Only returned for Weibull
    Cov_alpha_beta : None
        Only returned for Weibull
    mu : float
        Only returned for Normal and Lognormal
    sigma : float
        Only returned for Normal and Lognormal
    Cov_mu_sigma : None
        Only returned for Normal and Lognormal
    Lambda : float
        Only returned for Exponential
    Lambda_SE : None
        Only returned for Exponential

    Notes
    -----
    This function only exists to convert a list or array of parameters into a
    class with the correct parameters for the probability plots to use.
    """

    def __init__(self, dist, params):
        if dist == "Weibull":
            self.alpha = params[0]
            self.beta = params[1]
            self.gamma = 0
            self.alpha_SE = None
            self.beta_SE = None
            self.Cov_alpha_beta = None
        elif dist == "Lognormal":
            self.mu = np.log(params[0])
            self.sigma = params[1]
            self.gamma = 0
            self.mu_SE = None
            self.sigma_SE = None
            self.Cov_mu_sigma = None
        elif dist == "Normal":
            self.mu = params[0]
            self.sigma = params[1]
            self.mu_SE = None
            self.sigma_SE = None
            self.Cov_mu_sigma = None
        elif dist == "Exponential":
            self.Lambda = 1 / params[0]
            self.Lambda_SE = None
            self.gamma = 0
        else:
            raise ValueError(
                "dist must be one of Weibull, Normal, Lognormal, Exponential"
            )


def ALT_prob_plot(
    dist,
    model,
    stresses_for_groups,
    failure_groups,
    right_censored_groups,
    life_func,
    shape,
    scale_for_change_df,
    shape_for_change_df,
    use_level_stress=None,
    ax=True,
):
    """
    Generates an ALT probability plot using the inputs provided.

    Parameters
    ----------
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal"
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    stresses_for_groups : list
        The stresses for the failure groups
    failure_groups : list
        The failure groups. This is a list of lists.
    right_censored_groups
        The failure groups. This is a list of lists.
    life_func : function
        The life function for the ALT life model.
    shape : float, int
        The shape parameter of the model.
    scale_for_change_df : array, list
        The list of scale parameters for the lines.
    shape_for_change_df
        The list of shape parameters for the lines.
    use_level_stress : float, int, array, list, None
        The use level stress. This must be an array or list for dual stress
        models. Default is None.
    ax : axis, bool, optional
        The axis handle to use. Default is True which will create a new plot.
        If False then no plot will be generated.

    Returns
    -------
    current_axis : axis
        The axis handle of the plot. If ax is specified in the inputs then this
        will be the same handle.
    """

    if ax is True or type(ax) == _axes.Axes:
        if type(ax) == _axes.Axes:
            plt.sca(ax=ax)  # use the axes passed
        else:
            plt.figure()  # if no axes is passed, make a new figure

        from reliability.Probability_plotting import plotting_positions

        if dist == "Weibull":
            from reliability.Probability_plotting import (
                Weibull_probability_plot as probplot,
            )
            from reliability.Distributions import Weibull_Distribution as Distribution
        elif dist == "Lognormal":
            from reliability.Probability_plotting import (
                Lognormal_probability_plot as probplot,
            )
            from reliability.Distributions import Lognormal_Distribution as Distribution
        elif dist == "Normal":
            from reliability.Probability_plotting import (
                Normal_probability_plot as probplot,
            )
            from reliability.Distributions import Normal_Distribution as Distribution
        elif dist == "Exponential":
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale as probplot,
            )
            from reliability.Distributions import (
                Exponential_Distribution as Distribution,
            )
        else:
            raise ValueError(
                "dist must be either Weibull, Lognormal, Normal, Exponential"
            )

        if model in ["Dual_Exponential", "Power_Exponential", "Dual_Power"]:
            dual_stress = True
        elif model in ["Exponential", "Eyring", "Power"]:
            dual_stress = False
        else:
            raise ValueError(
                "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power"
            )

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()[
            "color"
        ]  # gets the default color cycle
        x_array = []
        y_array = []
        if dual_stress is True:
            for i, stress in enumerate(stresses_for_groups):
                f = failure_groups[i]
                if right_censored_groups is None:
                    rc = None
                else:
                    rc = right_censored_groups[i]
                # get the plotting positions so they can be given to probability_plot_xylims for autoscaling
                x, y = plotting_positions(failures=f, right_censored=rc)
                x_array.extend(x)
                y_array.extend(y)
                # generate the probability plot and the line from the life-stress model
                fitted_dist_params = make_fitted_dist_params_for_ALT_probplots(
                    dist=dist, params=[life_func(S1=stress[0], S2=stress[1]), shape]
                )
                probplot(
                    failures=f,
                    right_censored=rc,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_cycle[i],
                    label=str(
                        round_and_string(stress[0])
                        + ", "
                        + round_and_string(stress[1])
                    ),
                )
                # plot the original fitted line
                if dist == "Exponential":
                    if scale_for_change_df[i] != "":
                        Distribution(1 / scale_for_change_df[i]).CDF(
                            linestyle="--", alpha=0.5, color=color_cycle[i]
                        )
                else:
                    if scale_for_change_df[i] != "":
                        Distribution(
                            scale_for_change_df[i], shape_for_change_df[i]
                        ).CDF(linestyle="--", alpha=0.5, color=color_cycle[i])

            if use_level_stress is not None:
                if dist in ["Weibull", "Normal"]:
                    distribution_at_use_stress = Distribution(
                        life_func(S1=use_level_stress[0], S2=use_level_stress[1]), shape
                    )
                elif dist == "Lognormal":
                    distribution_at_use_stress = Distribution(
                        np.log(
                            life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                        ),
                        shape,
                    )
                elif dist == "Exponential":
                    distribution_at_use_stress = Distribution(
                        1 / life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    )
                distribution_at_use_stress.CDF(
                    color=color_cycle[i + 1],
                    label=str(
                        round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                        + " (use stress)"
                    ),
                )
                x_array.extend(
                    [
                        distribution_at_use_stress.quantile(min(y_array)),
                        distribution_at_use_stress.quantile(max(y_array)),
                    ]
                )  # this ensures the plot limits include the use stress distribution

            plt.legend(title="     Stress 1, Stress 2")

        else:
            for i, stress in enumerate(stresses_for_groups):
                f = failure_groups[i]
                if right_censored_groups is None:
                    rc = None
                else:
                    rc = right_censored_groups[i]
                # get the plotting positions so they can be given to probability_plot_xylims for autoscaling
                x, y = plotting_positions(failures=f, right_censored=rc)
                x_array.extend(x)
                y_array.extend(y)
                # generate the probability plot and the line from the life-stress model
                fitted_dist_params = make_fitted_dist_params_for_ALT_probplots(
                    dist=dist, params=[life_func(S1=stress), shape]
                )
                probplot(
                    failures=f,
                    right_censored=rc,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_cycle[i],
                    label=round_and_string(stress),
                )
                # plot the original fitted line
                if dist == "Exponential":
                    if scale_for_change_df[i] != "":
                        Distribution(1 / scale_for_change_df[i]).CDF(
                            linestyle="--", alpha=0.5, color=color_cycle[i]
                        )
                else:
                    if scale_for_change_df[i] != "":
                        Distribution(
                            scale_for_change_df[i], shape_for_change_df[i]
                        ).CDF(linestyle="--", alpha=0.5, color=color_cycle[i])

            if use_level_stress is not None:
                if dist in ["Weibull", "Normal"]:
                    distribution_at_use_stress = Distribution(
                        life_func(S1=use_level_stress), shape
                    )
                elif dist == "Lognormal":
                    distribution_at_use_stress = Distribution(
                        np.log(life_func(S1=use_level_stress)), shape
                    )
                elif dist == "Exponential":
                    distribution_at_use_stress = Distribution(
                        1 / life_func(S1=use_level_stress)
                    )
                distribution_at_use_stress.CDF(
                    color=color_cycle[i + 1],
                    label=str(
                        round_and_string(use_level_stress) + " (use stress)"
                    ),
                )
                x_array.extend(
                    [
                        distribution_at_use_stress.quantile(min(y_array)),
                        distribution_at_use_stress.quantile(max(y_array)),
                    ]
                )  # this ensures the plot limits include the use stress distribution

            plt.legend(title="Stress")

        probplot_type = dist.lower()
        if dist == "Exponential":
            probplot_type = "weibull"

        probability_plot_xylims(x=x_array, y=y_array, dist=probplot_type, spacing=0.1)
        probability_plot_xyticks()
        plt.title("Probability plot\n" + dist + "_" + model + " Model")
        plt.tight_layout()
        return plt.gca()


def life_stress_plot(
    model,
    dist,
    life_func,
    failure_groups,
    stresses_for_groups,
    use_level_stress=None,
    ax=True,
):
    """
    Generates a life stress plot using the inputs provided. The life stress plot
    is an output from each of the ALT_fitters.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal"
    life_func : function
        The life function for the ALT life model.
    failure_groups : list
        The failure groups. This is a list of lists.
    stresses_for_groups : list
        The stresses for the failure groups
    use_level_stress : float, int, array, list, None
        The use level stress. This must be an array or list for dual stress
        models. Default is None.
    ax : axes, bool, optional
        The axes handle to use. Default is True which will create a new plot.
        If False then no plot will be generated.
        This is also used to flip the x and y axes to make a stress-life plot. Use ax='swap' to make the life-stress
        plot change its axes to be a stress-life plot (similar to SN diagram).

    Returns
    -------
    current_axes : axes
        The axes handle of the plot. If ax is specified in the inputs then this
        will be the same handle.
    """
    if ax in ['swap','swapped','flip','flipped','SWAP','SWAPPED','FLIP','FLIPPED','Swap','Swapped','Flip','Flipped']:
        ax = True
        swap_xy = True
    else:
        swap_xy = False

    if ax is True or type(ax) == _axes.Axes:
        if model in ["Dual_Exponential", "Power_Exponential", "Dual_Power"]:
            dual_stress = True
        elif model in ["Exponential", "Eyring", "Power"]:
            dual_stress = False
        else:
            raise ValueError(
                "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power"
            )

        if type(ax) == _axes.Axes:
            if dual_stress is False:
                if hasattr(ax, "get_zlim") is False:
                    plt.sca(ax=ax)  # use the axes passed if 2d
                else:
                    colorprint(
                        "WARNING: The axes passed to the life_stress_plot has been ignored as it contains 3d projection. Only specify 3d projection in life stress plots for dual stress models.",
                        text_color="red",
                    )
                    plt.figure(figsize=(9, 9))
            else:  # dual stress models require 3d projection
                if hasattr(ax, "get_zlim") is True:
                    plt.sca(ax=ax)  # use the axes passed if 3d
                else:
                    colorprint(
                        "WARNING: The axes passed to the life_stress_plot has been ignored as it does not have 3d projection. This is a requirement of life stress plots for all dual stress models.",
                        text_color="red",
                    )
                    fig = plt.figure(figsize=(9, 9))
                    ax = fig.add_subplot(111, projection="3d")
        else:
            fig = plt.figure(figsize=(9, 9))  # if no axes is passed, make a new figure
            if dual_stress is True:
                ax = fig.add_subplot(111, projection="3d")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()[
            "color"
        ]  # gets the default color cycle

        if dist == "Weibull":
            line_label = r"$\alpha$"
        elif dist == "Lognormal":
            line_label = r"$ln(\sigma)$"
        elif dist == "Normal":
            line_label = r"$\sigma$"
        elif dist == "Exponential":
            line_label = r"$1/\lambda$"
        else:
            raise ValueError(
                "dist must be either Weibull, Lognormal, Normal, Exponential"
            )

        if dual_stress is True:
            # collect all the stresses so we can find their min and max
            stress_1_array0 = []
            stress_2_array0 = []
            for stress in stresses_for_groups:
                stress_1_array0.append(stress[0])
                stress_2_array0.append(stress[1])
            if use_level_stress is not None:
                stress_1_array0.append(use_level_stress[0])
                stress_2_array0.append(use_level_stress[1])
            min_stress_1 = min(stress_1_array0)
            max_stress_1 = max(stress_1_array0)
            min_stress_2 = min(stress_2_array0)
            max_stress_2 = max(stress_2_array0)
            # find the upper and lower limits so we can generate the grid of points for the surface
            stress_1_delta_log = np.log(max_stress_1) - np.log(min_stress_1)
            stress_2_delta_log = np.log(max_stress_2) - np.log(min_stress_2)
            stress_1_array_lower = np.exp(
                np.log(min_stress_1) - stress_1_delta_log * 0.2
            )
            stress_2_array_lower = np.exp(
                np.log(min_stress_2) - stress_2_delta_log * 0.2
            )
            stress_1_array_upper = np.exp(
                np.log(max_stress_1) + stress_1_delta_log * 0.2
            )
            stress_2_array_upper = np.exp(
                np.log(max_stress_2) + stress_2_delta_log * 0.2
            )
            stress_1_array = np.linspace(stress_1_array_lower, stress_1_array_upper, 50)
            stress_2_array = np.linspace(stress_2_array_lower, stress_2_array_upper, 50)
            X, Y = np.meshgrid(stress_1_array, stress_2_array)
            Z = life_func(S1=X, S2=Y)
            # plot the surface showing stress_1 and stress_2 vs life
            normalized_colors = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
            ax.plot_surface(
                X,
                Y,
                Z,
                cmap="jet_r",
                norm=normalized_colors,
                linewidth=1,
                antialiased=False,
                alpha=0.5,
                zorder=0,
            )
            for i, stress in enumerate(stresses_for_groups):
                # plot the failures as a scatter plot
                ax.scatter(
                    stress[0],
                    stress[1],
                    failure_groups[i],
                    color=color_cycle[i],
                    s=30,
                    label=str(
                        "Failures at stress of "
                        + round_and_string(stress[0])
                        + ", "
                        + round_and_string(stress[1])
                    ),
                    zorder=1,
                )
            if use_level_stress is not None:
                # plot the use level stress
                ax.scatter(
                    use_level_stress[0],
                    use_level_stress[1],
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]),
                    color=color_cycle[i + 1],
                    s=30,
                    label=str(
                        "Use stress of "
                        + round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                    ),
                    marker="^",
                    zorder=2,
                )
            ax.set_zlabel("Life")
            ax.set_zlim(bottom=0)
            ax.computed_zorder = False  # computed_zorder ensures the specified zorder is respected so that the scatter plot sits on top of the surface
            plt.xlabel("Stress 1")
            plt.ylabel("Stress 2")
            plt.xlim(min(stress_1_array), max(stress_1_array))
            plt.ylim(min(stress_2_array), max(stress_2_array))
            plt.legend(loc="upper right")
            plt.title("Life-stress plot\n" + dist + "_" + model + " model")

        else:  # single stress model
            if use_level_stress is not None:
                min_stress = min(min(stresses_for_groups), use_level_stress)
            else:
                min_stress = min(stresses_for_groups)
            max_stress = max(stresses_for_groups)
            stress_delta_log = np.log(max_stress) - np.log(min_stress)
            # lower and upper lim
            stress_array_lower = np.exp(np.log(min_stress) - stress_delta_log * 0.2)
            stress_array_upper = np.exp(np.log(max_stress) + stress_delta_log * 0.2)
            # array for the life-stress line
            stress_array = np.linspace(0, stress_array_upper * 10, 1000)
            life_array = life_func(S1=stress_array)
            if swap_xy is True:
                plt.plot(
                    life_array,
                    stress_array,
                    label=str("Characteristic life (" + line_label + ")"),
                    color="k",
                )
                plt.xlabel("Life")
                plt.ylabel("Stress")
            else:
                plt.plot(
                    stress_array,
                    life_array,
                    label=str("Characteristic life (" + line_label + ")"),
                    color="k",
                )
                plt.ylabel("Life")
                plt.xlabel("Stress")
            for i, stress in enumerate(stresses_for_groups):
                failure_points = failure_groups[i]
                stress_points = np.ones_like(failure_points) * stress
                if swap_xy is True:
                    plt.scatter(
                        failure_points,
                        stress_points,
                        color=color_cycle[i],
                        alpha=0.7,
                        label=str(
                            "Failures at stress of " + round_and_string(stress)
                        ),
                    )
                else:
                    plt.scatter(
                        stress_points,
                        failure_points,
                        color=color_cycle[i],
                        alpha=0.7,
                        label=str(
                            "Failures at stress of " + round_and_string(stress)
                        ),
                    )
            if use_level_stress is not None:
                alpha_at_use_stress = life_func(S1=use_level_stress)
                if swap_xy is True:
                    plt.plot(
                        [-1e20, alpha_at_use_stress, alpha_at_use_stress],
                        [use_level_stress, use_level_stress, plt.xlim()[0]],
                        label=str(
                            "Use stress of " + round_and_string(use_level_stress)
                        ),
                        color=color_cycle[i + 1],
                    )
                else:
                    plt.plot(
                        [use_level_stress, use_level_stress, plt.xlim()[0]],
                        [-1e20, alpha_at_use_stress, alpha_at_use_stress],
                        label=str(
                            "Use stress of " + round_and_string(use_level_stress)
                        ),
                        color=color_cycle[i + 1],
                    )
            # this is a list comprehension to flatten the list of lists. np.ravel won't work here
            flattened_failure_groups = [
                item for sublist in failure_groups for item in sublist
            ]
            if swap_xy is True:
                plt.xlim(
                    0,
                    1.2
                    * max(life_func(S1=stress_array_lower), max(flattened_failure_groups)),
                )
                plt.ylim(stress_array_lower, stress_array_upper)
                plt.title("Stress-life plot\n" + dist + "-" + model + " model")
            else:
                plt.ylim(
                    0,
                    1.2
                    * max(life_func(S1=stress_array_lower), max(flattened_failure_groups)),
                )
                plt.xlim(stress_array_lower, stress_array_upper)
                plt.title("Life-stress plot\n" + dist + "-" + model + " model")
            plt.legend(loc="upper right")
            plt.tight_layout()
        return plt.gca()


def xy_downsample(x, y, downsample_factor=None, default_max_values=1000):
    """
    This function downsamples the x and y arrays. This exists to make plotting
    much faster, particularly when matplotlib becomes very slow for tens of
    thousands of datapoints.

    Parameters
    ----------
    x : array, list
        The x values
    y : array, list
        The y values
    downsample_factor : int, optional
        How must downsampling to do. See Notes for more detail.
    default_max_values : int, optional
        The maximum number of values to be returned if downsample_factor is
        None. See Notes for more detail.

    Returns
    -------
    x : array
        The downsampled x values
    y : array
        The downsampled y values

    Notes
    -----
    Downsampling is done using the downsample_factor. If the down_sample factor
    is 2 then every second value will be returned, if 3 then every third value
    will be returned. The first and last items will always be included in the
    downsampled dataset.

    If downsample_factor is not specified, downsampling will only occur if there
    are more than default_max_values. The downsample factor will aim for the
    number of values to be returned to be between default_max_values/2 and
    default_max_values. By default this is between 500 and 1000.
    """

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    len_x = len(x)
    if downsample_factor is False:
        return x, y
    if downsample_factor in [None, True] and len_x < default_max_values:
        return x, y
    else:
        if downsample_factor in [None, True]:
            downsample_factor = np.floor(len_x / (0.5 * default_max_values))
        if len_x / downsample_factor < 2:
            return x, y
        else:
            indices = np.arange(
                start=0, stop=len_x, step=int(np.floor(downsample_factor)), dtype=int
            )

            if len_x - 1 not in indices:
                indices[-1] = len_x - 1
            x_downsample = []
            y_downsample = []
            for idx in indices:
                x_downsample.append(x_sorted[idx])
                y_downsample.append(y_sorted[idx])
            return x_downsample, y_downsample


def distributions_input_checking(
    self,
    func,
    xvals,
    xmin,
    xmax,
    show_plot=None,
    plot_CI=None,
    CI_type=None,
    CI=None,
    CI_y=None,
    CI_x=None,
):
    """
    Performs checks and sets default values for the inputs to distributions
    sub function (PDF, CDF, SF, HF, CHF)

    Parameters
    ----------
    self : object
        Distribution object created by reliability.Distributions
    func : str
        Must be either 'PDF','CDF', 'SF', 'HF', 'CHF'
    xvals : array, list
        x-values for plotting.
    xmin : int, float
        minimum x-value for plotting.
    xmax : int, float
        maximum x-value for plotting.
    show_plot : bool
        Whether the plot is to be shown.
    plot_CI : bool, optional
        Whether the confidence intervals are to be shown on the plot. Default is
        None.
    CI_type : str, optional
        If specified, it must be "time" or "reliability". Default is None
    CI : float, optional
        The confidence intervals. If specified, it must be between 0 and 1.
        Default is None.
    CI_y : list, array, optional
        The confidence interval y-values to trace. Default is None.
    CI_x : list, array, optional
        The confidence interval x-values to trace. Default is None.

    Returns
    -------
    X : array
        An array of the x-values for the plot. Created using generate_X_array
    xvals : array, list
        x-values for plotting.
    xmin : int, float
        minimum x-value for plotting.
    xmax : int, float
        maximum x-value for plotting.
    show_plot : bool
        Whether the plot is to be shown. Default is True. Only returned if func
        is 'PDF','CDF', 'SF', 'HF, or 'CHF'
    plot_CI : bool
        Whether the confidence intervals are to be shown on the plot. Default is
        True. Only returned if func is 'CDF', 'SF', or 'CHF' and self.name
        !='Beta'.
    CI_type : str
        The type of confidence interval. Will be either "time", "reliability", or "none".
        Default is "time". Only returned if func is 'CDF', 'SF', or 'CHF'.
        If self.name =='Beta' or self.name=='Exponential' it will return 'none'. If
        self.CI_type is specified and CI_type is not specified then self.CI_type will be
        used for CI_type.
    CI : float
        The confidence intervals between 0 and 1. Default is 0.95. Only returned
        if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'. If self.CI is
        specified and CI is None then self.CI will be used for CI.
    CI_y : list, array, float, int
        The confidence interval y-values to trace. Default is None. Only
        returned if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'.
    CI_x : list, array, float, int
        The confidence interval x-values to trace. Default is None. Only
        returned if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'.
    """
    if func not in ["PDF", "CDF", "SF", "HF", "CHF", "ALL"]:
        raise ValueError("func must be either 'PDF','CDF', 'SF', 'HF', 'CHF', 'ALL'")

    # type checking
    if type(xvals) not in [type(None), list, np.ndarray, int, float]:
        raise ValueError("xvals must be an int, float, list, or array. Default is None")
    if type(xmin) not in [type(None), int, float]:
        raise ValueError("xmin must be an int or float. Default is None")
    if type(xmax) not in [type(None), int, float]:
        raise ValueError("xmax must be an int or float. Default is None")
    if type(show_plot) not in [type(None), bool]:
        raise ValueError("show_plot must be True or False. Default is True")
    if type(plot_CI) not in [type(None), bool]:
        raise ValueError(
            "plot_CI must be True or False. Default is True. Only used if the distribution object was created by Fitters."
        )
    if type(CI_type) not in [type(None), str]:
        raise ValueError(
            'CI_type must be "time" or "reliability". Default is "time". Only used if the distribution object was created by Fitters.'
        )
    if CI is True:
        CI = 0.95
    if CI is False:
        CI = 0.95
        plot_CI = False
    if type(CI) not in [type(None), float]:
        raise ValueError(
            "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if the distribution object was created by Fitters."
        )
    if type(CI_y) not in [type(None), list, np.ndarray, float, int]:
        raise ValueError(
            'CI_y must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="time".'
        )
    if type(CI_x) not in [type(None), list, np.ndarray, float, int]:
        raise ValueError(
            'CI_x must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="reliability".'
        )

    # default values
    if (
        xmin is None
        and xmax is None
        and type(xvals) not in [list, np.ndarray, type(None)]
    ):
        X = xvals
        show_plot = False
    else:
        X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

    if CI is None and self.Z is None:
        CI = 0.95
    elif CI is not None:  # CI takes precedence over Z
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1")
    else:  # CI is None and Z is not None
        CI = 1 - ss.norm.cdf(-self.Z) * 2  # converts Z to CI

    if show_plot is None:
        show_plot = True

    if plot_CI is None:
        plot_CI == True

    no_CI_array = ['None','NONE','none','OFF','Off','off']
    if self.name == "Exponential":
        if CI_type not in no_CI_array and CI_type is not None:
            colorprint(
                "WARNING: CI_type is not required for the Exponential distribution since the confidence intervals of time and reliability are identical",
                text_color="red",
            )
        CI_type = None
    elif self.name == "Beta":
        if CI_type not in no_CI_array and CI_type is not None:
            colorprint(
                "WARNING: CI_type is not used for the Beta distribution since the confidence intervals are not implemented",
                text_color="red",
            )
        CI_type = None
    else:
        if CI_type is None:
            if self.CI_type in no_CI_array or self.CI_type is None:
                CI_type = None
            else:
                CI_type = self.CI_type
        elif CI_type in no_CI_array:
            CI_type = None

        if type(CI_type) == str:
            if CI_type.upper() in ["T", "TIME"]:
                CI_type = "time"
            elif CI_type.upper() in ["R", "REL", "RELIABILITY"]:
                CI_type = "reliability"
            else:
                colorprint(
                    "WARNING: CI_type is not recognised. Accepted values are 'time', 'reliability' and 'none'",
                    text_color="red",
                )
                CI_type = None

    if CI_x is not None and CI_y is not None:
        if CI_type == "reliability":
            colorprint(
                "WARNING: CI_x and CI_y can not be specified at the same time. CI_y has been reset to None and the results for CI_x will be provided.",
                text_color="red",
            )
            CI_y = None
        else:
            colorprint(
                "WARNING: CI_x and CI_y can not be specified at the same time. CI_x has been reset to None and the results for CI_y will be provided.",
                text_color="red",
            )
            CI_x = None

    if CI_x is not None:
        if type(CI_x) in [float, int]:
            if CI_x <= 0 and self.name not in ["Normal", "Gumbel"]:
                raise ValueError("CI_x must be greater than 0")
            CI_x = np.array([CI_x])  # package as array. Will be unpacked later
        else:
            CI_x = np.asarray(CI_x)
            if min(CI_x) <= 0 and self.name not in ["Normal", "Gumbel"]:
                raise ValueError("CI_x values must all be greater than 0")

    if CI_y is not None:
        if type(CI_y) in [float, int]:
            if CI_y <= 0:
                raise ValueError("CI_y must be greater than 0")
            if CI_y >= 1 and func in ["CDF", "SF"]:
                raise ValueError("CI_y must be less than 1")
            CI_y = np.array([CI_y])  # package as array. Will be unpacked later
        else:
            CI_y = np.asarray(CI_y)
            if min(CI_y) <= 0:
                raise ValueError("CI_y values must all be above 0")
            if max(CI_y) >= 1 and func in ["CDF", "SF"]:
                raise ValueError("CI_y values must all be below 1")

    if self.name == "Beta":
        if func in ["PDF", "CDF", "SF", "HF", "CHF"]:
            return X, xvals, xmin, xmax, show_plot
        else:  # func ='ALL' which is used for the .plot() method
            return X, xvals, xmin, xmax
    else:  # everything except the Beta distribution
        if func in ["PDF", "HF"]:
            return X, xvals, xmin, xmax, show_plot
        elif func in ["CDF", "SF", "CHF"]:
            return X, xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x
        else:  # func ='ALL' which is used for the .plot() method
            return X, xvals, xmin, xmax


def extract_CI(dist, func="CDF", CI_type="time", CI=0.95, CI_y=None, CI_x=None):
    """
    Extracts the confidence bounds at CI_x or CI_y.

    Parameters
    ----------
    dist : object
        Distribution object from reliability.Distributions
    func : str
        Must be either 'CDF', 'SF', 'CHF'
    CI_type : str
        Must be either 'time' or 'reliability'
    CI : float
        The confidence interval. Must be between 0 and 1.
    CI_y : list, array
        The y-values from which to extract the confidence interval (x-values)
        for bounds on time.
    CI_x : list, array
        The x-values from which to extract the confidence interval (y-values)
        for bounds on reliability.

    Returns
    -------
    lower : array
        An array of the lower confidence bounds at CI_x or CI_y
    upper : array
        An array of the upper confidence bounds at CI_x or CI_y

    Notes
    -----
    If CI_type="time" then CI_y must be specified in order to extract the
    confidence bounds on time.

    If CI_type="reliability" then CI_x must be specified in order to extract the
    confidence bounds on reliability.
    """

    if dist.name == "Exponential":
        if CI_y is not None and CI_x is not None:
            raise ValueError(
                "Both CI_x and CI_y have been provided. Please provide only one."
            )
        if CI_y is not None:
            if func == "SF":
                q = np.asarray(CI_y)
            elif func == "CDF":
                q = 1 - np.asarray(CI_y)
            elif func == "CHF":
                q = np.exp(-np.asarray(CI_y))
            else:
                raise ValueError("func must be CDF, SF, or CHF")
            SF_time = distribution_confidence_intervals.exponential_CI(
                self=dist, CI=CI, q=q
            )
            lower, upper = SF_time[0], SF_time[1]
        elif CI_x is not None:
            SF_rel = distribution_confidence_intervals.exponential_CI(
                self=dist, CI=CI, x=CI_x
            )
            if func == "SF":
                lower, upper = SF_rel[1], SF_rel[0]
            elif func == "CDF":
                lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
            elif func == "CHF":
                lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            else:
                raise ValueError("func must be CDF, SF, or CHF")
        else:
            lower, upper = None, None
    else:
        if CI_y is not None and CI_x is not None:
            raise ValueError(
                "Both CI_x and CI_y have been provided. Please provide only one."
            )
        if CI_x is not None and CI_y is None and CI_type == "time":
            colorprint(
                'WARNING: If CI_type="time" then CI_y must be specified in order to extract the confidence bounds on time.',
                text_color="red",
            )
            lower, upper = None, None
        elif CI_y is not None and CI_x is None and CI_type == "reliability":
            colorprint(
                'WARNING: If CI_type="reliability" then CI_x must be specified in order to extract the confidence bounds on reliability.',
                text_color="red",
            )
            lower, upper = None, None
        elif (CI_y is not None and CI_type == "time") or (
            CI_x is not None and CI_type == "reliability"
        ):
            if CI_type == "time":
                if func == "SF":
                    q = np.asarray(CI_y)
                elif func == "CDF":
                    q = 1 - np.asarray(CI_y)
                elif func == "CHF":
                    q = np.exp(-np.asarray(CI_y))
                else:
                    raise ValueError("func must be CDF, SF, or CHF")
            if dist.name == "Weibull":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.weibull_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.weibull_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Normal":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.normal_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.normal_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[1], SF_rel[0]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            elif dist.name == "Lognormal":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.lognormal_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.lognormal_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[1], SF_rel[0]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            elif dist.name == "Gamma":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.gamma_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.gamma_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Gumbel":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.gumbel_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.gumbel_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Loglogistic":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.loglogistic_CI(
                        self=dist, CI_type="time", CI=CI, q=q
                    )
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.loglogistic_CI(
                        self=dist, CI_type="reliability", CI=CI, x=CI_x
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            else:
                raise ValueError("Unknown distribution")
        else:
            lower, upper = None, None
    if type(lower) is not type(None):
        if len(lower) == 1:  # unpack arrays of length 1
            lower, upper = lower[0], upper[0]
    return lower, upper


def unpack_single_arrays(array):
    """
    Unpacks arrays with a single element to return just that element

    Parameters
    ----------
    array : float, int, list, array
        The value for unpacking

    Returns
    -------
    output : float, list, int, array
        If the input was a single length numpy array then the output will be the
        item from the array. If the input was anything else then the output will
        match the input
    """
    if type(array) == np.ndarray:
        if len(array) == 1:
            out = array[0]
        else:
            out = array
    else:
        out = array
    return out


def reshow_figure(handle):
    """
    Shows a figure from an axes handle or figure handle.
    This is useful if the handle is saved to a variable but the figure has been
    closed.
    Note that the Navigation Toolbar (for pan, zoom, and save) is still
    connected to the old figure. There is no known work around for this issue.

    Parameters
    ----------
    handle : object
        The axes handle (type(_axes.Axes)) or figure handle (type(Figure))

    Returns
    -------
    None
        The figure is automatically shown using plt.show().
    """
    type(ax) == _axes.Axes
    if type(handle) is not Figure and type(handle) != _axes.Axes:
        # check that the handle is either an axes or a figure
        raise ValueError("handle must be an axes handle or a figure handle")
    elif type(handle) == _axes.Axes:
        # if the handle is an axes then extract the Figure
        handle = handle.figure

    # rebuild the figure
    figsize = handle.get_size_inches()
    fig_new = plt.figure()
    new_manager = fig_new.canvas.manager
    new_manager.canvas.figure = handle
    handle.set_canvas(new_manager.canvas)
    handle.set_size_inches(figsize)
    plt.show()
