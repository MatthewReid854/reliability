"""
Utils (utilities)

This is a collection of utilities that are used throughout the python reliability library.
Functions have been placed here as to declutter the dropdown lists of your IDE and to provide a common resource across multiple modules.
It is not expected that users will be using any utils directly.

Included functions are:
round_to_decimals - applies different rounding rules to numbers above and below 1 so that small numbers do not get rounded to 0.
transform_spaced - Creates linearly spaced array (in transform space) based on a specified transform. This is like np.logspace but it can make an array that is weibull spaced, normal spaced, etc.
axes_transforms - Custom scale functions used in Probability_plotting
get_axes_limits - gets the current axes limits
restore_axes_limits - restores the axes limits based on values from get_axes_limits()
generate_X_array - generates the X values for all distributions
zeroise_below_gamma - sets all y values to zero when x < gamma. Used when the HF and CHF equations are specified
xy_transform - provides conversions between spatial (-inf,inf) and axes coordinates (0,1).
probability_plot_xylims - sets the x and y limits on probability plots
probability_plot_xyticks - sets the x and y ticks on probability plots
anderson_darling - calculates the Anderson-Darling goodness of fit statistic
colorprint - prints to the console in color, bold, italic, and underline
fitters_input_checking - error checking and default values for all the fitters
fill_no_autoscale - creates a shaded region without adding it to the global list of objects to consider when autoscale is calculated
line_no_autoscale - creates a line without adding it to the global list of objects to consider when autoscale is calculated
distribution_confidence_intervals - calculated and plots the confidence intervals for the distributions
linear_regression - given x and y data it will return slope and intercept of line of best fit. Includes options to specify slope or intercept.
least_squares - provides parameter estimates for distributions using the method of least squares. Used extensively by Fitters.
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import ticker
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

warnings.filterwarnings(
    action="ignore", category=OptimizeWarning
)  # ignores the optimize warning that curve_fit sometimes outputs when there are 3 data points to fit a 3P curve
warnings.filterwarnings(
    action="ignore", category=RuntimeWarning
)  # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required


def round_to_decimals(number, decimals=5, integer_floats_to_ints=True):
    """
    This function is used to round a number to a specified number of decimals. It is used heavily in the formatting of the parameter titles within reliability.Distributions
    It is not the same as rounding to a number of significant figures as it keeps preceeding zeros for numbers less than 1.

    Inputs:
    number - the number to be rounded
    decimals - the number of decimals (not including preceeding zeros) that are to be in the output
    integer_floats_to_ints - True/False. Default is True. Removes trailing zeros if there are no significant decimals (eg. 12.0 becomes 12).

    examples (with decimals = 5):
    1234567.1234567 ==> 1234567.12345
    0.0001234567 ==> 0.00012345
    1234567 ==> 1234567
    0.00 ==> 0
    """

    if np.isfinite(number):  # check the input is not NaN
        if number < 0:
            sign = -1
            num = number * -1
            skip_to_end = False
        elif number > 0:
            sign = 1
            num = number
            skip_to_end = False
        else:  # number == 0
            if integer_floats_to_ints is True:
                out = int(number)
            else:
                out = number
            sign = 0
            skip_to_end = True
        if skip_to_end is False:
            if num > 1:
                decimal = num % 1
                whole = num - decimal
                if decimal == 0:
                    if integer_floats_to_ints is True:
                        out = int(whole)
                    else:
                        out = whole
                else:
                    out = np.round(num, decimals)
            else:  # num<1
                out = np.round(num, decimals - int(np.floor(np.log10(abs(num)))) - 1)
        output = out * sign
    else:
        output = number
    return output


def transform_spaced(
    transform, y_lower=1e-8, y_upper=1 - 1e-8, num=1000, alpha=None, beta=None
):
    """
    Creates linearly spaced array based on a specified transform
    This is similar to np.linspace or np.logspace but is designed for weibull space, exponential space, normal space, gamma space, loglogistic space, and beta space.
    It is useful if the points generated are going to be plotted on axes that are scaled using the same transform and need to look equally spaced in the transform space
    Note that lognormal is the same as normal, since the x-axis is what is transformed, not the y-axis.

    :param transform: the transform name. Must be either weibull, exponential, normal, gamma, or beta.
    :param y_upper: the lower bound (must be within the bounds 0 to 1). Default is 1e-8
    :param y_lower: the upper bound (must be within the bounds 0 to 1). Default is 1-1e-8
    :param num: the number of values in the array
    :param alpha: the alpha value of the beta distribution. Only used if the transform is beta
    :param beta: the alpha value of the beta or gamma distribution. Only used if the transform is beta or gamma
    :return: linearly spaced array (appears linearly spaced when plotted in transform space)
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
            fwd = lambda x: ss.beta.ppf(x, a=alpha, b=beta)
            inv = lambda x: ss.beta.cdf(x, a=alpha, b=beta)
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
    This function gets the previous xlim and ylim and also checks whether there was a previous plot (based on whether the default 0,1 axes had been changed.
    It returns a list of items that are used by restore_axes_limits after the plot has been performed
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
    This function works in a pair with get_axes_limits
    Inputs:
    limits - a list of xlim, ylim, use_prev_lims created by get_axes_limits
    dist - the distribution object to which it is applied
    X - the xvalues of the plot
    Y - the yvalues of the plot
    xvals - the xvals specified. May be None if not specified

    No scaling will be done if the axes are not linear due to errors that result from log and function scaled axes when a limit of 0 is used.
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
        elif not np.isfinite(Y[-1]):  # asymptote on the right
            ylim_upper = max(Y)
        elif dist._pdf0 == np.inf or dist._pdf0 > 10:  # asymptote on the left
            idx = np.where(X >= dist.quantile(0.1))[0][0]
            ylim_upper = Y[idx]
        else:  # an increasing pdf. Not asymptote
            ylim_upper = max(Y) * top_spacing
    elif func in ["cdf", "CDF", "SF", "sf"]:
        ylim_upper = top_spacing
    elif func in ["hf", "HF"]:
        if not np.isfinite(dist._hf0) and not np.isfinite(
            Y[-1]
        ):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif not np.isfinite(Y[-1]):  # asymptote of the right
            ylim_upper = max(Y)
        elif dist._hf0 == np.inf or dist._hf0 > 10:  # asymptote on the left
            idx = np.where(X >= dist.quantile(0.1))[0][0]
            ylim_upper = Y[idx]
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
    generates the array of X values for each of the PDf, CDF, SF, HF, CHF functions within reliability.Distributions
    This is done with a variety of cases in order to ensure that for regions of high gradient (particularly asymptotes to inf) the points are more concentrated.
    This ensures that the line always looks as smooth as possible using only 200 data points
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


def zeroise_below_gamma(X, Y, gamma):
    """
    This will make all Y values 0 for the corresponding X values being below gamma.
    Used by HF and CHF which need to be zeroized if the gamma shifted form of the equation is used.
    """
    if gamma > 0:
        if len(np.where(X > gamma)[0]) == 0:
            Y[0::] = 0  # zeroize everything if there is no X values above gamma
        else:
            Y[0 : (np.where(X > gamma)[0][0])] = 0  # zeroize below X=gamma
    return Y


def xy_transform(value, direction="forward", axis="x"):
    """
    Converts between data values and axes coordinates (based on xlim() or ylim()).
    If direction is forward the returned value will always be between 0 and 1 provided value is on the plot.
    If direction is reverse the input should be between 0 and 1 and the returned value will be the data value based on the current plot lims
    axis must be x or y
    """
    if direction not in ["reverse", "inverse", "inv", "rev", "forward", "fwd"]:
        raise ValueError('direction must be "forward" or "reverse"')
    if axis not in ["X", "x", "Y", "y"]:
        raise ValueError("axis must be x or y. Default is x")

    ax = plt.gca()
    if direction in ["reverse", "inverse", "inv", "rev"]:
        if type(value) in [int, float, np.float64]:
            if axis == "x":
                transformed_values = ax.transData.inverted().transform(
                    (ax.transAxes.transform((value, 0.5))[0], 0.5)
                )[
                    0
                ]  # x transform
            else:
                transformed_values = ax.transData.inverted().transform(
                    (1, ax.transAxes.transform((1, value))[1])
                )[
                    1
                ]  # y transform
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
    finds the x and y limits of probability plots.
    This function is called by probability_plotting
    """

    # x limits
    if dist in ["weibull", "lognormal", "loglogistic"]:
        min_x_log = np.log10(min(x))
        max_x_log = np.log10(max(x))
        dx_log = max_x_log - min_x_log
        xlim_lower = 10 ** (min_x_log - dx_log * spacing)
        xlim_upper = 10 ** (max_x_log + dx_log * spacing)
        if xlim_lower == xlim_upper:
            xlim_lower = 10 ** (np.log10(xlim_lower) - 10 * spacing)
            xlim_upper = 10 ** (np.log10(xlim_upper) + 10 * spacing)
    elif dist in ["normal", "gamma", "exponential", "beta", "gumbel"]:
        min_x = min(x)
        max_x = max(x)
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
    plt.xlim(xlim_lower, xlim_upper)

    # y limits
    if dist == "weibull":
        min_y_tfm = axes_transforms.weibull_forward(min(y))
        max_y_tfm = axes_transforms.weibull_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.weibull_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.weibull_inverse(max_y_tfm + dy_tfm * spacing)
    if dist == "exponential":
        min_y_tfm = axes_transforms.exponential_forward(min(y))
        max_y_tfm = axes_transforms.exponential_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.exponential_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.exponential_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gamma":
        min_y_tfm = axes_transforms.gamma_forward(min(y), gamma_beta)
        max_y_tfm = axes_transforms.gamma_forward(max(y), gamma_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gamma_inverse(
            min_y_tfm - dy_tfm * spacing, gamma_beta
        )
        ylim_upper = axes_transforms.gamma_inverse(
            max_y_tfm + dy_tfm * spacing, gamma_beta
        )
    elif dist in ["normal", "lognormal"]:
        min_y_tfm = axes_transforms.normal_forward(min(y))
        max_y_tfm = axes_transforms.normal_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.normal_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.normal_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gumbel":
        min_y_tfm = axes_transforms.gumbel_forward(min(y))
        max_y_tfm = axes_transforms.gumbel_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gumbel_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.gumbel_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "beta":
        min_y_tfm = axes_transforms.beta_forward(min(y), beta_alpha, beta_beta)
        max_y_tfm = axes_transforms.beta_forward(max(y), beta_alpha, beta_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.beta_inverse(
            min_y_tfm - dy_tfm * spacing, beta_alpha, beta_beta
        )
        ylim_upper = axes_transforms.beta_inverse(
            max_y_tfm + dy_tfm * spacing, beta_alpha, beta_beta
        )
    elif dist == "loglogistic":
        min_y_tfm = axes_transforms.loglogistic_forward(min(y))
        max_y_tfm = axes_transforms.loglogistic_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.loglogistic_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.loglogistic_inverse(max_y_tfm + dy_tfm * spacing)
    if ylim_upper == ylim_lower:
        dx = min(1 - ylim_upper, ylim_upper - 1)
        ylim_upper = ylim_upper - spacing * dx
        ylim_lower = ylim_lower + spacing * dx
    plt.ylim(ylim_lower, ylim_upper)


def probability_plot_xyticks(yticks=None):
    """
    Sets the x and y ticks for probability plots
    X ticks are selected using either MaxNLocator or LogLocator.
    X ticks are formatted using a custom formatter.
    Y ticks are specified with FixedLocator due to their irregular spacing. Minor y ticks use MaxNLocator
    Y ticks are formatted using a custom Percent Formatter that handles decimals better
    This function is called by probability_plotting
    """

    def get_tick_locations(major_or_minor, in_lims=True, axis="x"):
        """
        returns the major or minor tick locations for the current axis
        if in_lims=True then it will only return the ticks that are within the current xlim() or ylim(). Default is True
        axis must be x or y. Default is x
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
            multiplier = value / (10 ** exponent)
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
        Slightly different than PercentFormatter as it does not force a particular number of decimals. ie. 99.00 becomes 99 while 99.99 still displays as such.
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
        "finds the sum of the distance (in axes coords (0 to 1)) of the distances from the edge ticks to the edges"
        xtick_locations = get_tick_locations("major", axis="x")
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
    Calculates the Anderson-Darling goodness of fit statistic
    These formulas are based on the method used in MINITAB which gives an adjusted form of the original AD statistic described on Wikipedia
    """
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
        lnZi * FnZi_1 ** 2
        - np.log(1 - Zi) * FnZi_1 ** 2
        - lnZi_1 * FnZi_1 ** 2
        + np.log(1 - Zi_1) * FnZi_1 ** 2
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
    colorprint - Provides easy access to color printing in the console
    Parameter names are self explanatory. Color options are grey, red, green, yellow, blue, pink, turquoise.
    Some flexibility in color names is allowed. eg. red and r will both give red.
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
    performs error checking and some basic default operations for all the inputs given to each of the fitters
    """

    def __init__(
        self,
        dist,
        failures,
        right_censored=None,
        method=None,
        optimizer=None,
        CI=0.95,
        percentiles=False,
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
                "right_censored must be a list or array of right_censored failure data"
            )
        failures = np.asarray(failures).astype(np.float)
        right_censored = np.asarray(right_censored).astype(np.float)

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
                        "WARNING: failures contained zeros. These have been removed to enable fitting of all distributions.",
                        text_color="red",
                    )
                else:
                    colorprint(
                        str(
                            "WARNING: failures contained zeros. These have been removed to enable fitting of the "
                            + dist
                            + " distribution."
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
                failures = f1[f1 != 0]
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
        if optimizer is None:
            frac_cens = len(right_censored) / (len(failures) + len(right_censored))
            if frac_cens > 0.97:
                optimizer = "TNC"  # default optimizer above 97% right censored data
            else:
                optimizer = (
                    "L-BFGS-B"  # default optimizer below 97% right censored data
                )
        elif optimizer.upper() not in ["L-BFGS-B", "TNC", "POWELL"]:
            raise ValueError(
                'optimizer must be either "L-BFGS-B", "TNC", or "powell". Default is "L-BFGS-B" below 97% censored data and "TNC" above 97% censored data.'
            )

        # error checking for method
        if method is not None:
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

        # percentiles error checking
        if type(percentiles) in [str, bool]:
            if percentiles in ["auto", True, "default", "on"]:
                percentiles = np.array(
                    [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
                )  # percentiles to be used as the defaults in the table of percentiles
        elif percentiles is not None:
            if type(percentiles) not in [list, np.ndarray]:
                raise ValueError("percentiles must be a list or array")
            percentiles = np.asarray(percentiles)
            if max(percentiles) >= 100 or min(percentiles) <= 0:
                raise ValueError("percentiles must be between 0 and 100")

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
        if CI_type is not None:
            if CI_type in ["t", "time", "T", "TIME"]:
                CI_type = "time"
            elif CI_type in ["r", "R", "rel", "REL", "reliability", "RELIABILITY"]:
                CI_type = "reliability"
            else:
                raise ValueError('CI_type must be "time" or "reliability"')

        # return everything
        self.failures = failures
        self.right_censored = right_censored
        self.CI = CI
        self.method = method
        self.optimizer = optimizer
        self.percentiles = percentiles
        self.force_beta = force_beta
        self.force_sigma = force_sigma
        self.CI_type = CI_type


def fill_no_autoscale(xlower, xupper, ylower, yupper, plot_type='CDF', **kwargs):
    """
    creates a filled region (polygon) without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when autoscale sets the range
    """
    # this trims the x and y arrays based on the y arrays containing nan and inf. Required by CHF due to -np.log(SF)=inf when SF=0
    if any(np.logical_not(np.isfinite(ylower))):
        idx_ylower = np.where(np.isfinite(ylower) == False)[0][0]
        ylower = ylower[0:idx_ylower]
        xlower = xlower[0:idx_ylower]
    if any(np.logical_not(np.isfinite(yupper))):
        idx_yupper = np.where(np.isfinite(yupper) == False)[0][0]
        yupper = yupper[0:idx_yupper]
        xupper = xupper[0:idx_yupper]
    if any(np.logical_not(np.isfinite(xlower))):
        idx_xlower = np.where(np.isfinite(xlower) == False)[0][0]
        ylower = ylower[0:idx_xlower]
        xlower = xlower[0:idx_xlower]
    if any(np.logical_not(np.isfinite(xupper))):
        idx_xupper = np.where(np.isfinite(xupper) == False)[0][0]
        yupper = yupper[0:idx_xupper]
        xupper = xupper[0:idx_xupper]

    if (
        len(xlower) != len(xupper)
        or len(xlower) != len(yupper)
        or len(xlower) != len(ylower)
        or len(xlower) < 2
    ):
        colorprint(
            "ERROR in fill_no_autoscale: Confidence intervals could not be plotted due to the presence of too many NaNs in the arrays.",
            text_color="red",
        )
    else:
        # this trims the y arrays free of 1's since the probability plot can't handle 1 as it's equivalent to inf. This should not be applied for the CHF.
        if plot_type.upper() in ['CDF','SF']:
            if max(ylower) >= 1:
                idx_ylower_1 = np.where(ylower >= 1)[0][0]
                ylower = ylower[0:idx_ylower_1]
                xlower = xlower[0:idx_ylower_1]
            if max(yupper) >= 1:
                idx_yupper_1 = np.where(yupper >= 1)[0][0]
                yupper = yupper[0:idx_yupper_1]
                xupper = xupper[0:idx_yupper_1]

        if (
            len(xlower) != len(xupper)
            or len(xlower) != len(yupper)
            or len(xlower) != len(ylower)
            or len(xlower) < 2
        ):
            colorprint(
                "ERROR in fill_no_autoscale: Confidence intervals could not be plotted due to the presence of too many NaNs in the arrays.",
                text_color="red",
            )
        else:
            # generate the polygon
            polygon = np.column_stack(
                [np.hstack([xlower, xupper[::-1]]), np.hstack([ylower, yupper[::-1]])]
            )  # this is equivalent to fill as it makes a polygon
            col = PolyCollection([polygon], **kwargs)
            plt.gca().add_collection(col, autolim=False)


def line_no_autoscale(x, y, **kwargs):
    """
    creates a line without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when autoscale sets the range
    """
    # this trims the x and y arrays based on the y arrays containing nan and inf. Required by CHF due to -np.log(SF)=inf when SF=0
    if any(np.logical_not(np.isfinite(y))):
        idx_y = np.where(np.isfinite(y) == False)[0][0]
        y = y[0:idx_y]
        x = x[0:idx_y]
    line = np.column_stack([x, y])  # this is equivalent to plot as it makes a line
    col = LineCollection([line], **kwargs)
    plt.gca().add_collection(col, autolim=False)


class distribution_confidence_intervals:
    """
    Contains functions that provide all the confidence intervals for CDF, SF, CHF for each distribution for which it is implemented
    """

    @staticmethod
    def CI_kwarg_handler(self, kwargs):
        """
        Processes specific arguments from kwargs and self to ensure the CI_type and plot_CI are extracted appropriately and passed to the confidence interval methods, without being passed to the plot method.
        This function is used within each CDF, SF, CHF before the plt.plot method is used.
        """
        kwargs_list = kwargs.keys()
        if "plot_CI" in kwargs_list:
            plot_CI = kwargs.pop("plot_CI")
        elif "show_CI" in kwargs_list:
            plot_CI = kwargs.pop("show_CI")
        else:
            plot_CI = True  # default
        if plot_CI not in [True, False]:
            colorprint(
                "WARNING: unexpected value in plot_CI. To show/hide the CI you can specify either show_CI=True/False or plot_CI=True/False",
                text_color="red",
            )
            plot_CI = True

        if "CI" in kwargs_list:
            CI = kwargs.pop(
                "CI"
            )  # this allows CI in the CDF,SF,CHF to override CI from above (in the fitter)
        elif self.Z is not None:
            CI = 1 - ss.norm.cdf(-self.Z) * 2  # converts Z to CI
        else:
            CI = 0.95

        if self.name == "Exponential":
            if "CI_type" in kwargs_list:
                colorprint(
                    "WARNING: CI_type is not required for the Exponential distribution since the confidence intervals of time and reliability are identical",
                    text_color="red",
                )
                CI_type = kwargs.pop("CI_type")  # remove it
            else:
                CI_type = (
                    None  # this will not be used but it is required for the output
                )
        else:
            if "CI_type" in kwargs_list:
                CI_type = kwargs.pop(
                    "CI_type"
                )  # this allows CI_type in the CDF,SF,CHF to override CI_type from above (either the default of time if unspecified or whatever came from the probability plot)
            elif self.CI_type is not None:
                CI_type = self.CI_type
            else:
                CI_type = "time"
        return CI_type, plot_CI, CI

    @staticmethod
    def exponential_CI(
        self, func, plot_CI=None, CI=None, text_title="", color=None, q=None
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Exponential CDF, SF, and CHF functions.
        """
        points = 200

        if func not in ["CDF", "SF", "CHF"]:
            raise ValueError("func must be either CDF, SF, or CHF")
        if type(q) not in [list, np.ndarray, type(None)]:
            raise ValueError("q must be a list or array of quantiles. Default is None")
        if q is not None:
            q = np.asarray(q)

        # this section plots the confidence interval
        if (
            self.Lambda_SE is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
        ):

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds"
            )  # Adds the CI and CI_type to the title
            plt.title(
                text_title
            )  # add a line to the plot title to include the confidence bounds information
            plt.subplots_adjust(top=0.81)

            Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
            Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))

            t = np.geomspace(
                self.quantile(0.00001) - self.gamma,
                self.quantile(0.99999) - self.gamma,
                points,
            )

            if func == "CDF":
                yy_upper = 1 - np.exp(-Lambda_upper * t)
                yy_lower = 1 - np.exp(-Lambda_lower * t)
            elif func == "SF":
                yy_upper = np.exp(-Lambda_upper * t)
                yy_lower = np.exp(-Lambda_lower * t)
            elif func == "CHF":
                yy_upper = -np.log(np.exp(-Lambda_upper * t))  # same as -np.log(SF)
                yy_lower = -np.log(np.exp(-Lambda_lower * t))

            if (
                q is not None
            ):  # calculates the values for the table of percentiles in the fitter
                t_lower = -np.log(q) / Lambda_upper + self.gamma
                t_upper = -np.log(q) / Lambda_lower + self.gamma

            if plot_CI is True:
                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_lower,color='blue',marker='.')
                # plt.scatter(t + self.gamma, yy_upper, color='red', marker='.')
            elif plot_CI is None and q is not None:
                return t_lower, t_upper

    @staticmethod
    def weibull_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Weibull CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.alpha_SE is not None
            and self.beta_SE is not None
            and self.Cov_alpha_beta is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
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
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
                    + 2
                    * du_da(v, self.alpha, self.beta)
                    * du_db(v, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
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
                            "weibull", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is ln(t)
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma  # transform back from ln(t)
                t_upper = np.exp(v_upper) + self.gamma

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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.geomspace(
                    self.quantile(0.00001) - self.gamma,
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

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_upper, color='red')
                # plt.scatter(t + self.gamma, yy_lower, color='blue')

    @staticmethod
    def gamma_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Gamma CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.alpha_SE is not None
            and self.beta_SE is not None
            and self.Cov_alpha_beta is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            def u(t, alpha, beta):  # u = gammainccinv(beta,R)
                return t / alpha

            def v(R, alpha, beta):  # v = t
                return agammainccinv(beta, R) * alpha

            du_da = jac(u, 1)  # derivative wrt alpha (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt alpha (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
                    + 2
                    * du_da(v, self.alpha, self.beta)
                    * du_db(v, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
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
                            "weibull", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is t
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = v_lower + self.gamma
                t_upper = v_upper + self.gamma

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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            if (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.geomspace(
                    self.quantile(0.00001) - self.gamma,
                    self.quantile(0.99999) - self.gamma,
                    points,
                )

                # u is reliability (gammainccinv(beta,R))
                u_lower = (
                    u(t, self.alpha, self.beta) + Z * var_u(self, t) ** 0.5
                )  # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Weibull_2P across
                u_upper = u(t, self.alpha, self.beta) - Z * var_u(self, t) ** 0.5

                Y_lower = agammaincc(
                    self.beta, u_lower
                )  # transform back from u = gammainccinv(beta,R)
                Y_upper = agammaincc(self.beta, u_upper)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_upper, color='red')
                # plt.scatter(t + self.gamma, yy_lower, color='blue')

    @staticmethod
    def normal_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Normal CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.mu_SE is not None
            and self.sigma_SE is not None
            and self.Cov_mu_sigma is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
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
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
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
                            "normal", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is t
                t_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                t_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.linspace(self.quantile(0.00001), self.quantile(0.99999), points)

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t,
                    xupper=t,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t, yy_upper, color='red')
                # plt.scatter(t, yy_lower, color='blue')

    @staticmethod
    def lognormal_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Lognormal CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.mu_SE is not None
            and self.sigma_SE is not None
            and self.Cov_mu_sigma is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
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
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
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
                            "normal", y_lower=1e-8, y_upper=1 - 1e-8, num=points
                        )

                # v is ln(t)
                v_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma
                t_upper = np.exp(v_upper) + self.gamma

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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.geomspace(
                    self.quantile(0.00001) - self.gamma,
                    self.quantile(0.99999) - self.gamma,
                    points,
                )

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t+ self.gamma, yy_upper, color='red')
                # plt.scatter(t+ self.gamma, yy_lower, color='blue')

    @staticmethod
    def loglogistic_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Loglogistic CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.alpha_SE is not None
            and self.beta_SE is not None
            and self.Cov_alpha_beta is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
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
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
                    + 2
                    * du_da(v, self.alpha, self.beta)
                    * du_db(v, self.alpha, self.beta)
                    * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE ** 2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE ** 2
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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.geomspace(
                    self.quantile(0.00001) - self.gamma,
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

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t + self.gamma, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_upper, color='red')
                # plt.scatter(t + self.gamma, yy_lower, color='blue')

    @staticmethod
    def gumbel_CI(
        self,
        func,
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
    ):
        """
        Generates the confidence intervals for CDF, SF, and CHF
        This is a utility function intended only for use by the Gumbel CDF, SF, and CHF functions.
        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            self.mu_SE is not None
            and self.sigma_SE is not None
            and self.Cov_mu_sigma is not None
            and self.Z is not None
            and (plot_CI is True or q is not None)
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
            if q is not None:
                q = np.asarray(q)

            CI_100 = round(
                CI * 100, 4
            )  # formats the confidence interval value ==> 0.95 becomes 95
            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z
            if CI_100 % 1 == 0:
                CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
            text_title = str(
                text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type
            )  # Adds the CI and CI_type to the title
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
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
                    + 2
                    * du_da(v, self.mu, self.sigma)
                    * du_db(v, self.mu, self.sigma)
                    * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE ** 2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE ** 2
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
                        plot_type=func,
                    )
                    line_no_autoscale(
                        x=t_lower, y=yy, color=color, linewidth=0
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper, y=yy, color=color, linewidth=0
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')
                elif plot_CI is None and q is not None:
                    return t_lower, t_upper

            elif (
                CI_type == "reliability"
            ):  # Confidence bounds on Reliability (in terms of time)
                t = np.linspace(self.quantile(0.00001), self.quantile(0.99999), points)

                # u is reliability u = ln(-ln(R))
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = np.exp(-np.exp(u_lower))  # transform back from ln(-ln(R))
                Y_upper = np.exp(-np.exp(u_upper))

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                fill_no_autoscale(
                    xlower=t,
                    xupper=t,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    plot_type=func,
                )
                line_no_autoscale(
                    x=t, y=yy_lower, color=color, linewidth=0
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t, y=yy_upper, color=color, linewidth=0
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t, yy_upper, color='red')
                # plt.scatter(t, yy_lower, color='blue')


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
    linear algebra solution to find line of best fix passing through points (x,y)
    specify slope or intercept to force these parameters.
    Rank regression can be on X (RRX) or Y (RRY). Default is RRX.
    note that slope depends on RRX_or_RRY. If you use RRY then slope is dy/dx but if you use RRX then slope is dx/dy.
    :returns slope,intercept in terms of Y = slope * X + intercept

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
        solution = (
            np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)
        )  # linear regression formula for RRY
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
        solution = (
            np.linalg.inv(yy.T.dot(yy)).dot(yy.T).dot(xx)
        )  # linear regression formula for RRX
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
                + str(round_to_decimals(m, 2))
                + ".x + "
                + str(round_to_decimals(c, 2))
            )
        plt.plot(xvals, yvals, label=label, **kwargs)
        plt.xlim(min(x) - delta_x * 0.2, max(x) + delta_x * 0.2)
        plt.ylim(min(y) - delta_y * 0.2, max(y) + delta_y * 0.2)
    return m, c


def least_squares(dist, failures, right_censored, method="RRX", force_shape=None):
    """
    Uses least squares or non-linear least squares estimation to fit the parameters of the distribution to the plotting positions.
    Plotting positions are based on failures and right_censored so while least squares estimation does not consider the right_censored data in the same way as MLE, the plotting positions do.
    The output of this method may be used as the initial guess for the MLE method.
    method may be RRX or RRY. Default is RRX.

    return the model's parameters in a list.
        E.g. for "Weibull_2P" it will return [alpha,beta]
             for "Weibull_3P" it will return [alpha,beta,gamma]
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
        raise ValueError('method muct be either "RRX" or "RRY". Default is RRX.')

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
                jac="cs",
                method="dogbox",
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
                jac="cs",
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
            return 1 - (r ** 2)

        # NLLS for Normal_2P which is used by Lognormal_3P by taking the log of the data. This is more accurate than doing it with Lognormal_3P.
        def __normal_2P_CDF(t, mu, sigma):
            return (1 + erf(((t - mu) / sigma) / 2 ** 0.5)) / 2

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
                jac="cs",
                method="dogbox",
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
        beta_guess = abs(0.6932 * LS_beta ** 2 - 0.0908 * LS_beta + 0.2804)
        alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess ** 2 + 1.1119 * beta_guess))

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
                method="dogbox",
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
        beta_guess = abs(0.6932 * LS_beta ** 2 - 0.0908 * LS_beta + 0.2804)
        alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess ** 2 + 1.1119 * beta_guess))

        def __perform_curve_fit_gamma_2P():  # separated out for repeated use
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
                method="dogbox",
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


class LS_optimisation:
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


class MLE_optimisation:
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
        else:
            raise ValueError(
                'func_name is not recognised. Use the correct name e.g. "Weibull_2P"'
            )

        n = len(failures) + len(right_censored)
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        guess = initial_guess  # set the current guess as the initial guess and then update the current guess each iteration
        if force_shape is None:
            k = len(bounds)
            while (
                delta_BIC > 0.001 and runs < 5
            ):  # exits after BIC convergence or 5 iterations
                runs += 1
                result = minimize(
                    value_and_grad(LL_func),
                    guess,
                    args=(failures, right_censored),
                    jac=True,
                    method=optimizer,
                    bounds=bounds,
                )
                params = result.x
                if func_name == "Exponential_1P":
                    guess = [params[0]]
                elif func_name in [
                    "Weibull_2P",
                    "Gamma_2P",
                    "Beta_2P",
                    "Loglogistic_2P",
                    "Lognormal_2P",
                    "Exponential_2P",
                ]:
                    guess = [params[0], params[1]]
                elif func_name in [
                    "Weibull_3P",
                    "Gamma_3P",
                    "Loglogistic_3P",
                    "Lognormal_3P",
                ]:
                    guess = [params[0], params[1], params[2]]
                elif func_name == "Weibull_mixture":
                    guess = [params[0], params[1], params[2], params[3], params[4]]
                elif func_name == "Weibull_CR":
                    guess = [params[0], params[1], params[2], params[3]]
                LL2 = 2 * LL_func(guess, failures, right_censored)
                BIC_array.append(np.log(n) * k + LL2)
                delta_BIC = abs(BIC_array[-1] - BIC_array[-2])
        else:  # this will only be run for Weibull_2P, Normal_2P, and Lognormal_2P so the guess is structured with this in mind
            bounds = [bounds[0]]  # bounds on the solution. Helps a lot with stability
            guess = [guess[0]]
            k = 1
            while (
                delta_BIC > 0.001 and runs < 5
            ):  # exits after BIC convergence or 5 iterations
                runs += 1
                result = minimize(
                    value_and_grad(LL_func_force),
                    guess,
                    args=(failures, right_censored, force_shape),
                    jac=True,
                    method=optimizer,
                    bounds=bounds,
                )
                params = result.x
                guess = [params[0]]
                LL2 = 2 * LL_func_force(guess, failures, right_censored, force_shape)
                BIC_array.append(np.log(n) * k + LL2)
                delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        if result.success is True:
            params = result.x
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
        else:  # if the bounded optimizer (L-BFGS-B, TNC, powell) fails then we have a second attempt using the slower but slightly more reliable nelder-mead optimizer.
            if force_shape is None:
                guess = initial_guess
                result = minimize(
                    value_and_grad(LL_func),
                    guess,
                    args=(failures, right_censored),
                    jac=True,
                    tol=1e-4,
                    method="nelder-mead",
                )
            else:
                guess = [initial_guess[0]]
                result = minimize(
                    value_and_grad(LL_func_force),
                    guess,
                    args=(failures, right_censored, force_shape),
                    jac=True,
                    tol=1e-4,
                    method="nelder-mead",
                )
            if result.success is True:
                params = result.x
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
            else:
                if func_name == "Weibull_mixture":
                    colorprint(
                        "WARNING: MLE estimates failed for Weibull_mixture. The initial estimates have been returned. These results may not be as accurate as MLE.",
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
                        "WARNING: MLE estimates failed for Weibull_CR. The initial estimates have been returned. These results may not be as accurate as MLE.",
                        text_color="red",
                    )
                    self.alpha_1 = initial_guess[0]
                    self.beta_1 = initial_guess[1]
                    self.alpha_2 = initial_guess[2]
                    self.beta_2 = initial_guess[3]
                else:
                    colorprint(
                        str(
                            "WARNING: MLE estimates failed for "
                            + func_name
                            + ". The least squares estimates have been returned. These results may not be as accurate as MLE."
                        ),
                        text_color="red",
                    )
                    if force_shape is None:
                        self.scale = initial_guess[0]  # alpha, mu, Lambda
                        if func_name not in ["Exponential_1P", "Exponential_2P"]:
                            self.shape = initial_guess[1]  # beta, sigma
                        else:
                            if func_name == "Exponential_2P":
                                self.gamma = initial_guess[
                                    1
                                ]  # gamma for Exponential_2P
                        if func_name in [
                            "Weibull_3P",
                            "Gamma_3P",
                            "Loglogistic_3P",
                            "Lognormal_3P",
                        ]:
                            self.gamma = initial_guess[
                                2
                            ]  # gamma for Weibull_3P, Gamma_3P, Loglogistic_3P, Lognormal_3P
                    else:  # this will only be reached for Weibull_2P, Normal_2P and Lognormal_2P so the scale and shape extraction is fine for these
                        self.scale = initial_guess[0]
                        self.shape = force_shape


def write_df_to_xlsx(df, path, **kwargs):
    """
    Writes a dataframe to an xlsx file
    For use exclusively by the Convert_data module
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
    removes NaNs from a list or array. This is better than simply using "x = x[numpy.logical_not(numpy.isnan(x))]" as numpy crashes for str and bool.
    returns a list or array of the same type as the input
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
