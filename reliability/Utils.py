'''
Utils (utilities)

This is a collection of utilities that are used throughout the python reliability library.
Functions have been placed here as to declutter the dropdown lists of your IDE and to provide a common resource across multiple modules.
It is not expected that users will be using any utils directly.

Included functions are:
round_to_decimals - applies different rounding rules to numbers above and below 1 so that small numbers do not get rounded to 0.
transform_spaced - Creates linearly spaced array (in transform space) based on a specified transform. This is like np.logspace but it can make an array that is weibull spaced, normal spaced, etc.
axes_transforms - Custom scale functions used in Probability_plotting
fill_no_autoscale - creates a shaded region without adding it to the global list of objects to consider when autoscale is calculated
line_no_autoscale - creates a line without adding it to the global list of objects to consider when autoscale is calculated
get_axes_limits - gets the current axes limits
restore_axes_limits - restores the axes limits based on values from get_axes_limits()
generate_X_array - generates the X values for all distributions
zeroise_below_gamma - sets all y values to zero when x < gamma. Used when the HF and CHF equations are specified
probability_plot_xylims - sets the x and y limits on probability plots
probability_plot_xyticks - sets the x and y ticks on probability plots
anderson_darling - calculates the Anderson-Darling goodness of fit statistic
'''

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import ticker


def round_to_decimals(number, decimals=5, integer_floats_to_ints=True):
    '''
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
    '''

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
    return out * sign


def transform_spaced(transform, y_lower=1e-8, y_upper=1 - 1e-8, num=1000, alpha=None, beta=None):
    '''
    Creates linearly spaced array based on a specified transform
    This is similar to np.linspace or np.logspace but is designed for weibull space, exponential space, normal space, gamma space, and beta space.
    It is useful if the points generated are going to be plotted on axes that are scaled using the same transform and need to look equally spaced in the transform space
    Note that lognormal is the same as normal, since the x-axis is what is transformed, not the y-axis.

    :param transform: the transform name. Must be either weibull, exponential, normal, gamma, or beta.
    :param y_upper: the lower bound (must be within the bounds 0 to 1). Default is 1e-8
    :param y_lower: the upper bound (must be within the bounds 0 to 1). Default is 1-1e-8
    :param num: the number of values in the array
    :param alpha: the alpha value of the beta distribution. Only used if the transform is beta
    :param beta: the alpha value of the beta or gamma distribution. Only used if the transform is beta or gamma
    :return: linearly spaced array (appears linearly spaced when plotted in transform space)
    '''
    np.seterr('ignore')  # this is required due to an error in scipy.stats
    if y_lower > y_upper:
        y_lower, y_upper = y_upper, y_lower
    if y_lower <= 0 or y_upper >= 1:
        raise ValueError('y_lower and y_upper must be within the range 0 to 1')
    if num <= 2:
        raise ValueError('num must be greater than 2')
    if transform in ['normal', 'Normal', 'norm', 'Norm']:
        fwd = lambda x: ss.norm.ppf(x)
        inv = lambda x: ss.norm.cdf(x)
    elif transform in ['weibull', 'Weibull', 'weib', 'Weib', 'wbl']:
        fwd = lambda x: np.log(-np.log(1 - x))
        inv = lambda x: 1 - np.exp(-np.exp(x))
    elif transform in ['exponential', 'Exponential', 'expon', 'Expon', 'exp', 'Exp']:
        fwd = lambda x: ss.expon.ppf(x)
        inv = lambda x: ss.expon.cdf(x)
    elif transform in ['gamma', 'Gamma', 'gam', 'Gam']:
        if beta is None:
            raise ValueError('beta must be specified to use the gamma transform')
        else:
            fwd = lambda x: ss.gamma.ppf(x, a=beta)
            inv = lambda x: ss.gamma.cdf(x, a=beta)
    elif transform in ['beta', 'Beta']:
        if alpha is None or beta is None:
            raise ValueError('alpha and beta must be specified to use the beta transform')
        else:
            fwd = lambda x: ss.beta.ppf(x, a=alpha, b=beta)
            inv = lambda x: ss.beta.cdf(x, a=alpha, b=beta)
    elif transform in ['lognormal', 'Lognormal', 'LN', 'ln', 'lognorm', 'Lognorm']:  # the transform is the same, it's just the xscale that is ln for lognormal
        return ValueError('the Lognormal transform is the same as the normal transform. Specify normal and try again')
    else:
        raise ValueError('transform must be either exponential, normal, weibull, gamma, or beta')

    # find the value of the bounds in tranform space
    upper = fwd(y_upper)
    lower = fwd(y_lower)
    # generate the array in transform space
    arr = np.linspace(lower, upper, num)
    # convert the array back from transform space
    transform_array = inv(arr)
    return transform_array


class axes_transforms:
    '''
    Custom scale functions used in Probability_plotting
    '''

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
    def expon_forward(F):
        return ss.expon.ppf(F)

    @staticmethod
    def expon_inverse(R):
        return ss.expon.cdf(R)

    @staticmethod
    def normal_forward(F):
        return ss.norm.ppf(F)

    @staticmethod
    def normal_inverse(R):
        return ss.norm.cdf(R)

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


def fill_no_autoscale(xlower, xupper, ylower, yupper, **kwargs):
    '''
    creates a filled region (polygon) without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when autoscale sets the range
    '''
    # this trims the x and y arrays based on the y arrays containing nan and inf. Required by CHF due to -np.log(SF)=inf when SF=0
    if any(np.logical_not(np.isfinite(ylower))):
        idx_ylower = np.where(np.isfinite(ylower) == False)[0][0]
        ylower = ylower[0:idx_ylower]
        xlower = xlower[0:idx_ylower]
    if any(np.logical_not(np.isfinite(yupper))):
        idx_yupper = np.where(np.isfinite(yupper) == False)[0][0]
        yupper = yupper[0:idx_yupper]
        xupper = xupper[0:idx_yupper]
    polygon = np.column_stack([np.hstack([xlower, xupper[::-1]]), np.hstack([ylower, yupper[::-1]])])  # this is equivalent to fill as it makes a polygon
    col = PolyCollection([polygon], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def line_no_autoscale(x, y, **kwargs):
    '''
    creates a line without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when autoscale sets the range
    '''
    # this trims the x and y arrays based on the y arrays containing nan and inf. Required by CHF due to -np.log(SF)=inf when SF=0
    if any(np.logical_not(np.isfinite(y))):
        idx_y = np.where(np.isfinite(y) == False)[0][0]
        y = y[0:idx_y]
        x = x[0:idx_y]
    line = np.column_stack([x, y])  # this is equivalent to plot as it makes a line
    col = LineCollection([line], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def get_axes_limits():
    '''
    This function works in a pair with restore_axes_limits
    This function gets the previous xlim and ylim and also checks whether there was a previous plot (based on whether the default 0,1 axes had been changed.
    It returns a list of items that are used by restore_axes_limits after the plot has been performed
    '''
    xlims = plt.xlim(auto=None)  # get previous xlim
    ylims = plt.ylim(auto=None)  # get previous ylim
    if xlims == (0, 1) and ylims == (0, 1):  # this checks if there was a previous plot. If the lims were 0,1 and 0,1 then there probably wasn't
        use_prev_lims = False
    else:
        use_prev_lims = True
    out = [xlims, ylims, use_prev_lims]
    return out


def restore_axes_limits(limits, dist, func, X, Y, xvals=None, xmin=None, xmax=None):
    '''
    This function works in a pair with get_axes_limits
    Inputs:
    limits - a list of xlim, ylim, use_prev_lims created by get_axes_limits
    dist - the distribution object to which it is applied
    X - the xvalues of the plot
    Y - the yvalues of the plot
    xvals - the xvals specified. May be None if not specified

    No scaling will be done if the axes are not linear due to errors that result from log and function scaled axes when a limit of 0 is used.
    '''
    xlims = limits[0]
    ylims = limits[1]
    use_prev_lims = limits[2]

    ################## XLIMS ########################
    # obtain the xlims as if we did not consider prev limits
    if xvals is None:
        if xmin is None:
            if dist.name in ['Weibull', 'Gamma', 'Loglogistic', 'Exponential', 'Lognormal']:
                if dist.gamma == 0:
                    xlim_lower = 0
                else:
                    diff = dist.quantile(0.999) - dist.quantile(0.001)
                    xlim_lower = max(0, dist.quantile(0.001) - diff * 0.1)
            elif dist.name == 'Normal':
                xlim_lower = dist.quantile(0.001)
            elif dist.name in ['Beta', 'Mixture', 'Competing risks']:
                xlim_lower = 0
            else:
                raise ValueError('Unrecognised distribution name')
        else:
            xlim_lower = xmin

        if xmax is None:
            if dist.name != 'Beta':
                xlim_upper = dist.quantile(0.999)
            else:
                xlim_upper = 1
        else:
            xlim_upper = xmax

        if xlim_lower > xlim_upper:
            xlim_lower, xlim_upper = xlim_upper, xlim_lower  # switch them if xmin and xmax were given in the wrong order
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

    if plt.gca().get_xscale() == 'linear' and len(X) > 1:
        plt.xlim(xlim_LOWER, xlim_UPPER, auto=None)

    ################## YLIMS ########################

    top_spacing = 1.1  # the amount of space between the max value and the upper axis limit. 1.1 means the axis lies 10% above the max value
    if func in ['pdf', 'PDF']:
        if not np.isfinite(dist._pdf0) and not np.isfinite(Y[-1]):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif not np.isfinite(Y[-1]):  # asymptote on the right
            ylim_upper = max(Y)
        elif dist._pdf0 == np.inf or dist._pdf0 > 10:  # asymptote on the left
            idx = np.where(X >= dist.quantile(0.1))[0][0]
            ylim_upper = Y[idx]
        else:  # an increasing pdf. Not asymptote
            ylim_upper = max(Y) * top_spacing
    elif func in ['cdf', 'CDF', 'SF', 'sf']:
        ylim_upper = top_spacing
    elif func in ['hf', 'HF']:
        if not np.isfinite(dist._hf0) and not np.isfinite(Y[-1]):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif not np.isfinite(Y[-1]):  # asymptote of the right
            ylim_upper = max(Y)
        elif dist._hf0 == np.inf or dist._hf0 > 10:  # asymptote on the left
            idx = np.where(X >= dist.quantile(0.1))[0][0]
            ylim_upper = Y[idx]
        elif max(Y) > Y[-1]:  # a peaked hf
            ylim_upper = max(Y) * top_spacing
        else:  # an increasing hf. Not an asymptote
            idx = np.where(X >= plt.xlim()[1])[0][0]
            ylim_upper = Y[idx] * top_spacing
    elif func in ['chf', 'CHF']:
        idx = np.where(X >= xlim_upper)[0][0]  # index of the chf where it is equal to b95
        if np.isfinite(Y[idx]):
            ylim_upper = Y[idx] * top_spacing
        else:
            ylim_upper = Y[idx - 1] * top_spacing
    else:
        raise ValueError('func is invalid')
    ylim_lower = 0

    # determine what to set the ylims based on whether to use_prev_lims
    if use_prev_lims == False:
        ylim_LOWER = ylim_lower
        ylim_UPPER = ylim_upper
    else:  # need to consider previous axes limits
        ylim_LOWER = min(ylim_lower, ylims[0])
        ylim_UPPER = max(ylim_upper, ylims[1])

    if plt.gca().get_yscale() == 'linear' and len(Y) > 1:
        if ylim_LOWER != ylim_UPPER and np.isfinite(ylim_UPPER):
            plt.ylim(ylim_LOWER, ylim_UPPER, auto=None)
        else:
            plt.ylim(bottom=ylim_LOWER, auto=None)


def generate_X_array(dist, xvals=None, xmin=None, xmax=None):
    '''
    generates the array of X values for each of the PDf, CDF, SF, HF, CHF functions within reliability.Distributions
    This is done with a variety of cases in order to ensure that for regions of high gradient (particularly asymptotes to inf) the points are more concentrated.
    This ensures that the line always looks as smooth as possible using only 200 data points
    '''

    # obtain the xvals array
    points = 200  # the number of points to use when generating the X array
    points_right = 25  # the number of points given to the area above QU. The total points is still equal to 'points' so the area below QU receives 'points - points_right'
    QL = dist.quantile(0.0001)  # quantile lower
    QU = dist.quantile(0.99)  # quantile upper
    if xvals is not None:
        X = xvals
        if type(X) in [float, int, np.float64]:
            if X < 0 and dist.name != 'Normal':
                raise ValueError('the value given for xvals is less than 0')
            if X > 1 and dist.name == 'Beta':
                raise ValueError('the value given for xvals is greater than 1. The beta distribution is bounded between 0 and 1.')
            X = np.array([X])
        elif type(X) is list:
            X = np.array(X)
        elif type(X) is np.ndarray:
            pass
        else:
            raise ValueError('unexpected type in xvals. Must be int, float, list, or array')
        if type(X) is np.ndarray and min(X) < 0 and dist.name != 'Normal':
            raise ValueError('xvals was found to contain values below 0')
        if type(X) is np.ndarray and max(X) > 1 and dist.name == 'Beta':
            raise ValueError('xvals was found to contain values above 1. The beta distribution is bounded between 0 and 1.')
    else:
        if dist.name in ['Weibull', 'Lognormal', 'Loglogistic', 'Exponential', 'Gamma']:
            if xmin is None:
                xmin = 0
            if xmin < 0:
                raise ValueError('xmin must be greater than or equal to 0 for all distributions except the Normal distribution')
            if xmax is None:
                xmax = dist.quantile(0.9999)
            if xmin > xmax:
                xmin, xmax = xmax, xmin  # switch them if they are given in the wrong order
            if (xmin < QL and xmax < QL) or (xmin >= QL and xmax <= QU) or (xmin > QU and xmax > QU):
                X = np.linspace(xmin, xmax, points)
            elif xmin < QL and xmax > QL and xmax < QU:
                if dist.gamma == 0:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                    else:  # pdf is asymptotic to inf at x=0
                        X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
                else:  # gamma > 0
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                    else:  # pdf is asymptotic to inf at x=0
                        detail = np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2) + dist.gamma
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail])
            elif xmin > QL and xmin < QU and xmax > QU:
                if dist._pdf0 == 0:
                    X = np.hstack([np.linspace(xmin, QU, points - points_right), np.linspace(QU, xmax, points_right)])
                else:  # pdf is asymptotic to inf at x=0
                    try:
                        detail = np.geomspace(xmin - dist.gamma, QU - dist.gamma, points - points_right) + dist.gamma
                        right = np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma
                    except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                        detail = np.linspace(xmin, QU, points - points_right)
                        right = np.linspace(QU, xmax, points_right)
                    X = np.hstack([detail, right])
            else:  # xmin < QL and xmax > QU
                if dist.gamma == 0:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, QU, points - (points_right + 1)), np.geomspace(QU, xmax, points_right)])
                    else:  # pdf is asymptotic to inf at x=0
                        try:
                            X = np.hstack([xmin, np.geomspace(QL, QU, points - (points_right + 1)), np.geomspace(QU, xmax, points_right)])
                        except ValueError:  # occurs for very low shape params causing QL to be zero
                            X = np.hstack([xmin, np.linspace(QL, QU, points - (points_right + 1)), np.geomspace(QU, xmax, points_right)])
                else:  # gamma > 0
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, QU, points - (points_right + 2)), np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma])
                    else:  # pdf is asymptotic to inf at x=0
                        try:
                            detail = np.geomspace(QL - dist.gamma, QU - dist.gamma, points - (points_right + 2)) + dist.gamma
                            right = np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma
                        except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                            detail = np.linspace(QL, QU, points - (points_right + 2))
                            right = np.linspace(QU, xmax, points_right)
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail, right])
        elif dist.name == 'Normal':
            if xmin is None:
                xmin = dist.quantile(0.0001)
            if xmax is None:
                xmax = dist.quantile(0.9999)
            if xmin > xmax:
                xmin, xmax = xmax, xmin  # switch them if they are given in the wrong order
            if xmin <= 0 or xmin > dist.quantile(0.0001):
                X = np.linspace(xmin, xmax, points)
            else:
                X = np.hstack([0, np.linspace(xmin, xmax, points - 1)])  # this ensures that the distribution is at least plotted from 0 if its xmin is above 0
        elif dist.name == 'Beta':
            if xmin is None:
                xmin = 0
            if xmax is None:
                xmax = 1
            if xmax > 1:
                raise ValueError('xmax must be less than or equal to 1 for the beta distribution')
            if xmin > xmax:
                xmin, xmax = xmax, xmin  # switch them if they are given in the wrong order
            X = np.linspace(xmin, xmax, points)
        else:
            raise ValueError('Unrecognised distribution name')
    return X


def zeroise_below_gamma(X, Y, gamma):
    '''
    This will make all Y values 0 for the corresponding X values being below gamma.
    Used by HF and CHF which need to be zeroized if the gamma shifted form of the equation is used.
    '''
    if gamma > 0:
        if len(np.where(X > gamma)[0]) == 0:
            Y[0::] = 0  # zeroize everything if there is no X values above gamma
        else:
            Y[0:(np.where(X > gamma)[0][0])] = 0  # zeroize below X=gamma
    return Y


def xy_transform(value, direction='forward', axis='x'):
    '''
    Converts between data values and axes coordinates (based on xlim() or ylim()).
    If direction is forward the returned value will always be between 0 and 1 provided value is on the plot.
    If direction is reverse the input should be between 0 and 1 and the returned value will be the data value based on the current plot lims
    axis must be x or y
    '''
    if direction not in ['reverse', 'inverse', 'inv', 'rev', 'forward', 'fwd']:
        raise ValueError('direction must be "forward" or "reverse"')
    if axis not in ['X', 'x', 'Y', 'y']:
        raise ValueError('axis must be x or y. Default is x')

    ax = plt.gca()
    if direction in ['reverse', 'inverse', 'inv', 'rev']:
        if type(value) in [int, float, np.float64]:
            if axis == 'x':
                transformed_values = ax.transData.inverted().transform((ax.transAxes.transform((value, 0.5))[0], 0.5))[0]  # x transform
            else:
                transformed_values = ax.transData.inverted().transform((1, ax.transAxes.transform((1, value))[1]))[1]  # y transform
        elif type(value) in [list, np.ndarray]:
            transformed_values = []
            for item in value:
                if axis == 'x':
                    transformed_values.append(ax.transData.inverted().transform((ax.transAxes.transform((item, 0.5))[0], 0.5))[0])  # x transform
                else:
                    transformed_values.append(ax.transData.inverted().transform((1, ax.transAxes.transform((1, item))[1]))[1])  # y transform
        else:
            raise ValueError('type of value is not recognized')
    else:  # direction is forward
        if type(value) in [int, float, np.float64]:
            if axis == 'x':
                transformed_values = ax.transAxes.inverted().transform(ax.transData.transform((value, 0.5)))[0]  # x transform
            else:
                transformed_values = ax.transAxes.inverted().transform(ax.transData.transform((1, value)))[1]  # y transform
        elif type(value) in [list, np.ndarray]:
            transformed_values = []
            for item in value:
                if axis == 'x':
                    transformed_values.append(ax.transAxes.inverted().transform(ax.transData.transform((item, 0.5)))[0])  # x transform
                else:
                    transformed_values.append(ax.transAxes.inverted().transform(ax.transData.transform((1, value)))[1])  # y transform
        else:
            raise ValueError('type of value is not recognized')
    return transformed_values


def probability_plot_xylims(x, y, dist, spacing=0.1, gamma_beta=None, beta_alpha=None, beta_beta=None):
    '''
    finds the x and y limits of probability plots.
    This function is called by probability_plotting
    '''
    # x limits
    if dist in ['weibull', 'lognormal', 'loglogistic']:
        min_x_log = np.log10(min(x))
        max_x_log = np.log10(max(x))
        dx_log = max_x_log - min_x_log
        xlim_lower = 10 ** (min_x_log - dx_log * spacing)
        xlim_upper = 10 ** (max_x_log + dx_log * spacing)
    elif dist in ['normal', 'gamma', 'exponential', 'beta']:
        min_x = min(x)
        max_x = max(x)
        dx = max_x - min_x
        xlim_lower = min_x - dx * spacing
        xlim_upper = max_x + dx * spacing
    else:
        raise ValueError('dist is unrecognised')
    if xlim_lower < 0 and dist != 'normal':
        xlim_lower = 0
    plt.xlim(xlim_lower, xlim_upper)

    # y limits
    if dist == 'weibull':
        min_y_tfm = axes_transforms.weibull_forward(min(y))
        max_y_tfm = axes_transforms.weibull_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.weibull_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.weibull_inverse(max_y_tfm + dy_tfm * spacing)
    if dist == 'exponential':
        min_y_tfm = axes_transforms.expon_forward(min(y))
        max_y_tfm = axes_transforms.expon_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.expon_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.expon_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == 'gamma':
        min_y_tfm = axes_transforms.gamma_forward(min(y), gamma_beta)
        max_y_tfm = axes_transforms.gamma_forward(max(y), gamma_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gamma_inverse(min_y_tfm - dy_tfm * spacing, gamma_beta)
        ylim_upper = axes_transforms.gamma_inverse(max_y_tfm + dy_tfm * spacing, gamma_beta)
    elif dist in ['normal', 'lognormal']:
        min_y_tfm = axes_transforms.normal_forward(min(y))
        max_y_tfm = axes_transforms.normal_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.normal_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.normal_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == 'beta':
        min_y_tfm = axes_transforms.beta_forward(min(y), beta_alpha, beta_beta)
        max_y_tfm = axes_transforms.beta_forward(max(y), beta_alpha, beta_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.beta_inverse(min_y_tfm - dy_tfm * spacing, beta_alpha, beta_beta)
        ylim_upper = axes_transforms.beta_inverse(max_y_tfm + dy_tfm * spacing, beta_alpha, beta_beta)
    elif dist == 'loglogistic':
        min_y_tfm = axes_transforms.loglogistic_forward(min(y))
        max_y_tfm = axes_transforms.loglogistic_forward(max(y))
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.loglogistic_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.loglogistic_inverse(max_y_tfm + dy_tfm * spacing)
    plt.ylim(ylim_lower, ylim_upper)


def probability_plot_xyticks(yticks=None):
    '''
    Sets the x and y ticks for probability plots
    X ticks are selected using either MaxNLocator or LogLocator.
    X ticks are formatted using a custom formatter.
    Y ticks are specified with FixedLocator due to their irregular spacing. Minor y ticks use MaxNLocator
    Y ticks are formatted using a custom Percent Formatter that handles decimals better
    This function is called by probability_plotting
    '''

    def get_tick_locations(major_or_minor, in_lims=True, axis='x'):
        '''
        returns the major or minor tick locations for the current axis
        if in_lims=True then it will only return the ticks that are within the current xlim() or ylim(). Default is True
        axis must be x or y. Default is x
        '''
        if axis == 'x':
            AXIS = ax.xaxis
            L = xlower
            U = xupper
        elif axis == 'y':
            AXIS = ax.yaxis
            L = ylower
            U = yupper
        else:
            raise ValueError('axis must be x or y. Default is x')

        if major_or_minor == 'major':
            all_locations = AXIS.get_major_locator().tick_values(L, U)
        elif major_or_minor == 'minor':
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
        '''
        Provides custom string formatting that is used for the xticks
        '''
        if value == 0:
            label = '0'
        elif abs(value) >= 10000 or abs(value) <= 0.0001:  # small numbers and big numbers are formatted with scientific notation
            if value < 0:
                sign = '-'
                value *= -1
            else:
                sign = ''
            exponent = int(np.floor(np.log10(value)))
            multiplier = value / (10 ** exponent)
            if multiplier % 1 < 0.0000001:
                multiplier = int(multiplier)
            if multiplier == 1:
                label = str((r'$%s%s^{%d}$') % (sign, 10, exponent))
            else:
                label = str((r'$%s%g\times%s^{%d}$') % (sign, multiplier, 10, exponent))
        else:  # numbers between 0.0001 and 10000 are formatted without scientific notation
            label = str('{0:g}'.format(value))
        return label

    def customPercentFormatter(value, _):
        '''
        Provides custom percent string formatting that is used for the yticks
        Slightly different than PercentFormatter as it does not force a particular number of decimals. ie. 99.00 becomes 99 while 99.99 still displays as such.
        '''
        value100 = value * 100
        value100dec = round(value100 % 1, 8)  # this breaks down after 8 decimal places due to python's auto rounding. Not likely to be an issue as we're rarely dealing with this many decimals
        if value100dec == 0:
            value100dec = int(value100dec)
        value100whole = int(value100 - value100dec)
        combined = value100dec + value100whole
        label = str(str(combined) + str('%'))
        return label

    def get_edge_distances():
        'finds the sum of the distance (in axes coords (0 to 1)) of the distances from the edge ticks to the edges'
        xtick_locations = get_tick_locations('major', axis='x')
        left_tick_distance = xy_transform(xtick_locations[0], direction='forward', axis='x') - xy_transform(xlower, direction='forward', axis='x')
        right_tick_distance = xy_transform(xupper, direction='forward', axis='x') - xy_transform(xtick_locations[-1], direction='forward', axis='x')
        return left_tick_distance + right_tick_distance

    ################# xticks
    MaxNLocator = ticker.MaxNLocator(nbins=10, min_n_ticks=2, steps=[1, 2, 5, 10])
    LogLocator = ticker.LogLocator()
    ax = plt.gca()
    xlower, xupper = plt.xlim()
    if xlower <= 0:  # can't use log scale if 0 (gamma) or negative numbers (normal and gumbel)
        loc_x = MaxNLocator
    elif xupper < 0.1:  # very small positive values
        loc_x = ticker.LogLocator()
    elif xupper < 1000 or np.log10(xupper) - np.log10(xlower) < 1.5:  # not too big and not too small OR it may be big but not too spread out
        loc_x = MaxNLocator
    else:  # it is really big (>1000) and spread out
        loc_x = ticker.LogLocator()
    ax.xaxis.set_major_locator(loc_x)  # apply the tick locator
    # do not apply a minor locator. It is never as good as the default

    if get_edge_distances() > 0.5:  # 0.5 means 50% of the axis is without ticks on either side. Above this is considered unacceptable
        # find which locator is better
        ax.xaxis.set_major_locator(MaxNLocator)
        edges_maxNLocator = get_edge_distances()
        ax.xaxis.set_major_locator(LogLocator)
        edges_LogLocator = get_edge_distances()
        if edges_LogLocator < edges_maxNLocator:
            ax.xaxis.set_major_locator(LogLocator)  # apply a new locator
        else:
            ax.xaxis.set_major_locator(MaxNLocator)  # apply a new locator
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(customFormatter))  # the custom formatter is always applied to the major ticks

    num_major_x_ticks_shown = len(get_tick_locations('major', axis='x'))
    num_minor_x_xticks_shown = len(get_tick_locations('minor', axis='x'))
    if max(abs(xlower), abs(xupper)) < 1000 and min(abs(xlower), abs(xupper)) > 0.001:
        max_minor_ticks = 15
    else:
        max_minor_ticks = 10
    if num_major_x_ticks_shown < 2 and num_minor_x_xticks_shown <= max_minor_ticks:
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(customFormatter))  # if there are less than 2 major ticks within the plotting limits then the minor ticks should be labeled. Only do this if there aren't too many minor ticks

    ################# yticks
    if yticks is None:
        yticks = [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.999999]
    loc_y = ticker.FixedLocator(yticks)
    loc_y_minor = ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10])
    ax.yaxis.set_major_locator(loc_y)  # sets the tick spacing
    ax.yaxis.set_minor_locator(loc_y_minor)  # sets the tick spacing
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.format_coord = lambda x, y: 'x={:g}, y={:.1%}'.format(x, y)  # sets the formatting of the axes coordinates in the bottom right of the figure. Without this the FuncFormatter raw strings make it into the axes coords and don't look good.


def anderson_darling(fitted_cdf, empirical_cdf):
    '''
    Calculates the Anderson-Darling goodness of fit statistic
    These formulas are based on the method used in MINITAB which gives an adjusted form of the original AD statistic described on Wikipedia
    '''
    Z = np.sort(np.asarray(fitted_cdf))
    Zi = np.hstack([Z, 1 - 1e-12])
    Zi_1 = (np.hstack([0, Zi]))[0:-1]  # Z_i-1
    FnZi = np.sort(np.asarray(empirical_cdf))
    FnZi_1 = np.hstack([0, FnZi])  # Fn(Z_i-1)
    lnZi = np.log(Zi)
    lnZi_1 = np.hstack([0, lnZi])[0:-1]

    A = -Zi - np.log(1 - Zi) + Zi_1 + np.log(1 - Zi_1)
    B = 2 * np.log(1 - Zi) * FnZi_1 - 2 * np.log(1 - Zi_1) * FnZi_1
    C = lnZi * FnZi_1 ** 2 - np.log(1 - Zi) * FnZi_1 ** 2 - lnZi_1 * FnZi_1 ** 2 + np.log(1 - Zi_1) * FnZi_1 ** 2
    n = len(fitted_cdf)
    AD = n * ((A + B + C).sum())
    return AD
