'''
Other Functions

This is a collection of utilities that are used throughout the python reliability library.
Functions have been placed here as to declutter the dropdown lists of your IDE.
It is not expected that users will be using any utils directly.

Included functions are:
round_to_decimals - applies different rounding rules to numbers above and below 1 so that small numbers do not get rounded to 0.
transform_spaced - Creates linearly spaced array (in transform space) based on a specified transform. This is like np.logspace but it can make an array that is weibull spaced, normal spaced, etc.
axes_transforms - Custom scale functions used in Probability_plotting
'''

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection, PathCollection


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

    def weibull_forward(F):
        return np.log(-np.log(1 - F))

    def weibull_inverse(R):
        return 1 - np.exp(-np.exp(R))

    def expon_forward(F):
        return ss.expon.ppf(F)

    def expon_inverse(R):
        return ss.expon.cdf(R)

    def normal_forward(F):
        return ss.norm.ppf(F)

    def normal_inverse(R):
        return ss.norm.cdf(R)

    def gamma_forward(F, beta):
        return ss.gamma.ppf(F, a=beta)

    def gamma_inverse(R, beta):
        return ss.gamma.cdf(R, a=beta)

    def beta_forward(F, alpha, beta):
        return ss.beta.ppf(F, a=alpha, b=beta)

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
            if dist.gamma == 0:
                xlim_lower = 0
            else:
                diff = dist.quantile(0.999) - dist.quantile(0.001)
                xlim_lower = max(0, dist.quantile(0.001) - diff * 0.1)
        else:
            xlim_lower = xmin
        if xmax is None:
            xlim_upper = dist.quantile(0.999)
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
    if func in ['pdf', 'PDF']:
        if dist._pdf0 == 0:  # an increasing pdf. Not asymptotic at x=0
            ylim_upper = max(Y) * 1.05
        else:  # asymptotic at x=0
            idx = np.where(X >= dist.b5)[0][0]  # sets ylim_upper as the value at the 5th quantile
            ylim_upper = Y[idx]
    elif func in ['cdf', 'CDF', 'SF', 'sf']:
        ylim_upper = 1.05
    elif func in ['hf', 'HF']:
        if dist._hf0 == 0:  # when the hazard function is increasing
            idx = np.where(X >= xlim_upper)[0][0]  # index of the hf where it is equal to b95
        else:  # when the hazard function is decreasing
            idx = np.where(X >= dist.quantile(0.01))[0][0]
        ylim_upper = Y[idx]
    elif func in ['chf', 'CHF']:
        idx = np.where(X >= xlim_upper)[0][0]  # index of the chf where it is equal to b95
        ylim_upper = Y[idx]
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
        if ylim_LOWER != ylim_UPPER:
            plt.ylim(ylim_LOWER, ylim_UPPER, auto=None)
        else:
            plt.ylim(bottom=ylim_LOWER, auto=None)


def generate_X_array(dist, func, xvals=None, xmin=None, xmax=None):
    '''
    generates the array of X values for each of the PDf, CDF, SF, HF, CHF functions within reliability.Distributions
    This is done with a variety of cases in order to ensure that for regions of high gradient (particularly asymptotes to inf) the points are more concentrated.
    This ensures that the line always looks as smooth as possible using only 100 data points
    '''

    # obtain the xvals array
    points = 100  # the number of points to use when generating the X array
    QL = dist.quantile(0.0001)  # quantile lower
    QU = dist.quantile(0.999)  # quantile upper
    if xvals is not None:
        X = xvals
        if type(X) in [float, int, np.float64]:
            if X < 0:
                raise ValueError('the value given for xvals is less than 0')
            X = np.array([X])
        elif type(X) is list:
            X = np.array(X)
        elif type(X) is np.ndarray:
            pass
        else:
            raise ValueError('unexpected type in xvals. Must be int, float, list, or array')
        if type(X) is np.ndarray and min(X) < 0:
            raise ValueError('xvals was found to contain values below 0')
    else:
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = QU + (QU - QL) * 0.1
        if xmin > xmax:
            xmin, xmax = xmax, xmin  # switch them if they are given in the wrong order
        if (xmin < QL and xmax < QL) or (xmin >= QL and xmax <= QU) or (xmin > QU and xmax > QU):
            X = np.linspace(xmin, xmax, points)
        elif xmin < QL and xmax > QL and xmax < QU:
            if dist.gamma == 0:
                if func in ['pdf', 'PDF', 'cdf', 'CDF', 'sf', 'SF']:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                    else:  # pdf is asymptotic to inf at x=0
                        X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
                elif func in ['hf', 'HF']:
                    if dist._hf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                    else:  # hf is asymptotic to inf at x=0
                        X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
                elif func in ['chf', 'CHF']:
                    X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                else:
                    raise ValueError('func is invalid')
            else:  # gamma > 0
                if func in ['pdf', 'PDF', 'cdf', 'CDF', 'sf', 'SF']:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                    else:  # pdf is asymptotic to inf at x=0
                        detail = np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2) + dist.gamma
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail])
                elif func in ['hf', 'HF']:
                    if dist._hf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                    else:  # hf is asymptotic to inf at x=0
                        detail = np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2) + dist.gamma
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail])
                elif func in ['chf', 'CHF']:
                    X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                else:
                    raise ValueError('func is invalid')
        elif xmin > QL and xmin < QU and xmax > QU:
            if func in ['pdf', 'PDF', 'cdf', 'CDF', 'sf', 'SF']:
                if dist._pdf0 == 0:
                    X = np.hstack([np.linspace(xmin, QU, points - 1), xmax])
                else:  # pdf is asymptotic to inf at x=0
                    detail = np.geomspace(xmin - dist.gamma, QU - dist.gamma, points - 1) + dist.gamma
                    X = np.hstack([detail, xmax])
            elif func in ['hf', 'HF']:
                if dist._hf0 == 0:
                    X = np.linspace(xmin, xmax, points)
                else:  # hf is asymptotic to inf at x=0
                    X = np.geomspace(xmin - dist.gamma, xmax - dist.gamma, points) + dist.gamma
            elif func in ['chf', 'CHF']:
                X = np.linspace(xmin, xmax, points)
            else:
                raise ValueError('func is invalid')
        else:  # xmin < QL and xmax > QU
            if dist.gamma == 0:
                if func in ['pdf', 'PDF', 'cdf', 'CDF', 'sf', 'SF']:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, QU, points - 2), xmax])
                    else:  # pdf is asymptotic to inf at x=0
                        X = np.hstack([xmin, np.geomspace(QL, QU, points - 2), xmax])
                elif func in ['hf', 'HF']:
                    if dist._hf0 == 0:
                        X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                    else:
                        X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
                elif func in ['chf', 'CHF']:
                    X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                else:
                    raise ValueError('func is invalid')
            else:  # gamma > 0
                if func in ['pdf', 'PDF', 'cdf', 'CDF', 'sf', 'SF']:
                    if dist._pdf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, QU, points - 3), xmax])
                    else:  # pdf is asymptotic to inf at x=0
                        detail = np.geomspace(QL - dist.gamma, QU - dist.gamma, points - 3) + dist.gamma
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail, xmax])
                elif func in ['hf', 'HF']:
                    if dist._hf0 == 0:
                        X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                    else:  # hf is asymptotic to inf at x=0
                        detail = np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2) + dist.gamma
                        X = np.hstack([xmin, dist.gamma - 1e-8, detail])
                elif func in ['chf', 'CHF']:
                    X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
                else:
                    raise ValueError('func is invalid')
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
