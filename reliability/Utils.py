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
from matplotlib.collections import PolyCollection, LineCollection


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
        raise ValueError('start and stop must be within the range 0 to 1')
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
    polygon = np.column_stack([np.hstack([xlower, xupper[::-1]]), np.hstack([ylower, yupper[::-1]])])  # this is equivalent to fill as it makes a polygon
    col = PolyCollection([polygon], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def line_no_autoscale(x, y, **kwargs):
    '''
    creates a line without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when autoscale sets the range
    '''
    line = np.column_stack([x, y])  # this is equivalent to plot as it makes a line
    col = LineCollection([line], **kwargs)
    plt.gca().add_collection(col, autolim=False)
