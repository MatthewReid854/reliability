"""
Probability plotting

This module contains the functions required to generate linearized probability
plots of the 8 standard distributions included in reliability. The most common
use of these type of probability plots is to assess goodness of fit. Also
included in this module are probability-probability (PP) plots and
quantile-quantile (QQ) plots.

The functions in this module are:
plotting_positions - generates an empirical estimate of the CDF
Weibull_probability_plot - Weibull_2P and Weibull_3P
Loglogistic_probability_plot - Loglogistic_2P and Loglogistic_3P
Normal_probability_plot - Normal_2P
Lognormal_probability_plot - Lognormal_2P and Lognormal_3P
Exponential_probability_plot - Exponential_1P and Exponential_2P
Exponential_probability_plot_Weibull_Scale - Exponential_1P and Exponential_2P
    with Weibull Scale makes multiple plots with different Lambda parameters
    be parallel.
Beta_probability_plot - Beta_2P
Gamma_probability_plot - Gamma_2P and Gamma_3P
Gumbel_probability_plot - Gumbel_2P
QQ_plot_parametric - quantile-quantile plot. Compares two parametric
    distributions using shared quantiles. Useful for Field-to-Test conversions
    in ALT.
QQ_plot_semiparametric - quantile-quantile plot. Compares failure data with a
    hypothesised parametric distribution. Useful to assess goodness of fit.
PP_plot_parametric - probability-probability plot. Compares two parametric
    distributions using their CDFs. Useful to understand the differences
    between the quantiles of the distributions.
PP_plot_semiparametric - probability-probability plot. Compares failure data
    with a hypothesised parametric distribution. Useful to assess goodness of
    fit.
plot_points - plots the failure points on a scatter plot. Useful to overlay
    the points with CDF, SF, or CHF. Does not scale the axis so can be used
    with methods like dist.SF()
"""

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from reliability.Distributions import (
    Weibull_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
    Gamma_Distribution,
    Beta_Distribution,
    Exponential_Distribution,
    Loglogistic_Distribution,
    Gumbel_Distribution,
    GeneralizedPareto_Distribution,
)
from reliability.Nonparametric import KaplanMeier, NelsonAalen, RankAdjustment
from reliability.Utils import (
    axes_transforms,
    round_to_decimals,
    probability_plot_xylims,
    probability_plot_xyticks,
    colorprint,
    xy_downsample,
)

np.seterr("ignore")
dec = 3  # number of decimals to use when rounding fitted parameters in labels


def plotting_positions(failures=None, right_censored=None, a=None):
    """
    Calculates the plotting positions for plotting on probability paper.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored failure data. Optional input. Default = None.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a) where k is the rank and n is the number of points.
        Optional input. Default is a = 0.3 which is the median rank method
        (same as the default in Minitab and Reliasoft). Must be in the range 0
        to 1. For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics

    Returns
    -------
    (array(x),array(y)) : tuple
        a tuple of two arrays. The arrays provide the x and y plotting
        positions. The x array will match the failures parameter while the y
        array will be the empirical estimate of the CDF at each of the failures.

    Notes
    -----
    This function is primarily used by the probability plotting functions. The
    order of the input data is preserved (not sorted).
    """

    # error checking the input
    if type(failures) in [list, np.ndarray]:
        f = np.asarray(failures)
    else:
        raise ValueError("failures must be specified as an array or list")

    if type(right_censored) == type(None):
        rc = np.array([])
    elif type(right_censored) in [np.ndarray, list]:
        rc = np.asarray(right_censored)
    else:
        raise ValueError("if specified, right_censored must be an array or list")

    if a is None:
        a = 0.3
    elif a < 0 or a > 1:
        raise ValueError(
            "a must be in the range 0 to 1. Default is 0.3 which gives the median rank. For more information see https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics"
        )

    # construct the dataframe for the rank adjustment method
    f_codes = np.ones_like(f)
    rc_codes = np.zeros_like(rc)
    cens_codes = np.hstack([f_codes, rc_codes])
    all_data = np.hstack([f, rc])
    n = len(all_data)
    data = {"times": all_data, "cens_codes": cens_codes}
    df = pd.DataFrame(data, columns=["times", "cens_codes"])
    df_sorted = df.sort_values(by="times")
    df_sorted["reverse_i"] = np.arange(1, n + 1)[::-1]
    failure_rows = df_sorted.loc[df_sorted["cens_codes"] == 1.0]
    reverse_i = failure_rows["reverse_i"].values
    len_reverse_i = len(reverse_i)
    leading_cens = np.where(df_sorted["cens_codes"].values == 1)[0][0]

    if leading_cens > 0:  # there are censored items before the first failure
        k = np.arange(1, len_reverse_i + 1)
        adjusted_rank2 = [0]
        rank_increment = [leading_cens / (n - 1)]
        for j in k:
            rank_increment.append((n + 1 - adjusted_rank2[-1]) / (1 + reverse_i[j - 1]))
            adjusted_rank2.append(adjusted_rank2[-1] + rank_increment[-1])
        adjusted_rank = adjusted_rank2[1:]
    else:  # the first item is a failure
        k = np.arange(1, len_reverse_i)
        adjusted_rank = [1]
        rank_increment = [1]
        for j in k:
            if j > 0:
                rank_increment.append((n + 1 - adjusted_rank[-1]) / (1 + reverse_i[j]))
                adjusted_rank.append(adjusted_rank[-1] + rank_increment[-1])
    F = []
    for i in adjusted_rank:
        F.append((i - a) / (n + 1 - 2 * a))

    # restore the original order of the points using the index from the sorted dataframe
    idx = failure_rows.index.values
    df2 = pd.DataFrame(
        data={"x": failure_rows.times.values, "y": F, "idx": idx},
        columns=["x", "y", "idx"],
    ).sort_values(by="idx")
    x = df2.x.values
    y = df2.y.values
    return x, y


def Weibull_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Weibull scaled probability paper so that the
    CDF of the distribution appears linear. This function can be used to show
    Weibull_2P or Weibull_3P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Weibull_3P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    # ensure the input data is arrays
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                alpha = __fitted_dist_params.alpha
                beta = __fitted_dist_params.beta
                alpha_SE = __fitted_dist_params.alpha_SE
                beta_SE = __fitted_dist_params.beta_SE
                Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            else:
                from reliability.Fitters import Fit_Weibull_2P

                fit = Fit_Weibull_2P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                alpha = fit.alpha
                beta = fit.beta
                alpha_SE = fit.alpha_SE
                beta_SE = fit.beta_SE
                Cov_alpha_beta = fit.Cov_alpha_beta
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Weibull_2P (α="
                    + str(round_to_decimals(alpha, dec))
                    + ", β="
                    + str(round_to_decimals(beta, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                alpha = __fitted_dist_params.alpha
                beta = __fitted_dist_params.beta
                gamma = __fitted_dist_params.gamma
                alpha_SE = __fitted_dist_params.alpha_SE
                beta_SE = __fitted_dist_params.beta_SE
                Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            else:
                from reliability.Fitters import Fit_Weibull_3P

                fit = Fit_Weibull_3P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                alpha = fit.alpha
                beta = fit.beta
                gamma = fit.gamma
                alpha_SE = fit.alpha_SE
                beta_SE = fit.beta_SE
                Cov_alpha_beta = fit.Cov_alpha_beta

            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Weibull_3P\n(α="
                    + str(round_to_decimals(alpha, dec))
                    + ", β="
                    + str(round_to_decimals(beta, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )

            xlabel = "Time - gamma"
            failures = failures - gamma
            if right_censored is not None:
                right_censored = right_censored - gamma
        wbf = Weibull_Distribution(
            alpha=alpha,
            beta=beta,
            alpha_SE=alpha_SE,
            beta_SE=beta_SE,
            Cov_alpha_beta=Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(axes_transforms.weibull_forward, axes_transforms.weibull_inverse),
    )
    plt.xscale("log")
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    plt.gcf().set_size_inches(9, 9)
    if show_fitted_distribution is True:
        wbf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nWeibull CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(
        xlabel
    )  # needs to be set after plotting the CDF to override the default 'xvals'
    probability_plot_xylims(x=x, y=y, dist="weibull", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Loglogistic_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Loglogistically scaled probability paper so
    that the CDF of the distribution appears linear. This function can be used
    to show Loglogistic_2P or Loglogistic_3P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Loglogistic_3P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    # ensure the input data is arrays
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                alpha = __fitted_dist_params.alpha
                beta = __fitted_dist_params.beta
                alpha_SE = __fitted_dist_params.alpha_SE
                beta_SE = __fitted_dist_params.beta_SE
                Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            else:
                from reliability.Fitters import Fit_Loglogistic_2P

                fit = Fit_Loglogistic_2P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                alpha = fit.alpha
                beta = fit.beta
                alpha_SE = fit.alpha_SE
                beta_SE = fit.beta_SE
                Cov_alpha_beta = fit.Cov_alpha_beta
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Loglogistic_2P (α="
                    + str(round_to_decimals(alpha, dec))
                    + ", β="
                    + str(round_to_decimals(beta, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                alpha = __fitted_dist_params.alpha
                beta = __fitted_dist_params.beta
                gamma = __fitted_dist_params.gamma
                alpha_SE = __fitted_dist_params.alpha_SE
                beta_SE = __fitted_dist_params.beta_SE
                Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            else:
                from reliability.Fitters import Fit_Loglogistic_3P

                fit = Fit_Loglogistic_3P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                alpha = fit.alpha
                beta = fit.beta
                gamma = fit.gamma
                alpha_SE = fit.alpha_SE
                beta_SE = fit.beta_SE
                Cov_alpha_beta = fit.Cov_alpha_beta

            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Loglogistic_3P\n(α="
                    + str(round_to_decimals(alpha, dec))
                    + ", β="
                    + str(round_to_decimals(beta, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )
            xlabel = "Time - gamma"
            failures = failures - gamma
            if right_censored is not None:
                right_censored = right_censored - gamma
        llf = Loglogistic_Distribution(
            alpha=alpha,
            beta=beta,
            alpha_SE=alpha_SE,
            beta_SE=beta_SE,
            Cov_alpha_beta=Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(
            axes_transforms.loglogistic_forward,
            axes_transforms.loglogistic_inverse,
        ),
    )
    plt.xscale("log")
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        llf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nLoglogistic CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(
        xlabel
    )  # needs to be set after plotting the CDF to override the default 'xvals'
    probability_plot_xylims(x=x, y=y, dist="loglogistic", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Exponential_probability_plot_Weibull_Scale(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Weibull scaled probability paper so that the
    CDF of the distribution appears linear. This differs from the Exponential
    probability plot on Exponential scaled probability paper as the Weibull
    paper will make multiple distributions with different Lambda parameters
    appear as parallel lines rather than as lines radiating from the origin.
    This change in scale has applications in ALT probability plotting.
    This function can be used to show Exponential_1P or Exponential_2P
    distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Exponential_2P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    This function works because a Weibull Distribution with alpha = x and
    beta = 1 is identical to an Exponential Distribution with Lambda = 1/x.

    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    CI_type is not required as the Exponential distribution has the same
    confidence interval bounds on both time and reliability.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    # ensure the input data is arrays
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 1 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 1"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                Lambda_SE = __fitted_dist_params.Lambda_SE
            else:
                from reliability.Fitters import Fit_Exponential_1P

                fit = Fit_Exponential_1P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_1P (λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                Lambda_SE = __fitted_dist_params.Lambda_SE
                gamma = __fitted_dist_params.gamma
            else:
                from reliability.Fitters import Fit_Exponential_2P

                fit = Fit_Exponential_2P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
                gamma = fit.gamma

            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_2P\n(λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )
            xlabel = "Time - gamma"
            failures = (
                failures - gamma + 0.009
            )  # this 0.009 adjustment is to avoid taking the log of 0. It causes negligible difference to the fit and plot. 0.009 is chosen to be the same as Weibull_Fit_3P adjustment.
            if right_censored is not None:
                right_censored = (
                    right_censored - gamma + 0.009
                )  # this 0.009 adjustment is to avoid taking the log of 0. It causes negligible difference to the fit and plot. 0.009 is chosen to be the same as Weibull_Fit_3P adjustment.

        ef = Exponential_Distribution(Lambda=Lambda, Lambda_SE=Lambda_SE, CI=CI)

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(axes_transforms.weibull_forward, axes_transforms.weibull_inverse),
    )
    plt.xscale("log")
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        ef.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.ylabel("Fraction failing")
    plt.title("Probability plot\nExponential CDF (Weibull Scale)")
    plt.xlabel(
        xlabel
    )  # needs to be set after plotting the CDF to override the default 'xvals'
    probability_plot_xylims(x=x, y=y, dist="weibull", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Gumbel_probability_plot(
    failures=None,
    right_censored=None,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Gumbel scaled probability paper so that the
    CDF of the distribution appears linear.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            mu = __fitted_dist_params.mu
            sigma = __fitted_dist_params.sigma
            mu_SE = __fitted_dist_params.mu_SE
            sigma_SE = __fitted_dist_params.sigma_SE
            Cov_mu_sigma = __fitted_dist_params.Cov_mu_sigma
        else:
            from reliability.Fitters import Fit_Gumbel_2P

            fit = Fit_Gumbel_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
            )
            mu = fit.mu
            sigma = fit.sigma
            mu_SE = fit.mu_SE
            sigma_SE = fit.sigma_SE
            Cov_mu_sigma = fit.Cov_mu_sigma

        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str(
                "Fitted Gumbel_2P (μ="
                + str(round_to_decimals(mu, dec))
                + ", σ="
                + str(round_to_decimals(sigma, dec))
                + ")"
            )
        gbf = Gumbel_Distribution(
            mu=mu,
            sigma=sigma,
            mu_SE=mu_SE,
            sigma_SE=sigma_SE,
            Cov_mu_sigma=Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(axes_transforms.gumbel_forward, axes_transforms.gumbel_inverse),
    )
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        gbf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nGumbel CDF")
    plt.xlabel("Time")
    plt.ylabel("Fraction failing")
    probability_plot_xylims(x=x, y=y, dist="gumbel", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Normal_probability_plot(
    failures=None,
    right_censored=None,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Normal scaled probability paper so that the
    CDF of the distribution appears linear.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            mu = __fitted_dist_params.mu
            sigma = __fitted_dist_params.sigma
            mu_SE = __fitted_dist_params.mu_SE
            sigma_SE = __fitted_dist_params.sigma_SE
            Cov_mu_sigma = __fitted_dist_params.Cov_mu_sigma
        else:
            from reliability.Fitters import Fit_Normal_2P

            fit = Fit_Normal_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
            )
            mu = fit.mu
            sigma = fit.sigma
            mu_SE = fit.mu_SE
            sigma_SE = fit.sigma_SE
            Cov_mu_sigma = fit.Cov_mu_sigma
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str(
                "Fitted Normal_2P (μ="
                + str(round_to_decimals(mu, dec))
                + ", σ="
                + str(round_to_decimals(sigma, dec))
                + ")"
            )
        nf = Normal_Distribution(
            mu=mu,
            sigma=sigma,
            mu_SE=mu_SE,
            sigma_SE=sigma_SE,
            Cov_mu_sigma=Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(axes_transforms.normal_forward, axes_transforms.normal_inverse),
    )
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        nf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nNormal CDF")
    plt.xlabel("Time")
    plt.ylabel("Fraction failing")
    probability_plot_xylims(x=x, y=y, dist="normal", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Lognormal_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Lognormal scaled probability paper so that
    the CDF of the distribution appears linear. This function can be used to
    show Lognormal_2P or Lognormal_3P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Lognormal_3P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                mu = __fitted_dist_params.mu
                sigma = __fitted_dist_params.sigma
                mu_SE = __fitted_dist_params.mu_SE
                sigma_SE = __fitted_dist_params.sigma_SE
                Cov_mu_sigma = __fitted_dist_params.Cov_mu_sigma
            else:
                from reliability.Fitters import Fit_Lognormal_2P

                fit = Fit_Lognormal_2P(
                    failures=failures,
                    right_censored=right_censored,
                    show_probability_plot=False,
                    print_results=False,
                )
                mu = fit.mu
                sigma = fit.sigma
                mu_SE = fit.mu_SE
                sigma_SE = fit.sigma_SE
                Cov_mu_sigma = fit.Cov_mu_sigma

            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Lognormal_2P (μ="
                    + str(round_to_decimals(mu, dec))
                    + ", σ="
                    + str(round_to_decimals(sigma, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                mu = __fitted_dist_params.mu
                sigma = __fitted_dist_params.sigma
                gamma = __fitted_dist_params.gamma
                mu_SE = __fitted_dist_params.mu_SE
                sigma_SE = __fitted_dist_params.sigma_SE
                Cov_mu_sigma = __fitted_dist_params.Cov_mu_sigma
            else:
                from reliability.Fitters import Fit_Lognormal_3P

                fit = Fit_Lognormal_3P(
                    failures=failures,
                    right_censored=right_censored,
                    show_probability_plot=False,
                    print_results=False,
                )
                mu = fit.mu
                sigma = fit.sigma
                gamma = fit.gamma
                mu_SE = fit.mu_SE
                sigma_SE = fit.sigma_SE
                Cov_mu_sigma = fit.Cov_mu_sigma
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Lognormal_3P (μ="
                    + str(round_to_decimals(mu, dec))
                    + ", σ="
                    + str(round_to_decimals(sigma, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )
            xlabel = "Time - gamma"
            failures = failures - gamma
            if right_censored is not None:
                right_censored = right_censored - gamma
        lnf = Lognormal_Distribution(
            mu=mu,
            sigma=sigma,
            mu_SE=mu_SE,
            sigma_SE=sigma_SE,
            Cov_mu_sigma=Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(axes_transforms.normal_forward, axes_transforms.normal_inverse),
    )
    plt.xscale("log")
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        lnf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nLognormal CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(xlabel)
    probability_plot_xylims(x=x, y=y, dist="lognormal", spacing=0.1)
    probability_plot_xyticks()
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Beta_probability_plot(
    failures=None,
    right_censored=None,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Beta scaled probability paper so that the
    CDF of the distribution appears linear.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
        times. Defaults = True.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    Both parameters of a Beta Distribution affect the axes scaling such that
    when two different Beta Distributions are plotted on the same Beta
    probability paper, one of them will always appear curved.

    If your plot does not appear automatically, use plt.show() to show it.

    Confidence intervals are not included for the Beta distribution.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"

    # We can't skip fitting when show_fitted_distribution = False because the axes scaling needs alpha and beta
    if CI <= 0 or CI >= 1:
        raise ValueError(
            "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
        )

    if __fitted_dist_params is not None:
        alpha = __fitted_dist_params.alpha
        beta = __fitted_dist_params.beta
    else:
        from reliability.Fitters import Fit_Beta_2P

        fit = Fit_Beta_2P(
            failures=failures,
            right_censored=right_censored,
            show_probability_plot=False,
            print_results=False,
        )
        alpha = fit.alpha
        beta = fit.beta
    if "label" in kwargs:
        label = kwargs.pop("label")
    else:
        label = str(
            "Fitted Beta_2P (α="
            + str(round_to_decimals(alpha, dec))
            + ", β="
            + str(round_to_decimals(beta, dec))
            + ")"
        )
    bf = Beta_Distribution(
        alpha=alpha,
        beta=beta,
    )

    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    f_beta = lambda x: axes_transforms.beta_forward(x, alpha, beta)
    fi_beta = lambda x: axes_transforms.beta_inverse(x, alpha, beta)
    plt.gca().set_yscale("function", functions=(f_beta, fi_beta))
    if show_fitted_distribution is True:
        bf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nBeta CDF")
    plt.xlabel("Time")
    plt.ylabel("Fraction failing")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    probability_plot_xylims(
        x=x, y=y, dist="beta", spacing=0.1, beta_alpha=alpha, beta_beta=beta
    )
    probability_plot_xyticks(
        yticks=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Gamma_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    CI_type="time",
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Gamma scaled probability paper so that the
    CDF of the distribution appears linear. This function can be used to show
    Gamma_2P or Gamma_3P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Gamma_3P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    The beta parameter of a Gamma Distribution affects the axes scaling such
    that when two Gamma Distributions with different beta paameters are plotted
    on the same Gamma probability paper, one of them will always appear curved.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    # ensure the input data is arrays
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 2 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 2"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted

    # We can't skip fitting when show_fitted_distribution = False because the axes scaling needs beta
    if CI <= 0 or CI >= 1:
        raise ValueError(
            "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
        )

    if __fitted_dist_params is not None:
        if __fitted_dist_params.gamma > 0:
            fit_gamma = True

    if fit_gamma is False:
        if __fitted_dist_params is not None:
            alpha = __fitted_dist_params.alpha
            beta = __fitted_dist_params.beta
            alpha_SE = __fitted_dist_params.alpha_SE
            beta_SE = __fitted_dist_params.beta_SE
            mu_SE = __fitted_dist_params.mu_SE
            Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            Cov_mu_beta = __fitted_dist_params.Cov_mu_beta
        else:
            from reliability.Fitters import Fit_Gamma_2P

            fit = Fit_Gamma_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
            )
            alpha = fit.alpha
            beta = fit.beta
            alpha_SE = fit.alpha_SE
            beta_SE = fit.beta_SE
            mu_SE = fit.mu_SE
            Cov_alpha_beta = fit.Cov_alpha_beta
            Cov_mu_beta = fit.Cov_mu_beta

        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str(
                "Fitted Gamma_2P (α="
                + str(round_to_decimals(alpha, dec))
                + ", β="
                + str(round_to_decimals(beta, dec))
                + ")"
            )
    elif fit_gamma is True:
        if __fitted_dist_params is not None:
            alpha = __fitted_dist_params.alpha
            beta = __fitted_dist_params.beta
            gamma = __fitted_dist_params.gamma
            alpha_SE = __fitted_dist_params.alpha_SE
            beta_SE = __fitted_dist_params.beta_SE
            mu_SE = __fitted_dist_params.mu_SE
            Cov_alpha_beta = __fitted_dist_params.Cov_alpha_beta
            Cov_mu_beta = __fitted_dist_params.Cov_mu_beta
        else:
            from reliability.Fitters import Fit_Gamma_3P

            fit = Fit_Gamma_3P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
            )
            alpha = fit.alpha
            beta = fit.beta
            gamma = fit.gamma
            alpha_SE = fit.alpha_SE
            beta_SE = fit.beta_SE
            mu_SE = fit.mu_SE
            Cov_alpha_beta = fit.Cov_alpha_beta
            Cov_mu_beta = fit.Cov_mu_beta

        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str(
                "Fitted Gamma_3P\n(α="
                + str(round_to_decimals(alpha, dec))
                + ", β="
                + str(round_to_decimals(beta, dec))
                + ", γ="
                + str(round_to_decimals(gamma, dec))
                + ")"
            )
        xlabel = "Time - gamma"
        failures = failures - gamma
        if right_censored is not None:
            right_censored = right_censored - gamma
    gf = Gamma_Distribution(
        alpha=alpha,
        beta=beta,
        alpha_SE=alpha_SE,
        beta_SE=beta_SE,
        mu_SE=mu_SE,
        Cov_alpha_beta=Cov_alpha_beta,
        Cov_mu_beta=Cov_mu_beta,
        CI=CI,
        CI_type=CI_type,
    )

    # plot the failure points and format the scale and axes
    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    f_gamma = lambda x: axes_transforms.gamma_forward(x, beta)
    fi_gamma = lambda x: axes_transforms.gamma_inverse(x, beta)
    plt.gca().set_yscale("function", functions=(f_gamma, fi_gamma))
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        gf.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nGamma CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(xlabel)
    probability_plot_xylims(x=x, y=y, dist="gamma", spacing=0.1, gamma_beta=beta)
    probability_plot_xyticks(
        yticks=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def Exponential_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Generates a probability plot on Exponentially scaled probability paper so
    that the CDF of the distribution appears linear. This differs from the
    Exponential_probability_plot_Weibull_Scale as Exponential paper will make
    multiple distributions with different Lambda parameters appear as lines
    radiating from the origin rather than as parallel lines. The parallel form
    is more convenient so the Weibull Scale is more commonly used than the
    Exponential Scale when plotting Exponential Distributions. This function can
    be used to show Exponential_1P or Exponential_2P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Exponential_2P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    CI_type is not required as the Exponential distribution has the same
    confidence interval bounds on both time and reliability.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 1 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 1"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                Lambda_SE = __fitted_dist_params.Lambda_SE
            else:
                from reliability.Fitters import Fit_Exponential_1P

                fit = Fit_Exponential_1P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_1P (λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                Lambda_SE = __fitted_dist_params.Lambda_SE
                gamma = __fitted_dist_params.gamma
            else:
                from reliability.Fitters import Fit_Exponential_2P

                fit = Fit_Exponential_2P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
                gamma = fit.gamma
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_2P\n(λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )
            xlabel = "Time - gamma"
            failures = failures - gamma
            if right_censored is not None:
                right_censored = right_censored - gamma
        ef = Exponential_Distribution(Lambda=Lambda, Lambda_SE=Lambda_SE, CI=CI)

    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    plt.gca().set_yscale(
        "function",
        functions=(
            axes_transforms.exponential_forward,
            axes_transforms.exponential_inverse,
        ),
    )
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        ef.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nExponential CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(
        xlabel
    )  # needs to be set after plotting the CDF to override the default 'xvals'
    probability_plot_xylims(x=x, y=y, dist="exponential", spacing=0.1)
    probability_plot_xyticks(
        yticks=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()

def GeneralizedPareto_probability_plot(
    failures=None,
    right_censored=None,
    fit_gamma=False,
    __fitted_dist_params=None,
    a=None,
    CI=0.95,
    show_fitted_distribution=True,
    show_scatter_points=True,
    downsample_scatterplot=False,
    **kwargs
):
    """
    TODO: this doesn't apply
    Generates a probability plot on Generalized Pareto scaled probability paper so
    that the CDF of the distribution appears linear. This differs from the
    Exponential_probability_plot_Weibull_Scale as Exponential paper will make
    multiple distributions with different Lambda parameters appear as lines
    radiating from the origin rather than as parallel lines. The parallel form
    is more convenient so the Weibull Scale is more commonly used than the
    Exponential Scale when plotting Exponential Distributions. This function can
    be used to show Exponential_1P or Exponential_2P distributions.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    fit_gamma : bool, optional
        Specify this as True in order to fit the Exponential_2P distribution and
        scale the x-axis to time - gamma. Default = False.
    show_fitted_distribution : bool, optional
        If True, the fitted distribution will be plotted on the probability
        plot. Defaults = True.
    show_scatter_points : bool, optional
        If True, the plot will include the scatter points from the failure
        times. Defaults = True.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    a : float, int, optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default = 0.3 which is the median rank method (same as
        the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    CI : float, optional
        The confidence interval for the bounds. Must be between 0 and 1.
        Optional input. Default = 0.95 for 95% CI.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the probability plot is returned as an object

    Notes
    -----
    There is a hidden parameter called __fitted_dist_params which is used to
    specify the parameters of the distribution that has already been fitted.
    Passing a distribution object to this parameter will bypass the fitting
    process and use the parameters of the distribution provided. When this is
    done the minimum length of failures can be 1. The distribution object must
    contain the SE and Cov of the parameters so it needs to be generated by the
    Fitters module.

    CI_type is not required as the Exponential distribution has the same
    confidence interval bounds on both time and reliability.

    If your plot does not appear automatically, use plt.show() to show it.
    """
    if failures is None or len(failures) == 0:
        raise ValueError("failures must be a list or array of the failure data.")
    elif len(failures) < 1 and __fitted_dist_params is None:
        raise ValueError(
            "Insufficient data to fit a distribution. Minimum number of points is 1"
        )

    if type(failures) not in [np.ndarray, list]:
        raise ValueError("failures must be a list or an array")
    failures = np.asarray(failures)

    if right_censored is not None:
        if type(right_censored) not in [np.ndarray, list]:
            raise ValueError("right_censored must be a list or an array")
        right_censored = np.asarray(right_censored)

    if show_fitted_distribution == False and fit_gamma == True:
        colorprint(
            "WARNING: fit_gamma has been reset to False because the distribution is not fitted when show_fitted_distribution = False.",
            text_color="red",
        )

    if "color" in kwargs:
        data_color = kwargs.get("color")
    else:
        data_color = "k"
    xlabel = "Time"  # this will be overridden if gamma is fitted
    if show_fitted_distribution is True:
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        if __fitted_dist_params is not None:
            if __fitted_dist_params.gamma > 0:
                fit_gamma = True

        if fit_gamma is False:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                Lambda_SE = __fitted_dist_params.Lambda_SE
            else:
                from reliability.Fitters import Fit_GeneralizedPareto_3P

                fit = Fit_GeneralizedPareto_3P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_1P (λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ")"
                )
        elif fit_gamma is True:
            if __fitted_dist_params is not None:
                Lambda = __fitted_dist_params.Lambda
                xi = __fitted_dist_params.xi
                Lambda_SE = __fitted_dist_params.Lambda_SE
                xi_SE = __fitted_dist_params.xi_SE
                gamma = __fitted_dist_params.gamma
            else:
                from reliability.Fitters import Fit_GeneralizedPareto_3P

                fit = Fit_GeneralizedPareto_3P(
                    failures=failures,
                    right_censored=right_censored,
                    CI=CI,
                    show_probability_plot=False,
                    print_results=False,
                )
                Lambda = fit.Lambda
                Lambda_SE = fit.Lambda_SE
                xi = fit.xi
                xi_SE = fit.xi_SE
                gamma = fit.gamma
            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = str(
                    "Fitted Exponential_2P\n(λ="
                    + str(round_to_decimals(Lambda, dec))
                    + ", γ="
                    + str(round_to_decimals(gamma, dec))
                    + ")"
                )
            xlabel = "Time - gamma"
            failures = failures - gamma
            if right_censored is not None:
                right_censored = right_censored - gamma
        gef = GeneralizedPareto_Distribution(Lambda=Lambda, Lambda_SE=Lambda_SE, xi=xi, xi_SE=xi_SE, CI=CI)

    x, y = plotting_positions(failures=failures, right_censored=right_censored, a=a)
    if show_scatter_points is True:
        x_scatter, y_scatter = xy_downsample(
            x, y, downsample_factor=downsample_scatterplot
        )
        plt.scatter(x_scatter, y_scatter, marker=".", linewidth=2, c=data_color)
    f_genpareto = lambda x: axes_transforms.generalizedpareto_forward(x, xi)
    fi_genpareto= lambda x: axes_transforms.generalizedpareto_inverse(x, xi)
    plt.gca().set_yscale( "function", functions=(f_genpareto, fi_genpareto))
    plt.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
    plt.grid(b=True, which="minor", color="k", alpha=0.08, linestyle="-")
    plt.gcf().set_size_inches(
        9, 9
    )  # adjust the figsize. This is done outside of figure creation so that layering of multiple plots is possible
    if show_fitted_distribution is True:
        gef.CDF(label=label, **kwargs)
        plt.legend(loc="upper left")
    plt.title("Probability plot\nExponential CDF")
    plt.ylabel("Fraction failing")
    plt.xlabel(
        xlabel
    )  # needs to be set after plotting the CDF to override the default 'xvals'
    probability_plot_xylims(x=x, y=y, dist="exponential", spacing=0.1)
    probability_plot_xyticks(
        yticks=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.12, right=0.94)
    return plt.gcf()


def PP_plot_parametric(
    X_dist=None,
    Y_dist=None,
    y_quantile_lines=None,
    x_quantile_lines=None,
    show_diagonal_line=False,
    downsample_scatterplot=False,
    **kwargs
):
    """
    The PP plot (probability-probability plot) consists of plotting the CDF of
    one distribution against the CDF of another distribution. If the
    distributions are similar, the PP plot will lie on the diagonal. This
    version of a PP plot is the fully parametric form in which we plot one
    distribution against another distribution. There is also a semi-parametric
    form offered in PP_plot_semiparametric.

    Parameters
    ----------
    X_dist : object
        A probability distribution object created using the
        reliability.Distributions module. The CDF of this distribution will be
        plotted along the X-axis.
    Y_dist : object
        A probability distribution object created using the
        reliability.Distributions module. The CDF of this distribution will be
        plotted along the Y-axis.
    y_quantile_lines : array, list, optional
        Starting points for the trace lines to find the X equivalent of the
        Y-quantile. Optional input. Default = None
    x_quantile_lines : array, list, optional
        Starting points for the trace lines to find the Y equivalent of the
        X-quantile. Optional input. Default = None
    show_diagonal_line : bool, optional
        If True the diagonal line will be shown on the plot. Default = False
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the PP plot is returned as an object

    Notes
    -----
    If your plot does not appear automatically, use plt.show() to show it.
    """

    if X_dist is None or Y_dist is None:
        raise ValueError(
            "X_dist and Y_dist must both be specified as probability distributions generated using the Distributions module"
        )
    if type(X_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ] or type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ]:
        raise ValueError(
            "Invalid probability distribution. X_dist and Y_dist must both be specified as probability distributions generated using the Distributions module"
        )

    # extract certain keyword arguments or specify them if they are not set
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"
    if "marker" in kwargs:
        marker = kwargs.pop("marker")
    else:
        marker = "."

    # generate plotting limits and create the PP_plot line
    dist_X_b01 = X_dist.quantile(0.01)
    dist_Y_b01 = Y_dist.quantile(0.01)
    dist_X_b99 = X_dist.quantile(0.99)
    dist_Y_b99 = Y_dist.quantile(0.99)
    xvals = np.linspace(min(dist_X_b01, dist_Y_b01), max(dist_X_b99, dist_Y_b99), 100)
    dist_X_CDF = X_dist.CDF(xvals=xvals, show_plot=False)
    dist_Y_CDF = Y_dist.CDF(xvals=xvals, show_plot=False)

    if downsample_scatterplot is not False:
        x_scatter, y_scatter = xy_downsample(
            dist_X_CDF, dist_Y_CDF, downsample_factor=downsample_scatterplot
        )
    else:
        x_scatter, y_scatter = dist_X_CDF, dist_Y_CDF
    plt.scatter(x_scatter, y_scatter, marker=marker, color=color)

    # this creates the labels for the axes using the parameters of the distributions
    X_label_str = str(X_dist.name + " CDF (" + X_dist.param_title + ")")
    Y_label_str = str(Y_dist.name + " CDF (" + Y_dist.param_title + ")")
    plt.xlabel(X_label_str)
    plt.ylabel(Y_label_str)

    ax = plt.gca()
    stick_to_y = blended_transform_factory(ax.transAxes, ax.transData)
    stick_to_x = blended_transform_factory(ax.transData, ax.transAxes)
    # this draws on the quantile lines
    if y_quantile_lines is not None:
        for q in y_quantile_lines:
            quantile = X_dist.CDF(xvals=Y_dist.quantile(q), show_plot=False)
            plt.plot(
                [-1000, quantile, quantile], [q, q, -1000], color="blue", linewidth=0.5
            )
            plt.text(
                s=str(round(quantile, 2)),
                x=quantile,
                y=0,
                transform=stick_to_x,
                clip_on=True,
            )
            plt.text(s=str(q), x=0, y=q, transform=stick_to_y, clip_on=True)

    if x_quantile_lines is not None:
        for q in x_quantile_lines:
            quantile = Y_dist.CDF(xvals=X_dist.quantile(q), show_plot=False)
            plt.plot(
                [q, q, -1000], [-1000, quantile, quantile], color="red", linewidth=0.5
            )
            plt.text(
                s=str(round(quantile, 2)),
                x=0,
                y=quantile,
                transform=stick_to_y,
                clip_on=True,
            )
            plt.text(s=str(q), x=q, y=0, transform=stick_to_x, clip_on=True)

    if show_diagonal_line is True:
        plt.plot([0, 1], [0, 1], color="red", alpha=0.7, label="Y = X")
    plt.title("Probability-Probability plot\nParametric")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return plt.gcf()


def QQ_plot_parametric(
    X_dist=None,
    Y_dist=None,
    show_fitted_lines=True,
    show_diagonal_line=False,
    downsample_scatterplot=False,
    **kwargs
):
    """
    A QQ plot (quantile-quantile plot) consists of plotting failure units vs
    failure units for shared quantiles. A quantile is simply the fraction
    failing (ranging from 0 to 1). To generate this plot we calculate the
    failure units (these may be units of time, strength, cycles, landings, etc.)
    at which a certain fraction has failed (0.01,0.02,0.03...0.99) for each
    distribution and plot them together. The time (or any other failure unit) at
    which a given fraction has failed is found using the inverse survival
    function. If the distributions are similar in shape, then the QQ plot should
    be a reasonably straight line. By plotting the failure times at equal
    quantiles for each distribution we can obtain a conversion between the two
    distributions which is useful for Field-to-Test conversions that are
    necessary during accelerated life testing (ALT).

    Parameters
    ----------
    X_dist : object
        A probability distribution object created using the
        reliability.Distributions module. The failure times at given quantiles
        from this distribution will be plotted along the X-axis.
    Y_dist : object
        A probability distribution object created using the
        reliability.Distributions module. The failure times at given quantiles
        from this distribution will be plotted along the Y-axis.
    show_fitted_lines : bool
        Default = True. These are the Y=mX and Y=mX+c lines of best fit.
    show_diagonal_line : bool
        Default = False. If True the diagonal line will be shown on the plot.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    model_parameters : list
        [m,m1,c1] - these are the values for the lines of best fit. m is used in
        Y=m.X, and m1 and c1 are used in Y=m1.X+c1

    Notes
    -----
    If your plot does not appear automatically, use plt.show() to show it.
    """

    if X_dist is None or Y_dist is None:
        raise ValueError(
            "dist_X and dist_Y must both be specified as probability distributions generated using the Distributions module"
        )
    if type(X_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ] or type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ]:
        raise ValueError(
            "dist_X and dist_Y must both be specified as probability distributions generated using the Distributions module"
        )
    xvals = np.linspace(0.01, 0.99, 100)

    # extract certain keyword arguments or specify them if they are not set
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"
    if "marker" in kwargs:
        marker = kwargs.pop("marker")
    else:
        marker = "."
    # calculate the failure times at the given quantiles
    dist_X_ISF = []
    dist_Y_ISF = []
    for x in xvals:
        dist_X_ISF.append(X_dist.inverse_SF(float(x)))
        dist_Y_ISF.append(Y_dist.inverse_SF(float(x)))
    dist_X_ISF = np.array(dist_X_ISF)
    dist_Y_ISF = np.array(dist_Y_ISF)

    if downsample_scatterplot is not False:
        x_scatter, y_scatter = xy_downsample(
            dist_X_ISF, dist_Y_ISF, downsample_factor=downsample_scatterplot
        )
    else:
        x_scatter, y_scatter = dist_X_ISF, dist_Y_ISF
    plt.scatter(x_scatter, y_scatter, marker=marker, color=color)

    # fit lines and generate text for equations to go in legend
    max_value = max(max(dist_X_ISF), max(dist_Y_ISF))
    x = dist_X_ISF[:, np.newaxis]
    y = dist_Y_ISF
    deg1 = np.polyfit(dist_X_ISF, dist_Y_ISF, deg=1)  # fit y=mx+c
    m = np.linalg.lstsq(x, y, rcond=-1)[0][0]  # fit y=mx
    x_fit = np.linspace(-max_value, max_value * 2, 100)
    y_fit = m * x_fit
    text_str = str("Y = " + str(round(m, 3)) + " X")
    y1_fit = deg1[0] * x_fit + deg1[1]
    if deg1[1] < 0:
        text_str1 = str(
            "Y = " + str(round(deg1[0], 3)) + " X" + " - " + str(round(-1 * deg1[1], 3))
        )
    else:
        text_str1 = str(
            "Y = " + str(round(deg1[0], 3)) + " X" + " + " + str(round(deg1[1], 3))
        )

    if show_diagonal_line is True:
        plt.plot(
            [-max_value, max_value * 2],
            [-max_value, max_value * 2],
            color="red",
            alpha=0.7,
            label="Y = X",
        )
    if show_fitted_lines is True:
        plt.plot(x_fit, y_fit, color="darkorange", alpha=0.5, label=text_str)
        plt.plot(x_fit, y1_fit, color="green", alpha=0.5, label=text_str1)
        plt.legend(title="Fitted lines:")

    # this creates the labels for the axes using the parameters of the distributions
    X_label_str = str(X_dist.name + " Quantiles (" + X_dist.param_title + ")")
    Y_label_str = str(Y_dist.name + " Quantiles (" + Y_dist.param_title + ")")
    plt.xlabel(X_label_str)
    plt.ylabel(Y_label_str)
    plt.title("Quantile-Quantile plot\nParametric")
    # plt.axis('equal')
    xmin, xmax = min(dist_X_ISF), max(dist_X_ISF)
    xdelta = xmax - xmin
    ymin, ymax = min(dist_Y_ISF), max(dist_Y_ISF)
    ydelta = ymax - ymin

    plt.xlim([xmin - 0.05 * xdelta, xmax + 0.05 * xdelta])
    plt.ylim([ymin - 0.05 * ydelta, ymax + 0.05 * ydelta])
    return [m, deg1[0], deg1[1]]


def PP_plot_semiparametric(
    X_data_failures=None,
    X_data_right_censored=None,
    Y_dist=None,
    show_diagonal_line=True,
    method="KM",
    downsample_scatterplot=False,
    **kwargs
):
    """
    A PP plot (probability-probability plot) consists of plotting the CDF of one
    distribution against the CDF of another distribution. If we have both
    distributions we can use the function PP_plot_parametric. This function is
    for when we want to compare a fitted distribution to an empirical
    distribution for a given set of data. If the fitted distribution is a good
    fit the PP plot will lie on the diagonal line. The main purpose of this type
    of plot is to assess the goodness of fit in a graphical way. To create a
    semi-parametric PP plot, we must provide the failure data and the method
    ('KM' for Kaplan-Meier, 'NA' for Nelson-Aalen, 'RA' for Rank Adjustment) to
    estimate the empirical CDF, and we must also provide the parametric
    distribution for the parametric CDF. The failure times are the limiting
    values here so the parametric CDF is only calculated at the failure times
    since that is the result from the empirical CDF.

    Parameters
    ----------
    X_data_failures : array, list
        The failure times.
    X_data_right_censored : array, list, optional
        The right censored failure times. Optional input.
    Y_dist : object
        A probability distribution created using the
        reliability.Distributions module. The CDF of this distribution will be
        plotted along the Y-axis.
    method : str, optional
        Must be 'KM', 'NA', or 'RA' for Kaplan-Meier, Nelson-Aalen, and Rank
        Adjustment respectively. Default = 'KM'.
    show_diagonal_line : bool
        Default = True. If True the diagonal line will be shown on the plot.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    figure : object
        The figure handle of the PP plot is returned as an object

    Notes
    -----
    The empirical CDF also accepts X_data_right_censored just as Kaplan-Meier,
    Nelson-Aalen, and Rank Adjustment will also accept right censored data.

    If your plot does not appear automatically, use plt.show() to show it.
    """

    if X_data_failures is None or Y_dist is None:
        raise ValueError(
            "X_data_failures and Y_dist must both be specified. X_data_failures can be an array or list of failure times. Y_dist must be a probability distribution generated using the Distributions module"
        )
    if type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ] or type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ]:
        raise ValueError(
            "Y_dist must be specified as a probability distribution generated using the Distributions module"
        )
    if type(X_data_failures) == list:
        X_data_failures = np.sort(np.array(X_data_failures))
    elif type(X_data_failures) == np.ndarray:
        X_data_failures = np.sort(X_data_failures)
    else:
        raise ValueError("X_data_failures must be an array or list")
    if type(X_data_right_censored) == list:
        X_data_right_censored = np.sort(np.array(X_data_right_censored))
    elif type(X_data_right_censored) == np.ndarray:
        X_data_right_censored = np.sort(X_data_right_censored)
    elif X_data_right_censored is None:
        pass
    else:
        raise ValueError("X_data_right_censored must be an array or list")
    # extract certain keyword arguments or specify them if they are not set
    if "a" in kwargs:  # rank adjustment heuristic
        a = kwargs.pop("a")
    else:
        a = None
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"
    if "marker" in kwargs:
        marker = kwargs.pop("marker")
    else:
        marker = "."
    if method == "KM":
        KM = KaplanMeier(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
        )
        df = KM.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Kaplan-Meier Estimate"].values)
        xlabel = "Empirical CDF (Kaplan-Meier estimate)"
    elif method == "NA":
        NA = NelsonAalen(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
        )
        df = NA.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Nelson-Aalen Estimate"].values)
        xlabel = "Empirical CDF (Nelson-Aalen estimate)"
    elif method == "RA":
        RA = RankAdjustment(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
            a=a,
        )
        df = RA.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Rank Adjustment Estimate"].values)
        xlabel = "Empirical CDF (Rank Adjustment estimate)"
    else:
        raise ValueError(
            'method must be "KM" for Kaplan-meier, "NA" for Nelson-Aalen, or "RA" for Rank Adjustment. Default is KM'
        )

    if show_diagonal_line is True:
        plt.plot([-1, 2], [-1, 2], color="red", alpha=0.7)

    CDF = Y_dist.CDF(X_data_failures, show_plot=False)

    if downsample_scatterplot is not False:
        x_scatter, y_scatter = xy_downsample(
            ecdf, CDF, downsample_factor=downsample_scatterplot
        )
    else:
        x_scatter, y_scatter = ecdf, CDF
    plt.scatter(x_scatter, y_scatter, marker=marker, color=color)

    Y_label_str = str(Y_dist.name + " CDF (" + Y_dist.param_title + ")")
    plt.ylabel(Y_label_str)
    plt.xlabel(xlabel)
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Probability-Probability Plot\nSemi-parametric")
    return plt.gcf()


def QQ_plot_semiparametric(
    X_data_failures=None,
    X_data_right_censored=None,
    Y_dist=None,
    show_fitted_lines=True,
    show_diagonal_line=False,
    method="KM",
    downsample_scatterplot=False,
    **kwargs
):
    """
    A QQ plot (quantile-quantile plot) consists of plotting failure units vs
    failure units for shared quantiles. A quantile is simply the fraction
    failing (ranging from 0 to 1). When we have two parametric distributions we
    can plot the failure times for common quanitles against one another using
    QQ_plot_parametric. QQ_plot_semiparametric is a semiparametric form of a
    QQ plot in which we obtain theoretical quantiles using a non-parametric
    estimate and a specified distribution. To generate this plot we begin with
    the failure units (these may be units of time, strength, cycles, landings,
    etc.). We then obtain an emprical CDF using either Kaplan-Meier,
    Nelson-Aalen, or Rank Adjustment. The empirical CDF gives us the quantiles
    we will use to equate the actual and theoretical failure times. Once we have
    the empirical CDF, we use the inverse survival function of the specified
    distribution to obtain the theoretical failure times and then plot the
    actual and theoretical failure times together. If the specified distribution
    is a good fit, then the QQ plot should be a reasonably straight line along
    the diagonal. The primary purpose of this plot is as a graphical goodness of
    fit test.

    Parameters
    ----------
    X_data_failures : list, array
        The failure times. These will be plotted along the X-axis.
    X_data_right_censored : list, array, optional
        The right censored failure times. Optional input.
    Y_dist : object
        A probability distribution created using the reliability.Distributions
        module. The quantiles of this distribution will be plotted along the
        Y-axis.
    method : str
        Must be either 'KM', 'NA', or 'RA' for Kaplan-Meier, Nelson-Aalen, and
        Rank-Adjustment respectively. Default = 'KM'.
    show_fitted_lines : bool
        Default = True. These are the Y=m.X and Y=m.X+c lines of best fit.
    show_diagonal_line : bool
        Default = False. If True the diagonal line will be shown on the plot.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    model_parameters : list
        [m,m1,c1] - these are the values for the lines of best fit. m is used in
        Y=m.X, and m1 and c1 are used in Y=m1.X+c1

    Notes
    -----
    If your plot does not appear automatically, use plt.show() to show it.
    """

    if X_data_failures is None or Y_dist is None:
        raise ValueError(
            "X_data_failures and Y_dist must both be specified. X_data_failures can be an array or list of failure times. Y_dist must be a probability distribution generated using the Distributions module"
        )
    if type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ] or type(Y_dist) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ]:
        raise ValueError(
            "Y_dist must be specified as a probability distribution generated using the Distributions module"
        )
    if type(X_data_failures) == list:
        X_data_failures = np.sort(np.array(X_data_failures))
    elif type(X_data_failures) == np.ndarray:
        X_data_failures = np.sort(X_data_failures)
    else:
        raise ValueError("X_data_failures must be an array or list")
    if type(X_data_right_censored) == list:
        X_data_right_censored = np.sort(np.array(X_data_right_censored))
    elif type(X_data_right_censored) == np.ndarray:
        X_data_right_censored = np.sort(X_data_right_censored)
    elif X_data_right_censored is None:
        pass
    else:
        raise ValueError("X_data_right_censored must be an array or list")
    if "a" in kwargs:  # rank adjustment heuristic
        a = kwargs.pop("a")
    else:
        a = None
    # extract certain keyword arguments or specify them if they are not set
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"
    if "marker" in kwargs:
        marker = kwargs.pop("marker")
    else:
        marker = "."

    if method == "KM":
        KM = KaplanMeier(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
        )
        df = KM.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Kaplan-Meier Estimate"].values)
        method_str = "Kaplan-Meier"
    elif method == "NA":
        NA = NelsonAalen(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
        )
        df = NA.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Nelson-Aalen Estimate"].values)
        method_str = "Nelson-Aalen"
    elif method == "RA":
        RA = RankAdjustment(
            failures=X_data_failures,
            right_censored=X_data_right_censored,
            show_plot=False,
            print_results=False,
            a=a,
        )
        df = RA.results
        failure_rows = df.loc[df["Censoring code (censored=0)"] == 1.0]
        ecdf = 1 - np.array(failure_rows["Rank Adjustment Estimate"].values)
        method_str = "Rank Adjustment"
    else:
        raise ValueError(
            'method must be "KM" for Kaplan-meier, "NA" for Nelson-Aalen, or "RA" for Rank Adjustment. Default is KM'
        )

    # calculate the failure times at the given quantiles
    dist_Y_ISF = []
    for q in ecdf:
        dist_Y_ISF.append(Y_dist.inverse_SF(float(q)))
    dist_Y_ISF = np.array(dist_Y_ISF[::-1])
    dist_Y_ISF[dist_Y_ISF == -np.inf] = 0

    if downsample_scatterplot is not False:
        x_scatter, y_scatter = xy_downsample(
            X_data_failures, dist_Y_ISF, downsample_factor=downsample_scatterplot
        )
    else:
        x_scatter, y_scatter = X_data_failures, dist_Y_ISF
    plt.scatter(x_scatter, y_scatter, marker=marker, color=color)

    plt.ylabel(
        str(
            "Theoretical Quantiles based on\n"
            + method_str
            + " estimate and "
            + Y_dist.name
            + " distribution"
        )
    )
    plt.xlabel("Actual Quantiles")
    plt.axis("square")
    max_value = max(max(dist_Y_ISF), max(X_data_failures))
    min_value = min(min(dist_Y_ISF), min(X_data_failures))
    if show_diagonal_line is True:
        plt.plot(
            [-max_value, max_value * 2],
            [-max_value, max_value * 2],
            color="red",
            alpha=0.7,
            label="Y = X",
        )

    # fit lines and generate text for equations to go in legend
    y = dist_Y_ISF[:, np.newaxis]
    x = X_data_failures[:, np.newaxis]
    deg1 = np.polyfit(X_data_failures, dist_Y_ISF, deg=1)  # fit y=mx+c
    m = np.linalg.lstsq(x, y, rcond=-1)[0][0][0]  # fit y=mx
    x_fit = np.linspace(-max_value, max_value * 2, 100)
    y_fit = m * x_fit
    text_str = str("Y = " + str(round(m, 3)) + " X")
    y1_fit = deg1[0] * x_fit + deg1[1]
    if deg1[1] < 0:
        text_str1 = str(
            "Y = " + str(round(deg1[0], 3)) + " X" + " - " + str(round(-1 * deg1[1], 3))
        )
    else:
        text_str1 = str(
            "Y = " + str(round(deg1[0], 3)) + " X" + " + " + str(round(deg1[1], 3))
        )
    if show_fitted_lines is True:
        plt.plot(x_fit, y_fit, color="red", alpha=0.5, label=text_str)
        plt.plot(x_fit, y1_fit, color="green", alpha=0.5, label=text_str1)
        plt.legend(title="Fitted lines:")
    delta = max_value - min_value
    lims = [min_value - 0.05 * delta, max_value + 0.05 * delta]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Quantile-Quantile Plot\nSemi-parametric")
    return [m, deg1[0], deg1[1]]


def plot_points(
    failures=None,
    right_censored=None,
    func="CDF",
    a=None,
    downsample_scatterplot=False,
    **kwargs
):
    """
    Plots the failure points as a scatter plot based on the plotting positions.
    This is similar to a probability plot, just without the axes scaling or the
    fitted distribution. It may be used to overlay the failure points with a
    fitted distribution on either the PDF, CDF, SF, HF, or CHF. If you choose to
    plot the points for PDF or HF the points will not form a smooth curve as
    this process requires integration of discrete points which leads to a
    discontinuous plot. The PDF and HF points are correct but not as useful as
    CDF, SF, and CHF.

    Parameters
    ----------
    failures : array, list
        The failure times. Minimum number of points allowed is 1.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    func : str
        The distribution function to plot. Choose either 'PDF,'CDF','SF','HF',
        or 'CHF'. Default = 'CDF'.
    a : float, int
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Default is a=0.3 which is the median rank method (same
        as the default in Minitab). For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is False which will
        result in no downsampling. This functionality makes plotting faster when
        there are very large numbers of points. It only affects the scatterplot
        not the calculations.
    kwargs
        Keyword arguments for the scatter plot. Defaults are set for color='k'
        and marker='.' These defaults can be changed using kwargs.

    Returns
    -------
    None

    Notes
    -----
    It is recommended that plot_points be used in conjunction with one of the
    plotting methods from a distribution (see the example below).

    .. code:: python

        from reliability.Fitters import Fit_Lognormal_2P
        from reliability.Probability_plotting import plot_points
        import matplotlib.pyplot as plt
        data = [8.0, 10.2, 7.1, 5.3, 8.5, 15.4, 17.7, 5.4, 5.8, 11.7, 4.4, 18.1, 8.5, 6.6, 9.7, 13.7, 8.2, 15.3, 2.9, 4.3]
        fitted_dist = Fit_Lognormal_2P(failures=data,show_probability_plot=False,print_results=False) #fit the Lognormal distribution to the failure data
        plot_points(failures=data,func='SF') #plot the failure points on the scatter plot
        fitted_dist.distribution.SF() #plot the distribution
        plt.show()
    """
    if failures is None or len(failures) < 1:
        raise ValueError(
            "failures must be an array or list with at least 1 failure time"
        )

    x, y = plotting_positions(
        failures=failures, right_censored=right_censored, a=a
    )  # get the plotting positions
    y = np.array(y)
    x = np.array(x)

    if func in [
        "pdf",
        "PDF",
    ]:  # the output of this looks messy because the derivative is of discrete points and not a continuous function
        dy = np.diff(np.hstack([[0], y]))
        dx = np.diff(np.hstack([[0], x]))
        y_adjusted = abs(dy / dx)  # PDF = dy/dx CDF
    elif func in ["cdf", "CDF"]:
        y_adjusted = y
    elif func in ["sf", "SF"]:
        y_adjusted = 1 - y  # SF = 1 - CDF
    elif func in [
        "hf",
        "HF",
    ]:  # the output of this looks messy because the derivative is of discrete points and not a continuous function
        dy = np.diff(np.hstack([[0], -np.log(1 - y)]))
        dx = np.diff(np.hstack([[0], x]))
        y_adjusted = abs(dy / dx)  # HF = dy/dx CHF
    elif func in ["chf", "CHF"]:
        y_adjusted = -np.log(1 - y)  # CHF = -ln(SF)
    else:
        raise ValueError("func must be either PDF, CDF, SF, HF, or CHF. Default is CDF")

    # set plotting defaults for keywords
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"
    if "marker" in kwargs:
        marker = kwargs.pop("marker")
    else:
        marker = "."

    # check the previous axes limits
    # this checks if there was a previous plot. If the lims were 0,1 and 0,1 then there probably wasn't.
    xlims = plt.xlim(auto=None)  # get previous xlim
    ylims = plt.ylim(auto=None)  # get previous ylim

    x_scatter, y_scatter = xy_downsample(
        x, y_adjusted, downsample_factor=downsample_scatterplot
    )
    plt.scatter(x_scatter, y_scatter, marker=marker, color=color, **kwargs)

    if xlims != (0, 1) or ylims != (0, 1):
        # Restore the previous limits if there were previous limits
        plt.xlim(*xlims, auto=None)
        plt.ylim(*ylims, auto=None)
