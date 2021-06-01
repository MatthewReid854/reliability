"""
Other Functions

This is a collection of several other functions that did not otherwise fit
within their own module. Included functions are:
stress_strength - stress-strength interference for any distributions (uses
    numerical integration).
stress_strength_normal - stress-strength interference two normal distributions
    (uses empirical method).
similar_distributions - finds the parameters of distributions that are similar
    to the input distribution and plots the results.
make_right_censored_data - a simple tool to right censor a complete dataset
    based on a threshold. Primarily used for testing Fitters when some right
    censored data is needed.
make_ALT_data - a tool to generate data for fitting ALT models. Primarily used
    for testing ALT_Fitters.
histogram - generates a histogram with optimal bin width and has an option to
    shade some bins white above a chosen threshold.
crosshairs - adds x,y crosshairs to plots based on mouse position
distribution_explorer - generates an interactive window to explore probability
    distributions using sliders for their parameters
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mplcursors import cursor
import warnings
from reliability.Distributions import (
    Weibull_Distribution,
    Normal_Distribution,
    Lognormal_Distribution,
    Exponential_Distribution,
    Gamma_Distribution,
    Beta_Distribution,
    Loglogistic_Distribution,
    Gumbel_Distribution,
)
from reliability.Fitters import Fit_Everything
from reliability.Utils import colorprint, round_to_decimals
from matplotlib.widgets import Slider, RadioButtons
import scipy.stats as ss
import time


def stress_strength(stress, strength, show_plot=True, print_results=True, warn=True):
    """
    Given the probability distributions for stress and strength, this module
    will find the probability of failure due to stress-strength interference.
    Failure is defined as when stress>strength. The calculation is achieved
    using numerical integration.

    Parameters
    ----------
    stress : object
        A probability distribution from the Distributions module
    strength : object
        A probability distribution from the Distributions module
    show_plot : bool, optional
        If True the distribution plot will be shown. Default = True.
    print_results : bool, optional
        If True, the results will be printed to console. Default = True.
    warn : bool, optional
        A warning will be issued if both stress and strength are Normal as you
        should use stress_strength_normal. You can supress this using warn=False
        A warning will be issued if the stress.mean > strength.mean as the user
        may have assigned the distributions to the wrong variables. You can
        supress this using warn=False. Default = True

    Returns
    -------
    probability_of_failure : float
        The probability of failure due to stress-strength interference

    Notes
    -----
    Example usage:

    .. code:: python

        from reliability.Distributions import Weibull_Distribution, Gamma_Distribution
        stress = Weibull_Distribution(alpha=2,beta=3,gamma=1)
        strength = Gamma_Distribution(alpha=2,beta=3,gamma=3)
        stress_strength(stress=stress, strength=strength)
    """

    if type(stress) not in [
        Weibull_Distribution,
        Normal_Distribution,
        Lognormal_Distribution,
        Exponential_Distribution,
        Gamma_Distribution,
        Beta_Distribution,
        Loglogistic_Distribution,
        Gumbel_Distribution,
    ] or type(strength) not in [
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
            "Stress and Strength must both be probability distributions. First define the distribution using reliability.Distributions.___"
        )
    if (
        type(stress) == Normal_Distribution
        and type(strength) == Normal_Distribution
        and warn is True
    ):  # supress the warning by setting warn=False
        colorprint(
            "WARNING: If strength and stress are both Normal distributions, it is more accurate to use the exact formula. The exact formula is supported in the function stress_strength_normal. To supress this warning set warn=False",
            text_color="red",
        )
    if stress.mean > strength.mean and warn == True:
        colorprint(
            "WARNING: The mean of the stress distribution is above the mean of the strength distribution. Please check you have assigned stress and strength to the correct variables. To supress this warning set warn=False",
            text_color="red",
        )

    x = np.linspace(
        min(stress.quantile(1e-8), strength.quantile(1e-8)),
        max(strength.quantile(1 - 1e-8), stress.quantile(1 - 1e-8)),
        1000,
    )  # we take the min and max here since there may be cases when stress > strength
    prob_of_failure = np.trapz(
        strength.PDF(x, show_plot=False) * stress.SF(x, show_plot=False), x
    )

    if show_plot is True:
        xlims = plt.xlim(auto=None)
        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        if abs(xmin) < (xmax - xmin) / 4:
            xmin = 0  # if the lower bound on xmin is near zero (relative to the entire range) then just make it zero
        if type(stress) == Beta_Distribution:
            xmin = 0
        if type(strength) == Beta_Distribution:
            xmax = 1
        xvals = np.linspace(xmin, xmax, 10000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
        Y = [
            (min(strength_PDF[i], stress_PDF[i])) for i in range(len(xvals))
        ]  # finds the lower of the two lines which is used as the upper boundary for fill_between
        plt.plot(xvals, stress_PDF, label="Stress")
        plt.plot(xvals, strength_PDF, label="Strength")
        intercept_idx = Y.index(max(Y))
        plt.fill_between(
            xvals,
            np.zeros_like(xvals),
            Y,
            color="salmon",
            alpha=1,
            linewidth=0,
            linestyle="--",
        )
        plt.fill_between(
            xvals[0:intercept_idx],
            strength_PDF[0:intercept_idx],
            stress_PDF[0:intercept_idx],
            color="steelblue",
            alpha=0.3,
            linewidth=0,
            linestyle="--",
        )
        plt.fill_between(
            xvals[intercept_idx::],
            stress_PDF[intercept_idx::],
            strength_PDF[intercept_idx::],
            color="darkorange",
            alpha=0.3,
            linewidth=0,
            linestyle="--",
        )
        failure_text = str(
            "Probability of\nfailure = " + str(round_to_decimals(prob_of_failure, 4))
        )
        plt.legend(title=failure_text)
        plt.title("Stress - Strength Interference Plot")
        plt.ylabel("Probability Density")
        plt.xlabel("Stress and Strength Units")
        plt.subplots_adjust(left=0.16)
        if xlims != (0, 1):
            plt.xlim(min(stress.b5, xlims[0]), max(strength.b95, xlims[1]), auto=None)
        else:
            plt.xlim(stress.b5, strength.b95, auto=None)
        plt.ylim(bottom=0, auto=None)

    if print_results is True:
        colorprint("Stress - Strength Interference", bold=True, underline=True)
        print("Stress Distribution:", stress.param_title_long)
        print("Strength Distribution:", strength.param_title_long)
        print(
            "Probability of failure (stress > strength):",
            round_to_decimals(prob_of_failure * 100),
            "%",
        )

    return prob_of_failure


def stress_strength_normal(
    stress, strength, show_plot=True, print_results=True, warn=True
):
    """
    Given the probability distributions for stress and strength, this module
    will find the probability of failure due to stress-strength interference.
    Failure is defined as when stress>strength. Uses the exact formula method
    which is only valid for two Normal Distributions. If you have distributions
    that are not both Normal Distributions, use the function stress_strength.

    Parameters
    ----------
    stress : object
        A probability distribution from the Distributions module
    strength : object
        A probability distribution from the Distributions module
    show_plot : bool, optional
        If True the distribution plot will be shown. Default = True.
    print_results : bool, optional
        If True, the results will be printed to console. Default = True.
    warn : bool, optional
        A warning will be issued if the stress.mean > strength.mean as the user
        may have assigned the distributions to the wrong variables. You can
        supress this using warn=False. Default = True

    Returns
    -------
    probability_of_failure : float
        The probability of failure due to stress-strength interference

    """
    if type(stress) is not Normal_Distribution:
        raise ValueError(
            "Both stress and strength must be a Normal_Distribution. If you need another distribution then use stress_strength rather than stress_strength_normal"
        )
    if type(strength) is not Normal_Distribution:
        raise ValueError(
            "Both stress and strength must be a Normal_Distribution. If you need another distribution then use stress_strength rather than stress_strength_normal"
        )
    if stress.mean > strength.mean and warn == True:
        colorprint(
            "WARNING: The mean of the stress distribution is above the mean of the strength distribution. Please check you have assigned stress and strength to the correct variables. To supress this warning set warn=False",
            text_color="red",
        )

    sigma_strength = strength.sigma
    mu_strength = strength.mu
    sigma_stress = stress.sigma
    mu_stress = stress.mu
    prob_of_failure = ss.norm.cdf(
        -(mu_strength - mu_stress) / ((sigma_strength ** 2 + sigma_stress ** 2) ** 0.5)
    )

    if show_plot is True:
        xlims = plt.xlim(auto=None)
        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        xvals = np.linspace(xmin, xmax, 1000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
        plt.plot(xvals, stress_PDF, label="Stress")
        plt.plot(xvals, strength_PDF, label="Strength")
        Y = [
            (min(strength_PDF[i], stress_PDF[i])) for i in range(len(xvals))
        ]  # finds the lower of the two lines which is used as the upper boundary for fill_between
        intercept_idx = Y.index(max(Y))
        plt.fill_between(
            xvals,
            np.zeros_like(xvals),
            Y,
            color="salmon",
            alpha=1,
            linewidth=0,
            linestyle="--",
        )
        plt.fill_between(
            xvals[0:intercept_idx],
            strength_PDF[0:intercept_idx],
            stress_PDF[0:intercept_idx],
            color="steelblue",
            alpha=0.3,
            linewidth=0,
            linestyle="--",
        )
        plt.fill_between(
            xvals[intercept_idx::],
            stress_PDF[intercept_idx::],
            strength_PDF[intercept_idx::],
            color="darkorange",
            alpha=0.3,
            linewidth=0,
            linestyle="--",
        )
        failure_text = str(
            "Probability of\nfailure = " + str(round_to_decimals(prob_of_failure, 4))
        )
        plt.legend(title=failure_text)
        plt.title("Stress - Strength Interference Plot")
        plt.ylabel("Probability Density")
        plt.xlabel("Stress and Strength Units")
        plt.subplots_adjust(left=0.15, right=0.93)
        if xlims != (0, 1):
            plt.xlim(min(stress.b5, xlims[0]), max(strength.b95, xlims[1]), auto=None)
        else:
            plt.xlim(stress.b5, strength.b95, auto=None)
        plt.ylim(bottom=0, auto=None)

    if print_results is True:
        colorprint("Stress - Strength Interference", bold=True, underline=True)
        print("Stress Distribution:", stress.param_title_long)
        print("Strength Distribution:", strength.param_title_long)
        print(
            "Probability of failure (stress > strength):",
            round_to_decimals(prob_of_failure * 100),
            "%",
        )

    return prob_of_failure


class similar_distributions:
    """
    This is a tool to find similar distributions when given an input
    distribution. It is useful to see how similar one distribution is to
    another. For example, you may look at a Weibull distribution and think it
    looks like a Normal distribution. Using this tool you can determine the
    parameters of the Normal distribution that most closely matches your Weibull
    distribution.

    Parameters
    ----------
    distribution : object
        A distribution object created using the reliability.Distributions
        module.
    include_location_shifted : bool, optional
        When set to True it will include Weibull_3P, Lognormal_3P, Gamma_3P,
        Exponential_2P, and Loglogistic_3P. Default = True
    show_plot : bool, optional
        If True it will show the PDF and CDF of the input distributions and the
        most similar distributions. Default = True.
    print_results : bool, optional
        If True the results will be printed to the console. Default = True.
    number_of_distributions_to_show : int, optional
        The number of similar distributions to show. Default = 3. If the number
        specified exceeds the number available (typically 10), then the number
        specified will automatically be reduced. Must be > 1.

    Returns
    -------
    results : array
        An array of distributions objects ranked in order of best fit.
    most_similar_distribution : object
        A distribution object. This is the first item from results.

    Notes
    -----
    The following example shows the distributions most similar to the input
    Weibull Distribution.

    .. code:: python

        from reliability.Distributions import Weibull_Distribution
        from reliability.Other_functions import similar_distributions
        dist = Weibull_Distribution(alpha=50,beta=3.3)
        similar_distributions(distribution=dist)
    """

    def __init__(
        self,
        distribution,
        include_location_shifted=True,
        show_plot=True,
        print_results=True,
        number_of_distributions_to_show=3,
    ):
        # ensure the input is a distribution object
        if type(distribution) not in [
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
                "distribution must be a probability distribution object from the reliability.Distributions module. First define the distribution using Reliability.Distributions.___"
            )

        if (
            type(number_of_distributions_to_show) is not int
            or number_of_distributions_to_show < 2
        ):
            raise ValueError(
                "number_of_distributions_to_show must be an integer greater than 1"
            )

        # sample the CDF from 0.001 to 0.999. These samples will be used to fit all other distributions.
        RVS = distribution.quantile(np.linspace(0.001, 0.999, 698))
        # 698 samples is the ideal number for the points to align. Evidenced using plot_points.

        # filter out negative values
        RVS_filtered = []
        negative_values_error = False
        for item in RVS:
            if item > 0:
                RVS_filtered.append(item)
            else:
                negative_values_error = True

        if len(RVS_filtered) < 175:
            raise ValueError(
                "The input distribution has more than 75% of its area in the negative domain (x<0). Comparison with distributions bounded by the positive domain (x>0) is not possible."
            )

        if negative_values_error is True:
            colorprint(
                "WARNING: The input distribution has non-negligible area for x<0. Samples from this region have been discarded to enable other distributions to be fitted.",
                text_color="red",
            )

        if include_location_shifted is False:
            # fit all distributions excluding location shifted distributions to the filtered samples
            fitted_results = Fit_Everything(
                failures=RVS_filtered,
                exclude=[
                    "Weibull_3P",
                    "Lognormal_3P",
                    "Exponential_2P",
                    "Gamma_3P",
                    "Loglogistic_3P",
                ],
                print_results=False,
                show_probability_plot=False,
                show_histogram_plot=False,
                show_PP_plot=False,
                show_best_distribution_probability_plot=False,
            )
        else:
            # fit all distributions to the filtered samples
            fitted_results = Fit_Everything(
                failures=RVS_filtered,
                print_results=False,
                show_probability_plot=False,
                show_histogram_plot=False,
                show_PP_plot=False,
                show_best_distribution_probability_plot=False,
            )
        ranked_distributions = list(fitted_results.results.Distribution.values)

        # removes the fitted version of the original distribution
        if distribution.name2 in ranked_distributions:
            ranked_distributions.remove(distribution.name2)

        ranked_distributions_objects = []
        ranked_distributions_labels = []
        sigfig = 2
        for dist_name in ranked_distributions:
            if dist_name == "Weibull_2P":
                ranked_distributions_objects.append(
                    Weibull_Distribution(
                        alpha=fitted_results.Weibull_2P_alpha,
                        beta=fitted_results.Weibull_2P_beta,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Weibull_2P (α="
                        + str(round(fitted_results.Weibull_2P_alpha, sigfig))
                        + ",β="
                        + str(round(fitted_results.Weibull_2P_beta, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Gamma_2P":
                ranked_distributions_objects.append(
                    Gamma_Distribution(
                        alpha=fitted_results.Gamma_2P_alpha,
                        beta=fitted_results.Gamma_2P_beta,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Gamma_2P (α="
                        + str(round(fitted_results.Gamma_2P_alpha, sigfig))
                        + ",β="
                        + str(round(fitted_results.Gamma_2P_beta, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Normal_2P":
                ranked_distributions_objects.append(
                    Normal_Distribution(
                        mu=fitted_results.Normal_2P_mu,
                        sigma=fitted_results.Normal_2P_sigma,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Normal_2P (μ="
                        + str(round(fitted_results.Normal_2P_mu, sigfig))
                        + ",σ="
                        + str(round(fitted_results.Normal_2P_sigma, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Lognormal_2P":
                ranked_distributions_objects.append(
                    Lognormal_Distribution(
                        mu=fitted_results.Lognormal_2P_mu,
                        sigma=fitted_results.Lognormal_2P_sigma,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Lognormal_2P (μ="
                        + str(round(fitted_results.Lognormal_2P_mu, sigfig))
                        + ",σ="
                        + str(round(fitted_results.Lognormal_2P_sigma, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Exponential_1P":
                ranked_distributions_objects.append(
                    Exponential_Distribution(
                        Lambda=fitted_results.Exponential_1P_lambda
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Exponential_1P (lambda="
                        + str(round(fitted_results.Exponential_1P_lambda, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Beta_2P":
                ranked_distributions_objects.append(
                    Beta_Distribution(
                        alpha=fitted_results.Beta_2P_alpha,
                        beta=fitted_results.Beta_2P_beta,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Beta_2P (α="
                        + str(round(fitted_results.Beta_2P_alpha, sigfig))
                        + ",β="
                        + str(round(fitted_results.Beta_2P_beta, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Loglogistic_2P":
                ranked_distributions_objects.append(
                    Loglogistic_Distribution(
                        alpha=fitted_results.Loglogistic_2P_alpha,
                        beta=fitted_results.Loglogistic_2P_beta,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Loglogistic_2P (α="
                        + str(round(fitted_results.Loglogistic_2P_alpha, sigfig))
                        + ",β="
                        + str(round(fitted_results.Loglogistic_2P_beta, sigfig))
                        + ")"
                    )
                )
            elif dist_name == "Gumbel_2P":
                ranked_distributions_objects.append(
                    Gumbel_Distribution(
                        mu=fitted_results.Gumbel_2P_mu,
                        sigma=fitted_results.Gumbel_2P_sigma,
                    )
                )
                ranked_distributions_labels.append(
                    str(
                        "Gumbel_2P (μ="
                        + str(round(fitted_results.Gumbel_2P_mu, sigfig))
                        + ",σ="
                        + str(round(fitted_results.Gumbel_2P_sigma, sigfig))
                        + ")"
                    )
                )

            if include_location_shifted is True:
                if dist_name == "Weibull_3P":
                    if fitted_results.Weibull_3P_gamma != 0:
                        ranked_distributions_objects.append(
                            Weibull_Distribution(
                                alpha=fitted_results.Weibull_3P_alpha,
                                beta=fitted_results.Weibull_3P_beta,
                                gamma=fitted_results.Weibull_3P_gamma,
                            )
                        )
                        ranked_distributions_labels.append(
                            str(
                                "Weibull_3P (α="
                                + str(round(fitted_results.Weibull_3P_alpha, sigfig))
                                + ",β="
                                + str(round(fitted_results.Weibull_3P_beta, sigfig))
                                + ",γ="
                                + str(round(fitted_results.Weibull_3P_gamma, sigfig))
                                + ")"
                            )
                        )
                elif dist_name == "Gamma_3P":
                    if fitted_results.Gamma_3P_gamma != 0:
                        ranked_distributions_objects.append(
                            Gamma_Distribution(
                                alpha=fitted_results.Gamma_3P_alpha,
                                beta=fitted_results.Gamma_3P_beta,
                                gamma=fitted_results.Gamma_3P_gamma,
                            )
                        )
                        ranked_distributions_labels.append(
                            str(
                                "Gamma_3P (α="
                                + str(round(fitted_results.Gamma_3P_alpha, sigfig))
                                + ",β="
                                + str(round(fitted_results.Gamma_3P_beta, sigfig))
                                + ",γ="
                                + str(round(fitted_results.Gamma_3P_gamma, sigfig))
                                + ")"
                            )
                        )
                elif dist_name == "Lognormal_3P":
                    if fitted_results.Lognormal_3P_gamma != 0:
                        ranked_distributions_objects.append(
                            Lognormal_Distribution(
                                mu=fitted_results.Lognormal_3P_mu,
                                sigma=fitted_results.Lognormal_3P_sigma,
                                gamma=fitted_results.Lognormal_3P_gamma,
                            )
                        )
                        ranked_distributions_labels.append(
                            str(
                                "Lognormal_3P (μ="
                                + str(round(fitted_results.Lognormal_3P_mu, sigfig))
                                + ",σ="
                                + str(round(fitted_results.Lognormal_3P_sigma, sigfig))
                                + ",γ="
                                + str(round(fitted_results.Lognormal_3P_gamma, sigfig))
                                + ")"
                            )
                        )
                elif dist_name == "Exponential_2P":
                    if fitted_results.Exponential_2P_gamma != 0:
                        ranked_distributions_objects.append(
                            Exponential_Distribution(
                                Lambda=fitted_results.Exponential_1P_lambda,
                                gamma=fitted_results.Exponential_2P_gamma,
                            )
                        )
                        ranked_distributions_labels.append(
                            str(
                                "Exponential_1P (lambda="
                                + str(
                                    round(fitted_results.Exponential_1P_lambda, sigfig)
                                )
                                + ",γ="
                                + str(
                                    round(fitted_results.Exponential_2P_gamma, sigfig)
                                )
                                + ")"
                            )
                        )
                elif dist_name == "Loglogistic_3P":
                    if fitted_results.Loglogistic_3P_gamma != 0:
                        ranked_distributions_objects.append(
                            Loglogistic_Distribution(
                                alpha=fitted_results.Loglogistic_3P_alpha,
                                beta=fitted_results.Loglogistic_3P_beta,
                                gamma=fitted_results.Loglogistic_3P_gamma,
                            )
                        )
                        ranked_distributions_labels.append(
                            str(
                                "Loglogistic_3P (α="
                                + str(
                                    round(fitted_results.Loglogistic_3P_alpha, sigfig)
                                )
                                + ",β="
                                + str(round(fitted_results.Loglogistic_3P_beta, sigfig))
                                + ",γ="
                                + str(
                                    round(fitted_results.Loglogistic_3P_gamma, sigfig)
                                )
                                + ")"
                            )
                        )

        number_of_distributions_fitted = len(ranked_distributions_objects)
        self.results = np.array(ranked_distributions_objects)
        self.most_similar_distribution = ranked_distributions_objects[0]
        if print_results is True:
            colorprint("Results from similar_distributions:", bold=True, underline=True)
            print("The input distribution was:")
            print(distribution.param_title_long)
            if number_of_distributions_fitted < number_of_distributions_to_show:
                number_of_distributions_to_show = number_of_distributions_fitted
            print(
                "\nThe top",
                number_of_distributions_to_show,
                "most similar distributions are:",
            )
            counter = 0
            while (
                counter < number_of_distributions_to_show
                and counter < number_of_distributions_fitted
            ):
                dist = ranked_distributions_objects[counter]
                print(dist.param_title_long)
                counter += 1

        if show_plot is True:
            plt.figure(figsize=(14, 6))
            plt.suptitle(
                str("Plot of similar distributions to " + distribution.param_title_long)
            )
            counter = 0
            xlower = distribution.quantile(0.001)
            xupper = distribution.quantile(0.999)
            x_delta = xupper - xlower
            plt.subplot(121)
            distribution.PDF(
                label=str("Input distribution [" + distribution.name2 + "]"),
                linestyle="--",
            )
            while (
                counter < number_of_distributions_to_show
                and counter < number_of_distributions_fitted
            ):
                ranked_distributions_objects[counter].PDF(
                    label=ranked_distributions_labels[counter]
                )
                counter += 1
            plt.xlim([xlower - x_delta * 0.1, xupper + x_delta * 0.1])
            plt.legend()
            plt.title("PDF")
            counter = 0
            plt.subplot(122)
            distribution.CDF(
                label=str("Input distribution [" + distribution.name2 + "]"),
                linestyle="--",
            )
            while (
                counter < number_of_distributions_to_show
                and counter < number_of_distributions_fitted
            ):
                ranked_distributions_objects[counter].CDF(
                    label=ranked_distributions_labels[counter]
                )
                counter += 1
            plt.xlim([xlower - x_delta * 0.1, xupper + x_delta * 0.1])
            plt.legend()
            plt.title("CDF")
            plt.subplots_adjust(left=0.08, right=0.95)
            plt.show()


def histogram(
    data, white_above=None, bins=None, density=True, cumulative=False, **kwargs
):
    """
    Plots a histogram using the data specified. This is similar to plt.hist
    except that it sets better defaults and also shades the bins white above a
    specified value (white_above). This is useful for representing complete data
    as right censored data in a histogram.

    Parameters
    ----------
    data : array, list
        The data to plot in the histogram.
    white_above : float, int, optional
        Bins above this value will be shaded white to represent right censored
        data. Default = None.
    bins : array, string, optional
        An array of bin edges or a string to specify how to calculate the bin
        edges. Acceptable strings are 'auto','fd','doane','scott','stone',
        'rice','sturges','sqrt'. Default = 'auto'. For more information on these
        methods, see the numpy documentation:
        https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
    density : bool, optional
        Determines whether to plot a density histogram or a count histogram.
        Default = True which is required when plotting a PDF or CDF.
    cumulative : bool, optional
        Use False for PDF and True for CDF. Default = False.
    kwargs
        Plotting kwargs for the histogram (color, alpha, etc.) which are passed
        to matplotlib.

    Returns
    -------
    None
    """

    if type(data) not in [np.ndarray, list]:
        raise ValueError("data must be an array or list")

    if white_above is not None:
        if type(white_above) not in [int, float, np.float64]:
            raise ValueError("white_above must be int or float")
        if white_above < min(data):
            raise ValueError("white_above must be greater than min(data)")

    if bins is None:
        bins = "auto"  # uses numpy to calculate bin edges: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges

    if "color" in kwargs:
        color = kwargs.pop("color")
    elif "c" in kwargs:
        color = kwargs.pop("c")
    else:
        color = "lightgrey"

    if "edgecolor" in kwargs:
        edgecolor = kwargs.pop("edgecolor")
    else:
        edgecolor = "k"

    if "linewidth" in kwargs:
        linewidth = kwargs.pop("linewidth")
    elif "lw" in kwargs:
        linewidth = kwargs.pop("lw")
    else:
        linewidth = 0.5

    _, bins_out, patches = plt.hist(
        data,
        density=density,
        cumulative=cumulative,
        color=color,
        bins=bins,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs
    )  # plots the histogram of the data

    if white_above is not None:
        for i in range(np.argmin(abs(np.array(bins_out) - white_above)), len(patches)):
            # this is to shade part of the histogram as white
            patches[i].set_facecolor("white")


class make_right_censored_data:
    """
    This function is used to create right censored data from complete data. It
    will right censor the data based on a specified threshold or fraction to
    censor.

    Parameters
    ----------
    data : list, array
        The complete data.
    threshold : int, float, optional
        This is the point to right censor (right censoring is done if data >
        threshold). This is known as "singly censored data" as everything is
        censored at a single point. Default is None in which case the
        fraction_censored will be used. See the notes below.
    fraction_censored : float, optional
        Must be between 0 and 1. Default = 0.5. Censoring is done randomly. This
        is known as "multiply censored data" as there are multiple times at
        which censoring occurs. See the notes below.
    seed : int, optional
        Sets the random seed. This is used for multiply censored data (i.e. when
        threshold is None). The data is shuffled to remove censoring bias that
        may be caused by any pre-sorting. Specifying the seed ensures a
        repeatable random shuffle. Default is None which will result in a
        different censoring each time. The seed is only used when threshold is
        not specified and the data is being multiply censored based on the
        fraction_censored.

    Returns
    -------
    failures : array
        The array of failure data
    right_censored : array
        The array of right censored data

    Notes
    -----
    If both threshold and fraction_censored are None, fraction_censored will
    default to 0.5 to produce multiply censored data. If both threshold and
    fraction_censored are specified, an error will be raised since these methods
    conflict.
    """

    def __init__(self, data, threshold=None, fraction_censored=None, seed=None):
        if type(data) not in [list, np.ndarray]:
            raise ValueError("data must be a list or array")
        data = np.asarray(data)

        if threshold is not None and fraction_censored is not None:
            raise ValueError(
                "threshold is used to control censoring above a set limit. fraction_censored is used to control the fraction of the values that will be censored. These cannot both be specified as they are conflicting methods of censoring"
            )
        if threshold is None and fraction_censored is None:
            fraction_censored = 0.5  # default to 50% multiply censored

        # multiply censored
        if threshold is None:
            if seed is not None:
                np.random.seed(seed)
            data = list(data)
            np.random.shuffle(
                data
            )  # randomize the order of the data in case it was ordered
            # place a limit on the amount of the data that can be censored
            if fraction_censored <= 0 or fraction_censored >= 1:
                raise ValueError(
                    "fraction_censored must be between 0 and 1. The default is 0.5 which will right censor half the data"
                )
            number_of_items_to_censor = np.floor(len(data) * fraction_censored)
            right_censored = []
            while len(right_censored) < number_of_items_to_censor:
                item = data.pop(0)  # draw an item from the start of the list
                th = np.random.rand() * max(
                    data
                )  # randomly choose a censoring time between 0 and the max of the data
                if item > th:  # check if the item exceeds the threshold for censoring
                    right_censored.append(th)  # if it does, censor at the threshold
                else:
                    data.append(
                        item
                    )  # if it doesn't, then return it to the end of the list in case it is needed for another draw
            self.failures = np.array(data)  # what's leftover
            self.right_censored = np.array(right_censored)

        # singly censored
        else:
            self.failures = data[data <= threshold]
            self.right_censored = np.ones_like(data[data > threshold]) * threshold


class make_ALT_data:
    """
    Generates Accelerated Life Test (ALT) data based on model parameters. This
    function is primarily used when testing the functions in ALT_fitters.

    Parameters
    ----------
    distribution : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal".
    life_stress_model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power"
    stress_1 : array, list
        The stresses for the ALT data. eg. [100,50,10].
    stress_2 : array, list
        The stresses for the ALT data. eg. [0.8,0.6,0.4]. Required only if using
        a dual stress model. Must match the length of stress_1.
    a : float, int
        Parameter from all models.
    b : float, int, optional
        Parameter from Exponential and Dual_Exponential models.
    c : float, int, optional
        Parameter from Eyring, Dual_Exponential, Power_Exponential, and
        Dual_Power models.
    n : float, int, optional
        Parameter from Power, Power_Exponential, and Dual_Power models.
    m : float, int, optional
        Parameter from Dual_Power model.
    beta : float, int, optional
        Shape parameter for Weibull distribution.
    sigma : float, int, optional
        Shape parameter for Normal or Lognormal distributions.
    use_level_stress : float, int, list, array, optional
        A float or int (if single stress) or a list or array (if dual stress).
        Optional input. Default = None.
    number_of_samples : int, optional
        The number of samples to generate for each stress. Default = 100. The
        total data points will be equal to the number of samples x number of
        stress levels
    fraction_censored : int, float, optional
        Use 0 for no censoring or specify a float between 0 and 1 for right
        censoring. Censoring is "multiply censored" meaning that there is no
        threshold above which all the right censored values will occur. Default
        = 0.5.
    seed : int, optional
        The random seed for repeatability. Default = None.

    Returns
    -------
    failures : list
        The failure data.
    failure_stresses : list
        The failure stresses that are paired with the failure data. Only
        provided if using a single stress model.
    failure_stresses_1 : list
        The failure stresses for stress_1 that are paired with the failure data.
        Only provided if using a dual stress model.
    failure_stresses_2 : list
        The failure stresses for stress_2 that are paired with the failure data.
        Only provided if using a dual stress model.
    right_censored : list
        The right censored data. This is only provided if fraction_censored > 0.
    right_censored_stresses : list
        The failure stresses that are paired with the right censored data. This
        is only provided if fraction_censored > 0. Only provided if using a
        single stress model.
    right_censored_stresses_1 : list
        The failure stresses that are paired with the right censored data. This
        is only provided if fraction_censored > 0. Only provided if using a
        dual stress model.
    right_censored_stresses_2 : list
        The failure stresses that are paired with the right censored data. This
        is only provided if fraction_censored > 0. Only provided if using a
        dual stress model.
    mean_life_at_use_stress : float
        This is only provided if use_level_stress is provided.
    """

    def __init__(
        self,
        distribution,
        life_stress_model,
        stress_1,
        stress_2=None,
        a=None,
        b=None,
        c=None,
        n=None,
        m=None,
        beta=None,
        sigma=None,
        use_level_stress=None,
        number_of_samples=100,
        fraction_censored=0.5,
        seed=None,
    ):

        # input error checking
        life_stress_model = life_stress_model.title()
        if life_stress_model in ["Exponential", "Eyring", "Power"]:
            dual_stress = False
        elif life_stress_model in [
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
        ]:
            dual_stress = True
        else:
            raise ValueError(
                "life_stress_model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power"
            )

        if type(stress_1) not in [list, np.ndarray]:
            raise ValueError("stress_1 must be a list or array of the stress levels")
        stress_1 = np.asarray(stress_1)
        num_stresses = len(stress_1)

        if use_level_stress is not None:
            if dual_stress is False:
                stress_1 = np.append(stress_1, use_level_stress)
            else:
                stress_1 = np.append(stress_1, use_level_stress[0])

        if dual_stress is True:
            if type(stress_2) not in [list, np.ndarray]:
                raise ValueError(
                    "stress_2 must be a list or array of the stress levels"
                )
            stress_2 = np.asarray(stress_2)
            if use_level_stress is not None:
                if len(use_level_stress) != 2:
                    raise ValueError(
                        "use_level_stress must be a list or array with 2 elements if using a dual-stress model"
                    )
                stress_2 = np.append(stress_2, use_level_stress[1])

            if len(stress_2) != len(stress_1):
                raise ValueError("stress_1 and stress_2 must be the same length")

        if fraction_censored < 0 or fraction_censored >= 1:
            raise ValueError(
                "fraction_censored must be 0 for no censoring or between 0 and 1 for right censoring"
            )

        # life stress model calculations
        if life_stress_model == "Exponential":
            if a is None or b is None:
                raise ValueError(
                    "a and b must be specified for the Exponential life-stress model"
                )
            if b <= 0:
                raise ValueError("b must be positive")
            life_model = b * np.exp(a / stress_1)
        elif life_stress_model == "Eyring":
            if a is None or c is None:
                raise ValueError(
                    "a and c must be specified for the Eyring life-stress model"
                )
            life_model = (1 / stress_1) * np.exp(-(c - a / stress_1))
        elif life_stress_model == "Power":
            if a is None or n is None:
                raise ValueError(
                    "a and n must be specified for the Power life-stress model"
                )
            if a <= 0:
                raise ValueError("a must be positive")
            life_model = a * stress_1 ** float(n)
        elif life_stress_model == "Dual_Exponential":
            if a is None or b is None or c is None:
                raise ValueError(
                    "a, b, and c must be specified for the Dual_Exponential life-stress model"
                )
            if c <= 0:
                raise ValueError("c must be positive")
            life_model = c * np.exp(a / stress_1 + b / stress_2)
        elif life_stress_model == "Power_Exponential":
            if a is None or c is None or n is None:
                raise ValueError(
                    "a, c, and n must be specified for the Power_Exponential life-stress model"
                )
            if c <= 0:
                raise ValueError("c must be positive")
            life_model = c * (stress_2 ** float(n)) * np.exp(a / stress_1)
        elif life_stress_model == "Dual_Power":
            if c is None or n is None or m is None:
                raise ValueError(
                    "c, n, and m must be specified for the Dual_Power life-stress model"
                )
            if c <= 0:
                raise ValueError("c must be positive")
            life_model = c * (stress_1 ** float(m)) * (stress_2 ** float(n))

        # data sampling
        failures = []
        right_censored = []
        failure_stresses_1 = []
        right_censored_stresses_1 = []
        failure_stresses_2 = []
        right_censored_stresses_2 = []
        np.random.seed(seed)
        seeds = np.random.randint(
            low=0, high=1000000, size=num_stresses
        )  # need a seed for each stress or the points will be the same just shifted horizontally

        def __make_dist(life):
            if distribution == "Weibull":
                if beta is None:
                    raise ValueError(
                        "beta must be specified for the Weibull distribution"
                    )
                dist = Weibull_Distribution(alpha=life, beta=beta)
            elif distribution == "Lognormal":
                if sigma is None:
                    raise ValueError(
                        "sigma must be specified for the Lognormal distribution"
                    )
                dist = Lognormal_Distribution(mu=np.log(life), sigma=sigma)
            elif distribution == "Normal":
                if sigma is None:
                    raise ValueError(
                        "sigma must be specified for the Normal distribution"
                    )
                dist = Normal_Distribution(mu=life, sigma=sigma)
            elif distribution == "Exponential":
                dist = Exponential_Distribution(Lambda=1 / life)
            else:
                raise ValueError(
                    "distribution must be one of Weibull, Lognormal, Normal, Exponential"
                )
            return dist

        for i in range(num_stresses):
            dist = __make_dist(life=life_model[i])
            raw_data = dist.random_samples(
                number_of_samples=number_of_samples, seed=seeds[i]
            )
            if min(raw_data) <= 0:
                raise ValueError(
                    "The values entered for the ALT model will result in negative failure times.\n"
                    "While this is acceptable for a pure Normal Distribution, it is not acceptable for an ALT model utilising the Normal Distribution.\n"
                    "Please modify your input parameters to create a model that does not result in the generation of negative failure times."
                )

            if fraction_censored == 0:
                failures.extend(raw_data)
                failure_stresses_1.extend(list(np.ones_like(raw_data) * stress_1[i]))
                if dual_stress is True:
                    failure_stresses_2.extend(
                        list(np.ones_like(raw_data) * stress_2[i])
                    )
            else:
                data = make_right_censored_data(
                    raw_data, fraction_censored=fraction_censored, seed=seeds[i]
                )
                failures.extend(data.failures)
                right_censored.extend(data.right_censored)
                failure_stresses_1.extend(
                    list(np.ones_like(data.failures) * stress_1[i])
                )
                right_censored_stresses_1.extend(
                    list(np.ones_like(data.right_censored) * stress_1[i])
                )
                if dual_stress is True:
                    failure_stresses_2.extend(
                        list(np.ones_like(data.failures) * stress_2[i])
                    )
                    right_censored_stresses_2.extend(
                        list(np.ones_like(data.right_censored) * stress_2[i])
                    )

        if dual_stress is False:
            self.failures = failures
            self.failure_stresses = failure_stresses_1
            if fraction_censored > 0:
                self.right_censored = right_censored
                self.right_censored_stresses = right_censored_stresses_1
        else:
            self.failures = failures
            self.failure_stresses_1 = failure_stresses_1
            self.failure_stresses_2 = failure_stresses_2
            if fraction_censored > 0:
                self.right_censored = right_censored
                self.right_censored_stresses_1 = right_censored_stresses_1
                self.right_censored_stresses_2 = right_censored_stresses_2
        if use_level_stress is not None:
            use_dist = __make_dist(life=life_model[-1])
            self.mean_life_at_use_stress = use_dist.mean


class crosshairs:
    """
    Adds interactive crosshairs to matplotlib plots

    Parameters
    ----------
    xlabel : str, optional
        The xlabel for annotations. Default is 'x'.
    ylabel : str, optional
        The ylabel for annotations. Default is 'y'.
    decimals : int, optional
        The number of decimals for rounding. Default is 2.
    dateformat : str, optional
        The datetime format. If specified the x crosshair and label will be
        formatted as a date using the format provided. Default is None which
        results in no date format being used on x.
    kwargs : optional
        plotting kwargs to change the style of the crosshairs (eg. color,
        linestyle, etc.).

    Returns
    -------
    None

    Notes
    -----
    Ensure this is used after you plot everything as anything plotted after
    crosshairs() is called will not be recognised by the snap-to feature. For a
    list of acceptable dateformat strings see
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    """

    def __init__(self, xlabel=None, ylabel=None, decimals=2, dateformat=None, **kwargs):

        if type(dateformat) not in [str, type(None)]:
            raise ValueError(
                "dateformat type must be str or None. For acceptable strings see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes"
            )
        if type(decimals) is not int:
            raise ValueError("decimals must be int")
        if type(xlabel) not in [str, type(None)]:
            raise ValueError("xlabel must be string or None")
        if type(ylabel) not in [str, type(None)]:
            raise ValueError("ylabel must be string or None")

        warnings.simplefilter(
            "ignore"
        )  # required when using fill_between due to warning in mplcursors: "UserWarning: Pick support for PolyCollection is missing."
        ch = cursor(hover=True)
        add_lines_and_text_with_kwargs = (
            lambda _: crosshairs.__add_lines_and_text_to_crosshairs(
                _, decimals, dateformat, **kwargs
            )
        )  # adds the line's kwargs before connecting it to cursor
        ch.connect("add", add_lines_and_text_with_kwargs)
        plt.gcf().canvas.mpl_connect(
            "axes_leave_event", crosshairs.__hide_crosshairs
        )  # hide the crosshairs and text when the mouse leaves the axes

        # this does the annotation part
        if xlabel is None:
            xlabel = "x"
        if ylabel is None:
            ylabel = "y"
        warnings.simplefilter(
            "ignore"
        )  # required when using fill_between due to warning in mplcursors: "UserWarning: Pick support for PolyCollection is missing."
        annot = cursor(multiple=True, bindings={"toggle_visible": "h"})
        format_annotation_labeled = lambda _: crosshairs.__format_annotation(
            _, decimals, dateformat, [xlabel, ylabel]
        )  # adds the labels to the 'format_annotation' function before connecting it to cursor
        annot.connect("add", format_annotation_labeled)

    @staticmethod
    def __add_lines_and_text_to_crosshairs(sel, decimals, dateformat, **kwargs):
        # set the default properties of the lines and text if they were not provided as kwargs
        if "c" in kwargs:
            color = kwargs.pop("c")
        elif "color" in kwargs:
            color = kwargs.pop("color")
        else:
            color = "k"
        if "lw" in kwargs:
            linewidth = kwargs.pop("lw")
        elif "linewidth" in kwargs:
            linewidth = kwargs.pop("linewidth")
        else:
            linewidth = 0.5
        if "ls" in kwargs:
            linestyle = kwargs.pop("ls")
        elif "linestyle" in kwargs:
            linestyle = kwargs.pop("linestyle")
        else:
            linestyle = "--"
        if "size" in kwargs:
            fontsize = kwargs.pop("size")
        elif "fontsize" in kwargs:
            fontsize = kwargs.pop("fontsize")
        else:
            fontsize = 10
        if "fontweight" in kwargs:
            fontweight = kwargs.pop("fontweight")
        elif "weight" in kwargs:
            fontweight = kwargs.pop("weight")
        else:
            fontweight = 0
        if "fontstyle" in kwargs:
            fontstyle = kwargs.pop("fontstyle")
        elif "style" in kwargs:
            fontstyle = kwargs.pop("style")
        else:
            fontstyle = "normal"

        sel.annotation.set(visible=False)  # Hide the normal annotation during hover
        try:
            ax = sel.artist.axes
        except AttributeError:
            ax = sel.annotation.axes  # this exception occurs for bar charts

        x, y = sel.target
        lines = [
            Line2D(
                [x, x],
                [0, 1],
                transform=ax.get_xaxis_transform(),
                c=color,
                lw=linewidth,
                ls=linestyle,
                **kwargs
            ),
            Line2D(
                [0, 1],
                [y, y],
                transform=ax.get_yaxis_transform(),
                c=color,
                lw=linewidth,
                ls=linestyle,
                **kwargs
            ),
        ]

        if type(dateformat) is str:
            x_string = time.strftime(dateformat, time.gmtime(x * 24 * 3600))
        else:
            x_string = round(x, decimals)

        texts = [
            ax.text(
                s=round(y, decimals),
                x=0,
                y=y,
                transform=ax.get_yaxis_transform(),
                color=color,
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                **kwargs
            ),
            ax.text(
                s=x_string,
                x=x,
                y=0,
                transform=ax.get_xaxis_transform(),
                color=color,
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                **kwargs
            ),
        ]

        for i in [0, 1]:
            line = lines[i]
            text = texts[i]
            ax.add_line(line)
            # the lines and text need to be registered with sel so that they are updated during mouse motion events
            sel.extras.append(line)
            sel.extras.append(text)

    @staticmethod
    def __format_annotation(sel, decimals, dateformat, label):
        # this is some simple formatting for the annotations (applied on mouse click)
        [x, y] = sel.annotation.xy

        if type(dateformat) is str:
            x_string = time.strftime(dateformat, time.gmtime(x * 24 * 3600))
        else:
            x_string = round(x, decimals)

        text = str(
            label[0]
            + " = "
            + str(x_string)
            + "\n"
            + label[1]
            + " = "
            + str(round(y, decimals))
        )
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(fc="white")

    @staticmethod
    def __hide_crosshairs(event):
        ax = event.inaxes  # this gets the axes where the event occurred.
        if len(ax.texts) >= 2:  # the lines can't be deleted if they haven't been drawn.
            if (
                ax.texts[-1].get_position()[1] == 0
                and ax.texts[-2].get_position()[0] == 0
            ):  # this identifies the texts (crosshair text coords) based on their combination of unique properties
                ax.lines[-1].set_visible(False)
                ax.lines[-2].set_visible(False)
                ax.texts[-1].set_visible(False)
                ax.texts[-2].set_visible(False)
        event.canvas.draw()


class distribution_explorer:
    """
    Generates an interactive plot of PDF, CDF, SF, HF, CHF for the selected
    distribution. Parameters can be changed using slider widgets. Distributions
    can be changed using radio button widget.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Example usage:

    .. code:: python

        from reliability.Other_functions import distribution_explorer
        distribution_explorer()
    """

    def __init__(self):
        # initialise the 5 plots
        plt.figure("Distribution Explorer", figsize=(12, 7))
        self.name = "Weibull"  # starting value
        dist = Weibull_Distribution(alpha=100, beta=2, gamma=0)
        plt.suptitle(dist.param_title_long, fontsize=15)
        self.ax_pdf = plt.subplot(231)
        dist.PDF()
        plt.title("PDF")
        plt.xlabel("")
        plt.ylabel("")
        self.ax_cdf = plt.subplot(232)
        dist.CDF()
        plt.title("CDF")
        plt.xlabel("")
        plt.ylabel("")
        self.ax_sf = plt.subplot(233)
        dist.SF()
        plt.title("SF")
        plt.xlabel("")
        plt.ylabel("")
        self.ax_hf = plt.subplot(234)
        dist.HF()
        plt.title("HF")
        plt.xlabel("")
        plt.ylabel("")
        self.ax_chf = plt.subplot(235)
        dist.CHF()
        plt.title("CHF")
        plt.xlabel("")
        plt.ylabel("")
        plt.subplots_adjust(
            left=0.07, right=0.98, top=0.9, bottom=0.25, wspace=0.18, hspace=0.30
        )

        # initialise the sliders
        x0 = 0.1
        width = 0.8
        height = 0.03
        self.active_color = "steelblue"
        self.background_color = "whitesmoke"
        self.ax0 = plt.axes([x0, 0.15, width, height], facecolor=self.background_color)
        self.ax1 = plt.axes([x0, 0.1, width, height], facecolor=self.background_color)
        self.ax2 = plt.axes([x0, 0.05, width, height], facecolor=self.background_color)
        self.s0 = Slider(
            self.ax0,
            "Alpha",
            valmin=0.1,
            valmax=500,
            valinit=dist.alpha,
            facecolor=self.active_color,
        )
        self.s1 = Slider(
            self.ax1,
            "Beta",
            valmin=0.2,
            valmax=25,
            valinit=dist.beta,
            facecolor=self.active_color,
        )
        self.s2 = Slider(
            self.ax2,
            "Gamma",
            valmin=0,
            valmax=500,
            valinit=dist.gamma,
            facecolor=self.active_color,
        )
        plt.subplots_adjust(
            left=0.07, right=0.98, top=0.9, bottom=0.25, wspace=0.18, hspace=0.30
        )

        # initialise the radio button
        radio_ax = plt.axes([0.708, 0.25, 0.27, 0.28], facecolor=self.background_color)
        radio_ax.set_title("Distribution")
        self.radio = RadioButtons(
            radio_ax,
            (
                "Weibull",
                "Gamma",
                "Normal",
                "Lognormal",
                "Beta",
                "Exponential",
                "Loglogistic",
                "Gumbel",
            ),
            active=0,
            activecolor=self.active_color,
        )

        # begin the interactive section
        distribution_explorer.__interactive(self, initial_run=True)

    @staticmethod
    def __update_distribution(name, self):
        self.name = name
        if self.name == "Weibull":
            dist = Weibull_Distribution(alpha=100, beta=2, gamma=0)
            param_names = ["Alpha", "Beta", "Gamma"]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0.1, valmax=500, valinit=dist.alpha
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.2, valmax=25, valinit=dist.beta
            )
            plt.sca(self.ax2)
            plt.cla()
            self.ax2.set_visible(True)
            self.s2 = Slider(
                self.ax2, param_names[2], valmin=0, valmax=500, valinit=dist.gamma
            )
        elif self.name == "Gamma":
            dist = Gamma_Distribution(alpha=100, beta=5, gamma=0)
            param_names = ["Alpha", "Beta", "Gamma"]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0.1, valmax=500, valinit=dist.alpha
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.2, valmax=25, valinit=dist.beta
            )
            plt.sca(self.ax2)
            plt.cla()
            self.ax2.set_visible(True)
            self.s2 = Slider(
                self.ax2, param_names[2], valmin=0, valmax=500, valinit=dist.gamma
            )
        elif self.name == "Loglogistic":
            dist = Loglogistic_Distribution(alpha=100, beta=8, gamma=0)
            param_names = ["Alpha", "Beta", "Gamma"]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0.1, valmax=500, valinit=dist.alpha
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.2, valmax=50, valinit=dist.beta
            )
            plt.sca(self.ax2)
            plt.cla()
            self.ax2.set_visible(True)
            self.s2 = Slider(
                self.ax2, param_names[2], valmin=0, valmax=500, valinit=dist.gamma
            )
        elif self.name == "Lognormal":
            dist = Lognormal_Distribution(mu=2.5, sigma=0.5, gamma=0)
            param_names = ["Mu", "Sigma", "Gamma"]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0, valmax=5, valinit=dist.mu
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.01, valmax=2, valinit=dist.sigma
            )
            plt.sca(self.ax2)
            plt.cla()
            self.ax2.set_visible(True)
            self.s2 = Slider(
                self.ax2, param_names[2], valmin=0, valmax=500, valinit=dist.gamma
            )
        elif self.name == "Normal":
            dist = Normal_Distribution(mu=0, sigma=10)
            param_names = ["Mu", "Sigma", ""]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=-100, valmax=100, valinit=dist.mu
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.01, valmax=20, valinit=dist.sigma
            )
            self.ax2.set_visible(False)
        elif self.name == "Gumbel":
            dist = Gumbel_Distribution(mu=0, sigma=10)
            param_names = ["Mu", "Sigma", ""]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=-100, valmax=100, valinit=dist.mu
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.01, valmax=20, valinit=dist.sigma
            )
            self.ax2.set_visible(False)
        elif self.name == "Exponential":
            dist = Exponential_Distribution(Lambda=1, gamma=0)
            param_names = ["Lambda", "Gamma", ""]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0.001, valmax=5, valinit=dist.Lambda
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0, valmax=500, valinit=dist.gamma
            )
            self.ax2.set_visible(False)
        elif self.name == "Beta":
            dist = Beta_Distribution(alpha=2, beta=2)
            param_names = ["Alpha", "Beta", ""]
            plt.sca(self.ax0)
            plt.cla()
            self.s0 = Slider(
                self.ax0, param_names[0], valmin=0.01, valmax=5, valinit=dist.alpha
            )
            plt.sca(self.ax1)
            plt.cla()
            self.s1 = Slider(
                self.ax1, param_names[1], valmin=0.01, valmax=5, valinit=dist.beta
            )
            self.ax2.set_visible(False)
        else:
            raise ValueError(str(self.name + " is an unknown distribution name"))
        plt.suptitle(dist.param_title_long, fontsize=15)
        distribution_explorer.__update_params(None, self)
        distribution_explorer.__interactive(self)
        plt.draw()

    @staticmethod
    def __update_params(_, self):
        value1 = self.s0.val
        value2 = self.s1.val
        value3 = self.s2.val
        if self.name == "Weibull":
            dist = Weibull_Distribution(alpha=value1, beta=value2, gamma=value3)
        elif self.name == "Loglogistic":
            dist = Loglogistic_Distribution(alpha=value1, beta=value2, gamma=value3)
        elif self.name == "Gamma":
            dist = Gamma_Distribution(alpha=value1, beta=value2, gamma=value3)
        elif self.name == "Loglogistic":
            dist = Loglogistic_Distribution(alpha=value1, beta=value2, gamma=value3)
        elif self.name == "Lognormal":
            dist = Lognormal_Distribution(mu=value1, sigma=value2, gamma=value3)
        elif self.name == "Beta":
            dist = Beta_Distribution(alpha=value1, beta=value2)
        elif self.name == "Normal":
            dist = Normal_Distribution(mu=value1, sigma=value2)
        elif self.name == "Gumbel":
            dist = Gumbel_Distribution(mu=value1, sigma=value2)
        elif self.name == "Exponential":
            dist = Exponential_Distribution(Lambda=value1, gamma=value2)
        else:
            raise ValueError(str(self.name + " is an unknown distribution name"))
        plt.sca(self.ax_pdf)
        plt.cla()
        dist.PDF()
        plt.title("PDF")
        plt.xlabel("")
        plt.ylabel("")
        plt.sca(self.ax_cdf)
        plt.cla()
        dist.CDF()
        plt.title("CDF")
        plt.xlabel("")
        plt.ylabel("")
        plt.sca(self.ax_sf)
        plt.cla()
        dist.SF()
        plt.title("SF")
        plt.xlabel("")
        plt.ylabel("")
        plt.sca(self.ax_hf)
        plt.cla()
        dist.HF()
        plt.title("HF")
        plt.xlabel("")
        plt.ylabel("")
        plt.sca(self.ax_chf)
        plt.cla()
        dist.CHF()
        plt.title("CHF")
        plt.xlabel("")
        plt.ylabel("")
        plt.subplots_adjust(
            left=0.07, right=0.98, top=0.9, bottom=0.25, wspace=0.18, hspace=0.30
        )
        plt.suptitle(dist.param_title_long, fontsize=15)
        plt.draw()

    def __interactive(self, initial_run=False):
        update_params_wrapper = lambda _: distribution_explorer.__update_params(_, self)
        update_distribution_wrapper = (
            lambda name: distribution_explorer.__update_distribution(name, self)
        )
        self.s0.on_changed(update_params_wrapper)
        self.s1.on_changed(update_params_wrapper)
        self.s2.on_changed(update_params_wrapper)
        if initial_run == True:
            self.radio.on_clicked(update_distribution_wrapper)
            plt.show()
        else:
            plt.draw()
