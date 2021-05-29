"""
Probability Distributions Module

Standard distributions are:
    Weibull_Distribution
    Normal_Distribution
    Lognormal_Distribution
    Exponential_Distribution
    Gamma_Distribution
    Beta_Distribution
    Loglogistic_Distribution
    Gumbel_Distribution

Mixture distributions are:
    Mixture_Model - this must be created using 2 or more of the above standard distributions
    Competing_Risks_Model - this must be created using 2 or more of the above standard distributions

Example usage:
dist = Weibull_Distribution(alpha = 8, beta = 1.2)
print(dist.mean)
    >> 7.525246866054174
print(dist.quantile(0.05))
    >> 0.6731943793488804
print(dist.mean_residual_life(15))
    >> 5.556500198354015
dist.plot()
    >> A figure of 5 plots and descriptive statistics will be displayed
dist.CHF()
    >> Cumulative Hazard Function plot will be displayed
values = dist.random_samples(number_of_samples=10000)
    >> random values will be generated from the distribution
"""

import scipy.stats as ss
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from reliability.Utils import (
    round_to_decimals,
    get_axes_limits,
    restore_axes_limits,
    generate_X_array,
    zeroise_below_gamma,
    distribution_confidence_intervals,
    colorprint,
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings


class Weibull_Distribution:
    """
    Weibull probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    alpha : float, int
        Scale parameter. Must be > 0
    beta : float, int
        Shape parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Weibull'
    name2 : 'str
        'Weibull_2P' or 'Weibull_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Weibull Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta,gamma]
    alpha : float
    beta : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, alpha=None, beta=None, gamma=0, **kwargs):
        self.name = "Weibull"
        if alpha is None or beta is None:
            raise ValueError(
                "Parameters alpha and beta must be specified. Eg. Weibull_Distribution(alpha=5,beta=2)"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.parameters = np.array([self.alpha, self.beta, self.gamma])
        mean, var, skew, kurt = ss.weibull_min.stats(
            self.beta, scale=self.alpha, loc=self.gamma, moments="mvsk"
        )
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.weibull_min.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = (
                self.alpha * ((self.beta - 1) / self.beta) ** (1 / self.beta)
                + self.gamma
            )
        else:
            self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
            )
            self.param_title_long = str(
                "Weibull Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
                + ")"
            )
            self.name2 = "Weibull_3P"
        else:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
            )
            self.param_title_long = str(
                "Weibull Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ")"
            )
            self.name2 = "Weibull_2P"
        self.b5 = ss.weibull_min.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.weibull_min.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "alpha_SE" in kwargs:
            self.alpha_SE = kwargs.pop("alpha_SE")
        else:
            self.alpha_SE = None
        if "beta_SE" in kwargs:
            self.beta_SE = kwargs.pop("beta_SE")
        else:
            self.beta_SE = None
        if "Cov_alpha_beta" in kwargs:
            self.Cov_alpha_beta = kwargs.pop("Cov_alpha_beta")
        else:
            self.Cov_alpha_beta = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are alpha_SE, beta_SE, Cov_alpha_beta, CI, and CI_type."
                ),
                text_color="red",
            )
        self._pdf0 = ss.weibull_min.pdf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.weibull_min.pdf(
            0, self.beta, scale=self.alpha, loc=0
        ) / ss.weibull_min.sf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """

        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = (self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (
            self.beta - 1
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        chf = ((X - self.gamma) / self.alpha) ** self.beta
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        plt.figure(figsize=(9, 7))
        text_title = str("Weibull Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Weibull Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Weibull Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.weibull_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Weibull Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.weibull_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        hf = (self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (
            self.beta - 1
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        self._hf = hf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Weibull Distribution\n" + " Hazard Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        chf = ((X - self.gamma) / self.alpha) ** self.beta
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Weibull Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.weibull_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.weibull_min.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.weibull_min.isf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.weibull_min.sf(x, self.beta, scale=self.alpha, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            print(
                "Descriptive statistics for Weibull distribution with alpha =",
                self.alpha,
                "and beta =",
                self.beta,
            )
        else:
            print(
                "Descriptive statistics for Weibull distribution with alpha =",
                self.alpha,
                ", beta =",
                self.beta,
                ", and gamma =",
                self.gamma,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.weibull_min.rvs(
            self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples
        )
        return RVS


class Normal_Distribution:
    """
    Normal probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    mu : float, int
        Location parameter
    sigma : float, int
        Scale parameter. Must be > 0

    Returns
    -------
    name : str
        'Normal'
    name2 : 'str
        'Normal_2P'
    param_title_long : str
        'Normal Distribution (μ=5,σ=2)'
    param_title : str
        'μ=5,σ=2'
    parameters : list
        [mu,sigma]
    mu : float
    sigma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, mu=None, sigma=None, **kwargs):
        self.name = "Normal"
        self.name2 = "Normal_2P"
        if mu is None or sigma is None:
            raise ValueError(
                "Parameters mu and sigma must be specified. Eg. Normal_Distribution(mu=5,sigma=2)"
            )
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.parameters = np.array([self.mu, self.sigma])
        self.mean = mu
        self.variance = sigma ** 2
        self.standard_deviation = sigma
        self.skewness = 0
        self.kurtosis = 3
        self.excess_kurtosis = 0
        self.median = mu
        self.mode = mu
        self.param_title = str(
            "μ="
            + str(round_to_decimals(self.mu, dec))
            + ",σ="
            + str(round_to_decimals(self.sigma, dec))
        )
        self.param_title_long = str(
            "Normal Distribution (μ="
            + str(round_to_decimals(self.mu, dec))
            + ",σ="
            + str(round_to_decimals(self.sigma, dec))
            + ")"
        )
        self.b5 = ss.norm.ppf(0.05, loc=self.mu, scale=self.sigma)
        self.b95 = ss.norm.ppf(0.95, loc=self.mu, scale=self.sigma)

        # extracts values for confidence interval plotting
        if "mu_SE" in kwargs:
            self.mu_SE = kwargs.pop("mu_SE")
        else:
            self.mu_SE = None
        if "sigma_SE" in kwargs:
            self.sigma_SE = kwargs.pop("sigma_SE")
        else:
            self.sigma_SE = None
        if "Cov_mu_sigma" in kwargs:
            self.Cov_mu_sigma = kwargs.pop("Cov_mu_sigma")
        else:
            self.Cov_mu_sigma = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + "is not recognised as an appropriate entry in kwargs. Appropriate entries are mu_SE, sigma_SE, Cov_mu_sigma, CI, and CI_type."
                ),
                text_color="red",
            )

        self._pdf0 = 0  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = 0  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.norm.pdf(X, self.mu, self.sigma)
        cdf = ss.norm.cdf(X, self.mu, self.sigma)
        sf = ss.norm.sf(X, self.mu, self.sigma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Normal Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        pdf = ss.norm.pdf(X, self.mu, self.sigma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Normal Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        cdf = ss.norm.cdf(X, self.mu, self.sigma)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Normal Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        sf = ss.norm.sf(X, self.mu, self.sigma)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Normal Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        hf = ss.norm.pdf(X, self.mu, self.sigma) / ss.norm.sf(X, self.mu, self.sigma)

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Normal Distribution\n" + " Hazard Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        chf = -np.log(ss.norm.sf(X, self.mu, self.sigma))
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Normal Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.norm.ppf(q, loc=self.mu, scale=self.sigma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.norm.isf(q, loc=self.mu, scale=self.sigma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.norm.sf(x, loc=self.mu, scale=self.sigma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        print(
            "Descriptive statistics for Normal distribution with mu =",
            self.mu,
            "and sigma =",
            self.sigma,
        )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.norm.rvs(loc=self.mu, scale=self.sigma, size=number_of_samples)
        return RVS


class Lognormal_Distribution:
    """
    Lognormal probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    mu : float, int
        Location parameter
    sigma : float, int
        Scale parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Lognormal'
    name2 : 'str
        'Lognormal_2P' or 'Lognormal_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Lognormal Distribution (μ=5,σ=2)'
    param_title : str
        'μ=5,σ=2'
    parameters : list
        [mu,sigma,gamma]
    mu : float
    sigma : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, mu=None, sigma=None, gamma=0, **kwargs):
        self.name = "Lognormal"
        if mu is None or sigma is None:
            raise ValueError(
                "Parameters mu and sigma must be specified. Eg. Lognormal_Distribution(mu=5,sigma=2)"
            )
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.parameters = np.array([self.mu, self.sigma, self.gamma])
        mean, var, skew, kurt = ss.lognorm.stats(
            self.sigma, self.gamma, np.exp(self.mu), moments="mvsk"
        )
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.lognorm.median(self.sigma, self.gamma, np.exp(self.mu))
        self.mode = np.exp(self.mu - self.sigma ** 2) + self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "μ="
                + str(round_to_decimals(self.mu, dec))
                + ",σ="
                + str(round_to_decimals(self.sigma, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
            )
            self.param_title_long = str(
                "Lognormal Distribution (μ="
                + str(round_to_decimals(self.mu, dec))
                + ",σ="
                + str(round_to_decimals(self.sigma, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
                + ")"
            )
            self.name2 = "Lognormal_3P"
        else:
            self.param_title = str(
                "μ="
                + str(round_to_decimals(self.mu, dec))
                + ",σ="
                + str(round_to_decimals(self.sigma, dec))
            )
            self.param_title_long = str(
                "Lognormal Distribution (μ="
                + str(round_to_decimals(self.mu, dec))
                + ",σ="
                + str(round_to_decimals(self.sigma, dec))
                + ")"
            )
            self.name2 = "Lognormal_2P"
        self.b5 = ss.lognorm.ppf(
            0.05, self.sigma, self.gamma, np.exp(self.mu)
        )  # note that scipy uses mu in a log way compared to most other software, so we must take the exp of the input
        self.b95 = ss.lognorm.ppf(0.95, self.sigma, self.gamma, np.exp(self.mu))

        # extracts values for confidence interval plotting
        if "mu_SE" in kwargs:
            self.mu_SE = kwargs.pop("mu_SE")
        else:
            self.mu_SE = None
        if "sigma_SE" in kwargs:
            self.sigma_SE = kwargs.pop("sigma_SE")
        else:
            self.sigma_SE = None
        if "Cov_mu_sigma" in kwargs:
            self.Cov_mu_sigma = kwargs.pop("Cov_mu_sigma")
        else:
            self.Cov_mu_sigma = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + "is not recognised as an appropriate entry in kwargs. Appropriate entries are mu_SE, sigma_SE, Cov_mu_sigma, CI, and CI_type."
                ),
                text_color="red",
            )

        self._pdf0 = ss.lognorm.pdf(
            0, self.sigma, 0, np.exp(self.mu)
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.lognorm.pdf(0, self.sigma, 0, np.exp(self.mu)) / ss.lognorm.sf(
            0, self.sigma, 0, np.exp(self.mu)
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))
        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))
        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Lognormal Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title("Probability Density\nFunction")
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )

        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title("Cumulative Distribution\nFunction")
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )

        plt.subplot(233)
        plt.plot(X, sf)
        plt.title("Survival Function")
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )

        plt.subplot(234)
        plt.plot(X, hf)
        plt.title("Hazard Function")
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )

        plt.subplot(235)
        plt.plot(X, chf)
        plt.title("Cumulative Hazard\nFunction")
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Lognormal Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Lognormal Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Lognormal Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        hf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu)) / ss.lognorm.sf(
            X, self.sigma, self.gamma, np.exp(self.mu)
        )

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Lognormal Distribution\n"
                + " Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        chf = -np.log(ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu)))
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Lognormal Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.lognorm.ppf(q, self.sigma, self.gamma, np.exp(self.mu))

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.lognorm.isf(q, self.sigma, self.gamma, np.exp(self.mu))

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.lognorm.sf(x, self.sigma, self.gamma, np.exp(self.mu))
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            print(
                "Descriptive statistics for Lognormal distribution with mu =",
                self.mu,
                "and sigma =",
                self.sigma,
            )
        else:
            print(
                "Descriptive statistics for Lognormal distribution with mu =",
                self.mu,
                ", sigma =",
                self.sigma,
                ", and gamma =",
                self.gamma,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.lognorm.rvs(
            self.sigma, self.gamma, np.exp(self.mu), size=number_of_samples
        )
        return RVS


class Exponential_Distribution:
    """
    Exponential probability distribution. Creates a probability distribution
    object.

    Parameters
    ----------
    Lambda : float, int
        Scale parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Exponential'
    name2 : 'str
        'Exponential_1P' or 'Exponential_2P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Exponential Distribution (λ=5)'
    param_title : str
        'λ=5'
    parameters : list
        [Lambda,gamma]
    Lambda : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, Lambda=None, gamma=0, **kwargs):
        self.name = "Exponential"
        if Lambda is None:
            raise ValueError(
                "Parameter Lambda must be specified. Eg. Exponential_Distribution(Lambda=3)"
            )
        self.Lambda = float(Lambda)
        self.gamma = float(gamma)
        self.parameters = np.array([self.Lambda, self.gamma])
        mean, var, skew, kurt = ss.expon.stats(
            scale=1 / self.Lambda, loc=self.gamma, moments="mvsk"
        )
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.expon.median(scale=1 / self.Lambda, loc=self.gamma)
        self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "λ="
                + str(round_to_decimals(self.Lambda, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
            )
            self.param_title_long = str(
                "Exponential Distribution (λ="
                + str(round_to_decimals(self.Lambda, dec))
                + ",γ="
                + str(round_to_decimals(gamma, dec))
                + ")"
            )
            self.name2 = "Exponential_2P"
        else:
            self.param_title = str("λ=" + str(round_to_decimals(self.Lambda, dec)))
            self.param_title_long = str(
                "Exponential Distribution (λ="
                + str(round_to_decimals(self.Lambda, dec))
                + ")"
            )
            self.name2 = "Exponential_1P"
        self.b5 = ss.expon.ppf(0.05, scale=1 / self.Lambda, loc=self.gamma)
        self.b95 = ss.expon.ppf(0.95, scale=1 / self.Lambda, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "Lambda_SE" in kwargs:
            self.Lambda_SE = kwargs.pop("Lambda_SE")
        else:
            self.Lambda_SE = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are Lambda_SE and CI."
                ),
                text_color="red",
            )
        self._pdf0 = ss.expon.pdf(
            0, scale=1 / self.Lambda, loc=0
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array.
        self._hf0 = (
            self.Lambda
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """

        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)
        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)
        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        hf = np.ones_like(X) * self.Lambda
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        chf = (X - self.gamma) * self.Lambda
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        plt.figure(figsize=(9, 7))
        text_title = str("Exponential Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Exponential Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)

        if show_plot == False:
            return cdf
        else:
            _, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Exponential Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.exponential_CI(
                self,
                func="CDF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        if show_plot == False:
            return sf
        else:
            _, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Exponential Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.exponential_CI(
                self,
                func="SF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        hf = np.ones_like(X) * self.Lambda
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Exponential Distribution\n"
                + " Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        chf = (X - self.gamma) * self.Lambda
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        if show_plot == False:
            return chf
        else:
            _, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Exponential Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.exponential_CI(
                self,
                func="CHF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.expon.ppf(q, scale=1 / self.Lambda, loc=self.gamma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.expon.isf(q, scale=1 / self.Lambda, loc=self.gamma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.expon.sf(x, scale=1 / self.Lambda, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            print(
                "Descriptive statistics for Exponential distribution with lambda =",
                self.Lambda,
            )
        else:
            print(
                "Descriptive statistics for Exponential distribution with lambda =",
                self.Lambda,
                ", and gamma =",
                self.gamma,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.expon.rvs(
            scale=1 / self.Lambda, loc=self.gamma, size=number_of_samples
        )
        return RVS


class Gamma_Distribution:
    """
    Gamma probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    alpha : float, int
        Scale parameter. Must be > 0
    beta : float, int
        Shape parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Gamma'
    name2 : 'str
        'Gamma_2P' or 'Gamma_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Gamma Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta,gamma]
    alpha : float
    beta : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, alpha=None, beta=None, gamma=0, **kwargs):
        self.name = "Gamma"
        if alpha is None or beta is None:
            raise ValueError(
                "Parameters alpha and beta must be specified. Eg. Gamma_Distribution(alpha=5,beta=2)"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.parameters = np.array([self.alpha, self.beta, self.gamma])
        mean, var, skew, kurt = ss.gamma.stats(
            self.beta, scale=self.alpha, loc=self.gamma, moments="mvsk"
        )
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.gamma.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = (self.beta - 1) * self.alpha + self.gamma
        else:
            self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
            )
            self.param_title_long = str(
                "Gamma Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
                + ")"
            )
            self.name2 = "Gamma_3P"
        else:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
            )
            self.param_title_long = str(
                "Gamma Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ")"
            )
            self.name2 = "Gamma_2P"
        self.b5 = ss.gamma.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.gamma.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "alpha_SE" in kwargs:
            self.alpha_SE = kwargs.pop("alpha_SE")
        else:
            self.alpha_SE = None
        if "beta_SE" in kwargs:
            self.beta_SE = kwargs.pop("beta_SE")
        else:
            self.beta_SE = None
        if "Cov_alpha_beta" in kwargs:
            self.Cov_alpha_beta = kwargs.pop("Cov_alpha_beta")
        else:
            self.Cov_alpha_beta = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are alpha_SE, beta_SE, Cov_alpha_beta, CI, and CI_type"
                ),
                text_color="red",
            )

        self._pdf0 = ss.gamma.pdf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.gamma.pdf(0, self.beta, scale=self.alpha, loc=0) / ss.gamma.sf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.gamma.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Gamma Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        pdf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Gamma Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        cdf = ss.gamma.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Gamma Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gamma_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        sf = ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Gamma Distribution\n" + " Survival Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gamma_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        hf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma) / ss.gamma.sf(
            X, self.beta, scale=self.alpha, loc=self.gamma
        )

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Gamma Distribution\n" + " Hazard Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        chf = -np.log(ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma))
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Gamma Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gamma_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.gamma.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.gamma.isf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.gamma.sf(x, self.beta, scale=self.alpha, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            print(
                "Descriptive statistics for Gamma distribution with alpha =",
                self.alpha,
                "and beta =",
                self.beta,
            )
        else:
            print(
                "Descriptive statistics for Gamma distribution with alpha =",
                self.alpha,
                ", beta =",
                self.beta,
                ", and gamma =",
                self.gamma,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.gamma.rvs(
            self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples
        )
        return RVS


class Beta_Distribution:
    """
    Beta probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    alpha : float, int
        Shape parameter 1. Must be > 0
    beta : float, int
        Shape parameter 2. Must be > 0

    Returns
    -------
    name : str
        'Beta'
    name2 : 'str
        'Beta_2P'
    param_title_long : str
        'Beta Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta]
    alpha : float
    beta : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, alpha=None, beta=None, **kwargs):
        self.name = "Beta"
        self.name2 = "Beta_2P"
        if alpha is None or beta is None:
            raise ValueError(
                "Parameters alpha and beta must be specified. Eg. Beta_Distribution(alpha=5,beta=2)"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.parameters = np.array([self.alpha, self.beta])
        mean, var, skew, kurt = ss.beta.stats(
            self.alpha, self.beta, 0, 1, moments="mvsk"
        )
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.beta.median(self.alpha, self.beta, 0, 1)
        if self.alpha > 1 and self.beta > 1:
            self.mode = (self.alpha - 1) / (self.beta + self.alpha - 2)
        else:
            self.mode = r"No mode exists unless $\alpha$ > 1 and $\beta$ > 1"
        self.param_title = str(
            "α="
            + str(round_to_decimals(self.alpha, dec))
            + ",β="
            + str(round_to_decimals(self.beta, dec))
        )
        self.param_title_long = str(
            "Beta Distribution (α="
            + str(round_to_decimals(self.alpha, dec))
            + ",β="
            + str(round_to_decimals(self.beta, dec))
            + ")"
        )
        self.b5 = ss.beta.ppf(0.05, self.alpha, self.beta, 0, 1)
        self.b95 = ss.beta.ppf(0.95, self.alpha, self.beta, 0, 1)

        # extracts values for confidence interval plotting
        if "alpha_SE" in kwargs:
            self.alpha_SE = kwargs.pop("alpha_SE")
        else:
            self.alpha_SE = None
        if "beta_SE" in kwargs:
            self.beta_SE = kwargs.pop("beta_SE")
        else:
            self.beta_SE = None
        if "Cov_alpha_beta" in kwargs:
            self.Cov_alpha_beta = kwargs.pop("Cov_alpha_beta")
        else:
            self.Cov_alpha_beta = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are alpha_SE, beta_SE, Cov_alpha_beta, CI, and CI_type"
                ),
                text_color="red",
            )

        # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._pdf0 = ss.beta.pdf(0, self.alpha, self.beta, 0, 1)
        # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.beta.pdf(0, self.alpha, self.beta, 0, 1) / ss.beta.sf(
            0, self.alpha, self.beta, 0, 1
        )

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)
        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)
        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Beta Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        if type(self.mode) == str:
            text_mode = str("Mode = " + str(self.mode))  # required when mode is str
        else:
            text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Beta Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Beta Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.beta_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Beta Distribution\n" + " Survival Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.beta_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        hf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1) / ss.beta.sf(
            X, self.alpha, self.beta, 0, 1
        )

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Beta Distribution\n" + " Hazard Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        chf = -np.log(ss.beta.sf(X, self.alpha, self.beta, 0, 1))
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Beta Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.beta_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.beta.ppf(q, self.alpha, self.beta, 0, 1)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.beta.isf(q, self.alpha, self.beta, 0, 1)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.beta.sf(x, self.alpha, self.beta, 0, 1)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        print(
            "Descriptive statistics for Beta distribution with alpha =",
            self.alpha,
            "and beta =",
            self.beta,
        )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.beta.rvs(self.alpha, self.beta, 0, 1, size=number_of_samples)
        return RVS


class Loglogistic_Distribution:
    """
    Loglogistic probability distribution. Creates a probability distribution
    object.

    Parameters
    ----------
    alpha : float, int
        Scale parameter. Must be > 0
    beta : float, int
        Shape parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Loglogistic'
    name2 : 'str
        'Loglogistic_2P' or 'Loglogistic_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Loglogistic Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta,gamma]
    alpha : float
    beta : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, alpha=None, beta=None, gamma=0, **kwargs):
        self.name = "Loglogistic"
        if alpha is None or beta is None:
            raise ValueError(
                "Parameters alpha and beta must be specified. Eg. Loglogistic_Distribution(alpha=5,beta=2)"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.parameters = np.array([self.alpha, self.beta, self.gamma])

        if self.beta > 1:
            self.mean = float(
                ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="m")
            )
        else:
            self.mean = r"no mean when $\beta \leq 1$"
        if self.beta > 2:
            self.variance = float(
                ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="v")
            )
            self.standard_deviation = self.variance ** 0.5
        else:
            self.variance = r"no variance when $\beta \leq 2$"
            self.standard_deviation = r"no stdev when $\beta \leq 2$"
        if self.beta > 3:
            self.skewness = float(
                ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="s")
            )
        else:
            self.skewness = r"no skewness when $\beta \leq 3$"
        if self.beta > 4:
            self.excess_kurtosis = float(
                ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="k")
            )
            self.kurtosis = self.excess_kurtosis + 3
        else:
            self.excess_kurtosis = r"no kurtosis when $\beta \leq 4$"
            self.kurtosis = r"no kurtosis when $\beta \leq 4$"

        self.median = ss.fisk.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = (
                self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta)
                + self.gamma
            )
        else:
            self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
            )
            self.param_title_long = str(
                "Loglogistic Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ",γ="
                + str(round_to_decimals(self.gamma, dec))
                + ")"
            )
            self.name2 = "Loglogistic_3P"
        else:
            self.param_title = str(
                "α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
            )
            self.param_title_long = str(
                "Loglogistic Distribution (α="
                + str(round_to_decimals(self.alpha, dec))
                + ",β="
                + str(round_to_decimals(self.beta, dec))
                + ")"
            )
            self.name2 = "Loglogistic_2P"
        self.b5 = ss.fisk.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.fisk.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "alpha_SE" in kwargs:
            self.alpha_SE = kwargs.pop("alpha_SE")
        else:
            self.alpha_SE = None
        if "beta_SE" in kwargs:
            self.beta_SE = kwargs.pop("beta_SE")
        else:
            self.beta_SE = None
        if "Cov_alpha_beta" in kwargs:
            self.Cov_alpha_beta = kwargs.pop("Cov_alpha_beta")
        else:
            self.Cov_alpha_beta = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING:"
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are alpha_SE, beta_SE, Cov_alpha_beta, CI, and CI_type."
                ),
                text_color="red",
            )
        self._pdf0 = ss.fisk.pdf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.fisk.pdf(0, self.beta, scale=self.alpha, loc=0) / ss.fisk.sf(
            0, self.beta, scale=self.alpha, loc=0
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """

        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.fisk.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.fisk.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.fisk.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = (self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (
            self.beta - 1
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        chf = np.log(1 + ((X - self.gamma) / self.alpha) ** self.beta)
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        plt.figure(figsize=(9, 7))
        text_title = str("Loglogistic Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))

        if type(self.mean) == str:
            text_mean = str("Mean = " + str(self.mean))  # required when mean is str
        else:
            text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))

        if type(self.standard_deviation) == str:
            text_std = str(
                "Standard deviation = " + str(self.standard_deviation)
            )  # required when standard deviation is str
        else:
            text_std = str(
                "Standard deviation = "
                + str(round_to_decimals(float(self.standard_deviation), dec))
            )

        if type(self.variance) == str:
            text_var = str(
                "Variance = " + str(self.variance)
            )  # required when variance is str
        else:
            text_var = str(
                "Variance = " + str(round_to_decimals(float(self.variance), dec))
            )

        if type(self.skewness) == str:
            text_skew = str(
                "Skewness = " + str(self.skewness)
            )  # required when skewness is str
        else:
            text_skew = str(
                "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
            )

        if type(self.excess_kurtosis) == str:
            text_ex_kurt = str(
                "Excess kurtosis = " + str(self.excess_kurtosis)
            )  # required when excess kurtosis is str
        else:
            text_ex_kurt = str(
                "Excess kurtosis = "
                + str(round_to_decimals(float(self.excess_kurtosis), dec))
            )

        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        pdf = ss.fisk.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Loglogistic Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf = ss.fisk.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Loglogistic Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        sf = ss.fisk.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Loglogistic Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        hf = (self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (
            self.beta - 1
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Loglogistic Distribution\n"
                + " Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        # obtain the X array
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

        chf = np.log(1 + ((X - self.gamma) / self.alpha) ** self.beta)
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Loglogistic Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.fisk.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.fisk.isf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.fisk.sf(x, self.beta, scale=self.alpha, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            print(
                "Descriptive statistics for Weibull distribution with alpha =",
                self.alpha,
                "and beta =",
                self.beta,
            )
        else:
            print(
                "Descriptive statistics for Weibull distribution with alpha =",
                self.alpha,
                ", beta =",
                self.beta,
                ", and gamma =",
                self.gamma,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.fisk.rvs(
            self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples
        )
        return RVS


class Gumbel_Distribution:
    """
    Gumbel probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    mu : float, int
        Location parameter
    sigma : float, int
        Scale parameter. Must be > 0

    Returns
    -------
    name : str
        'Gumbel'
    name2 : 'str
        'Gumbel_2P'
    param_title_long : str
        'Gumbel Distribution (μ=5,σ=2)'
    param_title : str
        'μ=5,σ=2'
    parameters : list
        [mu,sigma]
    mu : float
    sigma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals
    """

    def __init__(self, mu=None, sigma=None, **kwargs):
        self.name = "Gumbel"
        self.name2 = "Gumbel_2P"
        if mu is None or sigma is None:
            raise ValueError(
                "Parameters mu and sigma must be specified. Eg. Gumbel_Distribution(mu=5,sigma=2)"
            )
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.parameters = np.array([self.mu, self.sigma])
        mean, var, skew, kurt = ss.gumbel_l.stats(self.mu, self.sigma, moments="mvsk")
        self.mean = float(mean)
        self.standard_deviation = float(var ** 0.5)
        self.variance = float(var)
        self.skewness = float(skew)
        self.kurtosis = float(kurt + 3)
        self.excess_kurtosis = float(kurt)
        self.median = mu + sigma * np.log(np.log(2))
        self.mode = mu
        self.param_title = str(
            "μ="
            + str(round_to_decimals(self.mu, dec))
            + ",σ="
            + str(round_to_decimals(self.sigma, dec))
        )
        self.param_title_long = str(
            "Gumbel Distribution (μ="
            + str(round_to_decimals(self.mu, dec))
            + ",σ="
            + str(round_to_decimals(self.sigma, dec))
            + ")"
        )
        self.b5 = ss.gumbel_l.ppf(0.05, loc=self.mu, scale=self.sigma)
        self.b95 = ss.gumbel_l.ppf(0.95, loc=self.mu, scale=self.sigma)

        # extracts values for confidence interval plotting
        if "mu_SE" in kwargs:
            self.mu_SE = kwargs.pop("mu_SE")
        else:
            self.mu_SE = None
        if "sigma_SE" in kwargs:
            self.sigma_SE = kwargs.pop("sigma_SE")
        else:
            self.sigma_SE = None
        if "Cov_mu_sigma" in kwargs:
            self.Cov_mu_sigma = kwargs.pop("Cov_mu_sigma")
        else:
            self.Cov_mu_sigma = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs.keys():
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + "is not recognised as an appropriate entry in kwargs. Appropriate entries are mu_SE, sigma_SE, Cov_mu_sigma, CI, and CI_type."
                ),
                text_color="red",
            )

        self._pdf0 = 0  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = 0  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        X = generate_X_array(
            dist=self, xvals=xvals, xmin=xmin, xmax=xmax
        )  # obtain the X array

        pdf = ss.gumbel_l.pdf(X, self.mu, self.sigma)
        cdf = ss.gumbel_l.cdf(X, self.mu, self.sigma)
        sf = ss.gumbel_l.sf(X, self.mu, self.sigma)
        hf = np.exp((X - self.mu) / self.sigma) / self.sigma
        chf = np.exp((X - self.mu) / self.sigma)

        plt.figure(figsize=(9, 7))
        text_title = str("Gumbel Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = "
            + str(round_to_decimals(self.standard_deviation, dec))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        pdf = ss.gumbel_l.pdf(X, self.mu, self.sigma)

        if show_plot == False:
            return pdf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Gumbel Distribution\n"
                + " Probability Density Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        cdf = ss.gumbel_l.cdf(X, self.mu, self.sigma)

        if show_plot == False:
            return cdf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Gumbel Distribution\n"
                + " Cumulative Distribution Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gumbel_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        sf = ss.gumbel_l.sf(X, self.mu, self.sigma)

        if show_plot == False:
            return sf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(
                "Gumbel Distribution\n"
                + " Survival Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gumbel_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        hf = np.exp((X - self.mu) / self.sigma) / self.sigma

        if show_plot == False:
            return hf
        else:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(
                "Gumbel Distribution\n" + " Hazard Function " + "\n" + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        if (
            xmin is None
            and xmax is None
            and type(xvals) not in [list, np.ndarray, type(None)]
        ):
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(
                dist=self, xvals=xvals, xmin=xmin, xmax=xmax
            )  # obtain the X array

        chf = np.exp((X - self.mu) / self.sigma)
        self._X = X
        self._chf = chf

        if show_plot == False:
            return chf
        else:
            CI_type, plot_CI, CI = distribution_confidence_intervals.CI_kwarg_handler(
                self, kwargs
            )

            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(
                "Gumbel Distribution\n"
                + " Cumulative Hazard Function "
                + "\n"
                + self.param_title
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.gumbel_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

            return chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.gumbel_l.ppf(q, loc=self.mu, scale=self.sigma)

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return ss.gumbel_l.isf(q, loc=self.mu, scale=self.sigma)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """
        R = lambda x: ss.gumbel_l.sf(x, loc=self.mu, scale=self.sigma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        print(
            "Descriptive statistics for Gumbel distribution with mu =",
            self.mu,
            "and sigma =",
            self.sigma,
        )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.gumbel_l.rvs(loc=self.mu, scale=self.sigma, size=number_of_samples)
        return RVS


class Competing_Risks_Model:
    """
    The competing risks model is used to model the effect of multiple risks
    (expressed as probability distributions) that act on a system over time.
    The model is obtained using the product of the survival functions:

    :math:`SF_{total} = SF_1 × SF_2 × SF_3 × ... × SF_n`

    The output API is similar to the other probability distributions (Weibull,
    Normal, etc.) as shown below.

    Parameters
    ----------
    distributions : list, array
        a list or array of probability distribution objects used to construct
        the model

    Returns
    -------
    name : str
        'Competing risks'
    name2 : str
        'Competing risks using 3 distributions'. The exact name depends on the
        number of distributions used
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    An equivalent form of this model is to sum the hazard or cumulative hazard
    functions which will give the same result. In this way, we see the CDF, HF,
    and CHF of the overall model being equal to or higher than any of the
    constituent distributions. Similarly, the SF of the overall model will
    always be equal to or lower than any of the constituent distributions.
    The PDF occurs earlier in time since the earlier risks cause the population
    to fail sooner leaving less to fail due to the later risks.

    This model should be used when a data set has been divided by failure mode
    and each failure mode has been modelled separately. The competing risks
    model can then be used to recombine the constituent distributions into a
    single model. Unlike the mixture model, there are no proportions as the
    risks are competing to cause failure rather than being mixed.

    As this process is multiplicative for the survival function, and may accept
    many distributions of different types, the mathematical formulation quickly
    gets complex. For this reason, the algorithm combines the models numerically
    rather than empirically so there are no simple formulas for many of the
    descriptive statistics (mean, median, etc.). Also, the accuracy of the model
    is dependent on xvals. If the xvals array is small (<100 values) then the
    answer will be 'blocky' and inaccurate. The variable xvals is only accepted
    for PDF, CDF, SF, HF, CHF. The other methods (like random samples) use the
    default xvals for maximum accuracy. The default number of values generated
    when xvals is not given is 1000. Consider this carefully when specifying
    xvals in order to avoid inaccuracies in the results.
    """

    def __init__(self, distributions):
        if type(distributions) not in [list, np.ndarray]:
            raise ValueError(
                "distributions must be a list or array of distribution objects."
            )
        contains_normal_or_gumbel = False
        for dist in distributions:
            if type(dist) not in [
                Weibull_Distribution,
                Normal_Distribution,
                Lognormal_Distribution,
                Exponential_Distribution,
                Beta_Distribution,
                Gamma_Distribution,
                Loglogistic_Distribution,
                Gumbel_Distribution,
            ]:
                raise ValueError(
                    "distributions must be an array or list of probability distributions. Each distribution must be created using the reliability.Distributions module."
                )
            if type(dist) in [Normal_Distribution, Gumbel_Distribution]:
                contains_normal_or_gumbel = True  # check if we can have negative xvals (allowable if only normal and gumbel are in the mixture)
        self.__contains_normal_or_gumbel = contains_normal_or_gumbel
        self.distributions = distributions  # this just passes the distributions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.name = "Competing risks"
        self.num_dists = len(distributions)
        self.name2 = str(
            "Competing risks using " + str(self.num_dists) + " distributions"
        )

        # This is essentially just the same as the __combiner method but more automated with a high amount of detail for the X array to minimize errors
        xmax = -1e100
        xmin = 1e100
        xmax999 = -1e100
        xmin001 = 1e100
        xmax_inf = -1e100
        for dist in distributions:
            xmax = max(xmax, dist.quantile(1 - 1e-10))
            xmin = min(xmin, dist.quantile(1e-10))
            xmax999 = max(xmax999, dist.quantile(0.999))
            xmin001 = min(xmin001, dist.quantile(0.001))
            xmax_inf = max(
                xmax_inf, dist.quantile(1 - 1e-10)
            )  # effective infinity used by MRL
        self.__xmax999 = xmax999
        self.__xmin001 = xmin001
        self.__xmax_inf = xmax_inf

        X = np.linspace(xmin, xmax, 1000000)
        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative_zeros = np.zeros_like(X_negative)
        Y_negative_ones = np.ones_like(X_negative)

        sf = np.ones_like(X)
        hf = np.zeros_like(X)
        # combine the distributions using the product of the survival functions: SF_total = SF_1 x SF_2 x SF_3 x ....x SF_n
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                sf *= distributions[i].SF(X, show_plot=False)
                hf += distributions[i].HF(X, show_plot=False)
            else:
                sf *= np.hstack(
                    [Y_negative_ones, distributions[i].SF(X_positive, show_plot=False)]
                )
                hf += np.hstack(
                    [Y_negative_zeros, distributions[i].HF(X_positive, show_plot=False)]
                )
        pdf = hf * sf
        np.nan_to_num(
            pdf, copy=False, nan=0.0, posinf=None, neginf=None
        )  # because the hf is nan (which is expected due to being pdf/sf=0/0)

        self.__xvals_init = X  # used by random_samples
        self.__pdf_init = pdf  # used by random_samples
        self.__sf_init = sf  # used by quantile and inverse_SF
        self.mean = integrate.simps(pdf * X, x=X)
        self.standard_deviation = (
            integrate.simps(pdf * (X - self.mean) ** 2, x=X)
        ) ** 0.5
        self.variance = self.standard_deviation ** 2
        self.skewness = integrate.simps(
            pdf * ((X - self.mean) / self.standard_deviation) ** 3, x=X
        )
        self.kurtosis = integrate.simps(
            pdf * ((X - self.mean) / self.standard_deviation) ** 4, x=X
        )
        self.mode = X[np.argmax(pdf)]
        self.median = X[np.argmin(abs(sf - 0.5))]
        self.excess_kurtosis = self.kurtosis - 3
        self.b5 = X[np.argmin(abs((1 - sf) - 0.05))]
        self.b95 = X[np.argmin(abs((1 - sf) - 0.95))]

    def __combiner(self, xvals=None, xmin=None, xmax=None):
        """
        This is a hidden function used to combine the distributions numerically.
        It is necessary to do this outside of the __init__ method as it needs to be called by each function (PDF, CDF...) so that xvals is used consistently.
        This approach keeps the API the same as the other probability distributions.
        Users should never need to access this function directly.
        """
        distributions = self.distributions

        # obtain the X values
        if xvals is not None:
            X = xvals
        else:
            if xmin is None:
                if (
                    self.__xmin001 > 0
                    and self.__xmin001 - (self.__xmax999 - self.__xmin001) * 0.3 < 0
                ):
                    xmin = 0  # if its positive but close to zero then just make it zero
                else:
                    xmin = self.__xmin001
            if xmax is None:
                xmax = self.__xmax999
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            X = np.linspace(
                xmin, xmax, 1000
            )  # this is a big array because everything is numerical rather than empirical. Small array sizes will lead to blocky (inaccurate) results.

        # convert to numpy array if given list. raise error for other types. check for values below 0.
        if type(X) not in [np.ndarray, list]:
            raise ValueError("unexpected type in xvals. Must be  list, or array")
        else:
            X = np.asarray(X)
        if min(X) < 0 and self.__contains_normal_or_gumbel is False:
            raise ValueError(
                "xvals was found to contain values below 0. This is only allowed if some of the mixture components are Normal or Gumbel distributions."
            )

        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative_zeros = np.zeros_like(X_negative)
        Y_negative_ones = np.ones_like(X_negative)

        sf = np.ones_like(X)
        hf = np.zeros_like(X)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                sf *= distributions[i].SF(X, show_plot=False)
                hf += distributions[i].HF(X, show_plot=False)
            else:
                sf *= np.hstack(
                    [Y_negative_ones, distributions[i].SF(X_positive, show_plot=False)]
                )
                hf += np.hstack(
                    [Y_negative_zeros, distributions[i].HF(X_positive, show_plot=False)]
                )
        pdf = sf * hf
        np.nan_to_num(
            pdf, copy=False, nan=0.0, posinf=None, neginf=None
        )  # because the hf may contain nan (which is expected due to being pdf/sf=0/0 for high xvals)

        # these are all hidden to the user but can be accessed by the other functions in this module
        self.__xvals = X
        self.__pdf = pdf
        self.__cdf = 1 - sf
        self.__sf = sf
        self.__hf = hf
        self.__chf = -np.log(sf)
        self._pdf0 = pdf[0]
        self._hf0 = hf[0]

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)
        plt.figure(figsize=(9, 7))
        text_title = str("Competing Risks Model")
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(self.__xvals, self.__pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=self.__xvals,
            Y=self.__pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(self.__xvals, self.__cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=self.__xvals,
            Y=self.__cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(self.__xvals, self.__sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=self.__xvals,
            Y=self.__sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(self.__xvals, self.__hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=self.__xvals,
            Y=self.__hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(self.__xvals, self.__chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=self.__xvals,
            Y=self.__chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):

        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__pdf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.PDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.PDF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__pdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(
                "Competing Risks Model\n" + " Probability Density Function"
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=self.__xvals,
                Y=self.__pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__pdf

    def CDF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """

        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__cdf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CDF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__cdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Competing Risks Model\n" + " Cumulative Distribution Function"
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=self.__xvals,
                Y=self.__cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__cdf

    def SF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__sf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.SF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.SF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__sf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Competing Risks Model\n" + " Survival Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=self.__xvals,
                Y=self.__sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__sf

    def HF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__hf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.HF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.HF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__hf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Competing Risks Model\n" + " Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=self.__xvals,
                Y=self.__hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__hf

    def CHF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__chf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CHF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CHF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__chf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative Hazard")
            text_title = str("Competing Risks Model\n" + " Cumulative Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=self.__xvals,
                Y=self.__chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return self.__xvals_init[np.argmin(abs((1 - self.__sf_init) - q))]

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return self.__xvals_init[np.argmin(abs(self.__sf_init - q))]

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        print("Descriptive statistics for Competing Risks Model")
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """

        def __subcombiner(X):
            """
            This function does what __combiner does but more efficiently and
            also accepts single values
            """
            if type(X) == np.ndarray:
                sf = np.ones_like(X)
                X_positive = X[X >= 0]
                X_negative = X[X < 0]
                Y_negative = np.ones_like(X_negative)
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        sf *= self.distributions[i].SF(X, show_plot=False)
                    else:
                        sf *= np.hstack(
                            [
                                Y_negative,
                                self.distributions[i].SF(X_positive, show_plot=False),
                            ]
                        )
            else:
                sf = 1
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        sf *= self.distributions[i].SF(X, show_plot=False)
                    elif X > 0:
                        sf *= self.distributions[i].SF(X, show_plot=False)
            return sf

        t_full = np.linspace(t, self.__xmax_inf, 1000000)
        sf_full = __subcombiner(t_full)
        sf_single = __subcombiner(t)
        MRL = integrate.simps(sf_full, x=t_full) / sf_single
        return MRL

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(
            self.__xvals_init,
            size=number_of_samples,
            p=self.__pdf_init / sum(self.__pdf_init),
        )


class Mixture_Model:
    """
    The mixture model is used to create a distribution that contains parts from
    multiple distributions. This allows for a more complex model to be
    constructed as the sum of other distributions, each multiplied by a
    proportion (where the proportions sum to 1). The model is obtained using the
    sum of the cumulative distribution functions:

    :math:`CDF_{total} = (CDF_1 × p_1) + (CDF_2 × p_2) + (CDF_3 × p_3) + ... + (CDF_n × p_n)`

    The output API is similar to the other probability distributions (Weibull,
    Normal, etc.) as shown below.

    Parameters
    ----------
    distributions : list, array
        List or array of probability distribution objects used to construct the
        model.
    proportions : list, array
        List or array of floats specifying how much of each distribution to
        add to the mixture. The sum of proportions must always be 1.

    Returns
    -------
    name : str
        'Mixture'
    name2 : str
        'Mixture using 3 distributions'. The exact name depends on the number of
        distributions used.
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    An equivalent form of this model is to sum the PDF. SF is obtained as 1-CDF.
    Note that you cannot simply sum the HF or CHF as this method would be
    equivalent to the competing risks model. In this way, we see the mixture
    model will always lie somewhere between the constituent models.

    This model should be used when a data set cannot be modelled by a single
    distribution, as evidenced by the shape of the PDF, CDF or probability plot
    (points do not form a straight line). Unlike the competing risks model, this
    model requires the proportions to be supplied.

    As this process is additive for the survival function, and may accept many
    distributions of different types, the mathematical formulation quickly gets
    complex. For this reason, the algorithm combines the models numerically
    ather than empirically so there are no simple formulas for many of the
    descriptive statistics (mean, median, etc.). Also, the accuracy of the model
    is dependent on xvals. If the xvals array is small (<100 values) then the
    answer will be 'blocky' and inaccurate. The variable xvals is only accepted
    for PDF, CDF, SF, HF, CHF. The other methods (like random samples) use the
    default xvals for maximum accuracy. The default number of values generated
    when xvals is not given is 1000. Consider this carefully when specifying
    xvals in order to avoid inaccuracies in the results.
    """

    def __init__(self, distributions, proportions=None):
        if type(distributions) not in [list, np.ndarray]:
            raise ValueError(
                "distributions must be a list or array of distribution objects."
            )
        contains_normal_or_gumbel = False
        for dist in distributions:
            if type(dist) not in [
                Weibull_Distribution,
                Normal_Distribution,
                Lognormal_Distribution,
                Exponential_Distribution,
                Beta_Distribution,
                Gamma_Distribution,
                Loglogistic_Distribution,
                Gumbel_Distribution,
            ]:
                raise ValueError(
                    "distributions must be an array or list of probability distributions. Each distribution must be created using the reliability.Distributions module."
                )
            if type(dist) in [Normal_Distribution, Gumbel_Distribution]:
                contains_normal_or_gumbel = True  # check if we can have negative xvals (allowable if only normal and gumbel are in the mixture)
        self.__contains_normal_or_gumbel = contains_normal_or_gumbel

        if proportions is not None:
            if sum(proportions) != 1:
                raise ValueError("the sum of the proportions must be 1")
            if len(proportions) != len(distributions):
                raise ValueError(
                    "the length of the proportions array must match the length of the distributions array"
                )
        else:
            proportions = np.ones_like(distributions) / len(
                distributions
            )  # if proportions are not specified they are assumed to all be the same proportion

        self.proportions = proportions  # this just passes the proportions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.distributions = distributions  # this just passes the distributions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.name = "Mixture"
        self.num_dists = len(distributions)
        self.name2 = str("Mixture using " + str(self.num_dists) + " distributions")

        # This is essentially just the same as the __combiner method but more automated with a high amount of detail for the X array to minimize errors
        xmax = -1e100
        xmin = 1e100
        xmax999 = -1e100
        xmin001 = 1e100
        xmax_inf = -1e100
        for dist in distributions:
            xmax = max(xmax, dist.quantile(1 - 1e-10))
            xmin = min(xmin, dist.quantile(1e-10))
            xmax999 = max(xmax999, dist.quantile(0.999))
            xmin001 = min(xmin001, dist.quantile(0.001))
            xmax_inf = max(
                xmax_inf, dist.quantile(1 - 1e-10)
            )  # effective infinity used by MRL
        self.__xmax999 = xmax999
        self.__xmin001 = xmin001
        self.__xmax_inf = xmax_inf

        X = np.linspace(xmin, xmax, 1000000)
        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative = np.zeros_like(X_negative)

        cdf = np.zeros_like(X)
        pdf = np.zeros_like(X)
        # combine the distributions using the sum of the cumulative distribution functions: CDF_total = (CDF_1 x p_1) + (CDF_2 x p2) x (CDF_3 x p3) + .... + (CDF_n x pn)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                cdf += distributions[i].CDF(X, show_plot=False) * proportions[i]
                pdf += distributions[i].PDF(X, show_plot=False) * proportions[i]
            else:
                cdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].CDF(X_positive, show_plot=False)
                        * proportions[i],
                    ]
                )
                pdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].PDF(X_positive, show_plot=False)
                        * proportions[i],
                    ]
                )
        self.__pdf_init = pdf
        self.__cdf_init = cdf
        self.__xvals_init = X
        self.mean = integrate.simps(pdf * X, x=X)
        self.standard_deviation = (
            integrate.simps(pdf * (X - self.mean) ** 2, x=X)
        ) ** 0.5
        self.variance = self.standard_deviation ** 2
        self.skewness = integrate.simps(
            pdf * ((X - self.mean) / self.standard_deviation) ** 3, x=X
        )
        self.kurtosis = integrate.simps(
            pdf * ((X - self.mean) / self.standard_deviation) ** 4, x=X
        )
        self.mode = X[np.argmax(pdf)]
        self.median = X[np.argmin(abs((1 - cdf) - 0.5))]
        self.excess_kurtosis = self.kurtosis - 3
        self.b5 = X[np.argmin(abs(cdf - 0.05))]
        self.b95 = X[np.argmin(abs(cdf - 0.95))]

    def __combiner(self, xvals=None, xmin=None, xmax=None):
        """
        This is a hidden function used to combine the distributions numerically.
        It is necessary to do this outside of the __init__ method as it needs to be called by each function (PDF, CDF...) so that xvals is used consistently.
        This approach keeps the API the same as the other probability distributions.
        Users should never need to access this function directly.
        """
        distributions = self.distributions
        proportions = self.proportions

        # obtain the X values
        if xvals is not None:
            X = xvals
        else:
            if xmin is None:
                if (
                    self.__xmin001 > 0
                    and self.__xmin001 - (self.__xmax999 - self.__xmin001) * 0.3 < 0
                ):
                    xmin = 0  # if its positive but close to zero then just make it zero
                else:
                    xmin = self.__xmin001
            if xmax is None:
                xmax = self.__xmax999
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            X = np.linspace(
                xmin, xmax, 1000
            )  # this is a big array because everything is numerical rather than empirical. Small array sizes will lead to blocky (inaccurate) results.

        # convert to numpy array if given list. raise error for other types. check for values below 0.
        if type(X) not in [np.ndarray, list]:
            raise ValueError("unexpected type in xvals. Must be list or array")
        else:
            X = np.asarray(X)
        if min(X) < 0 and self.__contains_normal_or_gumbel is False:
            raise ValueError(
                "xvals was found to contain values below 0. This is only allowed if some of the mixture components are Normal or Gumbel distributions."
            )

        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative = np.zeros_like(X_negative)

        cdf = np.zeros_like(X)
        pdf = np.zeros_like(X)
        # combine the distributions using the sum of the cumulative distribution functions: CDF_total = (CDF_1 x p_1) + (CDF_2 x p2) x (CDF_3 x p3) + .... + (CDF_n x pn)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                cdf += distributions[i].CDF(X, show_plot=False) * proportions[i]
                pdf += distributions[i].PDF(X, show_plot=False) * proportions[i]
            else:
                cdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].CDF(X_positive, show_plot=False)
                        * proportions[i],
                    ]
                )
                pdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].PDF(X_positive, show_plot=False)
                        * proportions[i],
                    ]
                )

        # these are all hidden to the user but can be accessed by the other functions in this module
        hf = pdf / (1 - cdf)
        self.__xvals = X
        self.__pdf = pdf
        self.__cdf = cdf
        self.__sf = 1 - cdf
        self.__hf = hf
        self.__chf = -np.log(1 - cdf)
        self._pdf0 = pdf[0]
        self._hf0 = hf[0]

    def plot(self, xvals=None, xmin=None, xmax=None):
        """
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.
        """

        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        plt.figure(figsize=(9, 7))
        text_title = str("Mixture Model")
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(self.__xvals, self.__pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="PDF",
            X=self.__xvals,
            Y=self.__pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(self.__xvals, self.__cdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CDF",
            X=self.__xvals,
            Y=self.__cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(self.__xvals, self.__sf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="SF",
            X=self.__xvals,
            Y=self.__sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(self.__xvals, self.__hf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="HF",
            X=self.__xvals,
            Y=self.__hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(self.__xvals, self.__chf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self,
            func="CHF",
            X=self.__xvals,
            Y=self.__chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + str(round_to_decimals(float(self.mean), dec)))
        text_median = str("Median = " + str(round_to_decimals(self.median, dec)))
        text_mode = str("Mode = " + str(round_to_decimals(self.mode, dec)))
        text_b5 = str("$5^{th}$ quantile = " + str(round_to_decimals(self.b5, dec)))
        text_b95 = str("$95^{th}$ quantile = " + str(round_to_decimals(self.b95, dec)))
        text_std = str(
            "Standard deviation = " + str(round_to_decimals(self.standard_deviation))
        )
        text_var = str(
            "Variance = " + str(round_to_decimals(float(self.variance), dec))
        )
        text_skew = str(
            "Skewness = " + str(round_to_decimals(float(self.skewness), dec))
        )
        text_ex_kurt = str(
            "Excess kurtosis = "
            + str(round_to_decimals(float(self.excess_kurtosis), dec))
        )
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__pdf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.PDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.PDF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Mixture model"

            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__pdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Mixture Model\n" + " Probability Density Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=self.__xvals,
                Y=self.__pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__pdf

    def CDF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__cdf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CDF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Mixture model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__cdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str("Mixture Model\n" + " Cumulative Distribution Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=self.__xvals,
                Y=self.__cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__cdf

    def SF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__sf
        else:
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.SF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.SF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Mixture model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__sf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Mixture Model\n" + " Survival Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=self.__xvals,
                Y=self.__sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__sf

    def HF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__hf
        else:
            limits = get_axes_limits()
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.HF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.HF(xvals=self.__xvals, label=dist.param_title_long)
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Mixture model"
            plt.plot(self.__xvals, self.__hf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Mixture Model\n" + " Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=self.__xvals,
                Y=self.__hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__hf

    def CHF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_components=False,
        **kwargs
    ):
        """
        Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array
            The y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).

        Notes
        -----
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.
        """
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if show_plot == False:
            return self.__chf
        else:
            limits = get_axes_limits()
            if (
                plot_components is True
            ):  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CHF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CHF(xvals=self.__xvals, label=dist.param_title_long)
                        print("here")
            if "label" in kwargs:
                textlabel = kwargs.pop("label")
            else:
                textlabel = "Mixture model"
            plt.plot(self.__xvals, self.__chf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative Hazard")
            text_title = str("Mixture Model\n" + " Cumulative Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=self.__xvals,
                Y=self.__chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            return self.__chf

    def quantile(self, q):
        """
        Quantile calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return self.__xvals_init[np.argmin(abs(self.__cdf_init - q))]

    def inverse_SF(self, q):
        """
        Inverse survival function calculator

        Parameters
        ----------
        q : float
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.
        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type int, float, list, array")
        return self.__xvals_init[np.argmin(abs((1 - self.__cdf_init) - q))]

    def stats(self):
        """
        Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        print("Descriptive statistics for Mixture Model")
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def mean_residual_life(self, t):
        """
        Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life
        """

        def __subcombiner(X):
            """
            This function does what __combiner does but more efficiently and
            also accepts single values.
            """
            if type(X) == np.ndarray:
                cdf = np.zeros_like(X)
                X_positive = X[X >= 0]
                X_negative = X[X < 0]
                Y_negative = np.zeros_like(X_negative)
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        cdf += (
                            self.distributions[i].CDF(X, show_plot=False)
                            * self.proportions[i]
                        )
                    else:
                        cdf += np.hstack(
                            [
                                Y_negative,
                                self.distributions[i].CDF(X_positive, show_plot=False)
                                * self.proportions[i],
                            ]
                        )
            else:
                cdf = 0
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        cdf += (
                            self.distributions[i].CDF(X, show_plot=False)
                            * self.proportions[i]
                        )
                    elif X > 0:
                        cdf += (
                            self.distributions[i].CDF(X, show_plot=False)
                            * self.proportions[i]
                        )
            return 1 - cdf

        t_full = np.linspace(t, self.__xmax_inf, 1000000)
        sf_full = __subcombiner(t_full)
        sf_single = __subcombiner(t)
        MRL = integrate.simps(sf_full, x=t_full) / sf_single
        return MRL

    def random_samples(self, number_of_samples, seed=None):
        """
        Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats
        """
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(
            self.__xvals_init,
            size=number_of_samples,
            p=self.__pdf_init / sum(self.__pdf_init),
        )
