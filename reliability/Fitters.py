"""
Fitters

This module contains custom fitting functions for parametric distributions which
support complete and right censored data.
The included functions are:

- Fit_Weibull_2P
- Fit_Weibull_3P
- Fit_Exponential_1P
- Fit_Exponential_2P
- Fit_Gamma_2P
- Fit_Gamma_3P
- Fit_Lognormal_2P
- Fit_Lognormal_3P
- Fit_Normal_2P
- Fit_Gumbel_2P
- Fit_Beta_2P
- Fit_Loglogistic_2P
- Fit_Loglogistic_3P
- Fit_Weibull_Mixture
- Fit_Weibull_CR
- Fit_Weibull_DS
- Fit_Weibull_ZI
- Fit_Weibull_DSZI

Note that the Beta distribution is only for data in the range 0 < t < 1.
There is also a Fit_Everything function which will fit all distributions (except
the Weibull_Mixture and Weibull_CR models) and will provide plots and a table of
values.

All functions in this module work using autograd to find the derivative of the
log-likelihood function. In this way, the code only needs to specify the log PDF
and log SF in order to obtain the fitted parameters. Initial guesses of the
parameters are essential for autograd and are obtained using least squares or
non-linear least squares (depending on the function). If the distribution is an
extremely bad fit or is heavily censored (>99%) then these guesses may be poor
and the fit might not be successful. Generally the fit achieved by autograd is
highly successful, and whenever it fails the initial guess will be used and a
warning will be displayed.
"""

import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as ss
from reliability.Distributions import (
    Weibull_Distribution,
    Gamma_Distribution,
    Beta_Distribution,
    Exponential_Distribution,
    Normal_Distribution,
    Lognormal_Distribution,
    Loglogistic_Distribution,
    Gumbel_Distribution,
    Mixture_Model,
    Competing_Risks_Model,
    DSZI_Model,
)
from reliability.Nonparametric import KaplanMeier
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import (
    round_to_decimals,
    anderson_darling,
    fitters_input_checking,
    fitters_input_checking,
    colorprint,
    least_squares,
    MLE_optimization,
    LS_optimization,
    xy_downsample,
    extract_CI,
)
import autograd.numpy as anp
from autograd import value_and_grad
from autograd.differential_operators import hessian
from autograd.scipy.special import gamma as agamma
from autograd.scipy.special import beta as abeta
from autograd.scipy.special import erf
from autograd_gamma import betainc
from autograd_gamma import gammaincc

anp.seterr("ignore")
dec = 3  # number of decimals to use when rounding fitted parameters in labels

# change pandas display options
pd.options.display.float_format = (
    "{:g}".format
)  # improves formatting of numbers in dataframe
pd.options.display.max_columns = 9  # shows the dataframe without ... truncation
pd.options.display.width = 200  # prevents wrapping after default 80 characters


class Fit_Everything:
    """
    This function will fit all available distributions to the data provided.
    The only distributions not fitted are Weibull_DSZI and Weibull_ZI. The
    Beta_2P distribution will only be fitted if the data are between 0 and 1.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements for all the 2 parameter
        distributions to be fitted and 3 elements for all distributions to be
        fitted.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    sort_by : str
        Goodness of fit test to sort results by. Must be 'BIC','AICc','AD', or
        'Log-likelihood'. Default is BIC.
    show_probability_plot : bool, optional
        Provides a probability plot of each of the fitted distributions. True or
        False. Default = True
    show_histogram_plot : bool, optional
        True or False. Default = True. Will show a histogram (scaled to account
        for censored data) with the PDF and CDF of each fitted distribution.
    show_PP_plot : bool, optional
        Provides a comparison of parametric vs non-parametric fit using
        Probability-Probability (PP) plot. True or False. Default = True.
    show_best_distribution_probability_plot : bool, optional
        Provides a probability plot in a new figure of the best fitting
        distribution. True or False. Default = True.
    exclude : list, array, optional
        List or array of strings specifying which distributions to exclude.
        Default is None. Options are Weibull_2P, Weibull_3P, Weibull_CR,
        Weibull_Mixture, Weibull_DS, Normal_2P, Gamma_2P, Loglogistic_2P,
        Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P,
        Exponential_2P, Exponential_1P, Beta_2P.
    print_results : bool, optional
        Will show the results of the fitted parameters and the goodness of fit
        tests in a dataframe. True/False. Defaults to True.
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.

    Returns
    -------
    results : dataframe
        a pandas dataframe of results. Fitted parameters in this dataframe may
        be accessed by name. See below example in Notes.
    best_distribution : object
        a reliability.Distributions object created based on the parameters of
        the best fitting distribution.
    best_distribution_name : str
        the name of the best fitting distribution. E.g. 'Weibull_3P'
    parameters and goodness of fit results : float
        This is provided for each fitted distribution. For example, the
        Weibull_3P distribution values are Weibull_3P_alpha, Weibull_3P_beta,
        Weibull_3P_gamma, Weibull_3P_BIC, Weibull_3P_AICc, Weibull_3P_AD,
        Weibull_3P_loglik
    excluded_distributions : list
        a list of strings of the excluded distributions.

    Notes
    -----
    All parametric models have the number of parameters in the name. For
    example, Weibull_2P uses alpha and beta, whereas Weibull_3P uses alpha,
    beta, and gamma. This is applied even for Normal_2P for consistency in
    naming conventions. From the results, the distributions are sorted based on
    their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the distribution to the data.

    If the data provided contains only 2 failures, the three parameter
    distributions will automatically be excluded.

    Example Usage:

    .. code:: python

        X = [5,3,8,6,7,4,5,4,2]
        output = Fit_Everything(X)
        print('Weibull Alpha =',output.Weibull_2P_alpha)
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        exclude=None,
        sort_by="BIC",
        method="MLE",
        optimizer=None,
        print_results=True,
        show_histogram_plot=True,
        show_PP_plot=True,
        show_probability_plot=True,
        show_best_distribution_probability_plot=True,
        downsample_scatterplot=True,
    ):

        inputs = fitters_input_checking(
            dist="Everything",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        method = inputs.method
        optimizer = inputs.optimizer

        if method in ["RRX", "RRY", "LS", "NLLS"]:
            method = "LS"

        if show_histogram_plot not in [True, False]:
            raise ValueError(
                "show_histogram_plot must be either True or False. Defaults to True."
            )
        if print_results not in [True, False]:
            raise ValueError(
                "print_results must be either True or False. Defaults to True."
            )
        if show_PP_plot not in [True, False]:
            raise ValueError(
                "show_PP_plot must be either True or False. Defaults to True."
            )
        if show_probability_plot not in [True, False]:
            raise ValueError(
                "show_probability_plot must be either True or False. Defaults to True."
            )
        if show_best_distribution_probability_plot not in [True, False]:
            raise ValueError(
                "show_best_distribution_probability_plot must be either True or False. Defaults to True."
            )

        self.failures = failures
        self.right_censored = right_censored
        self._all_data = np.hstack([failures, right_censored])
        # This is used for scaling the histogram when there is censored data
        self._frac_fail = len(failures) / len(self._all_data)
        # This is used for reporting the fraction censored in the printed output
        self._frac_cens = len(right_censored) / len(self._all_data)
        # sorting the failure data is necessary for plotting quantiles in order
        d = sorted(self._all_data)
        self.__downsample_scatterplot = downsample_scatterplot

        if exclude is None:
            exclude = []
        if type(exclude) == np.ndarray:
            exclude = list(exclude)
        if type(exclude) not in [list, np.ndarray]:
            raise ValueError(
                'exclude must be a list or array or strings that match the names of the distributions to be excluded. eg "Weibull_2P".'
            )
        if len(failures) < 3:
            exclude.extend(
                [
                    "Weibull_3P",
                    "Gamma_3P",
                    "Loglogistic_3P",
                    "Lognormal_3P",
                    "Weibull_Mixture",
                    "Weibull_CR",
                ]
            )  # do not fit the 3P distributions if there are only 2 failures
        # flexible name checking for excluded distributions
        excluded_distributions = []
        unknown_exclusions = []
        for item in exclude:
            if type(item) not in [str, np.str_]:
                raise ValueError(
                    "exclude must be a list or array of strings that specified the distributions to be excluded from fitting. Available strings are:"
                    "\nWeibull_2P\nWeibull_3P\nNormal_2P\nGamma_2P\nLoglogistic_2P\nGamma_3P\nLognormal_2P\nLognormal_3P\nLoglogistic_3P\nGumbel_2P\nExponential_2P\nExponential_1P\nBeta_2P\nWeibull_Mixture\nWeibull_CR\nWeibull_DS"
                )
            if item.upper() in ["WEIBULL_2P", "WEIBULL2P", "WEIBULL2"]:
                excluded_distributions.append("Weibull_2P")
            elif item.upper() in ["WEIBULL_3P", "WEIBULL3P", "WEIBULL3"]:
                excluded_distributions.append("Weibull_3P")
            elif item.upper() in ["GAMMA_2P", "GAMMA2P", "GAMMA2"]:
                excluded_distributions.append("Gamma_2P")
            elif item.upper() in ["GAMMA_3P", "GAMMA3P", "GAMMA3"]:
                excluded_distributions.append("Gamma_3P")
            elif item.upper() in ["LOGNORMAL_2P", "LOGNORMAL2P", "LOGNORMAL2"]:
                excluded_distributions.append("Lognormal_2P")
            elif item.upper() in ["LOGNORMAL_3P", "LOGNORMAL3P", "LOGNORMAL3"]:
                excluded_distributions.append("Lognormal_3P")
            elif item.upper() in [
                "EXPONENTIAL_1P",
                "EXPONENTIAL1P",
                "EXPONENTIAL1",
                "EXPON_1P",
                "EXPON1P",
                "EXPON1",
            ]:
                excluded_distributions.append("Exponential_1P")
            elif item.upper() in [
                "EXPONENTIAL_2P",
                "EXPONENTIAL2P",
                "EXPONENTIAL2",
                "EXPON_2P",
                "EXPON2P",
                "EXPON2",
            ]:
                excluded_distributions.append("Exponential_2P")
            elif item.upper() in ["NORMAL_2P", "NORMAL2P", "NORMAL2"]:
                excluded_distributions.append("Normal_2P")
            elif item.upper() in ["GUMBEL_2P", "GUMBEL2P", "GUMBEL2"]:
                excluded_distributions.append("Gumbel_2P")
            elif item.upper() in ["LOGLOGISTIC_2P", "LOGLOGISTIC2P", "LOGLOGISTIC2"]:
                excluded_distributions.append("Loglogistic_2P")
            elif item.upper() in ["LOGLOGISTIC_3P", "LOGLOGISTIC3P", "LOGLOGISTIC3"]:
                excluded_distributions.append("Loglogistic_3P")
            elif item.upper() in ["BETA_2P", "BETA2P", "BETA2"]:
                excluded_distributions.append("Beta_2P")
            elif item.upper() in [
                "WEIBULL MIXTURE",
                "WEIBULLMIXTURE",
                "WEIBULL_MIXTURE",
                "MIXTURE",
                "WEIBULLMIX",
                "WEIBULL_MIX",
                "MIX",
            ]:
                excluded_distributions.append("Weibull_Mixture")
            elif item.upper() in [
                "WEIBULL CR",
                "WEIBULLCR",
                "WEIBULL_CR",
                "WEIBULL_COMPETING_RISKS",
                "WEIBULL_COMPETINGRISKS",
                "WEIBULLCOMPETINGRISKS",
                "WEIBULL COMPETING RISKS",
                "WEIBULL COMPETINGRISKS",
                "COMPETINGRISKS",
                "COMPETING RISKS",
                "CR",
            ]:
                excluded_distributions.append("Weibull_CR")
            elif item.upper() in [
                "WEIBULLDS",
                "WEIBULL_DS",
                "WEIBULL DS",
                "WEIBULL_DEFECTIVE_SUBPOPULATION",
                "WEIBULL_DEFECTIVESUBPOPULATION",
                "WEIBULLDEFECTIVESUBPOPULATION",
                "WEIBULL DEFECTIVE SUBPOPULATION",
                "DEFECTIVE SUBPOPULATION",
                "DEFECTIVESUBPOPULATION",
                "DS",
            ]:
                excluded_distributions.append("Weibull_DS")
            else:
                unknown_exclusions.append(item)
        if len(unknown_exclusions) > 0:
            colorprint(
                str(
                    "WARNING: The following items were not recognised distributions to exclude: "
                    + str(unknown_exclusions)
                ),
                text_color="red",
            )
            colorprint(
                "Available distributions to exclude are: Weibull_2P, Weibull_3P, Normal_2P, Gamma_2P, Loglogistic_2P, Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P, Exponential_2P, Exponential_1P, Beta_2P, Weibull_Mixture, Weibull_CR, Weibull_DS",
                text_color="red",
            )
        if (
            "Beta_2P" not in excluded_distributions
        ):  # if Beta wasn't manually excluded, check if is needs to be automatically excluded based on data above 1
            if max(self._all_data) >= 1:
                excluded_distributions.append("Beta_2P")
        self.excluded_distributions = excluded_distributions

        # create an empty dataframe to append the data from the fitted distributions
        df = pd.DataFrame(
            columns=[
                "Distribution",
                "Alpha",
                "Beta",
                "Gamma",
                "Alpha 1",
                "Beta 1",
                "Alpha 2",
                "Beta 2",
                "Proportion 1",
                "DS",
                "Mu",
                "Sigma",
                "Lambda",
                "Log-likelihood",
                "AICc",
                "BIC",
                "AD",
                "optimizer",
            ]
        )
        # Fit the parametric models and extract the fitted parameters
        if "Weibull_3P" not in self.excluded_distributions:
            self.__Weibull_3P_params = Fit_Weibull_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_3P_alpha = self.__Weibull_3P_params.alpha
            self.Weibull_3P_beta = self.__Weibull_3P_params.beta
            self.Weibull_3P_gamma = self.__Weibull_3P_params.gamma
            self.Weibull_3P_loglik = self.__Weibull_3P_params.loglik
            self.Weibull_3P_BIC = self.__Weibull_3P_params.BIC
            self.Weibull_3P_AICc = self.__Weibull_3P_params.AICc
            self.Weibull_3P_AD = self.__Weibull_3P_params.AD
            self.Weibull_3P_optimizer = self.__Weibull_3P_params.optimizer
            self._parametric_CDF_Weibull_3P = self.__Weibull_3P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_3P"],
                            "Alpha": [self.Weibull_3P_alpha],
                            "Beta": [self.Weibull_3P_beta],
                            "Gamma": [self.Weibull_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_3P_loglik],
                            "AICc": [self.Weibull_3P_AICc],
                            "BIC": [self.Weibull_3P_BIC],
                            "AD": [self.Weibull_3P_AD],
                            "optimizer": [self.Weibull_3P_optimizer],
                        }
                    ),
                ]
            )

        if "Gamma_3P" not in self.excluded_distributions:
            self.__Gamma_3P_params = Fit_Gamma_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gamma_3P_alpha = self.__Gamma_3P_params.alpha
            self.Gamma_3P_beta = self.__Gamma_3P_params.beta
            self.Gamma_3P_mu = self.__Gamma_3P_params.mu
            self.Gamma_3P_gamma = self.__Gamma_3P_params.gamma
            self.Gamma_3P_loglik = self.__Gamma_3P_params.loglik
            self.Gamma_3P_BIC = self.__Gamma_3P_params.BIC
            self.Gamma_3P_AICc = self.__Gamma_3P_params.AICc
            self.Gamma_3P_AD = self.__Gamma_3P_params.AD
            self.Gamma_3P_optimizer = self.__Gamma_3P_params.optimizer
            self._parametric_CDF_Gamma_3P = self.__Gamma_3P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gamma_3P"],
                            "Alpha": [self.Gamma_3P_alpha],
                            "Beta": [self.Gamma_3P_beta],
                            "Gamma": [self.Gamma_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gamma_3P_loglik],
                            "AICc": [self.Gamma_3P_AICc],
                            "BIC": [self.Gamma_3P_BIC],
                            "AD": [self.Gamma_3P_AD],
                            "optimizer": [self.Gamma_3P_optimizer],
                        }
                    ),
                ]
            )

        if "Exponential_2P" not in self.excluded_distributions:
            self.__Exponential_2P_params = Fit_Exponential_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Exponential_2P_lambda = self.__Exponential_2P_params.Lambda
            self.Exponential_2P_gamma = self.__Exponential_2P_params.gamma
            self.Exponential_2P_loglik = self.__Exponential_2P_params.loglik
            self.Exponential_2P_BIC = self.__Exponential_2P_params.BIC
            self.Exponential_2P_AICc = self.__Exponential_2P_params.AICc
            self.Exponential_2P_AD = self.__Exponential_2P_params.AD
            self.Exponential_2P_optimizer = self.__Exponential_2P_params.optimizer
            self._parametric_CDF_Exponential_2P = (
                self.__Exponential_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Exponential_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [self.Exponential_2P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [self.Exponential_2P_lambda],
                            "Log-likelihood": [self.Exponential_2P_loglik],
                            "AICc": [self.Exponential_2P_AICc],
                            "BIC": [self.Exponential_2P_BIC],
                            "AD": [self.Exponential_2P_AD],
                            "optimizer": [self.Exponential_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Lognormal_3P" not in self.excluded_distributions:
            self.__Lognormal_3P_params = Fit_Lognormal_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Lognormal_3P_mu = self.__Lognormal_3P_params.mu
            self.Lognormal_3P_sigma = self.__Lognormal_3P_params.sigma
            self.Lognormal_3P_gamma = self.__Lognormal_3P_params.gamma
            self.Lognormal_3P_loglik = self.__Lognormal_3P_params.loglik
            self.Lognormal_3P_BIC = self.__Lognormal_3P_params.BIC
            self.Lognormal_3P_AICc = self.__Lognormal_3P_params.AICc
            self.Lognormal_3P_AD = self.__Lognormal_3P_params.AD
            self.Lognormal_3P_optimizer = self.__Lognormal_3P_params.optimizer
            self._parametric_CDF_Lognormal_3P = (
                self.__Lognormal_3P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Lognormal_3P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [self.Lognormal_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Lognormal_3P_mu],
                            "Sigma": [self.Lognormal_3P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Lognormal_3P_loglik],
                            "AICc": [self.Lognormal_3P_AICc],
                            "BIC": [self.Lognormal_3P_BIC],
                            "AD": [self.Lognormal_3P_AD],
                            "optimizer": [self.Lognormal_3P_optimizer],
                        }
                    ),
                ]
            )

        if "Normal_2P" not in self.excluded_distributions:
            self.__Normal_2P_params = Fit_Normal_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Normal_2P_mu = self.__Normal_2P_params.mu
            self.Normal_2P_sigma = self.__Normal_2P_params.sigma
            self.Normal_2P_loglik = self.__Normal_2P_params.loglik
            self.Normal_2P_BIC = self.__Normal_2P_params.BIC
            self.Normal_2P_AICc = self.__Normal_2P_params.AICc
            self.Normal_2P_AD = self.__Normal_2P_params.AD
            self.Normal_2P_optimizer = self.__Normal_2P_params.optimizer
            self._parametric_CDF_Normal_2P = self.__Normal_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Normal_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Normal_2P_mu],
                            "Sigma": [self.Normal_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Normal_2P_loglik],
                            "AICc": [self.Normal_2P_AICc],
                            "BIC": [self.Normal_2P_BIC],
                            "AD": [self.Normal_2P_AD],
                            "optimizer": [self.Normal_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Lognormal_2P" not in self.excluded_distributions:
            self.__Lognormal_2P_params = Fit_Lognormal_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Lognormal_2P_mu = self.__Lognormal_2P_params.mu
            self.Lognormal_2P_sigma = self.__Lognormal_2P_params.sigma
            self.Lognormal_2P_gamma = 0
            self.Lognormal_2P_loglik = self.__Lognormal_2P_params.loglik
            self.Lognormal_2P_BIC = self.__Lognormal_2P_params.BIC
            self.Lognormal_2P_AICc = self.__Lognormal_2P_params.AICc
            self.Lognormal_2P_AD = self.__Lognormal_2P_params.AD
            self.Lognormal_2P_optimizer = self.__Lognormal_2P_params.optimizer
            self._parametric_CDF_Lognormal_2P = (
                self.__Lognormal_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Lognormal_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Lognormal_2P_mu],
                            "Sigma": [self.Lognormal_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Lognormal_2P_loglik],
                            "AICc": [self.Lognormal_2P_AICc],
                            "BIC": [self.Lognormal_2P_BIC],
                            "AD": [self.Lognormal_2P_AD],
                            "optimizer": [self.Lognormal_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Gumbel_2P" not in self.excluded_distributions:
            self.__Gumbel_2P_params = Fit_Gumbel_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gumbel_2P_mu = self.__Gumbel_2P_params.mu
            self.Gumbel_2P_sigma = self.__Gumbel_2P_params.sigma
            self.Gumbel_2P_loglik = self.__Gumbel_2P_params.loglik
            self.Gumbel_2P_BIC = self.__Gumbel_2P_params.BIC
            self.Gumbel_2P_AICc = self.__Gumbel_2P_params.AICc
            self.Gumbel_2P_AD = self.__Gumbel_2P_params.AD
            self.Gumbel_2P_optimizer = self.__Gumbel_2P_params.optimizer
            self._parametric_CDF_Gumbel_2P = self.__Gumbel_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gumbel_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Gumbel_2P_mu],
                            "Sigma": [self.Gumbel_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gumbel_2P_loglik],
                            "AICc": [self.Gumbel_2P_AICc],
                            "BIC": [self.Gumbel_2P_BIC],
                            "AD": [self.Gumbel_2P_AD],
                            "optimizer": [self.Gumbel_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Weibull_2P" not in self.excluded_distributions:
            self.__Weibull_2P_params = Fit_Weibull_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_2P_alpha = self.__Weibull_2P_params.alpha
            self.Weibull_2P_beta = self.__Weibull_2P_params.beta
            self.Weibull_2P_gamma = 0
            self.Weibull_2P_loglik = self.__Weibull_2P_params.loglik
            self.Weibull_2P_BIC = self.__Weibull_2P_params.BIC
            self.Weibull_2P_AICc = self.__Weibull_2P_params.AICc
            self.Weibull_2P_AD = self.__Weibull_2P_params.AD
            self.Weibull_2P_optimizer = self.__Weibull_2P_params.optimizer
            self._parametric_CDF_Weibull_2P = self.__Weibull_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_2P"],
                            "Alpha": [self.Weibull_2P_alpha],
                            "Beta": [self.Weibull_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_2P_loglik],
                            "AICc": [self.Weibull_2P_AICc],
                            "BIC": [self.Weibull_2P_BIC],
                            "AD": [self.Weibull_2P_AD],
                            "optimizer": [self.Weibull_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Weibull_Mixture" not in self.excluded_distributions:
            self.__Weibull_Mixture_params = Fit_Weibull_Mixture(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_Mixture_alpha_1 = self.__Weibull_Mixture_params.alpha_1
            self.Weibull_Mixture_beta_1 = self.__Weibull_Mixture_params.beta_1
            self.Weibull_Mixture_alpha_2 = self.__Weibull_Mixture_params.alpha_2
            self.Weibull_Mixture_beta_2 = self.__Weibull_Mixture_params.beta_2
            self.Weibull_Mixture_proportion_1 = (
                self.__Weibull_Mixture_params.proportion_1
            )
            self.Weibull_Mixture_loglik = self.__Weibull_Mixture_params.loglik
            self.Weibull_Mixture_BIC = self.__Weibull_Mixture_params.BIC
            self.Weibull_Mixture_AICc = self.__Weibull_Mixture_params.AICc
            self.Weibull_Mixture_AD = self.__Weibull_Mixture_params.AD
            self.Weibull_Mixture_optimizer = self.__Weibull_Mixture_params.optimizer
            self._parametric_CDF_Weibull_Mixture = (
                self.__Weibull_Mixture_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_Mixture"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [self.Weibull_Mixture_alpha_1],
                            "Beta 1": [self.Weibull_Mixture_beta_1],
                            "Alpha 2": [self.Weibull_Mixture_alpha_2],
                            "Beta 2": [self.Weibull_Mixture_beta_2],
                            "Proportion 1": [self.Weibull_Mixture_proportion_1],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_Mixture_loglik],
                            "AICc": [self.Weibull_Mixture_AICc],
                            "BIC": [self.Weibull_Mixture_BIC],
                            "AD": [self.Weibull_Mixture_AD],
                            "optimizer": [self.Weibull_Mixture_optimizer],
                        }
                    ),
                ]
            )

        if "Weibull_CR" not in self.excluded_distributions:
            self.__Weibull_CR_params = Fit_Weibull_CR(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_CR_alpha_1 = self.__Weibull_CR_params.alpha_1
            self.Weibull_CR_beta_1 = self.__Weibull_CR_params.beta_1
            self.Weibull_CR_alpha_2 = self.__Weibull_CR_params.alpha_2
            self.Weibull_CR_beta_2 = self.__Weibull_CR_params.beta_2
            self.Weibull_CR_loglik = self.__Weibull_CR_params.loglik
            self.Weibull_CR_BIC = self.__Weibull_CR_params.BIC
            self.Weibull_CR_AICc = self.__Weibull_CR_params.AICc
            self.Weibull_CR_AD = self.__Weibull_CR_params.AD
            self.Weibull_CR_optimizer = self.__Weibull_CR_params.optimizer
            self._parametric_CDF_Weibull_CR = self.__Weibull_CR_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_CR"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [self.Weibull_CR_alpha_1],
                            "Beta 1": [self.Weibull_CR_beta_1],
                            "Alpha 2": [self.Weibull_CR_alpha_2],
                            "Beta 2": [self.Weibull_CR_beta_2],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_CR_loglik],
                            "AICc": [self.Weibull_CR_AICc],
                            "BIC": [self.Weibull_CR_BIC],
                            "AD": [self.Weibull_CR_AD],
                            "optimizer": [self.Weibull_CR_optimizer],
                        }
                    ),
                ]
            )

        if "Weibull_DS" not in self.excluded_distributions:
            self.__Weibull_DS_params = Fit_Weibull_DS(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_DS_alpha = self.__Weibull_DS_params.alpha
            self.Weibull_DS_beta = self.__Weibull_DS_params.beta
            self.Weibull_DS_DS = self.__Weibull_DS_params.DS
            self.Weibull_DS_loglik = self.__Weibull_DS_params.loglik
            self.Weibull_DS_BIC = self.__Weibull_DS_params.BIC
            self.Weibull_DS_AICc = self.__Weibull_DS_params.AICc
            self.Weibull_DS_AD = self.__Weibull_DS_params.AD
            self.Weibull_DS_optimizer = self.__Weibull_DS_params.optimizer
            self._parametric_CDF_Weibull_DS = self.__Weibull_DS_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_DS"],
                            "Alpha": [self.Weibull_DS_alpha],
                            "Beta": [self.Weibull_DS_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [self.Weibull_DS_DS],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_DS_loglik],
                            "AICc": [self.Weibull_DS_AICc],
                            "BIC": [self.Weibull_DS_BIC],
                            "AD": [self.Weibull_DS_AD],
                            "optimizer": [self.Weibull_DS_optimizer],
                        }
                    ),
                ]
            )

        if "Gamma_2P" not in self.excluded_distributions:
            self.__Gamma_2P_params = Fit_Gamma_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gamma_2P_alpha = self.__Gamma_2P_params.alpha
            self.Gamma_2P_beta = self.__Gamma_2P_params.beta
            self.Gamma_2P_mu = self.__Gamma_2P_params.mu
            self.Gamma_2P_gamma = 0
            self.Gamma_2P_loglik = self.__Gamma_2P_params.loglik
            self.Gamma_2P_BIC = self.__Gamma_2P_params.BIC
            self.Gamma_2P_AICc = self.__Gamma_2P_params.AICc
            self.Gamma_2P_AD = self.__Gamma_2P_params.AD
            self.Gamma_2P_optimizer = self.__Gamma_2P_params.optimizer
            self._parametric_CDF_Gamma_2P = self.__Gamma_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gamma_2P"],
                            "Alpha": [self.Gamma_2P_alpha],
                            "Beta": [self.Gamma_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gamma_2P_loglik],
                            "AICc": [self.Gamma_2P_AICc],
                            "BIC": [self.Gamma_2P_BIC],
                            "AD": [self.Gamma_2P_AD],
                            "optimizer": [self.Gamma_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Exponential_1P" not in self.excluded_distributions:
            self.__Exponential_1P_params = Fit_Exponential_1P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Exponential_1P_lambda = self.__Exponential_1P_params.Lambda
            self.Exponential_1P_gamma = 0
            self.Exponential_1P_loglik = self.__Exponential_1P_params.loglik
            self.Exponential_1P_BIC = self.__Exponential_1P_params.BIC
            self.Exponential_1P_AICc = self.__Exponential_1P_params.AICc
            self.Exponential_1P_AD = self.__Exponential_1P_params.AD
            self.Exponential_1P_optimizer = self.__Exponential_1P_params.optimizer
            self._parametric_CDF_Exponential_1P = (
                self.__Exponential_1P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Exponential_1P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [self.Exponential_1P_lambda],
                            "Log-likelihood": [self.Exponential_1P_loglik],
                            "AICc": [self.Exponential_1P_AICc],
                            "BIC": [self.Exponential_1P_BIC],
                            "AD": [self.Exponential_1P_AD],
                            "optimizer": [self.Exponential_1P_optimizer],
                        }
                    ),
                ]
            )

        if "Loglogistic_2P" not in self.excluded_distributions:
            self.__Loglogistic_2P_params = Fit_Loglogistic_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Loglogistic_2P_alpha = self.__Loglogistic_2P_params.alpha
            self.Loglogistic_2P_beta = self.__Loglogistic_2P_params.beta
            self.Loglogistic_2P_gamma = 0
            self.Loglogistic_2P_loglik = self.__Loglogistic_2P_params.loglik
            self.Loglogistic_2P_BIC = self.__Loglogistic_2P_params.BIC
            self.Loglogistic_2P_AICc = self.__Loglogistic_2P_params.AICc
            self.Loglogistic_2P_AD = self.__Loglogistic_2P_params.AD
            self.Loglogistic_2P_optimizer = self.__Loglogistic_2P_params.optimizer
            self._parametric_CDF_Loglogistic_2P = (
                self.__Loglogistic_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Loglogistic_2P"],
                            "Alpha": [self.Loglogistic_2P_alpha],
                            "Beta": [self.Loglogistic_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Loglogistic_2P_loglik],
                            "AICc": [self.Loglogistic_2P_AICc],
                            "BIC": [self.Loglogistic_2P_BIC],
                            "AD": [self.Loglogistic_2P_AD],
                            "optimizer": [self.Loglogistic_2P_optimizer],
                        }
                    ),
                ]
            )

        if "Loglogistic_3P" not in self.excluded_distributions:
            self.__Loglogistic_3P_params = Fit_Loglogistic_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Loglogistic_3P_alpha = self.__Loglogistic_3P_params.alpha
            self.Loglogistic_3P_beta = self.__Loglogistic_3P_params.beta
            self.Loglogistic_3P_gamma = self.__Loglogistic_3P_params.gamma
            self.Loglogistic_3P_loglik = self.__Loglogistic_3P_params.loglik
            self.Loglogistic_3P_BIC = self.__Loglogistic_3P_params.BIC
            self.Loglogistic_3P_AICc = self.__Loglogistic_3P_params.AICc
            self.Loglogistic_3P_AD = self.__Loglogistic_3P_params.AD
            self.Loglogistic_3P_optimizer = self.__Loglogistic_3P_params.optimizer
            self._parametric_CDF_Loglogistic_3P = (
                self.__Loglogistic_3P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Loglogistic_3P"],
                            "Alpha": [self.Loglogistic_3P_alpha],
                            "Beta": [self.Loglogistic_3P_beta],
                            "Gamma": [self.Loglogistic_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Loglogistic_3P_loglik],
                            "AICc": [self.Loglogistic_3P_AICc],
                            "BIC": [self.Loglogistic_3P_BIC],
                            "AD": [self.Loglogistic_3P_AD],
                            "optimizer": [self.Loglogistic_3P_optimizer],
                        }
                    ),
                ]
            )

        if "Beta_2P" not in self.excluded_distributions:
            self.__Beta_2P_params = Fit_Beta_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Beta_2P_alpha = self.__Beta_2P_params.alpha
            self.Beta_2P_beta = self.__Beta_2P_params.beta
            self.Beta_2P_loglik = self.__Beta_2P_params.loglik
            self.Beta_2P_BIC = self.__Beta_2P_params.BIC
            self.Beta_2P_AICc = self.__Beta_2P_params.AICc
            self.Beta_2P_AD = self.__Beta_2P_params.AD
            self.Beta_2P_optimizer = self.__Beta_2P_params.optimizer
            self._parametric_CDF_Beta_2P = self.__Beta_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Beta_2P"],
                            "Alpha": [self.Beta_2P_alpha],
                            "Beta": [self.Beta_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Beta_2P_loglik],
                            "AICc": [self.Beta_2P_AICc],
                            "BIC": [self.Beta_2P_BIC],
                            "AD": [self.Beta_2P_AD],
                            "optimizer": [self.Beta_2P_optimizer],
                        }
                    ),
                ]
            )

        # change to sorting by BIC if there is insufficient data to get the AICc for everything that was fitted
        if (
            sort_by in ["AIC", "aic", "aicc", "AICc"]
            and "Insufficient data" in df["AICc"].values
        ):
            sort_by = "BIC"
        # sort the dataframe by BIC, AICc, or AD. Smallest AICc, BIC, AD is better fit
        if type(sort_by) != str:
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', 'AD', or 'Log-likelihood'. Default is 'BIC'."
            )
        if sort_by.upper() == "BIC":
            df2 = df.sort_values(by="BIC")
        elif sort_by.upper() in ["AICC", "AIC"]:
            df2 = df.sort_values(by="AICc")
        elif sort_by.upper() == "AD":
            df2 = df.sort_values(by="AD")
        elif sort_by.upper() in [
            "LOGLIK",
            "LOG LIK",
            "LOG-LIKELIHOOD",
            "LL",
            "LOGLIKELIHOOD",
            "LOG LIKELIHOOD",
        ]:
            df["LLabs"] = abs(
                df["Log-likelihood"]
            )  # need to create a new column for the absolute value before sorting
            df2 = df.sort_values(by="LLabs")
            df2.drop(
                "LLabs", axis=1, inplace=True
            )  # remove the column created just for sorting
        else:
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', 'AD', or 'Log-likelihood'. Default is 'BIC'."
            )
        if len(df2.index.values) == 0:
            raise ValueError("You have excluded all available distributions")
        self.results = df2

        # creates a distribution object of the best fitting distribution and assigns its name
        best_dist = self.results["Distribution"].values[0]
        self.best_distribution_name = best_dist
        if best_dist == "Weibull_2P":
            self.best_distribution = Weibull_Distribution(
                alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta
            )
        elif best_dist == "Weibull_3P":
            self.best_distribution = Weibull_Distribution(
                alpha=self.Weibull_3P_alpha,
                beta=self.Weibull_3P_beta,
                gamma=self.Weibull_3P_gamma,
            )
        elif best_dist == "Weibull_Mixture":
            d1 = Weibull_Distribution(
                alpha=self.Weibull_Mixture_alpha_1, beta=self.Weibull_Mixture_beta_1
            )
            d2 = Weibull_Distribution(
                alpha=self.Weibull_Mixture_alpha_2, beta=self.Weibull_Mixture_beta_2
            )
            self.best_distribution = Mixture_Model(
                distributions=[d1, d2],
                proportions=[
                    self.Weibull_Mixture_proportion_1,
                    1 - self.Weibull_Mixture_proportion_1,
                ],
            )
        elif best_dist == "Weibull_CR":
            d1 = Weibull_Distribution(
                alpha=self.Weibull_CR_alpha_1, beta=self.Weibull_CR_beta_1
            )
            d2 = Weibull_Distribution(
                alpha=self.Weibull_CR_alpha_2, beta=self.Weibull_CR_beta_2
            )
            self.best_distribution = Competing_Risks_Model(distributions=[d1, d2])
        if best_dist == "Weibull_DS":
            d1 = Weibull_Distribution(
                alpha=self.Weibull_DS_alpha, beta=self.Weibull_DS_beta
            )
            self.best_distribution = DSZI_Model(distribution=d1, DS=self.Weibull_DS_DS)
        elif best_dist == "Gamma_2P":
            self.best_distribution = Gamma_Distribution(
                alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta
            )
        elif best_dist == "Gamma_3P":
            self.best_distribution = Gamma_Distribution(
                alpha=self.Gamma_3P_alpha,
                beta=self.Gamma_3P_beta,
                gamma=self.Gamma_3P_gamma,
            )
        elif best_dist == "Lognormal_2P":
            self.best_distribution = Lognormal_Distribution(
                mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma
            )
        elif best_dist == "Lognormal_3P":
            self.best_distribution = Lognormal_Distribution(
                mu=self.Lognormal_3P_mu,
                sigma=self.Lognormal_3P_sigma,
                gamma=self.Lognormal_3P_gamma,
            )
        elif best_dist == "Exponential_1P":
            self.best_distribution = Exponential_Distribution(
                Lambda=self.Exponential_1P_lambda
            )
        elif best_dist == "Exponential_2P":
            self.best_distribution = Exponential_Distribution(
                Lambda=self.Exponential_2P_lambda, gamma=self.Exponential_2P_gamma
            )
        elif best_dist == "Normal_2P":
            self.best_distribution = Normal_Distribution(
                mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma
            )
        elif best_dist == "Beta_2P":
            self.best_distribution = Beta_Distribution(
                alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta
            )
        elif best_dist == "Loglogistic_2P":
            self.best_distribution = Loglogistic_Distribution(
                alpha=self.Loglogistic_2P_alpha, beta=self.Loglogistic_2P_beta
            )
        elif best_dist == "Loglogistic_3P":
            self.best_distribution = Loglogistic_Distribution(
                alpha=self.Loglogistic_3P_alpha,
                beta=self.Loglogistic_3P_beta,
                gamma=self.Loglogistic_3P_gamma,
            )
        elif best_dist == "Gumbel_2P":
            self.best_distribution = Gumbel_Distribution(
                mu=self.Gumbel_2P_mu, sigma=self.Gumbel_2P_sigma
            )

        # print the results
        if print_results is True:  # printing occurs by default
            frac_cens = round_to_decimals(self._frac_cens * 100)
            colorprint("Results from Fit_Everything:", bold=True, underline=True)
            print("Analysis method:", method)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_cens) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")

        if show_histogram_plot is True:
            # plotting enabled by default
            Fit_Everything.__histogram_plot(self)

        if show_PP_plot is True:
            # plotting enabled by default
            Fit_Everything.__P_P_plot(self)

        if show_probability_plot is True:
            # plotting enabled by default
            Fit_Everything.__probability_plot(self)

        if show_best_distribution_probability_plot is True:
            # plotting enabled by default
            Fit_Everything.__probability_plot(self, best_only=True)

        if (
            show_histogram_plot is True
            or show_PP_plot is True
            or show_probability_plot is True
            or show_best_distribution_probability_plot is True
        ):
            plt.show()

    def __probplot_layout(self):
        """
        Internal function to provide layout formatting of the plots.
        """
        items = len(self.results.index.values)  # number of items fitted
        xx1, yy1 = 2.5, 2  # multipliers for easy adjustment of window sizes
        xx2, yy2 = 0.5, 0.5
        if items == 16:
            # figsizes are in (w,h) format using the above multipliers
            cols, rows, figsize, figsizePP = (
                6,
                3,
                (xx1 * 8, yy1 * 4),
                (xx2 * 23, yy2 * 15),
            )
        elif items in [13, 14, 15]:
            cols, rows, figsize, figsizePP = (
                5,
                3,
                (xx1 * 7, yy1 * 4),
                (xx2 * 20, yy2 * 15),
            )
        elif items in [10, 11, 12]:
            cols, rows, figsize, figsizePP = (
                4,
                3,
                (xx1 * 6, yy1 * 4),
                (xx2 * 17, yy2 * 15),
            )
        elif items in [7, 8, 9]:
            cols, rows, figsize, figsizePP = (
                3,
                3,
                (xx1 * 5, yy1 * 4),
                (xx2 * 14, yy2 * 15),
            )
        elif items in [5, 6]:
            cols, rows, figsize, figsizePP = (
                3,
                2,
                (xx1 * 5, yy1 * 3),
                (xx2 * 13, yy2 * 11),
            )
        elif items == 4:
            cols, rows, figsize, figsizePP = (
                2,
                2,
                (xx1 * 4, yy1 * 3),
                (xx2 * 12, yy2 * 11),
            )
        elif items == 3:
            cols, rows, figsize, figsizePP = (
                3,
                1,
                (xx1 * 5, yy1 * 2.5),
                (xx2 * 20, yy2 * 8),
            )
        elif items == 2:
            cols, rows, figsize, figsizePP = (
                2,
                1,
                (xx1 * 4, yy1 * 2),
                (xx2 * 12, yy2 * 8),
            )
        elif items == 1:
            cols, rows, figsize, figsizePP = (
                1,
                1,
                (xx1 * 3, yy1 * 2),
                (xx2 * 12, yy2 * 8),
            )
        return cols, rows, figsize, figsizePP

    def __histogram_plot(self):
        """
        Generates a histogram plot of PDF and CDF of teh fitted distributions.
        """
        X = self.failures
        # define plotting limits
        xmin = 0
        xmax = max(X) * 1.2

        plt.figure(figsize=(12, 6))
        # this is the order to plot things so that the legend matches the results dataframe
        plotting_order = self.results["Distribution"].values
        iqr = np.subtract(*np.percentile(X, [75, 25]))  # interquartile range
        # Freedman–Diaconis rule ==> https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        bin_width = 2 * iqr * len(X) ** -(1 / 3)
        num_bins = int(np.ceil((max(X) - min(X)) / bin_width))
        # we need to make the histogram manually (can't use plt.hist) due to need to scale the heights when there's censored data
        hist, bins = np.histogram(X, bins=num_bins, density=True)
        hist_cumulative = np.cumsum(hist) / sum(hist)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        # Probability Density Functions
        plt.subplot(132)
        plt.bar(
            center,
            hist * self._frac_fail,
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )
        counter = 0
        ls = "-"
        for item in plotting_order:
            counter += 1
            if counter > 10:
                ls = "--"
            if item == "Weibull_2P":
                self.__Weibull_2P_params.distribution.PDF(
                    label=r"Weibull_2P ($\alpha , \beta$)", linestyle=ls
                )
            elif item == "Weibull_3P":
                self.__Weibull_3P_params.distribution.PDF(
                    label=r"Weibull_3P ($\alpha , \beta , \gamma$)", linestyle=ls
                )
            elif item == "Weibull_Mixture":
                self.__Weibull_Mixture_params.distribution.PDF(
                    label=r"Weibull_Mixture ($\alpha_1 , \beta_1 , \alpha_2 , \beta_2 , p_1$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Weibull_CR":
                self.__Weibull_CR_params.distribution.PDF(
                    label=r"Weibull_CR ($\alpha_1 , \beta_1 , \alpha_2 , \beta_2$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Weibull_DS":
                self.__Weibull_DS_params.distribution.PDF(
                    label=r"Weibull_DS ($\alpha , \beta , DS$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Gamma_2P":
                self.__Gamma_2P_params.distribution.PDF(
                    label=r"Gamma_2P ($\alpha , \beta$)", linestyle=ls
                )
            elif item == "Gamma_3P":
                self.__Gamma_3P_params.distribution.PDF(
                    label=r"Gamma_3P ($\alpha , \beta , \gamma$)", linestyle=ls
                )
            elif item == "Exponential_1P":
                self.__Exponential_1P_params.distribution.PDF(
                    label=r"Exponential_1P ($\lambda$)", linestyle=ls
                )
            elif item == "Exponential_2P":
                self.__Exponential_2P_params.distribution.PDF(
                    label=r"Exponential_2P ($\lambda , \gamma$)", linestyle=ls
                )
            elif item == "Lognormal_2P":
                self.__Lognormal_2P_params.distribution.PDF(
                    label=r"Lognormal_2P ($\mu , \sigma$)", linestyle=ls
                )
            elif item == "Lognormal_3P":
                self.__Lognormal_3P_params.distribution.PDF(
                    label=r"Lognormal_3P ($\mu , \sigma , \gamma$)", linestyle=ls
                )
            elif item == "Normal_2P":
                self.__Normal_2P_params.distribution.PDF(
                    label=r"Normal_2P ($\mu , \sigma$)", linestyle=ls
                )
            elif item == "Gumbel_2P":
                self.__Gumbel_2P_params.distribution.PDF(
                    label=r"Gumbel_2P ($\mu , \sigma$)", linestyle=ls
                )
            elif item == "Loglogistic_2P":
                self.__Loglogistic_2P_params.distribution.PDF(
                    label=r"Loglogistic_2P ($\alpha , \beta$)", linestyle=ls
                )
            elif item == "Loglogistic_3P":
                self.__Loglogistic_3P_params.distribution.PDF(
                    label=r"Loglogistic_3P ($\alpha , \beta , \gamma$)", linestyle=ls
                )
            elif item == "Beta_2P":
                self.__Beta_2P_params.distribution.PDF(
                    label=r"Beta_2P ($\alpha , \beta$)", linestyle=ls
                )
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.gca().legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(-1.1, 1),
            frameon=False,
            title="Distribution Fitted\n",
        )
        lgd._legend_box.align = "left"
        plt.xlim(xmin, xmax)
        plt.ylim(0, max(hist * self._frac_fail) * 1.2)
        plt.title("Probability Density Function")
        plt.xlabel("Data")
        plt.ylabel("Probability density")

        # Cumulative Distribution Functions
        plt.subplot(133)
        _, ecdf_y = plotting_positions(
            failures=self.failures, right_censored=self.right_censored
        )
        plt.bar(
            center,
            hist_cumulative * max(ecdf_y),
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )

        counter = 0
        ls = "-"
        for item in plotting_order:
            counter += 1
            if counter > 10:
                ls = "--"
            if item == "Weibull_2P":
                self.__Weibull_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Weibull_3P":
                self.__Weibull_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Weibull_Mixture":
                self.__Weibull_Mixture_params.distribution.CDF(
                    linestyle=ls, xmax=xmax * 2
                )
            elif item == "Weibull_CR":
                self.__Weibull_CR_params.distribution.CDF(linestyle=ls, xmax=xmax * 2)
            elif item == "Weibull_DS":
                self.__Weibull_DS_params.distribution.CDF(linestyle=ls, xmax=xmax * 2)
            elif item == "Gamma_2P":
                self.__Gamma_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Gamma_3P":
                self.__Gamma_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Exponential_1P":
                self.__Exponential_1P_params.distribution.CDF(
                    plot_CI=False, linestyle=ls
                )
            elif item == "Exponential_2P":
                self.__Exponential_2P_params.distribution.CDF(
                    plot_CI=False, linestyle=ls
                )
            elif item == "Lognormal_2P":
                self.__Lognormal_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Lognormal_3P":
                self.__Lognormal_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Normal_2P":
                self.__Normal_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Gumbel_2P":
                self.__Gumbel_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Loglogistic_2P":
                self.__Loglogistic_2P_params.distribution.CDF(
                    plot_CI=False, linestyle=ls
                )
            elif item == "Loglogistic_3P":
                self.__Loglogistic_3P_params.distribution.CDF(
                    plot_CI=False, linestyle=ls
                )
            elif item == "Beta_2P":
                self.__Beta_2P_params.distribution.CDF(linestyle=ls)
        plt.xlim(xmin, xmax)
        plt.ylim(0, max(ecdf_y) * 1.2)
        plt.title("Cumulative Distribution Function")
        plt.xlabel("Data")
        plt.ylabel("Cumulative probability density")
        plt.suptitle("Histogram plot of each fitted distribution")
        plt.subplots_adjust(left=0, bottom=0.10, right=0.97, top=0.88, wspace=0.18)

    def __P_P_plot(self):
        """
        Generates a subplot of Probability-Probability plots to compare the
        parametric vs non-parametric plots of the fitted distributions.
        """
        # Kaplan-Meier estimate of quantiles. Used in P-P plot.
        nonparametric = KaplanMeier(
            failures=self.failures,
            right_censored=self.right_censored,
            print_results=False,
            show_plot=False,
        )
        nonparametric_CDF = 1 - nonparametric.KM  # change SF into CDF

        cols, rows, _, figsizePP = Fit_Everything.__probplot_layout(self)
        # this is the order to plot things which matches the results dataframe
        plotting_order = self.results["Distribution"].values
        plt.figure(figsize=figsizePP)
        plt.suptitle(
            "Semi-parametric Probability-Probability plots of each fitted distribution\nParametric (x-axis) vs Non-Parametric (y-axis)\n"
        )
        subplot_counter = 1
        for item in plotting_order:
            plt.subplot(rows, cols, subplot_counter)

            xx = nonparametric_CDF
            plotlim = max(xx)
            if item == "Exponential_1P":
                yy = self._parametric_CDF_Exponential_1P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Exponential_2P":
                yy = self._parametric_CDF_Exponential_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Lognormal_2P":
                yy = self._parametric_CDF_Lognormal_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Lognormal_3P":
                yy = self._parametric_CDF_Lognormal_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_2P":
                yy = self._parametric_CDF_Weibull_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_3P":
                yy = self._parametric_CDF_Weibull_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_Mixture":
                yy = self._parametric_CDF_Weibull_Mixture
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_CR":
                yy = self._parametric_CDF_Weibull_CR
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_DS":
                yy = self._parametric_CDF_Weibull_DS
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Loglogistic_2P":
                yy = self._parametric_CDF_Loglogistic_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Loglogistic_3P":
                yy = self._parametric_CDF_Loglogistic_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gamma_2P":
                yy = self._parametric_CDF_Gamma_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gamma_3P":
                yy = self._parametric_CDF_Gamma_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Normal_2P":
                yy = self._parametric_CDF_Normal_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gumbel_2P":
                yy = self._parametric_CDF_Gumbel_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Beta_2P":
                yy = self._parametric_CDF_Beta_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy

            # downsample if necessary
            x_scatter, y_scatter = xy_downsample(
                xx, yy, downsample_factor=self.__downsample_scatterplot
            )
            # plot the scatterplot
            plt.scatter(x_scatter, y_scatter, marker=".", color="k")
            plt.title(item)
            plt.plot([-1, 2], [-1, 2], "r", alpha=0.7)  # red diagonal line
            plt.axis("square")
            plt.yticks([])
            plt.xticks([])
            plt.xlim(-plotlim * 0.05, plotlim * 1.05)
            plt.ylim(-plotlim * 0.05, plotlim * 1.05)
            subplot_counter += 1
        plt.tight_layout()

    def __probability_plot(self, best_only=False):
        """
        Generates a subplot of all the probability plots
        """
        from reliability.Probability_plotting import (
            Weibull_probability_plot,
            Normal_probability_plot,
            Gamma_probability_plot,
            Exponential_probability_plot,
            Beta_probability_plot,
            Lognormal_probability_plot,
            Exponential_probability_plot_Weibull_Scale,
            Loglogistic_probability_plot,
            Gumbel_probability_plot,
        )

        plt.figure()
        if best_only is False:
            cols, rows, figsize, _ = Fit_Everything.__probplot_layout(self)
            # this is the order to plot to match the results dataframe
            plotting_order = self.results["Distribution"].values
            plt.suptitle("Probability plots of each fitted distribution\n\n")
            subplot_counter = 1
        else:
            plotting_order = [self.results["Distribution"].values[0]]

        # xvals is used by Weibull_Mixture, Weibull_CR, and Weibull_DS
        xvals = np.logspace(
            np.log10(min(self.failures)) - 3, np.log10(max(self.failures)) + 1, 1000
        )
        for item in plotting_order:
            if best_only is False:
                plt.subplot(rows, cols, subplot_counter)
            if item == "Exponential_1P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Exponential_1P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Exponential_2P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Exponential_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Lognormal_2P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Lognormal_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Lognormal_3P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Lognormal_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_2P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Weibull_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_3P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Weibull_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_Mixture":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_Mixture_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Weibull_CR":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_CR_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Weibull_DS":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_DS_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Loglogistic_2P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Loglogistic_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Loglogistic_3P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Loglogistic_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gamma_2P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gamma_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gamma_3P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gamma_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Normal_2P":
                Normal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Normal_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gumbel_2P":
                Gumbel_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gumbel_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Beta_2P":
                Beta_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Beta_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )

            if best_only is False:
                plt.title(item)
                ax = plt.gca()
                ax.set_yticklabels([], minor=False)
                ax.set_xticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.set_xticklabels([], minor=True)
                ax.set_ylabel("")
                ax.set_xlabel("")
                try:
                    ax.get_legend().remove()
                except AttributeError:
                    pass
                    # some plots don't have a legend added so this exception ignores them when trying to remove the legend
                subplot_counter += 1
            else:
                if self.best_distribution_name == "Weibull_Mixture":
                    title_detail = "Weibull Mixture Model"
                elif self.best_distribution_name == "Weibull_CR":
                    title_detail = "Weibull Competing Risks Model"
                elif self.best_distribution_name == "Weibull_DS":
                    title_detail = "Weibull Defective Subpopulation Model"
                else:
                    title_detail = self.best_distribution.param_title_long
                plt.title(str("Probability plot of best distribution\n" + title_detail))
        if best_only is False:
            plt.tight_layout()
            plt.gcf().set_size_inches(figsize)


class Fit_Weibull_2P:
    """
    Fits a two parameter Weibull distribution (alpha,beta) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_beta is not
        specified or at least 1 element if force_beta is specified.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_beta : float, int, optional
        Used to specify the beta value if you need to force beta to be a certain
        value. Used in ALT probability plotting. Optional input. If specified it
        must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [1, 5, 10,..., 95, 99] set quantiles as
        either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_2P alpha parameter
    beta : float
        the fitted Weibull_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Weibull_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        force_beta=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Weibull_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Weibull_2P",
            LL_func=Fit_Weibull_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_beta,
            LL_func_force=Fit_Weibull_2P.LL_fb,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Weibull_2P",
                LL_func=Fit_Weibull_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_beta,
                LL_func_force=Fit_Weibull_2P.LL_fb,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix[0][1]
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta

        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P.LL_fb)(
                np.array(tuple([self.alpha])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_beta])),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta
                self.beta_lower = self.beta
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Weibull_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        if force_beta is None:
            k = 2
            LL2 = 2 * Fit_Weibull_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Weibull_2P.LL_fb(params, failures, right_censored, force_beta)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter weibull)
        LL_f = Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Weibull_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(params, T_f, T_rc, force_beta):
        # log likelihood function (2 parameter weibull) FORCED BETA
        LL_f = Fit_Weibull_2P.logf(T_f, params[0], force_beta).sum()
        LL_rc = Fit_Weibull_2P.logR(T_rc, params[0], force_beta).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_2P_grouped:
    """
    Fits a two parameter Weibull distribution (alpha,beta) to the data provided.
    This function is similar to Fit_Weibull_2P however it accepts a dataframe
    which allows for efficient handling of grouped (repeated) data.

    Parameters
    ----------
    dataframe : dataframe
        a pandas dataframe of the appropriate format. See the example in Notes.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. The default optimizer is
        'TNC'. The option to use all these optimizers is not available (as it is
        in all the other Fitters). If the optimizer fails, the initial guess
        will be returned.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_beta : float, int, optional
        Used to specify the beta value if you need to force beta to be a certain
        value. Used in ALT probability plotting. Optional input. If specified it
        must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_2P alpha parameter
    beta : float
        the fitted Weibull_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Weibull_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    Requirements of the input dataframe:
    The column titles MUST be 'category', 'time', 'quantity'
    The category values MUST be 'F' for failure or 'C' for censored (right
    censored). The time values are the failure or right censored times.
    The quantity is the number of items at that time. This must be specified for
    all values even if the quantity is 1.

    Example of the input dataframe:

    +------------+------------+-----------+
    | category   | time       | quantity  |
    +============+============+===========+
    | F          | 24         | 1         |
    +------------+------------+-----------+
    | F          | 29         | 1         |
    +------------+------------+-----------+
    | F          | 34         | 1         |
    +------------+------------+-----------+
    | F          | 39         | 2         |
    +------------+------------+-----------+
    | F          | 40         | 1         |
    +------------+------------+-----------+
    | F          | 42         | 3         |
    +------------+------------+-----------+
    | F          | 44         | 1         |
    +------------+------------+-----------+
    | C          | 50         | 3         |
    +------------+------------+-----------+
    | C          | 55         | 5         |
    +------------+------------+-----------+
    | C          | 60         | 10        |
    +------------+------------+-----------+

    This is easiest to achieve by importing data from excel. An example of this
    is:

    .. code:: python

        import pandas as pd
        from reliability.Fitters import Fit_Weibull_2P_grouped
        filename = 'C:\\Users\\Current User\\Desktop\\data.xlsx'
        df = pd.read_excel(io=filename)
        Fit_Weibull_2P_grouped(dataframe=df)
    """

    def __init__(
        self,
        dataframe=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        force_beta=None,
        quantiles=None,
        method="MLE",
        optimizer=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):

        if dataframe is None or type(dataframe) is not pd.core.frame.DataFrame:
            raise ValueError(
                'dataframe must be a pandas dataframe with the columns "category" (F for failure or C for censored), "time" (the failure times), and "quantity" (the number of events at each time)'
            )
        for item in dataframe.columns.values:
            if item not in ["category", "time", "quantity"]:
                raise ValueError(
                    'The titles of the dataframe columns must be: "category" (F for failure or C for censored), "time" (the failure times), and "quantity" (the number of events at each time)'
                )
        categories = dataframe.category.unique()
        for item in categories:
            if item not in ["F", "C"]:
                raise ValueError(
                    'The category column must have values "F" or "C" for failure or censored (right censored) respectively. Other values were detected.'
                )

        # automatically filter out rows with zeros and print warning if zeros have been removed
        dataframe0 = dataframe
        dataframe = dataframe0[dataframe0["time"] > 0]
        if len(dataframe0.time.values) != len(dataframe.time.values):
            colorprint(
                "WARNING: dataframe contained zeros. These have been removed to enable fitting.",
                text_color="red",
            )

        # unpack the dataframe
        failures_df = dataframe[dataframe["category"] == "F"]
        right_censored_df = dataframe[dataframe["category"] == "C"]
        failure_times = failures_df.time.values
        failure_qty = failures_df.quantity.values
        right_censored_times = right_censored_df.time.values
        right_censored_qty = right_censored_df.quantity.values

        # recompile the data to get the plotting positions for the initial guess
        failures = np.array([])
        right_censored = np.array([])
        for i in range(len(failure_times)):
            failures = np.append(
                failures, failure_times[i] * np.ones(int(failure_qty[i]))
            )
        for i in range(len(right_censored_times)):
            right_censored = np.append(
                right_censored,
                right_censored_times[i] * np.ones(int(right_censored_qty[i])),
            )

        # perform input error checking for the rest of the inputs
        inputs = fitters_input_checking(
            dist="Weibull_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0
        if optimizer not in ["L-BFGS-B", "TNC", "powell", "nelder-mead"]:
            optimizer = "TNC"  # temporary correction for "best" and "all"

        if method == "RRX":
            guess = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_beta,
            )
            LS_method = "RRX"
        elif method == "RRY":
            guess = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_beta,
            )
            LS_method = "RRY"
        elif method in ["LS", "MLE"]:
            guess_RRX = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_beta,
            )
            guess_RRY = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_beta,
            )
            if force_beta is not None:
                loglik_RRX = -Fit_Weibull_2P_grouped.LL_fb(
                    guess_RRX,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                    force_beta,
                )
                loglik_RRY = -Fit_Weibull_2P_grouped.LL_fb(
                    guess_RRY,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                    force_beta,
                )
            else:
                loglik_RRX = -Fit_Weibull_2P_grouped.LL(
                    guess_RRX,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                )
                loglik_RRY = -Fit_Weibull_2P_grouped.LL(
                    guess_RRY,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                )
            # take the best one
            if abs(loglik_RRX) < abs(loglik_RRY):  # RRX is best
                LS_method = "RRX"
                guess = guess_RRX
            else:  # RRY is best
                LS_method = "RRY"
                guess = guess_RRY

        if method in ["LS", "RRX", "RRY"]:
            self.alpha = guess[0]
            self.beta = guess[1]
            self.method = str("Least Squares Estimation (" + LS_method + ")")
            self.optimizer = None
        elif method == "MLE":
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = optimizer
            n = sum(failure_qty) + sum(right_censored_qty)
            k = len(guess)
            initial_guess = guess
            if force_beta is None:
                bnds = [
                    (0, None),
                    (0, None),
                ]  # bounds on the solution. Helps a lot with stability
                runs = 0
                delta_BIC = 1
                BIC_array = [1000000]
                while (
                    delta_BIC > 0.001 and runs < 10
                ):  # exits after BIC convergence or 10 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(Fit_Weibull_2P_grouped.LL),
                        guess,
                        args=(
                            failure_times,
                            right_censored_times,
                            failure_qty,
                            right_censored_qty,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bnds,
                        options={"maxiter": 300},
                    )  # this includes maxiter as TNC often exceeds the default limit of 100
                    params = result.x
                    guess = [params[0], params[1]]
                    LL2 = 2 * Fit_Weibull_2P_grouped.LL(
                        guess,
                        failure_times,
                        right_censored_times,
                        failure_qty,
                        right_censored_qty,
                    )
                    BIC_array.append(np.log(n) * k + LL2)
                    delta_BIC = abs(BIC_array[-1] - BIC_array[-2])
            else:  # force beta is True
                bnds = [(0, None)]  # bounds on the solution. Helps a lot with stability
                runs = 0
                delta_BIC = 1
                BIC_array = [1000000]
                guess = [guess[0]]
                k = len(guess)
                while (
                    delta_BIC > 0.001 and runs < 10
                ):  # exits after BIC convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(Fit_Weibull_2P_grouped.LL_fb),
                        guess,
                        args=(
                            failure_times,
                            right_censored_times,
                            failure_qty,
                            right_censored_qty,
                            force_beta,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bnds,
                        options={"maxiter": 300},
                    )
                    params = result.x
                    guess = [params[0]]
                    LL2 = 2 * Fit_Weibull_2P_grouped.LL_fb(
                        guess,
                        failure_times,
                        right_censored_times,
                        failure_qty,
                        right_censored_qty,
                        force_beta,
                    )
                    BIC_array.append(np.log(n) * k + LL2)
                    delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

            # check if the optimizer was successful. If it failed then return the initial guess with a warning
            if result.success is True:
                params = result.x
                if force_beta is None:
                    self.alpha = params[0]
                    self.beta = params[1]
                else:
                    self.alpha = params[0]
                    self.beta = force_beta
            else:  # return the initial guess with a warning
                colorprint(
                    str(
                        "WARNING: MLE estimates failed for Fit_Weibull_2P_grouped. The least squares estimates have been returned. These results may not be as accurate as MLE. You may want to try another optimzer from 'L-BFGS-B','TNC','powell','nelder-mead'."
                    ),
                    text_color="red",
                )
                if force_beta is None:
                    self.alpha = initial_guess[0]
                    self.beta = initial_guess[1]
                else:
                    self.alpha = initial_guess[0]
                    self.beta = force_beta

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P_grouped.LL)(
                np.array(tuple(params)),
                np.array(tuple(failure_times)),
                np.array(tuple(right_censored_times)),
                np.array(tuple(failure_qty)),
                np.array(tuple(right_censored_qty)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix[0][1]
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta

        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P_grouped.LL_fb)(
                np.array(tuple([self.alpha])),
                np.array(tuple(failure_times)),
                np.array(tuple(right_censored_times)),
                np.array(tuple(failure_qty)),
                np.array(tuple(right_censored_qty)),
                np.array(tuple([force_beta])),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta
                self.beta_lower = self.beta
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Weibull_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = sum(failure_qty) + sum(right_censored_qty)
        if force_beta is None:
            k = 2
            LL2 = 2 * Fit_Weibull_2P_grouped.LL(
                params,
                failure_times,
                right_censored_times,
                failure_qty,
                right_censored_qty,
            )
        else:
            k = 1
            LL2 = 2 * Fit_Weibull_2P_grouped.LL_fb(
                params,
                failure_times,
                right_censored_times,
                failure_qty,
                right_censored_qty,
                force_beta,
            )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(sum(right_censored_qty) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Weibull_2P_grouped (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(sum(failure_qty)) + "/" + str(sum(right_censored_qty))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc, Q_f, Q_rc):
        # log likelihood function (2 parameter weibull) ==> T is for time, Q is for quantity
        LL_f = (Fit_Weibull_2P_grouped.logf(T_f, params[0], params[1]) * Q_f).sum()
        LL_rc = (Fit_Weibull_2P_grouped.logR(T_rc, params[0], params[1]) * Q_rc).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(params, T_f, T_rc, Q_f, Q_rc, force_beta):
        # log likelihood function (2 parameter weibull) FORCED BETA  ==> T is for time, Q is for quantity
        LL_f = (Fit_Weibull_2P_grouped.logf(T_f, params[0], force_beta) * Q_f).sum()
        LL_rc = (Fit_Weibull_2P_grouped.logR(T_rc, params[0], force_beta) * Q_rc).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_3P:
    """
    Fits a three parameter Weibull distribution (alpha,beta,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_3P alpha parameter
    beta : float
        the fitted Weibull_3P beta parameter
    gamma : float
        the fitted Weibull_3P gamma parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Weibull_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Weibull_3P results will
    be discarded and the Weibull_2P distribution will be fitted. The returned
    values for gamma and gamma_SE will be 0.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        optimizer=None,
        method="MLE",
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Weibull_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method

        LS_results = LS_optimization(
            func_name="Weibull_3P",
            LL_func=Fit_Weibull_3P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Weibull_3P",
                LL_func=Fit_Weibull_3P.LL,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Weibull_2P distribution. Can't proceed with Weibull_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Weibull_3P
            weibull_2P_results = Fit_Weibull_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = weibull_2P_results.alpha
            self.beta = weibull_2P_results.beta
            self.gamma = 0
            self.alpha_SE = weibull_2P_results.alpha_SE
            self.beta_SE = weibull_2P_results.beta_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = weibull_2P_results.Cov_alpha_beta
            self.alpha_upper = weibull_2P_results.alpha_upper
            self.alpha_lower = weibull_2P_results.alpha_lower
            self.beta_upper = weibull_2P_results.beta_upper
            self.beta_lower = weibull_2P_results.beta_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.alpha, self.beta]
            params_3P = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Weibull_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Weibull_3P.LL)(
                    np.array(tuple(params_3P)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix[0][1]
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
                self.gamma_upper = self.gamma * (
                    np.exp(Z * (self.gamma_SE / self.gamma))
                )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
                self.gamma_lower = self.gamma * (
                    np.exp(-Z * (self.gamma_SE / self.gamma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Weibull_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.gamma_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Alpha", "Beta", "Gamma"],
            "Point Estimate": [self.alpha, self.beta, self.gamma],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.gamma_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.gamma_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Weibull_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Weibull_3P.LL(params_3P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            fig = Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Weibull_3P was fitted. The default legend in the probability plot thinks Weibull_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Weibull_3P\n(α="
                        + str(round_to_decimals(self.alpha, dec))
                        + ", β="
                        + str(round_to_decimals(self.beta, dec))
                        + ", γ="
                        + str(round_to_decimals(self.gamma, dec))
                        + ")"
                    )
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, g):  # Log PDF (3 parameter Weibull)
        return (b - 1) * anp.log((t - g) / a) + anp.log(b / a) - ((t - g) / a) ** b

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Weibull)
        return -(((t - g) / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Weibull)
        LL_f = Fit_Weibull_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Mixture:
    """
    Fits a mixture of two Weibull_2P distributions (this does not fit the gamma
    parameter). Right censoring is supported, though care should be taken to
    ensure that there still appears to be two groups when plotting only the
    failure data. A second group cannot be made from a mostly or totally
    censored set of samples. Use this model when you think there are multiple
    failure modes acting to create the failure data.

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 4 failures,
        but it is highly recommended to use another model if you have less than
        20 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha_1 : float
        the fitted Weibull_2P alpha parameter for the first (left) group
    beta_1 : float
        the fitted Weibull_2P beta parameter for the first (left) group
    alpha_2 : float
        the fitted Weibull_2P alpha parameter for the second (right) group
    beta_2 : float
        the fitted Weibull_2P beta parameter for the second (right) group
    proportion_1 : float
        the fitted proportion of the first (left) group
    proportion_2 : float
        the fitted proportion of the second (right) group. Same as
        1-proportion_1
    alpha_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_1_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_2_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_2_SE :float
        the standard error (sqrt(variance)) of the parameter
    proportion_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    alpha_1_upper : float
        the upper CI estimate of the parameter
    alpha_1_lower : float
        the lower CI estimate of the parameter
    alpha_2_upper : float
        the upper CI estimate of the parameter
    alpha_2_lower : float
        the lower CI estimate of the parameter
    beta_1_upper : float
        the upper CI estimate of the parameter
    beta_1_lower : float
        the lower CI estimate of the parameter
    beta_2_upper : float
        the upper CI estimate of the parameter
    beta_2_lower : float
        the lower CI estimate of the parameter
    proportion_1_upper : float
        the upper CI estimate of the parameter
    proportion_1_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Mixture_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is different to the Weibull Competing Risks as the overall Survival
    Function is the sum of the individual Survival Functions multiplied by a
    proportion rather than being the product as is the case in the Weibull
    Competing Risks Model.

    Mixture Model: :math:`SF_{model} = (proportion_1 × SF_1) +
    ((1-proportion_1) × SF_2)`

    Competing Risks Model: :math:`SF_{model} = SF_1 × SF_2`

    Similar to the competing risks model, you can use this model when you think
    there are multiple failure modes acting to create the failure data.

    Whilst some failure modes may not be fitted as well by a Weibull
    distribution as they may be by another distribution, it is unlikely that a
    mixture of data from two distributions (particularly if they are
    overlapping) will be fitted noticeably better by other types of mixtures
    than would be achieved by a Weibull mixture. For this reason, other types
    of mixtures are not implemented.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Weibull_Mixture",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        n = len(failures) + len(right_censored)
        _, y = plotting_positions(
            failures=failures, right_censored=right_censored
        )  # this is only used to find AD

        # this algorithm is used to estimate the dividing line between the two groups
        # firstly it fits a gaussian kde to the histogram
        # then it draws two straight lines from the highest peak of the kde down to the lower and upper bounds of the failures
        # the dividing line is the point where the difference between the kde and the straight lines is greatest
        max_failures = max(failures)
        min_failures = min(failures)
        gkde = ss.gaussian_kde(failures)
        delta = max_failures - min_failures
        x_kde = np.linspace(min_failures - delta / 5, max_failures + delta / 5, 100)
        y_kde = gkde.evaluate(x_kde)
        peak_y = max(y_kde)
        peak_x = x_kde[np.where(y_kde == peak_y)][0]

        left_x = min_failures
        left_y = gkde.evaluate(left_x)
        left_m = (peak_y - left_y) / (peak_x - left_x)
        left_c = -left_m * left_x + left_y
        left_line_x = np.linspace(left_x, peak_x, 100)
        left_line_y = left_m * left_line_x + left_c  # y=mx+c
        left_kde = gkde.evaluate(left_line_x)
        left_diff = abs(left_line_y - left_kde)
        left_diff_max = max(left_diff)
        left_div_line = left_line_x[np.where(left_diff == left_diff_max)][0]

        right_x = max_failures
        right_y = gkde.evaluate(right_x)
        right_m = (right_y - peak_y) / (right_x - peak_x)
        right_c = -right_m * right_x + right_y
        right_line_x = np.linspace(peak_x, right_x, 100)
        right_line_y = right_m * right_line_x + right_c  # y=mx+c
        right_kde = gkde.evaluate(right_line_x)
        right_diff = abs(right_line_y - right_kde)
        right_diff_max = max(right_diff)
        right_div_line = right_line_x[np.where(right_diff == right_diff_max)][0]

        if left_diff_max > right_diff_max:
            dividing_line = left_div_line
        else:
            dividing_line = right_div_line

        number_of_items_in_group_1 = len(np.where(failures<dividing_line)[0])
        number_of_items_in_group_2 = len(failures) - number_of_items_in_group_1
        if number_of_items_in_group_1 < 2:
            failures_sorted = np.sort(failures)
            dividing_line = (failures_sorted[1]+failures_sorted[2])/2 # adjusts the dividing line in case there aren't enough failures in the first group
        if number_of_items_in_group_2 < 2:
            failures_sorted = np.sort(failures)
            dividing_line = (failures_sorted[-2]+failures_sorted[-3])/2 # adjusts the dividing line in case there aren't enough failures in the second group

        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < dividing_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < dividing_line:
                GROUP_1_right_cens.append(item)
            else:
                GROUP_2_right_cens.append(item)

        # get inputs for the guess by fitting a weibull to each of the groups with their respective censored data
        group_1_estimates = Fit_Weibull_2P(
            failures=GROUP_1_failures,
            right_censored=GROUP_1_right_cens,
            show_probability_plot=False,
            print_results=False,
            optimizer=optimizer,
        )
        group_2_estimates = Fit_Weibull_2P(
            failures=GROUP_2_failures,
            right_censored=GROUP_2_right_cens,
            show_probability_plot=False,
            print_results=False,
            optimizer=optimizer,
        )
        # proportion guess
        p_guess = (len(GROUP_1_failures) + len(GROUP_1_right_cens)) / n
        guess = [
            group_1_estimates.alpha,
            group_1_estimates.beta,
            group_2_estimates.alpha,
            group_2_estimates.beta,
            p_guess,
        ]  # A1,B1,A2,B2,P

        # solve it
        MLE_results = MLE_optimization(
            func_name="Weibull_mixture",
            LL_func=Fit_Weibull_Mixture.LL,
            initial_guess=guess,
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha_1 = MLE_results.alpha_1
        self.beta_1 = MLE_results.beta_1
        self.alpha_2 = MLE_results.alpha_2
        self.beta_2 = MLE_results.beta_2
        self.proportion_1 = MLE_results.proportion_1
        self.proportion_2 = MLE_results.proportion_2
        self.optimizer = MLE_results.optimizer
        dist_1 = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1)
        dist_2 = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2)
        self.distribution = Mixture_Model(
            distributions=[dist_1, dist_2],
            proportions=[self.proportion_1, self.proportion_2],
        )

        params = [
            self.alpha_1,
            self.beta_1,
            self.alpha_2,
            self.beta_2,
            self.proportion_1,
        ]
        LL2 = 2 * Fit_Weibull_Mixture.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        k = 5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Mixture.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
            self.proportion_1_SE = abs(covariance_matrix[4][4]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Weibull_Mixture model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_1_SE = 0
            self.beta_1_SE = 0
            self.alpha_2_SE = 0
            self.beta_2_SE = 0
            self.proportion_1_SE = 0

        self.alpha_1_upper = self.alpha_1 * (
            np.exp(Z * (self.alpha_1_SE / self.alpha_1))
        )
        self.alpha_1_lower = self.alpha_1 * (
            np.exp(-Z * (self.alpha_1_SE / self.alpha_1))
        )
        self.beta_1_upper = self.beta_1 * (np.exp(Z * (self.beta_1_SE / self.beta_1)))
        self.beta_1_lower = self.beta_1 * (np.exp(-Z * (self.beta_1_SE / self.beta_1)))
        self.alpha_2_upper = self.alpha_2 * (
            np.exp(Z * (self.alpha_2_SE / self.alpha_2))
        )
        self.alpha_2_lower = self.alpha_2 * (
            np.exp(-Z * (self.alpha_2_SE / self.alpha_2))
        )
        self.beta_2_upper = self.beta_2 * (np.exp(Z * (self.beta_2_SE / self.beta_2)))
        self.beta_2_lower = self.beta_2 * (np.exp(-Z * (self.beta_2_SE / self.beta_2)))
        self.proportion_1_upper = self.proportion_1 / (
            self.proportion_1
            + (1 - self.proportion_1)
            * (
                np.exp(
                    -Z
                    * self.proportion_1_SE
                    / (self.proportion_1 * (1 - self.proportion_1))
                )
            )
        )
        # ref: http://reliawiki.org/index.php/The_Mixed_Weibull_Distribution
        self.proportion_1_lower = self.proportion_1 / (
            self.proportion_1
            + (1 - self.proportion_1)
            * (
                np.exp(
                    Z
                    * self.proportion_1_SE
                    / (self.proportion_1 * (1 - self.proportion_1))
                )
            )
        )

        Data = {
            "Parameter": ["Alpha 1", "Beta 1", "Alpha 2", "Beta 2", "Proportion 1"],
            "Point Estimate": [
                self.alpha_1,
                self.beta_1,
                self.alpha_2,
                self.beta_2,
                self.proportion_1,
            ],
            "Standard Error": [
                self.alpha_1_SE,
                self.beta_1_SE,
                self.alpha_2_SE,
                self.beta_2_SE,
                self.proportion_1_SE,
            ],
            "Lower CI": [
                self.alpha_1_lower,
                self.beta_1_lower,
                self.alpha_2_lower,
                self.beta_2_lower,
                self.proportion_1_lower,
            ],
            "Upper CI": [
                self.alpha_1_upper,
                self.beta_1_upper,
                self.alpha_2_upper,
                self.beta_2_upper,
                self.proportion_1_upper,
            ],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_Mixture (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull MM "
                    + str(round_to_decimals(self.proportion_1, dec))
                    + r" ($\alpha_1=$"
                    + str(round_to_decimals(self.alpha_1, dec))
                    + r", $\beta_1=$"
                    + str(round_to_decimals(self.beta_1, dec))
                    + ")+\n                             "
                    + str(round_to_decimals(self.proportion_2, dec))
                    + r" ($\alpha_2=$"
                    + str(round_to_decimals(self.alpha_2, dec))
                    + r", $\beta_2=$"
                    + str(round_to_decimals(self.beta_2, dec))
                    + ")"
                )
            xvals = np.logspace(
                np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Mixture CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a1, b1, a2, b2, p):  # Log Mixture PDF (2 parameter Weibull)
        return anp.log(
            p * ((b1 * t ** (b1 - 1)) / (a1 ** b1)) * anp.exp(-((t / a1) ** b1))
            + (1 - p) * ((b2 * t ** (b2 - 1)) / (a2 ** b2)) * anp.exp(-((t / a2) ** b2))
        )

    @staticmethod
    def logR(t, a1, b1, a2, b2, p):  # Log Mixture SF (2 parameter Weibull)
        return anp.log(
            p * anp.exp(-((t / a1) ** b1)) + (1 - p) * anp.exp(-((t / a2) ** b2))
        )

    @staticmethod
    def LL(params, T_f, T_rc):
        # Log Mixture Likelihood function (2 parameter weibull)
        LL_f = Fit_Weibull_Mixture.logf(
            T_f, params[0], params[1], params[2], params[3], params[4]
        ).sum()
        LL_rc = Fit_Weibull_Mixture.logR(
            T_rc, params[0], params[1], params[2], params[3], params[4]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_CR:
    """
    Fits a Weibull Competing Risks Model consisting of two Weibull_2P
    distributions (this does not fit the gamma parameter). Similar to the
    mixture model, you can use this model when you think there are multiple
    failure modes acting to create the failure data.

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 4 failures,
        but it is highly recommended to use another model if you have less than
        20 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha_1 : float
        the fitted Weibull_2P alpha parameter for the first distribution
    beta_1 : float
        the fitted Weibull_2P beta parameter for the first distribution
    alpha_2 : float
        the fitted Weibull_2P alpha parameter for the second distribution
    beta_2 : float
        the fitted Weibull_2P beta parameter for the second distribution
    alpha_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_1_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_2_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_2_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_1_upper : float
        the upper CI estimate of the parameter
    alpha_1_lower : float
        the lower CI estimate of the parameter
    alpha_2_upper : float
        the upper CI estimate of the parameter
    alpha_2_lower : float
        the lower CI estimate of the parameter
    beta_1_upper : float
        the upper CI estimate of the parameter
    beta_1_lower : float
        the lower CI estimate of the parameter
    beta_2_upper : float
        the upper CI estimate of the parameter
    beta_2_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Competing_Risks_Model object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is different to the Weibull Mixture model as the overall Survival
    Function is the product of the individual Survival Functions rather than
    being the sum as is the case in the Weibull Mixture Model.

    Mixture Model: :math:`SF_{model} = (proportion_1 × SF_1) +
    ((1-proportion_1) × SF_2)`

    Competing Risks Model: :math:`SF_{model} = SF_1 × SF_2`

    Whilst some failure modes may not be fitted as well by a Weibull
    distribution as they may be by another distribution, it is unlikely that
    data from a competing risks model will be fitted noticeably better by other
    types of competing risks models than would be achieved by a Weibull
    Competing Risks model. For this reason, other types of competing risks
    models are not implemented.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Weibull_CR",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        n = len(failures) + len(right_censored)
        _, y = plotting_positions(
            failures=failures, right_censored=right_censored
        )  # this is only used to find AD

        # this algorithm is used to estimate the dividing line between the two groups
        # firstly it fits a gaussian kde to the histogram
        # then it draws two straight lines from the highest peak of the kde down to the lower and upper bounds of the failures
        # the dividing line is the point where the difference between the kde and the straight lines is greatest
        max_failures = max(failures)
        min_failures = min(failures)
        gkde = ss.gaussian_kde(failures)
        delta = max_failures - min_failures
        x_kde = np.linspace(min_failures - delta / 5, max_failures + delta / 5, 100)
        y_kde = gkde.evaluate(x_kde)
        peak_y = max(y_kde)
        peak_x = x_kde[np.where(y_kde == peak_y)][0]

        left_x = min_failures
        left_y = gkde.evaluate(left_x)
        left_m = (peak_y - left_y) / (peak_x - left_x)
        left_c = -left_m * left_x + left_y
        left_line_x = np.linspace(left_x, peak_x, 100)
        left_line_y = left_m * left_line_x + left_c  # y=mx+c
        left_kde = gkde.evaluate(left_line_x)
        left_diff = abs(left_line_y - left_kde)
        left_diff_max = max(left_diff)
        left_div_line = left_line_x[np.where(left_diff == left_diff_max)][0]

        right_x = max_failures
        right_y = gkde.evaluate(right_x)
        right_m = (right_y - peak_y) / (right_x - peak_x)
        right_c = -right_m * right_x + right_y
        right_line_x = np.linspace(peak_x, right_x, 100)
        right_line_y = right_m * right_line_x + right_c  # y=mx+c
        right_kde = gkde.evaluate(right_line_x)
        right_diff = abs(right_line_y - right_kde)
        right_diff_max = max(right_diff)
        right_div_line = right_line_x[np.where(right_diff == right_diff_max)][0]

        if left_diff_max > right_diff_max:
            dividing_line = left_div_line
        else:
            dividing_line = right_div_line

        number_of_items_in_group_1 = len(np.where(failures<dividing_line)[0])
        number_of_items_in_group_2 = len(failures) - number_of_items_in_group_1
        if number_of_items_in_group_1 < 2:
            failures_sorted = np.sort(failures)
            dividing_line = (failures_sorted[1]+failures_sorted[2])/2 # adjusts the dividing line in case there aren't enough failures in the first group
        if number_of_items_in_group_2 < 2:
            failures_sorted = np.sort(failures)
            dividing_line = (failures_sorted[-2]+failures_sorted[-3])/2 # adjusts the dividing line in case there aren't enough failures in the second group

        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < dividing_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < dividing_line:
                GROUP_1_right_cens.append(item)
            else:
                GROUP_2_right_cens.append(item)

        # get inputs for the guess by fitting a weibull to each of the groups with their respective censored data
        group_1_estimates = Fit_Weibull_2P(
            failures=GROUP_1_failures,
            right_censored=GROUP_1_right_cens,
            show_probability_plot=False,
            print_results=False,
        )
        group_2_estimates = Fit_Weibull_2P(
            failures=GROUP_2_failures,
            right_censored=GROUP_2_right_cens,
            show_probability_plot=False,
            print_results=False,
        )
        guess = [
            group_1_estimates.alpha,
            group_1_estimates.beta,
            group_2_estimates.alpha,
            group_2_estimates.beta,
        ]  # A1,B1,A2,B2

        # solve it
        MLE_results = MLE_optimization(
            func_name="Weibull_CR",
            LL_func=Fit_Weibull_CR.LL,
            initial_guess=guess,
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha_1 = MLE_results.alpha_1
        self.beta_1 = MLE_results.beta_1
        self.alpha_2 = MLE_results.alpha_2
        self.beta_2 = MLE_results.beta_2
        self.optimizer = MLE_results.optimizer
        dist_1 = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1)
        dist_2 = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2)
        self.distribution = Competing_Risks_Model(distributions=[dist_1, dist_2])

        params = [self.alpha_1, self.beta_1, self.alpha_2, self.beta_2]
        k = 4
        LL2 = 2 * Fit_Weibull_CR.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_CR.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Weibull_CR model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_1_SE = 0
            self.beta_1_SE = 0
            self.alpha_2_SE = 0
            self.beta_2_SE = 0

        self.alpha_1_upper = self.alpha_1 * (
            np.exp(Z * (self.alpha_1_SE / self.alpha_1))
        )
        self.alpha_1_lower = self.alpha_1 * (
            np.exp(-Z * (self.alpha_1_SE / self.alpha_1))
        )
        self.beta_1_upper = self.beta_1 * (np.exp(Z * (self.beta_1_SE / self.beta_1)))
        self.beta_1_lower = self.beta_1 * (np.exp(-Z * (self.beta_1_SE / self.beta_1)))
        self.alpha_2_upper = self.alpha_2 * (
            np.exp(Z * (self.alpha_2_SE / self.alpha_2))
        )
        self.alpha_2_lower = self.alpha_2 * (
            np.exp(-Z * (self.alpha_2_SE / self.alpha_2))
        )
        self.beta_2_upper = self.beta_2 * (np.exp(Z * (self.beta_2_SE / self.beta_2)))
        self.beta_2_lower = self.beta_2 * (np.exp(-Z * (self.beta_2_SE / self.beta_2)))

        Data = {
            "Parameter": ["Alpha 1", "Beta 1", "Alpha 2", "Beta 2"],
            "Point Estimate": [self.alpha_1, self.beta_1, self.alpha_2, self.beta_2],
            "Standard Error": [
                self.alpha_1_SE,
                self.beta_1_SE,
                self.alpha_2_SE,
                self.beta_2_SE,
            ],
            "Lower CI": [
                self.alpha_1_lower,
                self.beta_1_lower,
                self.alpha_2_lower,
                self.beta_2_lower,
            ],
            "Upper CI": [
                self.alpha_1_upper,
                self.beta_1_upper,
                self.alpha_2_upper,
                self.beta_2_upper,
            ],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_CR (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull CR "
                    + r" ($\alpha_1=$"
                    + str(round_to_decimals(self.alpha_1, dec))
                    + r", $\beta_1=$"
                    + str(round_to_decimals(self.beta_1, dec))
                    + ") ×\n                            "
                    + r" ($\alpha_2=$"
                    + str(round_to_decimals(self.alpha_2, dec))
                    + r", $\beta_2=$"
                    + str(round_to_decimals(self.beta_2, dec))
                    + ")"
                )
            xvals = np.logspace(
                np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            plt.title("Probability Plot\nWeibull Competing Risks CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a1, b1, a2, b2):  # Log PDF (Competing Risks)
        return anp.log(
            -(-(b2 * (t / a2) ** b2) / t - (b1 * (t / a1) ** b1) / t)
            * anp.exp(-((t / a2) ** b2) - (t / a1) ** b1)
        )

    @staticmethod
    def logR(t, a1, b1, a2, b2):  # Log SF (Competing Risks)
        return -((t / a1) ** b1) - ((t / a2) ** b2)

    @staticmethod
    def LL(params, T_f, T_rc):
        # Log Likelihood function (Competing Risks)
        LL_f = Fit_Weibull_CR.logf(
            T_f, params[0], params[1], params[2], params[3]
        ).sum()
        LL_rc = Fit_Weibull_CR.logR(
            T_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_DSZI:
    """
    Fits a Weibull Defective Subpopulation Zero Inflated (DSZI) distribution to
    the data provided. This is a 4 parameter distribution (alpha, beta, DS, ZI).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 non-zero
        failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_DSZI alpha parameter
    beta : float
        the fitted Weibull_DSZI beta parameter
    DS : float
        the fitted Weibull_DSZI DS parameter
    ZI : float
        the fitted Weibull_DSZI ZI parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    DS_SE :float
        the standard error (sqrt(variance)) of the parameter
    ZI_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    DS_upper : float
        the upper CI estimate of the parameter
    DS_lower : float
        the lower CI estimate of the parameter
    ZI_upper : float
        the upper CI estimate of the parameter
    ZI_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        # need to remove zeros before passing to fitters input checking
        failures = np.asarray(failures)
        failures_no_zeros = failures[failures != 0]
        failures_zeros = failures[failures == 0]

        inputs = fitters_input_checking(
            dist="Weibull_DSZI",
            failures=failures_no_zeros,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures_no_zeros = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        _, y_pts = plotting_positions(failures=failures, right_censored=right_censored)
        DS_guess = max(y_pts)

        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures_no_zeros,
            right_censored=None,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta
        ZI_guess = len(failures_zeros) / (len(failures) + len(right_censored))

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_DSZI",
            LL_func=Fit_Weibull_DSZI.LL,
            initial_guess=[alpha_guess, beta_guess, DS_guess, ZI_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.DS = MLE_results.DS
        self.ZI = MLE_results.ZI
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.DS, self.ZI]
        hessian_matrix = hessian(Fit_Weibull_DSZI.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures_zeros)),
            np.array(tuple(failures_no_zeros)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.DS_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.ZI_SE = abs(covariance_matrix[3][3]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Weibull_DSZI Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.DS_SE = 0
            self.ZI_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.DS == 1:
            self.DS_lower = 1  # DS=1 causes a divide by zero error for CIs
            self.DS_upper = 1
        else:
            self.DS_upper = self.DS / (
                self.DS
                + (1 - self.DS) * (np.exp(-Z * self.DS_SE / (self.DS * (1 - self.DS))))
            )
            self.DS_lower = self.DS / (
                self.DS
                + (1 - self.DS) * (np.exp(Z * self.DS_SE / (self.DS * (1 - self.DS))))
            )
        if self.ZI == 0:
            self.ZI_upper = 0  # ZI = 0 causes a divide by zero error for CIs
            self.ZI_lower = 0
        else:
            self.ZI_upper = self.ZI / (
                self.ZI
                + (1 - self.ZI) * (np.exp(-Z * self.ZI_SE / (self.ZI * (1 - self.ZI))))
            )
            self.ZI_lower = self.ZI / (
                self.ZI
                + (1 - self.ZI) * (np.exp(Z * self.ZI_SE / (self.ZI * (1 - self.ZI))))
            )

        results_data = {
            "Parameter": ["Alpha", "Beta", "DS", "ZI"],
            "Point Estimate": [self.alpha, self.beta, self.DS, self.ZI],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.DS_SE, self.ZI_SE],
            "Lower CI": [
                self.alpha_lower,
                self.beta_lower,
                self.DS_lower,
                self.ZI_lower,
            ],
            "Upper CI": [
                self.alpha_upper,
                self.beta_upper,
                self.DS_upper,
                self.ZI_upper,
            ],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            DS=self.DS,
            ZI=self.ZI,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 4
        LL2 = 2 * Fit_Weibull_DSZI.LL(
            params, failures_zeros, failures_no_zeros, right_censored
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        # moves all the y values for the x=0 points to be equal to the value of ZI.
        y = np.where(x == 0, self.ZI, y)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_DSZI (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Weibull_probability_plot,
                plot_points,
            )

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures_no_zeros,
                right_censored=rc,
                show_fitted_distribution=False,
                show_scatter_points=False,
                **kwargs,
            )

            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull_DSZI"
                    + r" ($\alpha=$"
                    + str(round_to_decimals(self.alpha, dec))
                    + r", $\beta=$"
                    + str(round_to_decimals(self.beta, dec))
                    + r", $DS=$"
                    + str(round_to_decimals(self.DS, dec))
                    + r", $ZI=$"
                    + str(round_to_decimals(self.ZI, dec))
                    + ")"
                )
            plot_points(
                failures=failures,
                right_censored=right_censored,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )

            xvals = np.logspace(
                np.log10(min(failures_no_zeros)) - 3,
                np.log10(max(failures_no_zeros)) + 1,
                1000,
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title(
                "Probability Plot\nWeibull Defective Subpopulation Zero Inflated CDF"
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, ds, zi):  # Log PDF (Weibull DSZI)
        return (
            (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(ds - zi)
        )

    @staticmethod
    def logR(t, a, b, ds, zi):  # Log SF (Weibull DSZI)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * (ds - zi) + zi))

    @staticmethod
    def LL(params, T_0, T_f, T_rc):
        # log likelihood function (Weibull DSZI)
        if params[3] > 0:
            LL_0 = anp.log(params[3]) * len(T_0)  # deals with t=0
        else:
            LL_0 = 0  # enables fitting when ZI = 0 to avoid log(0) error
        LL_f = Fit_Weibull_DSZI.logf(
            T_f, params[0], params[1], params[2], params[3]
        ).sum()
        LL_rc = Fit_Weibull_DSZI.logR(
            T_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_0 + LL_f + LL_rc)


class Fit_Weibull_DS:
    """
    Fits a Weibull Defective Subpopulation (DS) distribution to the data
    provided. This is a 3 parameter distribution (alpha, beta, DS).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_DS alpha parameter
    beta : float
        the fitted Weibull_DS beta parameter
    DS : float
        the fitted Weibull_DS DS parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    DS_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    DS_upper : float
        the upper CI estimate of the parameter
    DS_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Weibull_DS",
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        _, y_pts = plotting_positions(failures=failures, right_censored=right_censored)

        DS_guess = max(y_pts)
        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures,
            right_censored=None,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_DS",
            LL_func=Fit_Weibull_DS.LL,
            initial_guess=[alpha_guess, beta_guess, DS_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.DS = MLE_results.DS
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.DS]
        hessian_matrix = hessian(Fit_Weibull_DS.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.DS_SE = abs(covariance_matrix[2][2]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Weibull_DS Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.DS_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.DS == 1:
            self.DS_lower = 1  # DS=1 causes a divide by zero error for CIs
            self.DS_upper = 1
        else:
            self.DS_upper = self.DS / (
                self.DS
                + (1 - self.DS) * (np.exp(-Z * self.DS_SE / (self.DS * (1 - self.DS))))
            )
            self.DS_lower = self.DS / (
                self.DS
                + (1 - self.DS) * (np.exp(Z * self.DS_SE / (self.DS * (1 - self.DS))))
            )

        results_data = {
            "Parameter": ["Alpha", "Beta", "DS"],
            "Point Estimate": [self.alpha, self.beta, self.DS],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.DS_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.DS_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.DS_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            DS=self.DS,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Weibull_DS.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_DS (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull_DS"
                    + r" ($\alpha=$"
                    + str(round_to_decimals(self.alpha, dec))
                    + r", $\beta=$"
                    + str(round_to_decimals(self.beta, dec))
                    + r", $DS=$"
                    + str(round_to_decimals(self.DS, dec))
                    + ")"
                )
            xvals = np.logspace(
                np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Defective Subpopulation CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, ds):  # Log PDF (Weibull DS)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(ds)

    @staticmethod
    def logR(t, a, b, ds):  # Log SF (Weibull DS)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * ds))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (Weibull DS)
        LL_f = Fit_Weibull_DS.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_DS.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_ZI:
    """
    Fits a Weibull Zero Inflated (ZI) distribution to the data
    provided. This is a 3 parameter distribution (alpha, beta, ZI).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 non-zero
        failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Weibull_ZI alpha parameter
    beta : float
        the fitted Weibull_ZI beta parameter
    ZI : float
        the fitted Weibull_ZI ZI parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    ZI_SE :float
        the standard error (sqrt(variance)) of the parameter.
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    ZI_upper : float
        the upper CI estimate of the parameter.
    ZI_lower : float
        the lower CI estimate of the parameter.
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        # need to remove zeros before passing to fitters input checking
        failures = np.asarray(failures)
        failures_no_zeros = failures[failures != 0]
        failures_zeros = failures[failures == 0]

        inputs = fitters_input_checking(
            dist="Weibull_ZI",
            failures=failures_no_zeros,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures_no_zeros = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures_no_zeros,
            right_censored=right_censored,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
            CI=CI,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta
        ZI_guess = len(failures_zeros) / (len(failures) + len(right_censored))

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_ZI",
            LL_func=Fit_Weibull_ZI.LL,
            initial_guess=[alpha_guess, beta_guess, ZI_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.ZI = MLE_results.ZI
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.ZI]
        hessian_matrix = hessian(Fit_Weibull_ZI.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures_zeros)),
            np.array(tuple(failures_no_zeros)),
            np.array(tuple(right_censored)),
        )

        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.ZI_SE = abs(covariance_matrix[2][2]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Weibull_ZI Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.ZI_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.ZI == 0:
            self.ZI_upper = 0  # ZI = 0 causes a divide by zero error for CIs
            self.ZI_lower = 0
        else:
            self.ZI_upper = self.ZI / (
                self.ZI
                + (1 - self.ZI) * (np.exp(-Z * self.ZI_SE / (self.ZI * (1 - self.ZI))))
            )
            self.ZI_lower = self.ZI / (
                self.ZI
                + (1 - self.ZI) * (np.exp(Z * self.ZI_SE / (self.ZI * (1 - self.ZI))))
            )

        results_data = {
            "Parameter": ["Alpha", "Beta", "ZI"],
            "Point Estimate": [self.alpha, self.beta, self.ZI],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.ZI_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.ZI_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.ZI_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            ZI=self.ZI,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Weibull_ZI.LL(
            params, failures_zeros, failures_no_zeros, right_censored
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        # moves all the y values for the x=0 points to be equal to the value of ZI.
        y = np.where(x == 0, self.ZI, y)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_ZI (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Weibull_probability_plot,
                plot_points,
            )

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(
                failures=failures_no_zeros,
                right_censored=rc,
                show_fitted_distribution=False,
                show_scatter_points=False,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull_ZI"
                    + r" ($\alpha=$"
                    + str(round_to_decimals(self.alpha, dec))
                    + r", $\beta=$"
                    + str(round_to_decimals(self.beta, dec))
                    + r", $ZI=$"
                    + str(round_to_decimals(self.ZI, dec))
                    + ")"
                )
            plot_points(
                failures=failures,
                right_censored=right_censored,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            xvals = np.logspace(
                np.log10(min(failures_no_zeros)) - 3,
                np.log10(max(failures_no_zeros)) + 1,
                1000,
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Zero Inflated CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, zi):  # Log PDF (Weibull ZI)
        return (
            (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(1 - zi)
        )

    @staticmethod
    def logR(t, a, b, zi):  # Log SF (Weibull ZI)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * (1 - zi) + zi))

    @staticmethod
    def LL(params, T_0, T_f, T_rc):
        # log likelihood function (Weibull ZI)
        if params[2] > 0:
            LL_0 = anp.log(params[2]) * len(T_0)  # deals with t=0
        else:
            LL_0 = 0  # enables fitting when ZI = 0 to avoid log(0) error
        LL_f = Fit_Weibull_ZI.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_ZI.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_0 + LL_f + LL_rc)


class Fit_Exponential_1P:
    """
    Fits a one parameter Exponential distribution (Lambda) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for the model's parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    Lambda : float
        the fitted Exponential_1P Lambda parameter
    Lambda_inv : float
        the inverse of the fitted Exponential_1P Lambda parameter
    Lambda_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_SE_inv : float
        the standard error (sqrt(variance)) of the inverse of the parameter
    Lambda_upper : float
        the upper CI estimate of the parameter
    Lambda_lower : float
        the lower CI estimate of the parameter
    Lambda_upper_inv : float
        the upper CI estimate of the inverse of the parameter
    Lambda_lower_inv : float
        the lower CI estimate of the inverse of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Exponential_Distribution object with the parameter of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles. This is only produced if
        quantiles is not None. Since quantiles defaults to None, this output
        is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is a one parameter distribution, but the results provide both the
    parameter (Lambda) as well as the inverse (1/Lambda). This is provided for
    convenience as some other software (Minitab and scipy.stats) use 1/Lambda
    instead of Lambda. Lambda_SE_inv, Lambda_upper_inv, and Lambda_lower_inv are
    also provided for convenience.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Exponential_1P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Exponential_1P",
            LL_func=Fit_Exponential_1P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.Lambda = LS_results.guess[0]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Exponential_1P",
                LL_func=Fit_Exponential_1P.LL,
                initial_guess=[LS_results.guess[0]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.Lambda = MLE_results.scale
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.Lambda]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.Lambda_upper = self.Lambda * (
                np.exp(Z * (self.Lambda_SE / self.Lambda))
            )
            self.Lambda_lower = self.Lambda * (
                np.exp(-Z * (self.Lambda_SE / self.Lambda))
            )
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = abs(
                1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z
            )
            self.Lambda_lower_inv = 1 / self.Lambda_upper
            self.Lambda_upper_inv = 1 / self.Lambda_lower
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_1P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.Lambda_SE = 0
            self.Lambda_upper = self.Lambda
            self.Lambda_lower = self.Lambda
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = 0
            self.Lambda_lower_inv = 1 / self.Lambda
            self.Lambda_upper_inv = 1 / self.Lambda

        results_data = {
            "Parameter": ["Lambda", "1/Lambda"],
            "Point Estimate": [self.Lambda, self.Lambda_inv],
            "Standard Error": [self.Lambda_SE, self.Lambda_SE_inv],
            "Lower CI": [self.Lambda_lower, self.Lambda_lower_inv],
            "Upper CI": [self.Lambda_upper, self.Lambda_upper_inv],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Exponential_Distribution(
            Lambda=self.Lambda, Lambda_SE=self.Lambda_SE, CI=CI
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 1
        LL2 = 2 * Fit_Exponential_1P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_1P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale,
            )

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Exponential_probability_plot_Weibull_Scale(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, L):  # Log PDF (1 parameter Expon)
        return anp.log(L) - L * t

    @staticmethod
    def logR(t, L):  # Log SF (1 parameter Expon)
        return -(L * t)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (1 parameter Expon)
        LL_f = Fit_Exponential_1P.logf(T_f, params[0]).sum()
        LL_rc = Fit_Exponential_1P.logR(T_rc, params[0]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_2P:
    """
    Fits a two parameter Exponential distribution (Lambda, gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for the model's parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    Lambda : float
        the fitted Exponential_1P Lambda parameter
    Lambda_inv : float
        the inverse of the fitted Exponential_1P Lambda parameter
    gamma : float
        the fitted Exponential_2P gamma parameter
    Lambda_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_SE_inv : float
        the standard error (sqrt(variance)) of the inverse of the parameter
    gamma_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_upper : float
        the upper CI estimate of the parameter
    Lambda_lower : float
        the lower CI estimate of the parameter
    Lambda_upper_inv : float
        the upper CI estimate of the inverse of the parameter
    Lambda_lower_inv : float
        the lower CI estimate of the inverse of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Exponential_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles. This is only produced if
        quantiles is not None. Since quantiles defaults to None, this output
        is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is a two parameter distribution (Lambda, gamma), but the results
    provide both Lambda as well as the inverse (1/Lambda). This is provided for
    convenience as some other software (Minitab and scipy.stats) use 1/Lambda
    instead of Lambda. Lambda_SE_inv, Lambda_upper_inv, and Lambda_lower_inv are
    also provided for convenience.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        # To obtain the confidence intervals of the parameters, the gamma parameter is estimated by optimizing the log-likelihood function but
        # it is assumed as fixed because the variance-covariance matrix of the estimated parameters cannot be determined numerically. By assuming
        # the standard error in gamma is zero, we can use Exponential_1P to obtain the confidence intervals for Lambda. This is the same procedure
        # performed by both Reliasoft and Minitab. You may find the results are slightly different to Minitab and this is because the optimization
        # of gamma is done more efficiently here than Minitab does it. This is evidenced by comparing the log-likelihood for the same data input.

        inputs = fitters_input_checking(
            dist="Exponential_2P",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            quantiles=quantiles,
            method=method,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        quantiles = inputs.quantiles
        method = inputs.method
        optimizer = inputs.optimizer

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Exponential_2P",
            LL_func=Fit_Exponential_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.Lambda = LS_results.guess[0]
            self.gamma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            if (
                LS_results.guess[0] < 1
            ):  # The reason for having an inverted and non-inverted cases is due to the gradient being too shallow in some cases. If Lambda<1 we invert it so it's bigger. This prevents the gradient getting too shallow for the optimizer to find the correct minimum.
                MLE_results = MLE_optimization(
                    func_name="Exponential_2P",
                    LL_func=Fit_Exponential_2P.LL_inv,
                    initial_guess=[1 / LS_results.guess[0], LS_results.guess[1]],
                    failures=failures,
                    right_censored=right_censored,
                    optimizer=optimizer,
                )
                self.Lambda = 1 / MLE_results.scale
            else:
                MLE_results = MLE_optimization(
                    func_name="Exponential_2P",
                    LL_func=Fit_Exponential_2P.LL,
                    initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                    failures=failures,
                    right_censored=right_censored,
                    optimizer=optimizer,
                )
                self.Lambda = MLE_results.scale
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. Uses Exponential_1P because gamma (while optimized) cannot be used in the MLE solution as the solution is unbounded. This is why there are no CI limits on gamma.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_1P = [self.Lambda]
        params_2P = [self.Lambda, self.gamma]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params_1P)),
            np.array(tuple(failures - self.gamma)),
            np.array(tuple(right_censored - self.gamma)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.gamma_SE = 0
            self.Lambda_upper = self.Lambda * (
                np.exp(Z * (self.Lambda_SE / self.Lambda))
            )
            self.Lambda_lower = self.Lambda * (
                np.exp(-Z * (self.Lambda_SE / self.Lambda))
            )
            self.gamma_upper = self.gamma
            self.gamma_lower = self.gamma
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = abs(
                1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z
            )
            self.Lambda_lower_inv = 1 / self.Lambda_upper
            self.Lambda_upper_inv = 1 / self.Lambda_lower
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.Lambda_SE = 0
            self.gamma_SE = 0
            self.Lambda_upper = self.Lambda
            self.Lambda_lower = self.Lambda
            self.gamma_upper = self.gamma
            self.gamma_lower = self.gamma
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = 0
            self.Lambda_lower_inv = 1 / self.Lambda
            self.Lambda_upper_inv = 1 / self.Lambda

        results_data = {
            "Parameter": ["Lambda", "1/Lambda", "Gamma"],
            "Point Estimate": [self.Lambda, self.Lambda_inv, self.gamma],
            "Standard Error": [self.Lambda_SE, self.Lambda_SE_inv, self.gamma_SE],
            "Lower CI": [self.Lambda_lower, self.Lambda_lower_inv, self.gamma_lower],
            "Upper CI": [self.Lambda_upper, self.Lambda_upper_inv, self.gamma_upper],
        }

        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Exponential_Distribution(
            Lambda=self.Lambda, gamma=self.gamma, Lambda_SE=self.Lambda_SE, CI=CI
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Exponential_2P.LL(params_2P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale,
            )

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Exponential_probability_plot_Weibull_Scale(
                failures=failures,
                right_censored=rc,
                CI=CI,
                __fitted_dist_params=self,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, L, g):  # Log PDF (2 parameter Expon)
        return anp.log(L) - L * (t - g)

    @staticmethod
    def logR(t, L, g):  # Log SF (2 parameter Expon)
        return -(L * (t - g))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Expon)
        LL_f = Fit_Exponential_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Exponential_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    # #this is the inverted forms of the above functions. It simply changes Lambda to be 1/Lambda which is necessary when Lambda<<1
    @staticmethod
    def LL_inv(params, T_f, T_rc):
        # log likelihood function (2 parameter Expon)
        LL_f = Fit_Exponential_2P.logf(T_f, 1 / params[0], params[1]).sum()
        LL_rc = Fit_Exponential_2P.logR(T_rc, 1 / params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_2P:
    """
    Fits a two parameter Normal distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain
    (x<0). If you need an entirely positive distribution that is similar to
    Normal then consider using Weibull.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_sigma is not
        specified or at least 1 element if force_sigma is specified.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_sigma : float, int, optional
        Used to specify the beta value if you need to force sigma to be a
        certain value. Used in ALT probability plotting. Optional input. If
        specified it must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    mu : float
        the fitted Normal_2P mu parameter
    sigma : float
        the fitted Normal_2P sigma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Normal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Normal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Normal_2P",
            LL_func=Fit_Normal_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_sigma,
            LL_func_force=Fit_Normal_2P.LL_fs,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Normal_2P",
                LL_func=Fit_Normal_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_sigma,
                LL_func_force=Fit_Normal_2P.LL_fs,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Normal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (
                    Z * self.mu_SE
                )  # these are unique to normal and lognormal mu params
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (
                    np.exp(Z * (self.sigma_SE / self.sigma))
                )
                self.sigma_lower = self.sigma * (
                    np.exp(-Z * (self.sigma_SE / self.sigma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Normal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        else:
            hessian_matrix = hessian(Fit_Normal_2P.LL_fs)(
                np.array(tuple([self.mu])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_sigma])),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu + (
                    Z * self.mu_SE
                )  # these are unique to normal and lognormal mu params
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Normal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Normal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        if force_sigma is None:
            k = 2
            LL2 = 2 * Fit_Normal_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Normal_2P.LL_fs(
                [self.mu], failures, right_censored, force_sigma
            )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Normal_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Normal_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Normal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Normal)
        return anp.log(anp.exp(-0.5 * (((t - mu) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Normal)
        return anp.log((1 + erf(((mu - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Normal)
        LL_f = Fit_Normal_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Normal_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(params, T_f, T_rc, force_sigma):
        # log likelihood function (2 parameter Normal) FORCED SIGMA
        LL_f = Fit_Normal_2P.logf(T_f, params[0], force_sigma).sum()
        LL_rc = Fit_Normal_2P.logR(T_rc, params[0], force_sigma).sum()
        return -(LL_f + LL_rc)


class Fit_Gumbel_2P:
    """
    Fits a two parameter Gumbel distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain
    (x<0). If you need an entirely positive distribution that is similar to
    Gumbel then consider using Weibull.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Gumbel_2P mu parameter
    sigma : float
        the fitted Gumbel_2P sigma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gumbel_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    The Gumbel Distribution is similar to the Normal Distribution, with mu
    controlling the peak of the distribution between -inf < mu < inf.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Gumbel_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Gumbel_2P",
            LL_func=Fit_Gumbel_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Gumbel_2P",
                LL_func=Fit_Gumbel_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        hessian_matrix = hessian(Fit_Gumbel_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = covariance_matrix[0][1]
            self.mu_upper = self.mu + (
                Z * self.mu_SE
            )  # these are unique to gumbel, normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Gumbel_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.mu_SE = 0
            self.sigma_SE = 0
            self.Cov_mu_sigma = 0
            self.mu_upper = self.mu
            self.mu_lower = self.mu
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gumbel_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Gumbel_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gumbel_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gumbel_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Gumbel_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Gumbel)
        return -anp.log(sigma) + (t - mu) / sigma - anp.exp((t - mu) / sigma)

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Gumbel)
        return -anp.exp((t - mu) / sigma)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Gumbel)
        LL_f = Fit_Gumbel_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gumbel_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_2P:
    """
    Fits a two parameter Lognormal distribution (mu,sigma) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_sigma is not
        specified or at least 1 element if force_sigma is specified.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_sigma : float, int, optional
        Used to specify the sigma value if you need to force sigma to be a
        certain value. Used in ALT probability plotting. Optional input. If
        specified it must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Lognormal_2P alpha parameter
    sigma : float
        the fitted Lognormal_2P beta parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Lognormal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Lognormal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Lognormal_2P",
            LL_func=Fit_Lognormal_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_sigma,
            LL_func_force=Fit_Lognormal_2P.LL_fs,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Lognormal_2P",
                LL_func=Fit_Lognormal_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_sigma,
                LL_func_force=Fit_Lognormal_2P.LL_fs,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (
                    np.exp(Z * (self.sigma_SE / self.sigma))
                )  # sigma is strictly positive
                self.sigma_lower = self.sigma * (
                    np.exp(-Z * (self.sigma_SE / self.sigma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
        else:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL_fs)(
                np.array(tuple([self.mu])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_sigma])),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Lognormal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        if force_sigma is None:
            k = 2
            LL2 = 2 * Fit_Lognormal_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Lognormal_2P.LL_fs(
                [self.mu], failures, right_censored, force_sigma
            )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Lognormal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Lognormal)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - mu) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t) - mu) / (sigma * 2 ** 0.5)))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter lognormal)
        LL_f = Fit_Lognormal_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Lognormal_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(params, T_f, T_rc, force_sigma):
        # log likelihood function (2 parameter lognormal) FORCED SIGMA
        LL_f = Fit_Lognormal_2P.logf(T_f, params[0], force_sigma).sum()
        LL_rc = Fit_Lognormal_2P.logR(T_rc, params[0], force_sigma).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_3P:
    """
    Fits a three parameter Lognormal distribution (mu,sigma,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Lognormal_3P mu parameter
    sigma : float
        the fitted Lognormal_3P sigma parameter
    gamma : float
        the fitted Lognormal_3P gamma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Lognormal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Lognormal_3P results
    will be discarded and the Lognormal_2P distribution will be fitted. The
    returned values for gamma and gamma_SE will be 0.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        optimizer=None,
        method="MLE",
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Lognormal_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Lognormal_3P",
            LL_func=Fit_Lognormal_3P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Lognormal_3P",
                LL_func=Fit_Lognormal_3P.LL,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Lognormal_2P distribution. Can't proceed with Lognormal_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Lognormal_3P
            lognormal_2P_results = Fit_Lognormal_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.mu = lognormal_2P_results.mu
            self.sigma = lognormal_2P_results.sigma
            self.gamma = 0
            self.mu_SE = lognormal_2P_results.mu_SE
            self.sigma_SE = lognormal_2P_results.sigma_SE
            self.gamma_SE = 0
            self.Cov_mu_sigma = lognormal_2P_results.Cov_mu_sigma
            self.mu_upper = lognormal_2P_results.mu_upper
            self.mu_lower = lognormal_2P_results.mu_lower
            self.sigma_upper = lognormal_2P_results.sigma_upper
            self.sigma_lower = lognormal_2P_results.sigma_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.mu, self.sigma, self.gamma]

        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.mu, self.sigma]
            params_3P = [self.mu, self.sigma, self.gamma]
            # here we need to get mu_SE and sigma_SE from the Lognormal_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for mu_SE and sigma_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Lognormal_3P.LL)(
                    np.array(tuple(params_3P)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (
                    Z * self.mu_SE
                )  # Mu can be positive or negative.
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (
                    np.exp(Z * (self.sigma_SE / self.sigma))
                )  # sigma is strictly positive
                self.sigma_lower = self.sigma * (
                    np.exp(-Z * (self.sigma_SE / self.sigma))
                )
                self.gamma_upper = self.gamma * (
                    np.exp(Z * (self.gamma_SE / self.gamma))
                )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
                self.gamma_lower = self.gamma * (
                    np.exp(-Z * (self.gamma_SE / self.gamma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.gamma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Mu", "Sigma", "Gamma"],
            "Point Estimate": [self.mu, self.sigma, self.gamma],
            "Standard Error": [self.mu_SE, self.sigma_SE, self.gamma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower, self.gamma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Lognormal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            gamma=self.gamma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Lognormal_3P.LL(params_3P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            fig = Lognormal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Lognormal_3P was fitted. The default legend in the probability plot thinks Lognormal_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Lognormal_3P\n(μ="
                        + str(round_to_decimals(self.mu, dec))
                        + ", σ="
                        + str(round_to_decimals(self.sigma, dec))
                        + ", γ="
                        + str(round_to_decimals(self.gamma, dec))
                        + ")"
                    )
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma, gamma):  # Log PDF (3 parameter Lognormal)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t - gamma) - mu) / sigma) ** 2))
            / ((t - gamma) * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, mu, sigma, gamma):  # Log SF (3 parameter Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t - gamma) - mu) / (sigma * 2 ** 0.5)))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Lognormal)
        LL_f = Fit_Lognormal_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Lognormal_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Gamma_2P:
    """
    Fits a two parameter Gamma distribution (alpha,beta) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Gamma_2P alpha parameter
    beta : float
        the fitted Gamma_2P beta parameter
    mu : float
        mu = ln(alpha). Alternate parametrisation (mu, beta) used for the
        confidence intervals.
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    Cov_mu_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gamma_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    This is a two parameter distribution but it has two parametrisations. These
    are alpha,beta and mu,beta. The alpha,beta parametrisation is reported in
    the results table while the mu,beta parametrisation is accessible from the
    results by name. The reason for this is because the most common
    parametrisation (alpha,beta) should be reported while the less common
    parametrisation (mu,beta) is used by some other software so is provided
    for convenience of comparison. The mu = ln(alpha) relationship is simple
    but this relationship does not extend to the variances or covariances so
    additional calculations are required to find both solutions. The mu,beta
    parametrisation is used for the confidence intervals as it is more stable.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        method="MLE",
        optimizer=None,
        quantiles=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Gamma_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            CI_type=CI_type,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_results = LS_optimization(
            func_name="Gamma_2P",
            LL_func=Fit_Gamma_2P.LL_ab,
            failures=failures,
            right_censored=right_censored,
            method="LS",
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.mu = np.log(self.alpha)
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results_ab = MLE_optimization(
                func_name="Gamma_2P",
                LL_func=Fit_Gamma_2P.LL_ab,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results_ab.scale
            self.mu = np.log(MLE_results_ab.scale)
            self.beta = MLE_results_ab.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results_ab.optimizer

        # confidence interval estimates of parameters
        # this needs to be done in terms of alpha beta (ab) parametrisation
        # and mu beta (mb) parametrisation
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_ab = [self.alpha, self.beta]
        hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)(
            np.array(tuple(params_ab)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix_ab = np.linalg.inv(hessian_matrix_ab)
            self.alpha_SE = abs(covariance_matrix_ab[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix_ab[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix_ab[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

            params_mb = [self.mu, self.beta]
            hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)(
                np.array(tuple(params_mb)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix_mb = np.linalg.inv(hessian_matrix_mb)
            self.mu_SE = abs(covariance_matrix_mb[0][0]) ** 0.5
            self.Cov_mu_beta = covariance_matrix_mb[0][1]
            self.mu_upper = self.mu * (np.exp(Z * (self.mu_SE / self.mu)))
            self.mu_lower = self.mu * (np.exp(-Z * (self.mu_SE / self.mu)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Gamma_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.mu_SE = 0
            self.Cov_alpha_beta = 0
            self.Cov_mu_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta
            self.mu_upper = self.mu
            self.mu_lower = self.mu

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gamma_Distribution(
            alpha=self.alpha,
            mu=self.mu,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            mu_SE=self.mu_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            Cov_mu_beta=self.Cov_mu_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Gamma_2P.LL_ab(params_ab, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gamma_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI_type=CI_type,
                CI=CI,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf_ab(t, a, b):  # Log PDF (2 parameter Gamma) - alpha, beta parametrisation
        return anp.log(t ** (b - 1)) - anp.log((a ** b) * agamma(b)) - (t / a)

    @staticmethod
    def logR_ab(t, a, b):  # Log SF (2 parameter Gamma) - alpha, beta parametrisation
        return anp.log(gammaincc(b, t / a))

    @staticmethod
    def LL_ab(params, T_f, T_rc):
        # log likelihood function (2 parameter Gamma) - alpha, beta parametrisation
        LL_f = Fit_Gamma_2P.logf_ab(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gamma_2P.logR_ab(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def logf_mb(t, m, b):  # Log PDF (2 parameter Gamma) - mu, beta parametrisation
        return (
            anp.log(t ** (b - 1))
            - anp.log((anp.exp(m) ** b) * agamma(b))
            - (t / anp.exp(m))
        )

    @staticmethod
    def logR_mb(t, m, b):  # Log SF (2 parameter Gamma) - mu, beta parametrisation
        return anp.log(gammaincc(b, t / anp.exp(m)))

    @staticmethod
    def LL_mb(params, T_f, T_rc):
        # log likelihood function (2 parameter Gamma) - mu, beta parametrisation
        LL_f = Fit_Gamma_2P.logf_mb(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gamma_2P.logR_mb(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Gamma_3P:
    """
    Fits a three parameter Gamma distribution (alpha,beta,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Gamma_3P alpha parameter
    beta : float
        the fitted Gamma_3P beta parameter
    mu : float
        mu = ln(alpha). Alternate parametrisation (mu, beta) used for the
        confidence intervals.
    gamma : float
        the fitted Gamma_3P gamma parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    Cov_mu_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gamma_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Gamma_3P results will
    be discarded and the Gamma_2P distribution will be fitted. The returned
    values for gamma and gamma_SE will be 0.

    This is a three parameter distribution but it has two parametrisations.
    These are alpha,beta,gamma and mu,beta,gamma. The alpha,beta,gamma
    parametrisation is reported in the results table while the mu,beta,gamma
    parametrisation is accessible from the results by name. The reason for this
    is because the most common parametrisation (alpha,beta,gamma) should be
    reported while the less common parametrisation (mu,beta,gamma) is used by
    some other software so is provided for convenience of comparison. The
    mu = ln(alpha) relationship is simple but this relationship does not extend
    to the variances or covariances so additional calculations are required to
    find both solutions. The mu,beta,gamma parametrisation is used for the
    confidence intervals as it is more stable.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        method="MLE",
        quantiles=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Gamma_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            CI_type=CI_type,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Gamma_3P",
            LL_func=Fit_Gamma_3P.LL_abg,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.mu = np.log(self.alpha)
            self.beta = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results_abg = MLE_optimization(
                func_name="Gamma_3P",
                LL_func=Fit_Gamma_3P.LL_abg,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results_abg.scale
            self.mu = np.log(MLE_results_abg.scale)
            self.beta = MLE_results_abg.shape
            self.gamma = MLE_results_abg.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results_abg.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Gamma_2P distribution. Can't proceed with Gamma_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Gamma_3P
            gamma_2P_results = Fit_Gamma_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = gamma_2P_results.alpha
            self.beta = gamma_2P_results.beta
            self.mu = gamma_2P_results.mu
            self.gamma = 0
            self.alpha_SE = gamma_2P_results.alpha_SE
            self.beta_SE = gamma_2P_results.beta_SE
            self.mu_SE = gamma_2P_results.mu_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = gamma_2P_results.Cov_alpha_beta
            self.Cov_mu_beta = gamma_2P_results.Cov_mu_beta
            self.alpha_upper = gamma_2P_results.alpha_upper
            self.alpha_lower = gamma_2P_results.alpha_lower
            self.beta_upper = gamma_2P_results.beta_upper
            self.beta_lower = gamma_2P_results.beta_lower
            self.mu_upper = gamma_2P_results.mu_upper
            self.mu_lower = gamma_2P_results.mu_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P_abg = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P_ab = [self.alpha, self.beta]
            params_2P_mb = [self.mu, self.beta]
            params_3P_abg = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Gamma_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)(
                np.array(tuple(params_2P_ab)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix_ab = np.linalg.inv(hessian_matrix_ab)
                hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)(
                    np.array(tuple(params_2P_mb)),
                    np.array(tuple(failures - self.gamma)),
                    np.array(tuple(right_censored - self.gamma)),
                )
                covariance_matrix_mb = np.linalg.inv(hessian_matrix_mb)

                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Gamma_3P.LL_abg)(
                    np.array(tuple(params_3P_abg)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)

                self.alpha_SE = abs(covariance_matrix_ab[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix_ab[1][1]) ** 0.5
                self.mu_SE = abs(covariance_matrix_mb[0][0]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix_ab[0][1]
                self.Cov_mu_beta = covariance_matrix_mb[0][1]
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.mu_upper = self.mu * (np.exp(Z * (self.mu_SE / self.mu)))
                self.mu_lower = self.mu * (np.exp(-Z * (self.mu_SE / self.mu)))
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
                self.gamma_upper = self.gamma * (
                    np.exp(Z * (self.gamma_SE / self.gamma))
                )
                self.gamma_lower = self.gamma * (
                    np.exp(-Z * (self.gamma_SE / self.gamma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Gamma_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.mu_SE = 0
                self.gamma_SE = 0
                self.Cov_alpha_beta = 0
                self.Cov_mu_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.beta_upper = self.beta
                self.beta_lower = self.beta
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Alpha", "Beta", "Gamma"],
            "Point Estimate": [self.alpha, self.beta, self.gamma],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.gamma_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.gamma_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gamma_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            gamma=self.gamma,
            alpha_SE=self.alpha_SE,
            mu_SE=self.mu_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            Cov_mu_beta=self.Cov_mu_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Gamma_3P.LL_abg(params_3P_abg, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gamma_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            fig = Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Gamma_3P was fitted. The default legend in the probability plot thinks Gamma_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Gamma_3P\n(α="
                        + str(round_to_decimals(self.alpha, dec))
                        + ", β="
                        + str(round_to_decimals(self.beta, dec))
                        + ", γ="
                        + str(round_to_decimals(self.gamma, dec))
                        + ")"
                    )
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf_abg(t, a, b, g):  # Log PDF (3 parameter Gamma) - alpha,beta,gamma
        return (
            anp.log((t - g) ** (b - 1)) - anp.log((a ** b) * agamma(b)) - ((t - g) / a)
        )

    @staticmethod
    def logR_abg(t, a, b, g):  # Log SF (3 parameter Gamma) - alpha,beta,gamma
        return anp.log(gammaincc(b, (t - g) / a))

    @staticmethod
    def LL_abg(params, T_f, T_rc):
        # log likelihood function (3 parameter Gamma) - alpha,beta,gamma
        LL_f = Fit_Gamma_3P.logf_abg(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Gamma_3P.logR_abg(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def logf_mbg(t, m, b, g):  # Log PDF (3 parameter Gamma) - mu,beta,gamma
        return (
            anp.log((t - g) ** (b - 1))
            - anp.log((anp.exp(m) ** b) * agamma(b))
            - ((t - g) / anp.exp(m))
        )

    @staticmethod
    def logR_mbg(t, m, b, g):  # Log SF (3 parameter Gamma) - mu,beta,gamma
        return anp.log(gammaincc(b, (t - g) / anp.exp(m)))

    @staticmethod
    def LL_mbg(params, T_f, T_rc):
        # log likelihood function (3 parameter Gamma) - mu,beta,gamma
        LL_f = Fit_Gamma_3P.logf_mbg(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Gamma_3P.logR_mbg(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Beta_2P:
    """
    Fits a two parameter Beta distribution (alpha,beta) to the data provided.
    All data must be in the range 0 < x < 1.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        point estimates. Default is None which results in no output. To use
        default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set quantiles as either
        'auto', True, 'default', 'on'. If an array or list is specified then it
        will be used instead of the default array. Any array or list specified
        must contain values between 0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Beta_2P alpha parameter
    beta : float
        the fitted Beta_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Beta_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    Confidence intervals on the plots are not provided.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Beta_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Beta_2P",
            LL_func=Fit_Beta_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Beta_2P",
                LL_func=Fit_Beta_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            # for Beta_2P there are actually 2 shape parameters but this just uses the scale and shape nomenclature
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Beta_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Beta_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Beta_Distribution(
            alpha=self.alpha,
            beta=self.beta,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            quantile_data = {
                "Quantile": quantiles,
                "Point Estimate": point_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Point Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Beta_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Beta_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Beta_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Beta_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Beta)
        return anp.log(((t ** (a - 1)) * ((1 - t) ** (b - 1)))) - anp.log(abeta(a, b))

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Beta)
        return anp.log(1 - betainc(a, b, t))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter beta)
        LL_f = Fit_Beta_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Beta_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Loglogistic_2P:
    """
    Fits a two parameter Loglogistic distribution (alpha,beta) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Loglogistic_2P alpha parameter
    beta : float
        the fitted Loglogistic_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Loglogistic_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Loglogistic_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Loglogistic_2P",
            LL_func=Fit_Loglogistic_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Loglogistic_2P",
                LL_func=Fit_Loglogistic_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Loglogistic_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Loglogistic_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer."
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Loglogistic_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Loglogistic_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Loglogistic_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Loglogistic_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Loglogistic_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Loglogistic)
        return (
            anp.log(b / a) - (b + 1) * anp.log(t / a) - 2 * anp.log(1 + (t / a) ** -b)
        )

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Loglogistic)
        return -anp.log((1 + (t / a) ** b))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Loglogistic)
        LL_f = Fit_Loglogistic_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Loglogistic_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Loglogistic_3P:
    """
    Fits a three parameter Loglogistic distribution (alpha,beta,gamma) to the
    data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, None, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use None to turn off the confidence intervals. Must be either 'time',
        'reliability', or None. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Loglogistic_3P alpha parameter
    beta : float
        the fitted Loglogistic_3P beta parameter
    gamma : float
        the fitted Loglogistic_3P gamma parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Loglogistic_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Loglogistic_3P results
    will be discarded and the Loglogistic_2P distribution will be fitted. The
    returned values for gamma and gamma_SE will be 0.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        CI_type="time",
        optimizer=None,
        method="MLE",
        quantiles=None,
        downsample_scatterplot=True,
        **kwargs,
    ):

        inputs = fitters_input_checking(
            dist="Loglogistic_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimization(
            func_name="Loglogistic_3P",
            LL_func=Fit_Lognormal_3P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Loglogistic_3P",
                LL_func=Fit_Loglogistic_3P.LL,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Loglogistic_2P distribution. Can't proceed with Loglogistic_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Loglogistic_3P
            loglogistic_2P_results = Fit_Loglogistic_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = loglogistic_2P_results.alpha
            self.beta = loglogistic_2P_results.beta
            self.gamma = 0
            self.alpha_SE = loglogistic_2P_results.alpha_SE
            self.beta_SE = loglogistic_2P_results.beta_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = loglogistic_2P_results.Cov_alpha_beta
            self.alpha_upper = loglogistic_2P_results.alpha_upper
            self.alpha_lower = loglogistic_2P_results.alpha_lower
            self.beta_upper = loglogistic_2P_results.beta_upper
            self.beta_lower = loglogistic_2P_results.beta_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.alpha, self.beta]
            params_3P = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Loglogistic_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Loglogistic_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Loglogistic_3P.LL)(
                    np.array(tuple(params_3P)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix[0][1]
                self.alpha_upper = self.alpha * (
                    np.exp(Z * (self.alpha_SE / self.alpha))
                )
                self.alpha_lower = self.alpha * (
                    np.exp(-Z * (self.alpha_SE / self.alpha))
                )
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
                self.gamma_upper = self.gamma * (
                    np.exp(Z * (self.gamma_SE / self.gamma))
                )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
                self.gamma_lower = self.gamma * (
                    np.exp(-Z * (self.gamma_SE / self.gamma))
                )
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Loglogistic_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer."
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.gamma_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Alpha", "Beta", "Gamma"],
            "Point Estimate": [self.alpha, self.beta, self.gamma],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.gamma_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.gamma_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Loglogistic_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Loglogistic_3P.LL(params_3P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(
            fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y
        )
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(
            GoF_data, columns=["Goodness of fit", "Value"]
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Loglogistic_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(
                    str(
                        "Table of quantiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Loglogistic_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            fig = Loglogistic_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Loglogistic_3P was fitted. The default legend in the probability plot thinks Loglogistic_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Loglogistic_3P\n(α="
                        + str(round_to_decimals(self.alpha, dec))
                        + ", β="
                        + str(round_to_decimals(self.beta, dec))
                        + ", γ="
                        + str(round_to_decimals(self.gamma, dec))
                        + ")"
                    )
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, g):  # Log PDF (3 parameter Loglogistic)
        return (
            anp.log(b / a)
            - (b + 1) * anp.log((t - g) / a)
            - 2 * anp.log(1 + ((t - g) / a) ** -b)
        )

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Loglogistic)
        return -anp.log((1 + ((t - g) / a) ** b))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Loglogistic)
        LL_f = Fit_Loglogistic_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Loglogistic_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)
