"""
Fitters
This module contains custom fitting functions for parametric distributions which support complete and right censored data.
The included functions are:
Fit_Weibull_2P
Fit_Weibull_3P
Fit_Exponential_1P
Fit_Exponential_2P
Fit_Gamma_2P
Fit_Gamma_3P
Fit_Lognormal_2P
Fit_Lognormal_3P
Fit_Normal_2P
Fit_Gumbel_2P
Fit_Beta_2P
Fit_Loglogistic_2P
Fit_Loglogistic_3P
Fit_Weibull_Mixture
Fit_Weibull_CR

Note that the Beta distribution is only for data in the range 0 < t < 1.
There is also a Fit_Everything function which will fit all distributions (except the Weibull_Mixture and Weibull_CR models) and will provide plots and a table of values.

All functions in this module work using autograd to find the derivative of the log-likelihood function. In this way, the code only needs to specify
the log PDF and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained
using scipy or least squares (depending on the function). If the distribution is an extremely bad fit or is heavily censored (>99%) then these guesses may be poor and the fit might not be successful.
Generally the fit achieved by autograd is highly successful, and whenever it fails the initial guess will be used and a warning will be displayed.
"""

import numpy as np
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
)
from reliability.Nonparametric import KaplanMeier
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import (
    round_to_decimals,
    anderson_darling,
    distribution_confidence_intervals,
    fitters_input_checking,
    colorprint,
    least_squares,
    MLE_optimisation,
    LS_optimisation,
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
    Fit_Everything
    This function will fit all available distributions (excluding mixture and competing risks) for the data you enter, which may include right censored data.

    Inputs:
    failures - an array or list of the failure times (this does not need to be sorted).
    right_censored - an array or list of the right failure times (this does not need to be sorted).
    sort_by - goodness of fit test to sort results by. Must be 'BIC','AICc','AD', or 'Log-likelihood'. Default is BIC.
    method - 'LS' (least squares) or 'MLE' (maximum likelihood estimation). Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    print_results - True/False. Defaults to True. Will show the results of the fitted parameters and the goodness of fit
        tests in a dataframe.
    show_histogram_plot - True/False. Defaults to True. Will show a histogram (scaled to account for censored data) with
        the PDF and CDF of each fitted distribution
    show_PP_plot - True/False. Defaults to True.
        Provides a comparison of parametric vs non-parametric fit using Probability-Probability (PP) plot.
    show_probability_plot - True/False. Defaults to True. Provides a probability plot of each of the fitted distributions.
    show_best_distribution_probability_plot - True/False. Defaults to True. Provides a probability plot in a new figure of the best fitting distribution.
    exclude - list or array of strings specifying which distributions to exclude. Default is None. Options are Weibull_2P, Weibull_3P, Normal_2P,
        Gamma_2P, Loglogistic_2P, Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P, Exponential_2P, Exponential_1P, Beta_2P

    Outputs:
    results - the dataframe of results. Fitted parameters in this dataframe may be accessed by name. See below example.
        In displaying these results, the pandas dataframe is designed to use the common greek letter parametrisations
        rather than the scale, shape, location , threshold parametrisations which can become confusing for some
        distributions.
    best_distribution - a distribution object created based on the parameters of the best fitting distribution
    best_distribution_name - the name of the best fitting distribution. E.g. 'Weibull_3P'
    parameters and goodness of fit results for each fitted distribution. For example, the Weibull_3P distribution values are:
        Weibull_3P_alpha
        Weibull_3P_beta
        Weibull_3P_gamma
        Weibull_3P_BIC
        Weibull_3P_AICc
        Weibull_3P_AD
        Weibull_3P_loglik
    All parametric models have the number of parameters in the name. For example, Weibull_2P used alpha and beta, whereas Weibull_3P
    uses alpha, beta, and gamma. This is applied even for Normal_2P for consistency in naming conventions.
    From the results, the distributions are sorted based on their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the distribution to the data.

    Example Usage:
    X = [0.95892,1.43249,1.04221,0.67583,3.28411,1.03072,0.05826,1.81387,2.06383,0.59762,5.99005,1.92145,1.35179,0.50391]
    output = Fit_Everything(X)
    To extract the parameters of the Weibull distribution from the results dataframe, you may access the parameters by name:
    print('Weibull Alpha =',output.Weibull_2P_alpha,'\nWeibull Beta =',output.Weibull_2P_beta)
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
        self._frac_fail = len(failures) / len(
            self._all_data
        )  # This is used for scaling the histogram when there is censored data
        self._frac_cens = len(right_censored) / len(
            self._all_data
        )  # This is used for reporting the fraction censored in the printed output
        d = sorted(
            self._all_data
        )  # sorting the failure data is necessary for plotting quantiles in order

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
                ["Weibull_3P", "Gamma_3P", "Loglogistic_3P", "Lognormal_3P"]
            )  # do not fit the 3P distributions if there are only 2 failures
        # flexible name checking for excluded distributions
        excluded_distributions = []
        unknown_exclusions = []
        for item in exclude:
            if type(item) not in [str, np.str_]:
                raise ValueError(
                    "exclude must be a list or array of strings that specified the distributions to be excluded from fitting. Available strings are:"
                    "\nWeibull_2P\nWeibull_3P\nNormal_2P\nGamma_2P\nLoglogistic_2P\nGamma_3P\nLognormal_2P\nLognormal_3P\nLoglogistic_3P\nGumbel_2P\nExponential_2P\nExponential_1P\nBeta_2P"
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
                "Available distributions to exclude are: Weibull_2P, Weibull_3P, Normal_2P, Gamma_2P, Loglogistic_2P, Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P, Exponential_2P, Exponential_1P, Beta_2P",
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
                "Mu",
                "Sigma",
                "Lambda",
                "Log-likelihood",
                "AICc",
                "BIC",
                "AD",
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
            self._parametric_CDF_Weibull_3P = self.__Weibull_3P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Weibull_3P",
                    "Alpha": self.Weibull_3P_alpha,
                    "Beta": self.Weibull_3P_beta,
                    "Gamma": self.Weibull_3P_gamma,
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Weibull_3P_loglik,
                    "AICc": self.Weibull_3P_AICc,
                    "BIC": self.Weibull_3P_BIC,
                    "AD": self.Weibull_3P_AD,
                },
                ignore_index=True,
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
            self.Gamma_3P_gamma = self.__Gamma_3P_params.gamma
            self.Gamma_3P_loglik = self.__Gamma_3P_params.loglik
            self.Gamma_3P_BIC = self.__Gamma_3P_params.BIC
            self.Gamma_3P_AICc = self.__Gamma_3P_params.AICc
            self.Gamma_3P_AD = self.__Gamma_3P_params.AD
            self._parametric_CDF_Gamma_3P = self.__Gamma_3P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Gamma_3P",
                    "Alpha": self.Gamma_3P_alpha,
                    "Beta": self.Gamma_3P_beta,
                    "Gamma": self.Gamma_3P_gamma,
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Gamma_3P_loglik,
                    "AICc": self.Gamma_3P_AICc,
                    "BIC": self.Gamma_3P_BIC,
                    "AD": self.Gamma_3P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Exponential_2P = (
                self.__Exponential_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Exponential_2P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": self.Exponential_2P_gamma,
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": self.Exponential_2P_lambda,
                    "Log-likelihood": self.Exponential_2P_loglik,
                    "AICc": self.Exponential_2P_AICc,
                    "BIC": self.Exponential_2P_BIC,
                    "AD": self.Exponential_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Lognormal_3P = (
                self.__Lognormal_3P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Lognormal_3P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": self.Lognormal_3P_gamma,
                    "Mu": self.Lognormal_3P_mu,
                    "Sigma": self.Lognormal_3P_sigma,
                    "Lambda": "",
                    "Log-likelihood": self.Lognormal_3P_loglik,
                    "AICc": self.Lognormal_3P_AICc,
                    "BIC": self.Lognormal_3P_BIC,
                    "AD": self.Lognormal_3P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Normal_2P = self.__Normal_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Normal_2P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": "",
                    "Mu": self.Normal_2P_mu,
                    "Sigma": self.Normal_2P_sigma,
                    "Lambda": "",
                    "Log-likelihood": self.Normal_2P_loglik,
                    "AICc": self.Normal_2P_AICc,
                    "BIC": self.Normal_2P_BIC,
                    "AD": self.Normal_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Lognormal_2P = (
                self.__Lognormal_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Lognormal_2P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": "",
                    "Mu": self.Lognormal_2P_mu,
                    "Sigma": self.Lognormal_2P_sigma,
                    "Lambda": "",
                    "Log-likelihood": self.Lognormal_2P_loglik,
                    "AICc": self.Lognormal_2P_AICc,
                    "BIC": self.Lognormal_2P_BIC,
                    "AD": self.Lognormal_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Gumbel_2P = self.__Gumbel_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Gumbel_2P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": "",
                    "Mu": self.Gumbel_2P_mu,
                    "Sigma": self.Gumbel_2P_sigma,
                    "Lambda": "",
                    "Log-likelihood": self.Gumbel_2P_loglik,
                    "AICc": self.Gumbel_2P_AICc,
                    "BIC": self.Gumbel_2P_BIC,
                    "AD": self.Gumbel_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Weibull_2P = self.__Weibull_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Weibull_2P",
                    "Alpha": self.Weibull_2P_alpha,
                    "Beta": self.Weibull_2P_beta,
                    "Gamma": "",
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Weibull_2P_loglik,
                    "AICc": self.Weibull_2P_AICc,
                    "BIC": self.Weibull_2P_BIC,
                    "AD": self.Weibull_2P_AD,
                },
                ignore_index=True,
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
            self.Gamma_2P_gamma = 0
            self.Gamma_2P_loglik = self.__Gamma_2P_params.loglik
            self.Gamma_2P_BIC = self.__Gamma_2P_params.BIC
            self.Gamma_2P_AICc = self.__Gamma_2P_params.AICc
            self.Gamma_2P_AD = self.__Gamma_2P_params.AD
            self._parametric_CDF_Gamma_2P = self.__Gamma_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Gamma_2P",
                    "Alpha": self.Gamma_2P_alpha,
                    "Beta": self.Gamma_2P_beta,
                    "Gamma": "",
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Gamma_2P_loglik,
                    "AICc": self.Gamma_2P_AICc,
                    "BIC": self.Gamma_2P_BIC,
                    "AD": self.Gamma_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Exponential_1P = (
                self.__Exponential_1P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Exponential_1P",
                    "Alpha": "",
                    "Beta": "",
                    "Gamma": "",
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": self.Exponential_1P_lambda,
                    "Log-likelihood": self.Exponential_1P_loglik,
                    "AICc": self.Exponential_1P_AICc,
                    "BIC": self.Exponential_1P_BIC,
                    "AD": self.Exponential_1P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Loglogistic_2P = (
                self.__Loglogistic_2P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Loglogistic_2P",
                    "Alpha": self.Loglogistic_2P_alpha,
                    "Beta": self.Loglogistic_2P_beta,
                    "Gamma": "",
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Loglogistic_2P_loglik,
                    "AICc": self.Loglogistic_2P_AICc,
                    "BIC": self.Loglogistic_2P_BIC,
                    "AD": self.Loglogistic_2P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Loglogistic_3P = (
                self.__Loglogistic_3P_params.distribution.CDF(xvals=d, show_plot=False)
            )
            df = df.append(
                {
                    "Distribution": "Loglogistic_3P",
                    "Alpha": self.Loglogistic_3P_alpha,
                    "Beta": self.Loglogistic_3P_beta,
                    "Gamma": self.Loglogistic_3P_gamma,
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Loglogistic_3P_loglik,
                    "AICc": self.Loglogistic_3P_AICc,
                    "BIC": self.Loglogistic_3P_BIC,
                    "AD": self.Loglogistic_3P_AD,
                },
                ignore_index=True,
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
            self._parametric_CDF_Beta_2P = self.__Beta_2P_params.distribution.CDF(
                xvals=d, show_plot=False
            )
            df = df.append(
                {
                    "Distribution": "Beta_2P",
                    "Alpha": self.Beta_2P_alpha,
                    "Beta": self.Beta_2P_beta,
                    "Gamma": "",
                    "Mu": "",
                    "Sigma": "",
                    "Lambda": "",
                    "Log-likelihood": self.Beta_2P_loglik,
                    "AICc": self.Beta_2P_AICc,
                    "BIC": self.Beta_2P_BIC,
                    "AD": self.Beta_2P_AD,
                },
                ignore_index=True,
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
            df2 = df.reindex(df.BIC.sort_values().index)
        elif sort_by.upper() in ["AICC", "AIC"]:
            df2 = df.reindex(df.AICc.sort_values().index)
        elif sort_by.upper() == "AD":
            df2 = df.reindex(df.AD.sort_values().index)
        elif sort_by.upper() in [
            "LOGLIK",
            "LOG LIK",
            "LOG-LIKELIHOOD",
            "LL",
            "LOGLIKELIHOOD",
            "LOG LIKELIHOOD",
        ]:
            df2 = df.reindex(abs(df["Log-likelihood"]).sort_values().index)
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
            frac_cens = self._frac_cens * 100
            if frac_cens % 1 < 1e-10:
                frac_cens = int(frac_cens)
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
            # plotting occurs by default
            Fit_Everything.histogram_plot(self)

        if show_PP_plot is True:
            # plotting occurs by default
            Fit_Everything.P_P_plot(self)

        if show_probability_plot is True:
            # plotting occurs by default
            Fit_Everything.probability_plot(self)

        if show_best_distribution_probability_plot is True:
            # plotting occurs by default
            Fit_Everything.probability_plot(self, best_only=True)

        if (
            show_histogram_plot is True
            or show_PP_plot is True
            or show_probability_plot is True
            or show_best_distribution_probability_plot is True
        ):
            plt.show()

    def probplot_layout(self):
        items = len(self.results.index.values)  # number of items that were fitted
        if items == 13:  # ------------------------ w   , h    w , h
            cols, rows, figsize, figsizePP = 5, 3, (17.5, 8), (10, 7.5)
        elif items in [10, 11, 12]:
            cols, rows, figsize, figsizePP = 4, 3, (15, 8), (8.5, 7.5)
        elif items in [7, 8, 9]:
            cols, rows, figsize, figsizePP = 3, 3, (12.5, 8), (7, 7.5)
        elif items in [5, 6]:
            cols, rows, figsize, figsizePP = 3, 2, (12.5, 6), (6.5, 5.5)
        elif items == 4:
            cols, rows, figsize, figsizePP = 2, 2, (10, 6), (6, 5.5)
        elif items == 3:
            cols, rows, figsize, figsizePP = 3, 1, (12.5, 5), (10, 4)
        elif items == 2:
            cols, rows, figsize, figsizePP = 2, 1, (10, 4), (6, 4)
        elif items == 1:
            cols, rows, figsize, figsizePP = 1, 1, (7.5, 4), (6, 4)
        return cols, rows, figsize, figsizePP

    def histogram_plot(self):
        X = self.failures
        # define plotting limits
        delta = max(X) - min(X)
        xmin = 0
        if max(X) <= 1:
            xmax = 1  # this is the case when beta is fitted
        else:
            xmax = (
                max(X) + delta
            )  # this is when beta is not fitted so the upper xlim goes a bit more

        plt.figure(figsize=(14, 6))
        # we need to make the histogram manually (can't use plt.hist) due to need to scale the heights when there's censored data
        plotting_order = self.results[
            "Distribution"
        ].values  # this is the order to plot things so that the legend matches the results dataframe
        iqr = np.subtract(*np.percentile(X, [75, 25]))  # interquartile range
        bin_width = (
            2 * iqr * len(X) ** -(1 / 3)
        )  # Freedman–Diaconis rule ==> https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        num_bins = int(np.ceil((max(X) - min(X)) / bin_width))
        hist, bins = np.histogram(X, bins=num_bins, density=True)
        hist_cumulative = np.cumsum(hist) / sum(hist)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        # Probability Density Functions
        plt.subplot(121)
        plt.bar(
            center,
            hist * self._frac_fail,
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )
        for item in plotting_order:
            if item == "Weibull_2P":
                Weibull_Distribution(
                    alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta
                ).PDF(label=r"Weibull ($\alpha , \beta$)")
            elif item == "Weibull_3P":
                Weibull_Distribution(
                    alpha=self.Weibull_3P_alpha,
                    beta=self.Weibull_3P_beta,
                    gamma=self.Weibull_3P_gamma,
                ).PDF(label=r"Weibull ($\alpha , \beta , \gamma$)")
            elif item == "Gamma_2P":
                Gamma_Distribution(
                    alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta
                ).PDF(label=r"Gamma ($\alpha , \beta$)")
            elif item == "Gamma_3P":
                Gamma_Distribution(
                    alpha=self.Gamma_3P_alpha,
                    beta=self.Gamma_3P_beta,
                    gamma=self.Gamma_3P_gamma,
                ).PDF(label=r"Gamma ($\alpha , \beta , \gamma$)")
            elif item == "Exponential_1P":
                Exponential_Distribution(Lambda=self.Exponential_1P_lambda).PDF(
                    label=r"Exponential ($\lambda$)"
                )
            elif item == "Exponential_2P":
                Exponential_Distribution(
                    Lambda=self.Exponential_2P_lambda, gamma=self.Exponential_2P_gamma
                ).PDF(label=r"Exponential ($\lambda , \gamma$)")
            elif item == "Lognormal_2P":
                Lognormal_Distribution(
                    mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma
                ).PDF(label=r"Lognormal ($\mu , \sigma$)")
            elif item == "Lognormal_3P":
                Lognormal_Distribution(
                    mu=self.Lognormal_3P_mu,
                    sigma=self.Lognormal_3P_sigma,
                    gamma=self.Lognormal_3P_gamma,
                ).PDF(label=r"Lognormal ($\mu , \sigma , \gamma$)")
            elif item == "Normal_2P":
                Normal_Distribution(
                    mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma
                ).PDF(label=r"Normal ($\mu , \sigma$)")
            elif item == "Gumbel_2P":
                Gumbel_Distribution(
                    mu=self.Gumbel_2P_mu, sigma=self.Gumbel_2P_sigma
                ).PDF(label=r"Gumbel ($\mu , \sigma$)")
            elif item == "Loglogistic_2P":
                Loglogistic_Distribution(
                    alpha=self.Loglogistic_2P_alpha, beta=self.Loglogistic_2P_beta
                ).PDF(label=r"Loglogistic ($\alpha , \beta$)")
            elif item == "Loglogistic_3P":
                Loglogistic_Distribution(
                    alpha=self.Loglogistic_3P_alpha,
                    beta=self.Loglogistic_3P_beta,
                    gamma=self.Loglogistic_3P_gamma,
                ).PDF(label=r"Loglogistic ($\alpha , \beta, \gamma$)")
            elif item == "Beta_2P":
                Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).PDF(
                    label=r"Beta ($\alpha , \beta$)"
                )
        plt.xlim(xmin, xmax)
        plt.ylim(0, max(hist) * 1.5)
        plt.title("Probability Density Function")
        plt.xlabel("Data")
        plt.ylabel("Probability density")
        plt.legend()

        # Cumulative Distribution Functions
        plt.subplot(122)
        plt.bar(
            center,
            hist_cumulative * self._frac_fail,
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )
        for item in plotting_order:
            if item == "Weibull_2P":
                Weibull_Distribution(
                    alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta
                ).CDF(label=r"Weibull ($\alpha , \beta$)")
            elif item == "Weibull_3P":
                Weibull_Distribution(
                    alpha=self.Weibull_3P_alpha,
                    beta=self.Weibull_3P_beta,
                    gamma=self.Weibull_3P_gamma,
                ).CDF(label=r"Weibull ($\alpha , \beta , \gamma$)")
            elif item == "Gamma_2P":
                Gamma_Distribution(
                    alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta
                ).CDF(label=r"Gamma ($\alpha , \beta$)")
            elif item == "Gamma_3P":
                Gamma_Distribution(
                    alpha=self.Gamma_3P_alpha,
                    beta=self.Gamma_3P_beta,
                    gamma=self.Gamma_3P_gamma,
                ).CDF(label=r"Gamma ($\alpha , \beta , \gamma$)")
            elif item == "Exponential_1P":
                Exponential_Distribution(Lambda=self.Exponential_1P_lambda).CDF(
                    label=r"Exponential ($\lambda$)"
                )
            elif item == "Exponential_2P":
                Exponential_Distribution(
                    Lambda=self.Exponential_2P_lambda, gamma=self.Exponential_2P_gamma
                ).CDF(label=r"Exponential ($\lambda , \gamma$)")
            elif item == "Lognormal_2P":
                Lognormal_Distribution(
                    mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma
                ).CDF(label=r"Lognormal ($\mu , \sigma$)")
            elif item == "Lognormal_3P":
                Lognormal_Distribution(
                    mu=self.Lognormal_3P_mu,
                    sigma=self.Lognormal_3P_sigma,
                    gamma=self.Lognormal_3P_gamma,
                ).CDF(label=r"Lognormal ($\mu , \sigma , \gamma$)")
            elif item == "Normal_2P":
                Normal_Distribution(
                    mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma
                ).CDF(label=r"Normal ($\mu , \sigma$)")
            elif item == "Gumbel_2P":
                Gumbel_Distribution(
                    mu=self.Gumbel_2P_mu, sigma=self.Gumbel_2P_sigma
                ).CDF(label=r"Gumbel ($\mu , \sigma$)")
            elif item == "Loglogistic_2P":
                Loglogistic_Distribution(
                    alpha=self.Loglogistic_2P_alpha, beta=self.Loglogistic_2P_beta
                ).CDF(label=r"Loglogistic ($\alpha , \beta$)")
            elif item == "Loglogistic_3P":
                Loglogistic_Distribution(
                    alpha=self.Loglogistic_3P_alpha,
                    beta=self.Loglogistic_3P_beta,
                    gamma=self.Loglogistic_3P_gamma,
                ).CDF(label=r"Loglogistic ($\alpha , \beta, \gamma$)")
            elif item == "Beta_2P":
                Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).CDF(
                    label=r"Beta ($\alpha , \beta$)"
                )
        plt.xlim([xmin, xmax])
        plt.title("Cumulative Distribution Function")
        plt.xlabel("Data")
        plt.ylabel("Cumulative probability density")
        plt.legend()
        plt.suptitle("Histogram plot of each fitted distribution")
        plt.subplots_adjust(left=0.07, bottom=0.10, right=0.97, top=0.88, wspace=0.15)

    def P_P_plot(self):  # probability-probability plot of parametric vs non-parametric
        # Kaplan-Meier estimate of quantiles. Used in P-P plot.

        nonparametric = KaplanMeier(
            failures=self.failures,
            right_censored=self.right_censored,
            print_results=False,
            show_plot=False,
        )
        nonparametric_CDF = 1 - np.array(nonparametric.KM)  # change SF into CDF

        cols, rows, _, figsizePP = Fit_Everything.probplot_layout(self)
        plotting_order = self.results[
            "Distribution"
        ].values  # this is the order to plot things which matches the results dataframe
        plt.figure(figsize=figsizePP)
        plt.suptitle(
            "Semi-parametric Probability-Probability plots of each fitted distribution\nParametric (x-axis) vs Non-Parametric (y-axis)\n"
        )
        subplot_counter = 1
        for item in plotting_order:
            plt.subplot(rows, cols, subplot_counter)
            if item == "Exponential_1P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Exponential_1P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Exponential_1P,
                    marker=".",
                    color="k",
                )
            elif item == "Exponential_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Exponential_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Exponential_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Lognormal_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Lognormal_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Lognormal_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Lognormal_3P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Lognormal_3P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Lognormal_3P,
                    marker=".",
                    color="k",
                )
            elif item == "Weibull_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Weibull_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Weibull_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Weibull_3P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Weibull_3P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Weibull_3P,
                    marker=".",
                    color="k",
                )
            elif item == "Loglogistic_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Loglogistic_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Loglogistic_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Loglogistic_3P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Loglogistic_3P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Loglogistic_3P,
                    marker=".",
                    color="k",
                )
            elif item == "Gamma_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Gamma_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Gamma_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Gamma_3P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Gamma_3P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Gamma_3P,
                    marker=".",
                    color="k",
                )
            elif item == "Normal_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Normal_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Normal_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Gumbel_2P":
                xlim = max(
                    np.hstack([nonparametric_CDF, self._parametric_CDF_Gumbel_2P])
                )
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Gumbel_2P,
                    marker=".",
                    color="k",
                )
            elif item == "Beta_2P":
                xlim = max(np.hstack([nonparametric_CDF, self._parametric_CDF_Beta_2P]))
                plt.scatter(
                    nonparametric_CDF,
                    self._parametric_CDF_Beta_2P,
                    marker=".",
                    color="k",
                )
            else:
                raise ValueError("unknown item was fitted")
            plt.title(item)
            plt.plot(
                [-xlim, 2 * xlim], [-xlim, 2 * xlim], "r", alpha=0.7
            )  # red diagonal line
            plt.axis("square")
            plt.yticks([])
            plt.xticks([])
            plt.xlim(-xlim * 0.05, xlim * 1.05)
            plt.ylim(-xlim * 0.05, xlim * 1.05)
            subplot_counter += 1
        plt.tight_layout()

    def probability_plot(self, best_only=False):
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
            cols, rows, figsize, _ = Fit_Everything.probplot_layout(self)
            # this is the order to plot to match the results dataframe
            plotting_order = self.results["Distribution"].values
            plt.suptitle("Probability plots of each fitted distribution\n\n")
            subplot_counter = 1
        else:
            plotting_order = [self.results["Distribution"].values[0]]

        for item in plotting_order:
            if best_only is False:
                plt.subplot(rows, cols, subplot_counter)
            if item == "Exponential_1P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Exponential_1P_params,
                )
            elif item == "Exponential_2P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Exponential_2P_params,
                )
            elif item == "Lognormal_2P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Lognormal_2P_params,
                )
            elif item == "Lognormal_3P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Lognormal_3P_params,
                )
            elif item == "Weibull_2P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Weibull_2P_params,
                )
            elif item == "Weibull_3P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Weibull_3P_params,
                )
            elif item == "Loglogistic_2P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Loglogistic_2P_params,
                )
            elif item == "Loglogistic_3P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Loglogistic_3P_params,
                )
            elif item == "Gamma_2P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gamma_2P_params,
                )
            elif item == "Gamma_3P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gamma_3P_params,
                )
            elif item == "Normal_2P":
                Normal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Normal_2P_params,
                )
            elif item == "Gumbel_2P":
                Gumbel_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Gumbel_2P_params,
                )
            elif item == "Beta_2P":
                Beta_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    __fitted_dist_params=self.__Beta_2P_params,
                )
            else:
                raise ValueError("unknown item was fitted")
            if best_only is False:
                plt.title(item)
                ax = plt.gca()
                ax.set_yticklabels([], minor=False)
                ax.set_xticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.set_xticklabels([], minor=True)
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.get_legend().remove()
                subplot_counter += 1
            else:
                plt.title(
                    str(
                        "Probability plot of best distribution\n"
                        + self.best_distribution.param_title_long
                    )
                )
        if best_only is False:
            plt.tight_layout()
            plt.gcf().set_size_inches(figsize)


class Fit_Weibull_2P:
    """
    Fit_Weibull_2P

    Fits a 2-parameter Weibull distribution (alpha,beta) to the data provided.

    inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type - time, reliability, None. Default is time. This is the confidence bounds on time or on reliability. Use None to turn off the confidence intervals.
    force_beta - Use this to specify the beta value if you need to force beta to be a certain value. Used in ALT probability plotting. Optional input.
    percentiles - percentiles to produce a table of percentiles failed with lower, point, and upper estimates. Default is None which results in no output. True or 'auto' will use default array [1, 5, 10,..., 95, 99]. If an array or list is specified then it will be used instead of the default array.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    outputs:
    alpha - the fitted Weibull_2P alpha parameter
    beta - the fitted Weibull_2P beta parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Weibull_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        force_beta=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Weibull_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P.LL_fb)(
                np.array(tuple([self.alpha])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_beta])),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.weibull_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Weibull_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(
        params, T_f, T_rc, force_beta
    ):  # log likelihood function (2 parameter weibull) FORCED BETA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], force_beta).sum()  # failure times
        LL_rc += Fit_Weibull_2P.logR(
            T_rc, params[0], force_beta
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_2P_grouped:
    """
    Fit_Weibull_2P_grouped

    Fits a 2-parameter Weibull distribution (alpha,beta) to the data provided.
    The data input is a dataframe which allows for efficient handling of large volumes of grouped data.
    This is almost identical to Fit_Weibull_2P except for the input format.
    There may be small differences in results due to the initial guess (scipy or least squares), the optimizer (L-BFGS-B or TNC), and rounding errors when processing very heavily censored data.

    Inputs:
    dataframe - a pandas dataframe of the appropriate format. See the example below.
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    force_beta - Use this to specify the beta value if you need to force beta to be a certain value. Used in ALT probability plotting. Optional input.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    percentiles - percentiles to produce a table of percentiles failed with lower, point, and upper estimates. Default is None which results in no output. True or 'auto' will use default array [1, 5, 10,..., 95, 99]. If an array or list is specified then it will be used instead of the default array.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Weibull_2P alpha parameter
    beta - the fitted Weibull_2P beta parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Weibull_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.

    Requirements of the input dataframe:
    The column titles MUST be 'category', 'time', 'quantity'
    The category values MUST be 'F' for failure or 'C' for censored (right censored).
    The time values are the failure or right censored times.
    The quantity is the number of items at that time. This must be specified for all values even if the quantity is 1.

    Example of the input dataframe:
    category  time  quantity
           F    24         1
           F    29         1
           F    34         1
           F    39         2
           F    40         1
           F    42         3
           F    44         1
           C    50         3
           C    55         5
           C    60        10

    This is easiest to achieve by importing data from excel. An example of this is:
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
        percentiles=None,
        method="MLE",
        optimizer=None,
        CI_type="time",
        **kwargs
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
            percentiles=percentiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0

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
        elif method == "MLE":
            self.method = "Maximum Likelihood Estimation (MLE)"
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

            if result.success is True:
                params = result.x
                if force_beta is None:
                    self.alpha = params[0]
                    self.beta = params[1]
                else:
                    self.alpha = params[0]
                    self.beta = force_beta
            else:  # if the L-BFGS-B or TNC optimizer fails then we have a second attempt using the slower but slightly more reliable nelder-mead optimizer
                if force_beta is None:
                    guess = initial_guess
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
                        tol=1e-4,
                        method="nelder-mead",
                    )
                else:
                    guess = initial_guess[0]
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
                        tol=1e-4,
                        method="nelder-mead",
                    )
                if result.success is True:
                    params = result.x
                    if force_beta is None:
                        self.alpha = params[0]
                        self.beta = params[1]
                    else:
                        self.alpha = params[0]
                        self.beta = force_beta
                else:
                    colorprint(
                        str(
                            "WARNING: MLE estimates failed for Weibull_2P_grouped. The least squares estimates have been returned. These results may not be as accurate as MLE."
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
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P_grouped.LL_fb)(
                np.array(tuple([self.alpha])),
                np.array(tuple(failure_times)),
                np.array(tuple(right_censored_times)),
                np.array(tuple(failure_qty)),
                np.array(tuple(right_censored_qty)),
                np.array(tuple([force_beta])),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.weibull_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(sum(failure_qty)) + "/" + str(sum(right_censored_qty))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(
        params, T_f, T_rc, Q_f, Q_rc
    ):  # log likelihood function (2 parameter weibull) ==> T is for time, Q is for quantity
        LL_f = 0
        LL_rc = 0
        LL_f += (
            Fit_Weibull_2P_grouped.logf(T_f, params[0], params[1]) * Q_f
        ).sum()  # failure times
        LL_rc += (
            Fit_Weibull_2P_grouped.logR(T_rc, params[0], params[1]) * Q_rc
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(
        params, T_f, T_rc, Q_f, Q_rc, force_beta
    ):  # log likelihood function (2 parameter weibull) FORCED BETA  ==> T is for time, Q is for quantity
        LL_f = 0
        LL_rc = 0
        LL_f += (
            Fit_Weibull_2P_grouped.logf(T_f, params[0], force_beta) * Q_f
        ).sum()  # failure times
        LL_rc += (
            Fit_Weibull_2P_grouped.logR(T_rc, params[0], force_beta) * Q_rc
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_3P:
    """
    Fit_Weibull_3P
    Fits a 3-parameter Weibull distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type - 'time' or 'reliability'. Default is time. Used for the probability plot and the distribution object in the output.
    method - 'MLE' (maximum likelihood estimation), or 'LS' (least squares estimation). LS will perform non-linear least squares estimation. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    percentiles - percentiles to produce a table of percentiles failed with lower, point, and upper estimates. Default is None which results in no output. True or 'auto' will use default array [1, 5, 10,..., 95, 99]. If an array or list is specified then it will be used instead of the default array.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Weibull_3P alpha parameter
    beta - the fitted Weibull_3P beta parameter
    gamma - the fitted Weibull_3P gamma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Weibull_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    gamma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters alpha and beta
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    gamma_upper - the upper CI estimate of the parameter
    gamma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        CI_type="time",
        optimizer=None,
        method="MLE",
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Weibull_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
            self.gamma_upper = self.gamma * (
                np.exp(Z * (self.gamma_SE / self.gamma))
            )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
            self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.weibull_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )
            self.percentiles = percentiles.set_index("Percentile")

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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
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

    @staticmethod
    def logf(t, a, b, g):  # Log PDF (3 parameter Weibull)
        return (b - 1) * anp.log((t - g) / a) + anp.log(b / a) - ((t - g) / a) ** b

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Weibull)
        return -(((t - g) / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_3P.logf(
            T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_3P.logR(
            T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Mixture:
    """
    Fit_Weibull_Mixture
    Fits a mixture of 2 x Weibull_2P distributions (this does not fit the gamma parameter).
    Right censoring is supported, though care should be taken to ensure that there still appears to be two groups when plotting only the failure data.
    A second group cannot be made from a mostly or totally censored set of samples.
    Use this model when you think there are multiple failure modes acting to create the failure data.

    This is different to the Weibull Competing Risks as the overall Survival Function is the sum of the individual Survival Functions multiplied by a proportion
    rather than being the product as is the case in the Weibull Competing Risks Model.
    Mixture ==> SF_model = (proportion_1 x SF_1) + ((1-proportion_1) x SF_2)
    Competing Risks ==> SF_model = SF_1 x SF_2

    Similar to the competing risks model, you can use this model when you think there are multiple failure modes acting to create the failure data.

    Whilst some failure modes may not be fitted as well by a Weibull distribution as they may be by another distribution, it
    is unlikely that a mixture of data from two distributions (particularly if they are overlapping) will be fitted
    noticeably better by other types of mixtures than would be achieved by a Weibull mixture. For this reason, other types
    of mixtures are not implemented.

    Inputs:
    failures - an array or list of the failure data. There must be at least 4 failures, but it is highly recommended to use another model if you have
        less than 20 failures.
    right_censored - an array or list of right censored data
    print_results - True/False. This will print results to console. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    show_probability_plot - True/False. This will show the probability plot with the fitted mixture CDF. Default is True.

    Outputs:
    alpha_1 - the fitted Weibull_2P alpha parameter for the first (left) group
    beta_1 - the fitted Weibull_2P beta parameter for the first (left) group
    alpha_2 - the fitted Weibull_2P alpha parameter for the second (right) group
    beta_2 - the fitted Weibull_2P beta parameter for the second (right) group
    proportion_1 - the fitted proportion of the first (left) group
    proportion_2 - the fitted proportion of the second (right) group. Same as 1-proportion_1
    alpha_1_SE - the standard error on the parameter
    beta_1_SE - the standard error on the parameter
    alpha_2_SE - the standard error on the parameter
    beta_2_SE - the standard error on the parameter
    proportion_1_SE - the standard error on the parameter
    alpha_1_upper - the upper confidence interval estimate of the parameter
    alpha_1_lower - the lower confidence interval estimate of the parameter
    beta_1_upper - the upper confidence interval estimate of the parameter
    beta_1_lower - the lower confidence interval estimate of the parameter
    alpha_2_upper - the upper confidence interval estimate of the parameter
    alpha_2_lower - the lower confidence interval estimate of the parameter
    beta_2_upper - the upper confidence interval estimate of the parameter
    beta_2_lower - the lower confidence interval estimate of the parameter
    proportion_1_upper - the upper confidence interval estimate of the parameter
    proportion_1_lower - the lower confidence interval estimate of the parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
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

        # find the division line. This is to assign data to each group
        h = np.histogram(failures, bins=50, density=True)
        hist_counts = h[0]
        hist_bins = h[1]
        midbins = []
        for i in range(len(hist_bins)):
            if i > 0 and i < len(hist_bins):
                midbins.append((hist_bins[i] + hist_bins[i - 1]) / 2)
        peaks_x = []
        peaks_y = []
        batch_width = 8
        for i, x in enumerate(hist_counts):
            if i < batch_width:
                batch = hist_counts[0 : i + batch_width]
            elif i > batch_width and i > len(hist_counts - batch_width):
                batch = hist_counts[i - batch_width : len(hist_counts)]
            else:
                batch = hist_counts[
                    i - batch_width : i + batch_width
                ]  # the histogram counts are batched (actual batch size = 2 x batch_width)
            if (
                max(batch) == x
            ):  # if the current point is higher than the rest of the batch then it is counted as a peak
                peaks_x.append(midbins[i])
                peaks_y.append(x)
        if (
            len(peaks_x) > 2
        ):  # if there are more than 2 peaks, the mean is moved based on the height of the peaks. Higher peaks will attract the mean towards them more than smaller peaks.
            yfracs = np.array(peaks_y) / sum(peaks_y)
            division_line = sum(peaks_x * yfracs)
        else:
            division_line = np.average(peaks_x)
        self.division_line = division_line
        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < division_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < division_line:
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
        p_guess = (
            len(GROUP_1_failures) + len(GROUP_1_right_cens)
        ) / n  # proportion guess
        guess = [
            group_1_estimates.alpha,
            group_1_estimates.beta,
            group_2_estimates.alpha,
            group_2_estimates.beta,
            p_guess,
        ]  # A1,B1,A2,B2,P

        # solve it
        MLE_results = MLE_optimisation(
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
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.proportion_1_SE = abs(covariance_matrix[4][4]) ** 0.5
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
        )  # ref: http://reliawiki.org/index.php/The_Mixed_Weibull_Distribution
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
            print("Analysis method: MLE")
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
                failures=failures, right_censored=rc, show_fitted_distribution=False
            )
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
            self.distribution.CDF(
                xvals=xvals, label=label_str
            )  # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Mixture CDF")

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
    def LL(params, T_f, T_rc):  # Log Mixture Likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Mixture.logf(
            T_f, params[0], params[1], params[2], params[3], params[4]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Mixture.logR(
            T_rc, params[0], params[1], params[2], params[3], params[4]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_CR:
    """
    Fit_Weibull_CR
    Fits a Weibull Competing Risks Model consisting of 2 x Weibull_2P distributions (this does not fit the gamma parameter).
    This is different to the Weibull Mixture model as the overall Survival Function is the product of the individual Survival Functions rather than
    being the sum as is the case in the Weibull Mixture Model.
    Competing Risks ==> SF_model = SF_1 x SF_2
    Mixture ==> SF_model = (proportion_1 x SF_1) + ((1-proportion_1) x SF_2)

    Similar to the mixture model, you can use this model when you think there are multiple failure modes acting to create the failure data.

    Whilst some failure modes may not be fitted as well by a Weibull distribution as they may be by another distribution, it is unlikely that data
    from a competing risks model will be fitted noticeably better by other types of competing risks models than would be achieved by a Weibull
    Competing Risks model. For this reason, other types of competing risks models are not implemented.

    Inputs:
    failures - an array or list of the failure data. There must be at least 4 failures, but it is highly recommended to use another model if you have
        less than 20 failures.
    right_censored - an array or list of right censored data
    print_results - True/False. This will print results to console. Default is True.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    show_probability_plot - True/False. This will show the probability plot with the fitted Weibull_CR CDF. Default is True.

    Outputs:
    alpha_1 - the fitted Weibull_2P alpha parameter for the first distribution
    beta_1 - the fitted Weibull_2P beta parameter for the first distribution
    alpha_2 - the fitted Weibull_2P alpha parameter for the second distribution
    beta_2 - the fitted Weibull_2P beta parameter for the second distribution
    alpha_1_SE - the standard error on the parameter
    beta_1_SE - the standard error on the parameter
    alpha_2_SE - the standard error on the parameter
    beta_2_SE - the standard error on the parameter
    alpha_1_upper - the upper confidence interval estimate of the parameter
    alpha_1_lower - the lower confidence interval estimate of the parameter
    beta_1_upper - the upper confidence interval estimate of the parameter
    beta_1_lower - the lower confidence interval estimate of the parameter
    alpha_2_upper - the upper confidence interval estimate of the parameter
    alpha_2_lower - the lower confidence interval estimate of the parameter
    beta_2_upper - the upper confidence interval estimate of the parameter
    beta_2_lower - the lower confidence interval estimate of the parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
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
        # find the division line. This is to assign data to each group
        h = np.histogram(failures, bins=50, density=True)
        hist_counts = h[0]
        hist_bins = h[1]
        midbins = []
        for i in range(len(hist_bins)):
            if i > 0 and i < len(hist_bins):
                midbins.append((hist_bins[i] + hist_bins[i - 1]) / 2)
        peaks_x = []
        peaks_y = []
        batch_width = 8
        for i, x in enumerate(hist_counts):
            if i < batch_width:
                batch = hist_counts[0 : i + batch_width]
            elif i > batch_width and i > len(hist_counts - batch_width):
                batch = hist_counts[i - batch_width : len(hist_counts)]
            else:
                batch = hist_counts[
                    i - batch_width : i + batch_width
                ]  # the histogram counts are batched (actual batch size = 2 x batch_width)
            if (
                max(batch) == x
            ):  # if the current point is higher than the rest of the batch then it is counted as a peak
                peaks_x.append(midbins[i])
                peaks_y.append(x)
        if (
            len(peaks_x) > 2
        ):  # if there are more than 2 peaks, the mean is moved based on the height of the peaks. Higher peaks will attract the mean towards them more than smaller peaks.
            yfracs = np.array(peaks_y) / sum(peaks_y)
            division_line = sum(peaks_x * yfracs)
        else:
            division_line = np.average(peaks_x)
        self.division_line = division_line
        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < division_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < division_line:
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
        MLE_results = MLE_optimisation(
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
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
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
            print("Analysis method: MLE")
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
                failures=failures, right_censored=rc, show_fitted_distribution=False
            )
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
            self.distribution.CDF(xvals=xvals, label=label_str)
            plt.title("Probability Plot\nWeibull Competing Risks CDF")

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
    def LL(params, T_f, T_rc):  # Log Likelihood function (Competing Risks)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_CR.logf(
            T_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_CR.logR(
            T_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Expon_1P:
    """
    Deprecated Function due to renaming
    Use Fit_Exponential_1P instead
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        **kwargs
    ):
        warning_str = "DeprecationWarning: Fit_Expon_1P was renamed to Fit_Exponential_1P in version 0.5.4. Your function has still been run, however, Fit_Expon_1P will be fully deprecated in March 2021."
        colorprint(warning_str, text_color="red")
        Fit_Exponential_1P(
            failures=failures,
            right_censored=right_censored,
            show_probability_plot=show_probability_plot,
            print_results=print_results,
            CI=CI,
            percentiles=percentiles,
            **kwargs
        )


class Fit_Expon_2P:
    """
    Deprecated Function due to renaming
    Use Fit_Exponential_2P instead
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        **kwargs
    ):
        warning_str = "DeprecationWarning: Fit_Expon_2P was renamed to Fit_Exponential_2P in version 0.5.4. Your function has still been run, however, Fit_Expon_2P will be fully deprecated in March 2021."
        colorprint(warning_str, text_color="red")
        Fit_Exponential_2P(
            failures=failures,
            right_censored=right_censored,
            show_probability_plot=show_probability_plot,
            print_results=print_results,
            CI=CI,
            percentiles=percentiles,
            **kwargs
        )


class Fit_Exponential_1P:
    """
    Fit_Exponential_1P
    Fits a 1-parameter Exponential distribution (Lambda) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    percentiles - percentiles to produce a table of percentiles failed with lower, point, and upper estimates. Default is None which results in no output. True or 'auto' will use default array [1, 5, 10,..., 95, 99]. If an array or list is specified then it will be used instead of the default array.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    Lambda - the fitted Exponential_1P lambda parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - an Exponential_Distribution object with the parameters of the fitted distribution
    Lambda_SE - the standard error (sqrt(variance)) of the parameter
    Lambda_upper - the upper CI estimate of the parameter
    Lambda_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for the parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.

    *Note that this is a 1 parameter distribution but Lambda_inv is also provided as some programs (such as Minitab and scipy.stats) use this instead of Lambda
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        method="MLE",
        optimizer=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Exponential_1P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
                func_name="Exponential_1P",
                LL_func=Fit_Exponential_1P.LL,
                initial_guess=[LS_results.guess[0]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.Lambda = MLE_results.scale
            self.method = "Maximum Likelihood Estimation (MLE)"

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.Lambda]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
        self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
        SE_inv = abs(1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z)

        results_data = {
            "Parameter": ["Lambda", "1/Lambda"],
            "Point Estimate": [self.Lambda, 1 / self.Lambda],
            "Standard Error": [self.Lambda_SE, SE_inv],
            "Lower CI": [self.Lambda_lower, 1 / self.Lambda_upper],
            "Upper CI": [self.Lambda_upper, 1 / self.Lambda_lower],
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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.exponential_CI(
                self=self.distribution, func="CDF", CI=CI, q=1 - (percentiles / 100)
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, L):  # Log PDF (1 parameter Expon)
        return anp.log(L) - L * t

    @staticmethod
    def logR(t, L):  # Log SF (1 parameter Expon)
        return -(L * t)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (1 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_1P.logf(T_f, params[0]).sum()  # failure times
        LL_rc += Fit_Exponential_1P.logR(T_rc, params[0]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_2P:
    """
    Fit_Exponential_2P
    Fits a 2-parameter Exponential distribution (Lambda,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    percentiles - percentiles to produce a table of percentiles failed with lower, point, and upper estimates. Default is None which results in no output. True or 'auto' will use default array [1, 5, 10,..., 95, 99]. If an array or list is specified then it will be used instead of the default array.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    Lambda - the fitted Exponential_2P lambda parameter
    Lambda_inv - the inverse of the Lambda parameter (1/Lambda)
    gamma - the fitted Exponential_2P gamma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - an Exponential_Distribution object with the parameters of the fitted distribution
    Lambda_SE - the standard error (sqrt(variance)) of the parameter
    Lambda_SE_inv - the standard error of the Lambda_inv parameter
    gamma_SE - the standard error (sqrt(variance)) of the parameter. This will always be 0.
    Lambda_upper - the upper CI estimate of the parameter
    Lambda_lower - the lower CI estimate of the parameter
    Lambda_upper_inv - the upper CI estimate of the Lambda_inv  parameter
    Lambda_lower_inv - the lower CI estimate of the Lambda_inv parameter
    gamma_upper - the upper CI estimate of the parameter
    gamma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for the parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.

    *Note that this is a 2 parameter distribution but Lambda_inv is also provided as some programs (such as Minitab and scipy.stats) use this instead of Lambda
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        method="MLE",
        optimizer=None,
        **kwargs
    ):
        # To obtain the confidence intervals of the parameters, the gamma parameter is estimated by optimizing the log-likelihood function but
        # it is assumed as fixed because the variance-covariance matrix of the estimated parameters cannot be determined numerically. By assuming
        # the standard error in gamma is zero, we can use Exponential_1P to obtain the confidence intervals for Lambda. This is the same procedure
        # performed by both Reliasoft and Minitab. You may find the results are slightly different to Minitab and this is because the optimisation
        # of gamma is done more efficiently here than Minitab does it. This is evidenced by comparing the log-likelihood for the same data input.

        inputs = fitters_input_checking(
            dist="Exponential_2P",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            percentiles=percentiles,
            method=method,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        percentiles = inputs.percentiles
        method = inputs.method
        optimizer = inputs.optimizer

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            if (
                LS_results.guess[0] < 1
            ):  # The reason for having an inverted and non-inverted cases is due to the gradient being too shallow in some cases. If Lambda<1 we invert it so it's bigger. This prevents the gradient getting too shallow for the optimizer to find the correct minimum.
                MLE_results = MLE_optimisation(
                    func_name="Exponential_2P",
                    LL_func=Fit_Exponential_2P.LL_inv,
                    initial_guess=[1 / LS_results.guess[0], LS_results.guess[1]],
                    failures=failures,
                    right_censored=right_censored,
                    optimizer=optimizer,
                )
                self.Lambda = 1 / MLE_results.scale
            else:
                MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters. Uses Exponential_1P because gamma (while optimized) cannot be used in the MLE solution as the solution is unbounded
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_1P = [self.Lambda]
        params_2P = [self.Lambda, self.gamma]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params_1P)),
            np.array(tuple(failures - self.gamma)),
            np.array(tuple(right_censored - self.gamma)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.gamma_SE = 0
        self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
        self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
        self.gamma_upper = self.gamma
        self.gamma_lower = self.gamma
        self.Lambda_inv = 1 / self.Lambda
        self.Lambda_SE_inv = abs(
            1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z
        )
        self.Lambda_lower_inv = 1 / self.Lambda_upper
        self.Lambda_upper_inv = 1 / self.Lambda_lower

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.exponential_CI(
                self=self.distribution, func="CDF", CI=CI, q=1 - (percentiles / 100)
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, L, g):  # Log PDF (2 parameter Expon)
        return anp.log(L) - L * (t - g)

    @staticmethod
    def logR(t, L, g):  # Log SF (2 parameter Expon)
        return -(L * (t - g))

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_2P.logf(
            T_f, params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

    # #this is the inverted forms of the above functions. It simply changes Lambda to be 1/Lambda which is necessary when Lambda<<1
    @staticmethod
    def LL_inv(params, T_f, T_rc):  # log likelihood function (2 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_2P.logf(
            T_f, 1 / params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_2P.logR(
            T_rc, 1 / params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_2P:
    """
    Fit_Normal_2P
    Fits a 2-parameter Normal distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain (x<0).
    If you need an entirely positive distribution that is similar to Normal then consider using Weibull.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    force_sigma - Use this to specify the sigma value if you need to force sigma to be a certain value. Used in ALT probability plotting. Optional input.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    mu - the fitted Normal_2P mu parameter
    sigma - the fitted Normal_2P sigma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Normal_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma - the covariance between the parameters
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Normal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Normal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = abs(covariance_matrix[0][1])
            self.mu_upper = self.mu + (
                Z * self.mu_SE
            )  # these are unique to normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        else:
            hessian_matrix = hessian(Fit_Normal_2P.LL_fs)(
                np.array(tuple([self.mu])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_sigma])),
            )
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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.normal_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

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
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Normal_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(
        params, T_f, T_rc, force_sigma
    ):  # log likelihood function (2 parameter Normal) FORCED SIGMA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_2P.logf(T_f, params[0], force_sigma).sum()  # failure times
        LL_rc += Fit_Normal_2P.logR(
            T_rc, params[0], force_sigma
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Gumbel_2P:
    """
    Fit_Gumbel_2P
    Fits a 2-parameter Gumbel distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain (x<0).
    The Gumbel Distribution is similar to the Normal Distribution, with mu controlling the peak of the distribution between -inf < mu < inf

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    mu - the fitted Gumbel_2P mu parameter
    sigma - the fitted Gumbel_2P sigma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Gumbel_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma - the covariance between the parameters
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Gumbel_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        hessian_matrix = hessian(Fit_Gumbel_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.Cov_mu_sigma = abs(covariance_matrix[0][1])
        self.mu_upper = self.mu + (
            Z * self.mu_SE
        )  # these are unique to gumbel, normal and lognormal mu params
        self.mu_lower = self.mu + (-Z * self.mu_SE)
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.gumbel_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Gumbel)
        return -anp.log(sigma) + (t - mu) / sigma - anp.exp((t - mu) / sigma)

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Gumbel)
        return -anp.exp((t - mu) / sigma)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Gumbel)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gumbel_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Gumbel_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_2P:
    """
    Fit_Lognormal_2P
    Fits a 2-parameter Lognormal distribution (mu,sigma) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    force_sigma - Use this to specify the sigma value if you need to force sigma to be a certain value. Used in ALT probability plotting. Optional input.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    mu - the fitted Lognormal_2P mu parameter
    sigma - the fitted Lognormal_2P sigma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Lognormal_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma - the covariance between the parameters
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Lognormal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = abs(covariance_matrix[0][1])
            self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (
                np.exp(Z * (self.sigma_SE / self.sigma))
            )  # sigma is strictly positive
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        else:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL_fs)(
                np.array(tuple([self.mu])),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array(tuple([force_sigma])),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = 0
            self.Cov_mu_sigma = 0
            self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
            self.mu_lower = self.mu + (-Z * self.mu_SE)
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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.lognormal_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

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
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter lognormal)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Lognormal_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(
        params, T_f, T_rc, force_sigma
    ):  # log likelihood function (2 parameter lognormal) FORCED SIGMA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_2P.logf(
            T_f, params[0], force_sigma
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_2P.logR(
            T_rc, params[0], force_sigma
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_3P:
    """
    Fit_Lognormal_3P
    Fits a 3-parameter Lognormal distribution (mu,sigma,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    mu - the fitted Lognormal_3P mu parameter
    sigma - the fitted Lognormal_3P sigma parameter
    gamma - the fitted Lognormal_3P gamma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Lognormal_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    gamma_SE - the standard error (sqrt(variance)) of the parameter
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    gamma_upper - the upper CI estimate of the parameter
    gamma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        CI_type="time",
        optimizer=None,
        method="MLE",
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Lognormal_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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
            self.Cov_mu_sigma = abs(covariance_matrix[0][1])
            self.mu_upper = self.mu + (
                Z * self.mu_SE
            )  # Mu can be positive or negative.
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (
                np.exp(Z * (self.sigma_SE / self.sigma))
            )  # sigma is strictly positive
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
            self.gamma_upper = self.gamma * (
                np.exp(Z * (self.gamma_SE / self.gamma))
            )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
            self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.lognormal_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
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
    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Lognormal)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_3P.logf(
            T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_3P.logR(
            T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Gamma_2P:
    """
    Fit_Gamma_2P
    Fits a 2-parameter Gamma distribution (alpha,beta) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Gamma_2P alpha parameter
    beta - the fitted Gamma_2P beta parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Gamma_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
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
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Gamma_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        # percentiles = inputs.percentiles
        # CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_results = LS_optimisation(
            func_name="Gamma_2P",
            LL_func=Fit_Gamma_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method="LS",
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
                func_name="Gamma_2P",
                LL_func=Fit_Gamma_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Gamma_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.Cov_alpha_beta = abs(covariance_matrix[0][1])
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

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
        self.distribution = Gamma_Distribution(alpha=self.alpha, beta=self.beta)
        # self.distribution = Gamma_Distribution(alpha=self.alpha, beta=self.beta, alpha_SE=self.alpha_SE, beta_SE=self.beta_SE, Cov_alpha_beta=self.Cov_alpha_beta, CI=CI, CI_type=CI_type)

        # if percentiles is not None:
        #     point_estimate = self.distribution.quantile(q=percentiles / 100)
        #     lower_estimate, upper_estimate = distribution_confidence_intervals.gamma_CI(self=self.distribution, func='CDF', CI_type='time', CI=CI, q=1 - (percentiles / 100))
        #     percentile_data = {'Percentile': percentiles,
        #                        'Lower Estimate': lower_estimate,
        #                        'Point Estimate': point_estimate,
        #                        'Upper Estimate': upper_estimate}
        #     self.percentiles = pd.DataFrame(percentile_data, columns=['Percentile', 'Lower Estimate', 'Point Estimate', 'Upper Estimate'])

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Gamma_2P.LL(params, failures, right_censored)
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            # if percentiles is not None:
            #     print(str('Table of percentiles (' + str(CI_rounded) + '% CI bounds on time):'))
            #     print(self.percentiles.to_string(index=False), '\n')

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
                **kwargs
            )
            # Gamma_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self, CI=CI, CI_type=CI_type, **kwargs)

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Gamma)
        return anp.log(t ** (b - 1)) - anp.log((a ** b) * agamma(b)) - (t / a)

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Gamma)
        return anp.log(gammaincc(b, t / a))

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Gamma_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Gamma_3P:
    """
    Fit_Gamma_3P
    Fits a 3-parameter Gamma distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), or 'LS' (least squares estimation). LS will perform non-linear least squares estimation. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Gamma_3P alpha parameter
    beta - the fitted Gamma_3P beta parameter
    gamma - the fitted Gamma_3P gamma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Gamma_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    gamma_SE - the standard error (sqrt(variance)) of the parameter
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    gamma_upper - the upper CI estimate of the parameter
    gamma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
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
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Gamma_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        # percentiles = inputs.percentiles
        # CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
            func_name="Gamma_3P",
            LL_func=Fit_Gamma_3P.LL,
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
                func_name="Gamma_3P",
                LL_func=Fit_Gamma_3P.LL,
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
            self.gamma = 0
            self.alpha_SE = gamma_2P_results.alpha_SE
            self.beta_SE = gamma_2P_results.beta_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = gamma_2P_results.Cov_alpha_beta
            self.alpha_upper = gamma_2P_results.alpha_upper
            self.alpha_lower = gamma_2P_results.alpha_lower
            self.beta_upper = gamma_2P_results.beta_upper
            self.beta_lower = gamma_2P_results.beta_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.alpha, self.beta]
            params_3P = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Gamma_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Gamma_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            covariance_matrix = np.linalg.inv(hessian_matrix)
            # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
            hessian_matrix_for_gamma = hessian(Fit_Gamma_3P.LL)(
                np.array(tuple(params_3P)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
            self.gamma_upper = self.gamma * (
                np.exp(Z * (self.gamma_SE / self.gamma))
            )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
            self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

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
        # self.distribution = Gamma_Distribution(alpha=self.alpha, beta=self.beta, gamma=self.gamma, alpha_SE=self.alpha_SE, beta_SE=self.beta_SE, Cov_alpha_beta=self.Cov_alpha_beta, CI=CI, CI_type=CI_type)
        self.distribution = Gamma_Distribution(
            alpha=self.alpha, beta=self.beta, gamma=self.gamma
        )

        # if percentiles is not None:
        #     point_estimate = self.distribution.quantile(q=percentiles / 100)
        #     lower_estimate, upper_estimate = distribution_confidence_intervals.gamma_CI(self=self.distribution, func='CDF', CI_type='time', CI=CI, q=1 - (percentiles / 100))
        #     percentile_data = {'Percentile': percentiles,
        #                        'Lower Estimate': lower_estimate,
        #                        'Point Estimate': point_estimate,
        #                        'Upper Estimate': upper_estimate}
        #     self.percentiles = pd.DataFrame(percentile_data, columns=['Percentile', 'Lower Estimate', 'Point Estimate', 'Upper Estimate'])

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Gamma_3P.LL(params_3P, failures, right_censored)
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            # if percentiles is not None:
            #     print(str('Table of percentiles (' + str(CI_rounded) + '% CI bounds on time):'))
            #     print(self.percentiles.to_string(index=False), '\n')

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            # fig = Gamma_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self, CI=CI, CI_type=CI_type, **kwargs)
            fig = Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                **kwargs
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

    @staticmethod
    def logf(t, a, b, g):  # Log PDF (3 parameter Gamma)
        return (
            anp.log((t - g) ** (b - 1)) - anp.log((a ** b) * agamma(b)) - ((t - g) / a)
        )

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Gamma)
        return anp.log(gammaincc(b, (t - g) / a))

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_3P.logf(
            T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Gamma_3P.logR(
            T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Beta_2P:
    """
    Fit_Beta_2P
    Fits a 2-parameter Beta distribution (alpha,beta) to the data provided.
    All data must be in the range 0 < x < 1.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Beta_2P alpha parameter
    beta - the fitted Beta_2P beta parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Beta_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
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
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Beta_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        # percentiles = inputs.percentiles
        # CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
                func_name="Beta_2P",
                LL_func=Fit_Beta_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Beta_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.Cov_alpha_beta = abs(covariance_matrix[0][1])
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

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
        self.distribution = Beta_Distribution(alpha=self.alpha, beta=self.beta)
        # self.distribution = Beta_Distribution(alpha=self.alpha, beta=self.beta, alpha_SE=self.alpha_SE, beta_SE=self.beta_SE, Cov_alpha_beta=self.Cov_alpha_beta, CI=CI, CI_type=CI_type)

        # if percentiles is not None:
        #     point_estimate = self.distribution.quantile(q=percentiles / 100)
        #     lower_estimate, upper_estimate = distribution_confidence_intervals.beta_CI(self=self.distribution, func='CDF', CI_type='time', CI=CI, q=1 - (percentiles / 100))
        #     percentile_data = {'Percentile': percentiles,
        #                        'Lower Estimate': lower_estimate,
        #                        'Point Estimate': point_estimate,
        #                        'Upper Estimate': upper_estimate}
        #     self.percentiles = pd.DataFrame(percentile_data, columns=['Percentile', 'Lower Estimate', 'Point Estimate', 'Upper Estimate'])

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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            # if percentiles is not None:
            #     print(str('Table of percentiles (' + str(CI_rounded) + '% CI bounds on time):'))
            #     print(self.percentiles.to_string(index=False), '\n')

        if show_probability_plot is True:
            from reliability.Probability_plotting import Beta_probability_plot

            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            # Beta_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self, CI=CI, CI_type=CI_type, **kwargs)
            Beta_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                **kwargs
            )

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Beta)
        return anp.log(((t ** (a - 1)) * ((1 - t) ** (b - 1)))) - anp.log(abeta(a, b))

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Beta)
        return anp.log(1 - betainc(a, b, t))

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter beta)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Beta_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Beta_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Loglogistic_2P:
    """
    Fit_Loglogistic_2P

    Fits a 2-parameter Loglogistic distribution (alpha,beta) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    method - 'MLE' (maximum likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank regression on X), 'RRY' (Rank regression on Y). LS will perform both RRX and RRY and return the better one. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type - time, reliability, None. Default is time. This is the confidence bounds on time or on reliability. Use None to turn off the confidence intervals.
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    outputs:
    alpha - the fitted Loglogistic_2P alpha parameter
    beta - the fitted Loglogistic_2P beta parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - Log Likelihood * -2 (as used in JMP Pro)
    AICc - Akaike Information Criterion corrected
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Loglogistic_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        percentiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Loglogistic_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Loglogistic_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.Cov_alpha_beta = abs(covariance_matrix[0][1])
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.loglogistic_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
            )

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Loglogistic)
        return (
            anp.log(b / a) - (b + 1) * anp.log(t / a) - 2 * anp.log(1 + (t / a) ** -b)
        )

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Loglogistic)
        return -anp.log((1 + (t / a) ** b))

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Loglogistic)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Loglogistic_2P.logf(
            T_f, params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Loglogistic_2P.logR(
            T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Loglogistic_3P:
    """
    Fit_Loglogistic_3P
    Fits a 3-parameter Loglogistic distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type - 'time' or 'reliability'. Default is time. Used for the probability plot and the distribution object in the output.
    method - 'MLE' (maximum likelihood estimation), or 'LS' (least squares estimation). LS will perform non-linear least squares estimation. Default is 'MLE'.
    optimizer - 'L-BFGS-B', 'TNC', or 'powell'. These are all bound constrained methods. If the bounded method fails, nelder-mead will be used. If nelder-mead fails then the initial guess will be returned with a warning. For more information on optimizers see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    kwargs are accepted for the probability plot (eg. linestyle, label, color)

    Outputs:
    alpha - the fitted Loglogistic_3P alpha parameter
    beta - the fitted Loglogistic_3P beta parameter
    gamma - the fitted Loglogistic_3P gamma parameter
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 - LogLikelihood*-2 (as used in JMP Pro)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    AD - the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution - a Loglogistic_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    gamma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters alpha and beta
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    gamma_upper - the upper CI estimate of the parameter
    gamma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit values (Log-likelihood, AICc, BIC, AD).
    percentiles - a dataframe of the percentiles with bounds on time. This is only produced if percentiles is 'auto' or a list or array. Since percentiles defaults to None, this output is not normally produced.
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
        percentiles=None,
        **kwargs
    ):

        inputs = fitters_input_checking(
            dist="Loglogistic_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            percentiles=percentiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        percentiles = inputs.percentiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        if method == "MLE":
            LS_method = "LS"
        else:
            LS_method = method
        LS_results = LS_optimisation(
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

        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimisation(
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
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
            self.gamma_upper = self.gamma * (
                np.exp(Z * (self.gamma_SE / self.gamma))
            )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
            self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

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

        if percentiles is not None:
            point_estimate = self.distribution.quantile(q=percentiles / 100)
            (
                lower_estimate,
                upper_estimate,
            ) = distribution_confidence_intervals.loglogistic_CI(
                self=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                q=1 - (percentiles / 100),
            )
            percentile_data = {
                "Percentile": percentiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.percentiles = pd.DataFrame(
                percentile_data,
                columns=[
                    "Percentile",
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
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if percentiles is not None:
                print(
                    str(
                        "Table of percentiles ("
                        + str(CI_rounded)
                        + "% CI bounds on time):"
                    )
                )
                print(self.percentiles.to_string(index=False), "\n")

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
                **kwargs
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
    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Loglogistic)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Loglogistic_3P.logf(
            T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Loglogistic_3P.logR(
            T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)
