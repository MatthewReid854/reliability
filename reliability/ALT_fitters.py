"""
ALT_fitters

ALT_fitters includes the combinations of 4 life models (Weibull, Lognormal, Normal, Exponential) and 5 life-stress models (Exponential, Eyring, Power, Dual_Exponential, Power_Exponential)
These combinations produce 20 different ALT models with the following names:

Fit_Weibull_Exponential
Fit_Weibull_Eyring
Fit_Weibull_Power
Fit_Weibull_Dual_Exponential
Fit_Weibull_Power_Exponential

Fit_Lognormal_Exponential
Fit_Lognormal_Eyring
Fit_Lognormal_Power
Fit_Lognormal_Dual_Exponential
Fit_Lognormal_Power_Exponential

Fit_Normal_Exponential
Fit_Normal_Eyring
Fit_Normal_Power
Fit_Normal_Dual_Exponential
Fit_Normal_Power_Exponential

Fit_Exponential_Exponential
Fit_Exponential_Eyring
Fit_Exponential_Power
Fit_Exponential_Dual_Exponential
Fit_Exponential_Power_Exponential

For more informtaion on each model, use the help function.
Eg: help(Fit_Weibull_Exponential)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import scipy.stats as ss
import pandas as pd
import autograd.numpy as anp
import warnings
from autograd import value_and_grad
from autograd.scipy.special import erf
from autograd.differential_operators import hessian
from reliability.Fitters import (
    Fit_Weibull_2P,
    Fit_Lognormal_2P,
    Fit_Normal_2P,
    Fit_Exponential_1P,
)
from reliability.ALT_probability_plotting import (
    ALT_probability_plot_Weibull,
    ALT_probability_plot_Lognormal,
    ALT_probability_plot_Normal,
)
from reliability.Distributions import (
    Weibull_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
    Exponential_Distribution,
)
from reliability.Probability_plotting import (
    Weibull_probability_plot,
    Lognormal_probability_plot,
    Normal_probability_plot,
    Exponential_probability_plot_Weibull_Scale,
)
from reliability.Utils import (
    probability_plot_xyticks,
    probability_plot_xylims,
    colorprint,
)

anp.seterr("ignore")
color_list = [
    "steelblue",
    "darkorange",
    "red",
    "green",
    "purple",
    "blue",
    "grey",
    "deeppink",
    "cyan",
    "chocolate",
    "blueviolet",
    "indianred",
    "yellow",
    "olivedrab",
    "crimson",
    "black",
    "lightseagreen",
    "pink",
    "indigo",
    "darkgreen",
]

pd.set_option("display.width", 200)  # prevents wrapping after default 80 characters
pd.set_option("display.max_columns", 9)  # shows the dataframe without ... truncation


class Fit_Weibull_Exponential:
    """
    Fit_Weibull_Exponential

    This function will Fit the Weibull-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.
    If you are using this model for the Arrhenius equation, a = Ea/K_B. When results are printed Ea will be provided in eV.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Exponential model
    b - fitted parameter from the Exponential model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __expon(stress, a, b):
            return b * np.exp(a / stress)

        if initial_guess is None:
            initial_guess, _ = curve_fit(__expon, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, b].")

        # this gets the common beta for the initial guess using the functions already built into ALT_probability_plot_Weibull
        ALT_fit = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Weibull_Exponential.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Weibull-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Weibull_Exponential.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.beta = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.beta_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.beta_upper = self.beta * (
            np.exp(Z * (self.beta_SE / self.beta))
        )  # a and b can be +- but beta is strictly + so the formulas here are different for beta
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["a", "b", "beta"],
            "Point Estimate": [self.a, self.b, self.beta],
            "Standard Error": [self.a_SE, self.b_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.beta_upper],
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

        if use_level_stress is not None:
            self.alpha_at_use_stress = self.b * np.exp(self.a / use_level_stress)
            self.mean_life = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            ).mean

        if print_results is True:

            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Weibull_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV",
            )
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Weibull_probability_plot is expecting to receive
                class __make_fitted_dist_params_weibull:
                    def __init__(self2, alpha, beta):
                        self2.alpha = alpha
                        self2.beta = beta
                        self2.gamma = 0
                        self2.alpha_SE = None
                        self2.beta_SE = None
                        self2.Cov_alpha_beta = None

                life = self.b * np.exp(self.a / stress)
                fitted_dist_params = __make_fitted_dist_params_weibull(
                    alpha=life, beta=self.beta
                )
                original_fit = Fit_Weibull_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Weibull_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Weibull_Distribution(
                    alpha=self.alpha_at_use_stress, beta=self.beta
                ).CDF(label=use_label_str, color=color_list[i + 1], linestyle="--")
                x_array = np.hstack(
                    [
                        Weibull_Distribution(
                            alpha=self.alpha_at_use_stress, beta=self.beta
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Weibull-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, b, beta):  # Log PDF
        life = b * anp.exp(a / T)
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, T, a, b, beta):  # Log SF
        life = b * anp.exp(a / T)
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Eyring:
    """
    Fit_Weibull_Eyring

    This function will Fit the Weibull-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Eyring model
    c - fitted parameter from the Eyring model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).        alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __eyring(stress, a, c):
            return 1 / stress * np.exp(-(c - a / stress))

        if initial_guess is None:
            initial_guess, _ = curve_fit(__eyring, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, c].")

        # this gets the common beta for the initial guess using the functions already built into ALT_probability_plot_Weibull
        ALT_fit = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Weibull_Eyring.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Weibull-Eyring model. Try a better initial guess by specifying the parameter initial_guess = [a,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Weibull_Eyring.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.beta = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Eyring.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.beta_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.beta_upper = self.beta * (
            np.exp(Z * (self.beta_SE / self.beta))
        )  # a and c can be +- but beta is strictly + so the formulas here are different for beta
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["a", "c", "beta"],
            "Point Estimate": [self.a, self.c, self.beta],
            "Standard Error": [self.a_SE, self.c_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.beta_upper],
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

        if use_level_stress is not None:
            self.alpha_at_use_stress = (
                1 / use_level_stress * np.exp(-(self.c - self.a / use_level_stress))
            )
            self.mean_life = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Weibull_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Weibull_probability_plot is expecting to receive
                class __make_fitted_dist_params_weibull:
                    def __init__(self2, alpha, beta):
                        self2.alpha = alpha
                        self2.beta = beta
                        self2.gamma = 0
                        self2.alpha_SE = None
                        self2.beta_SE = None
                        self2.Cov_alpha_beta = None

                life = 1 / stress * np.exp(-(self.c - self.a / stress))
                fitted_dist_params = __make_fitted_dist_params_weibull(
                    alpha=life, beta=self.beta
                )
                original_fit = Fit_Weibull_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Weibull_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Weibull_Distribution(
                    alpha=self.alpha_at_use_stress, beta=self.beta
                ).CDF(label=use_label_str, color=color_list[i + 1], linestyle="--")
                x_array = np.hstack(
                    [
                        Weibull_Distribution(
                            alpha=self.alpha_at_use_stress, beta=self.beta
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Weibull-Eyring Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, c, beta):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, T, a, c, beta):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Eyring.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Power:
    """
    Fit_Weibull_Power

    This function will Fit the Weibull-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power model
    n - fitted parameter from the Power model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power(stress, a, n):
            return a * stress ** n

        if initial_guess is None:
            initial_guess, _ = curve_fit(__power, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, n].")

        # this gets the common beta for the initial guess using the functions already built into ALT_probability_plot_Weibull
        ALT_fit = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Weibull_Power.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Weibull-Power model. Try a better initial guess by specifying the parameter initial_guess = [a,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Weibull_Power.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.n = params[1]
        self.beta = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Power.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.beta_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.beta_upper = self.beta * (
            np.exp(Z * (self.beta_SE / self.beta))
        )  # a and n can be +- but beta is strictly + so the formulas here are different for beta
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["a", "n", "beta"],
            "Point Estimate": [self.a, self.n, self.beta],
            "Standard Error": [self.a_SE, self.n_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.beta_upper],
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

        if use_level_stress is not None:
            self.alpha_at_use_stress = self.a * use_level_stress ** self.n
            self.mean_life = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Weibull_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Weibull_probability_plot is expecting to receive
                class __make_fitted_dist_params_weibull:
                    def __init__(self2, alpha, beta):
                        self2.alpha = alpha
                        self2.beta = beta
                        self2.gamma = 0
                        self2.alpha_SE = None
                        self2.beta_SE = None
                        self2.Cov_alpha_beta = None

                life = self.a * stress ** self.n
                fitted_dist_params = __make_fitted_dist_params_weibull(
                    alpha=life, beta=self.beta
                )
                original_fit = Fit_Weibull_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Weibull_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Weibull_Distribution(
                    alpha=self.alpha_at_use_stress, beta=self.beta
                ).CDF(label=use_label_str, color=color_list[i + 1], linestyle="--")
                x_array = np.hstack(
                    [
                        Weibull_Distribution(
                            alpha=self.alpha_at_use_stress, beta=self.beta
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Weibull-Power Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, n, beta):  # Log PDF
        life = a * T ** n
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, T, a, n, beta):  # Log SF
        life = a * T ** n
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Power.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Dual_Exponential:
    """
    Fit_Weibull_Dual_Exponential

    This function will Fit the Weibull-Dual-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature-humidity. It is recommended that you ensure your temperature data are in Kelvin and humidity data range from 0 to 1.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as humidity) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as humidity) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Dual-Exponential model
    b - fitted parameter from the Dual-Exponential model
    c - fitted parameter from the Dual-Exponential model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_humidity]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_1) == list:
            failure_stress_1 = np.array(failure_stress_1)
        if type(failure_stress_1) != np.ndarray:
            raise TypeError(
                "failure_stress_1 must be a list or array of failure_stress data"
            )
        if type(failure_stress_2) == list:
            failure_stress_2 = np.array(failure_stress_2)
        if type(failure_stress_2) != np.ndarray:
            raise TypeError(
                "failure_stress_2 must be a list or array of failure_stress data"
            )
        if len(failure_stress_1) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_1"
            )
        if len(failure_stress_2) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_2"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_1) == list:
                right_censored_stress_1 = np.array(right_censored_stress_1)
            if type(right_censored_stress_1) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_1 must be a list or array of right censored failure_stress data"
                )
            if type(right_censored_stress_2) == list:
                right_censored_stress_2 = np.array(right_censored_stress_2)
            if type(right_censored_stress_2) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_2 must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress_1) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_1"
                )
            if len(right_censored_stress_2) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_2"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __dual_expon(stress, a, b, c):
            T = stress[0]
            H = stress[1]
            return c * np.exp(a / T + b / H)

        xdata = np.array(list(zip(failure_stress_1, failure_stress_2))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__dual_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, b, c]")

        # this gets the common beta for the initial guess using the functions already built into ALT_probability_plot_Weibull
        ALT_fit_1 = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_1,
            right_censored_stress=right_censored_stress_1,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_2,
            right_censored_stress=right_censored_stress_2,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Weibull_Dual_Exponential.LL),
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
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Weibull-Dual-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Weibull_Dual_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.beta = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Dual_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.c_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.beta_upper = self.beta * (
            np.exp(Z * (self.beta_SE / self.beta))
        )  # a and b can be +- but beta is strictly + so the formulas here are different for beta
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["a", "b", "c", "beta"],
            "Point Estimate": [self.a, self.b, self.c, self.beta],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.beta_upper],
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

        if use_level_stress is not None:
            self.alpha_at_use_stress = self.c * np.exp(
                self.a / use_level_stress[0] + self.b / use_level_stress[1]
            )
            self.mean_life = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Weibull_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack([failure_stress_1, right_censored_stress_1])
            STRESS_2 = np.hstack([failure_stress_2, right_censored_stress_2])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_1 and right_censored_stress_2 arrays contains pairs of values that are not found in the failure_stress_1 and failure_stress_2 arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Weibull_probability_plot is expecting to receive
                class __make_fitted_dist_params_weibull:
                    def __init__(self2, alpha, beta):
                        self2.alpha = alpha
                        self2.beta = beta
                        self2.gamma = 0
                        self2.alpha_SE = None
                        self2.beta_SE = None
                        self2.Cov_alpha_beta = None

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * np.exp(self.a / stress_1 + self.b / stress_2)
                fitted_dist_params = __make_fitted_dist_params_weibull(
                    alpha=life, beta=self.beta
                )
                original_fit = Fit_Weibull_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i],
                    label=str(stress_pair),
                    plot_CI=False,
                    xvals=xvals,
                )
                Weibull_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Weibull_Distribution(
                    alpha=self.alpha_at_use_stress, beta=self.beta
                ).CDF(label=use_label_str, color=color_list[i + 1], linestyle="--")
                x_array = np.hstack(
                    [
                        Weibull_Distribution(
                            alpha=self.alpha_at_use_stress, beta=self.beta
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="     Stress 1 , Stress 2")
            leg._legend_box.align = "left"
            plt.title("Weibull-Dual-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, b, c, beta):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, S1, S2, a, b, c, beta):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Power_Exponential:
    """
    Fit_Weibull_Power_Exponential

    This function will Fit the Weibull-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_thermal, stress_nonthermal]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power-Exponential model
    c - fitted parameter from the Power-Exponential model
    n - fitted parameter from the Power-Exponential model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_thermal=None,
        failure_stress_nonthermal=None,
        right_censored=None,
        right_censored_stress_thermal=None,
        right_censored_stress_nonthermal=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_voltage]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_thermal) == list:
            failure_stress_thermal = np.array(failure_stress_thermal)
        if type(failure_stress_thermal) != np.ndarray:
            raise TypeError(
                "failure_stress_thermal must be a list or array of thermal failure_stress data"
            )
        if type(failure_stress_nonthermal) == list:
            failure_stress_nonthermal = np.array(failure_stress_nonthermal)
        if type(failure_stress_nonthermal) != np.ndarray:
            raise TypeError(
                "failure_stress_nonthermal must be a list or array of nonthermal failure_stress data"
            )
        if len(failure_stress_thermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_thermal"
            )
        if len(failure_stress_nonthermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_nonthermal"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_thermal) == list:
                right_censored_stress_thermal = np.array(right_censored_stress_thermal)
            if type(right_censored_stress_thermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_thermal must be a list or array of right censored thermal failure_stress data"
                )
            if type(right_censored_stress_nonthermal) == list:
                right_censored_stress_nonthermal = np.array(
                    right_censored_stress_nonthermal
                )
            if type(right_censored_stress_nonthermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_nonthermal must be a list or array of right censored nonthermal failure_stress data"
                )
            if len(right_censored_stress_thermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_thermal"
                )
            if len(right_censored_stress_nonthermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_nonthermal"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power_expon(stress, a, c, n):
            T = stress[0]
            S = stress[1]
            return c * S ** n * np.exp(a / T)

        xdata = np.array(list(zip(failure_stress_thermal, failure_stress_nonthermal))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__power_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, c, n].")

        # this gets the common beta for the initial guess using the functions already built into ALT_probability_plot_Weibull
        ALT_fit_1 = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_thermal,
            right_censored_stress=right_censored_stress_thermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Weibull(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_nonthermal,
            right_censored_stress=right_censored_stress_nonthermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_thermal = []
            right_censored_stress_nonthermal = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Weibull_Power_Exponential.LL),
                guess,
                args=(
                    failures,
                    right_censored,
                    failure_stress_thermal,
                    failure_stress_nonthermal,
                    right_censored_stress_thermal,
                    right_censored_stress_nonthermal,
                ),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Weibull-Power-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,c,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Weibull_Power_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_thermal,
                failure_stress_nonthermal,
                right_censored_stress_thermal,
                right_censored_stress_nonthermal,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.n = params[2]
        self.beta = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Power_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_thermal)),
            np.array(tuple(failure_stress_nonthermal)),
            np.array(tuple(right_censored_stress_thermal)),
            np.array(tuple(right_censored_stress_nonthermal)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.beta_upper = self.beta * (
            np.exp(Z * (self.beta_SE / self.beta))
        )  # a and b can be +- but beta is strictly + so the formulas here are different for beta
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["a", "c", "n", "beta"],
            "Point Estimate": [self.a, self.c, self.n, self.beta],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.beta_upper],
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

        if use_level_stress is not None:
            self.alpha_at_use_stress = (
                self.c
                * (use_level_stress[1]) ** self.n
                * np.exp(self.a / use_level_stress[0])
            )
            self.mean_life = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Weibull_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack(
                [failure_stress_thermal, right_censored_stress_thermal]
            )
            STRESS_2 = np.hstack(
                [failure_stress_nonthermal, right_censored_stress_nonthermal]
            )
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_thermal and right_censored_stress_nonthermal arrays contains pairs of values that are not found in the failure_stress_thermal and failure_stress_nonthermal arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Weibull_probability_plot is expecting to receive
                class __make_fitted_dist_params_weibull:
                    def __init__(self2, alpha, beta):
                        self2.alpha = alpha
                        self2.beta = beta
                        self2.gamma = 0
                        self2.alpha_SE = None
                        self2.beta_SE = None
                        self2.Cov_alpha_beta = None

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * stress_2 ** self.n * np.exp(self.a / stress_1)
                fitted_dist_params = __make_fitted_dist_params_weibull(
                    alpha=life, beta=self.beta
                )
                original_fit = Fit_Weibull_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i],
                    label=str(stress_pair),
                    plot_CI=False,
                    xvals=xvals,
                )
                Weibull_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Weibull_Distribution(
                    alpha=self.alpha_at_use_stress, beta=self.beta
                ).CDF(label=use_label_str, color=color_list[i + 1], linestyle="--")
                x_array = np.hstack(
                    [
                        Weibull_Distribution(
                            alpha=self.alpha_at_use_stress, beta=self.beta
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="Thermal stress , Non-thermal stress")
            leg._legend_box.align = "left"
            plt.title("Weibull-Power-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, c, n, beta):  # Log PDF
        life = c * S2 ** n * anp.exp(a / S1)
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, S1, S2, a, c, n, beta):  # Log SF
        life = c * S2 ** n * anp.exp(a / S1)
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Weibull_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Exponential:
    """
    Fit_Lognormal_Exponential

    This function will Fit the Lognormal-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.
    If you are using this model for the Arrhenius equation, a = Ea/K_B. When results are printed Ea will be provided in eV.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Exponential model
    b - fitted parameter from the Exponential model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __expon(stress, a, b):
            return b * np.exp(a / stress)

        if initial_guess is None:
            initial_guess, _ = curve_fit(__expon, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, b].")

        # this gets the common shape for the initial guess using the functions already built into ALT_probability_plot_Lognormal
        ALT_fit = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Lognormal_Exponential.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Lognormal-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Lognormal_Exponential.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "b", "sigma"],
            "Point Estimate": [self.a, self.b, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            use_life = self.b * np.exp(self.a / use_level_stress)
            self.mu_at_use_stress = np.log(use_life)
            self.mean_life = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV",
            )
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common shape parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Lognormal_probability_plot is expecting to receive
                class __make_fitted_dist_params_lognormal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma
                        self2.gamma = 0

                life = self.b * np.exp(self.a / stress)
                fitted_dist_params = __make_fitted_dist_params_lognormal(
                    mu=np.log(life), sigma=self.sigma
                )
                original_fit = Fit_Lognormal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Lognormal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Lognormal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Lognormal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Lognormal-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="lognormal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, b, sigma):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, b, sigma):  # Log SF
        life = b * anp.exp(a / T)
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Eyring:
    """
    Fit_Lognormal_Eyring

    This function will Fit the Lognormal-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Eyring model
    c - fitted parameter from the Eyring model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __eyring(stress, a, c):
            return 1 / stress * np.exp(-(c - a / stress))

        if initial_guess is None:
            initial_guess, _ = curve_fit(__eyring, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, c].")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Lognormal
        ALT_fit = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Lognormal_Eyring.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Lognormal-Eyring model. Try a better initial guess by specifying the parameter initial_guess = [a,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Lognormal_Eyring.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_Eyring.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and c can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "c", "sigma"],
            "Point Estimate": [self.a, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            use_life = (
                1 / use_level_stress * np.exp(-(self.c - self.a / use_level_stress))
            )
            self.mu_at_use_stress = np.log(use_life)
            self.mean_life = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Lognormal_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Lognormal_probability_plot is expecting to receive
                class __make_fitted_dist_params_lognormal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma
                        self2.gamma = 0

                life = 1 / stress * np.exp(-(self.c - self.a / stress))
                fitted_dist_params = __make_fitted_dist_params_lognormal(
                    mu=np.log(life), sigma=self.sigma
                )
                original_fit = Fit_Lognormal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Lognormal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Lognormal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Lognormal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Lognormal-Eyring Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="lognormal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, c, sigma):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, c, sigma):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_Eyring.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Power:
    """
    Fit_Lognormal_Power

    This function will Fit the Lognormal-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power model
    n - fitted parameter from the Power model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power(stress, a, n):
            return a * stress ** n

        if initial_guess is None:
            initial_guess, _ = curve_fit(__power, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, n]")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Lognormal
        ALT_fit = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Lognormal_Power.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Lognormal-Power model. Try a better initial guess by specifying the parameter initial_guess = [a,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Lognormal_Power.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.n = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_Power.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and n can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "n", "sigma"],
            "Point Estimate": [self.a, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            use_life = self.a * use_level_stress ** self.n
            self.mu_at_use_stress = np.log(use_life)
            self.mean_life = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Lognormal_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Lognormal_probability_plot is expecting to receive
                class __make_fitted_dist_params_lognormal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma
                        self2.gamma = 0

                life = self.a * stress ** self.n
                fitted_dist_params = __make_fitted_dist_params_lognormal(
                    mu=np.log(life), sigma=self.sigma
                )
                original_fit = Fit_Lognormal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Lognormal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Lognormal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Lognormal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Lognormal-Power Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="lognormal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, n, sigma):  # Log PDF
        life = a * T ** n
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, n, sigma):  # Log SF
        life = a * T ** n
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_Power.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Dual_Exponential:
    """
    Fit_Lognormal_Dual_Exponential

    This function will Fit the Lognormal-Dual-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature-humidity. It is recommended that you ensure your temperature data are in Kelvin and humidity data range from 0 to 1.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as humidity) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as humidity) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Dual-Exponential model
    b - fitted parameter from the Dual-Exponential model
    c - fitted parameter from the Dual-Exponential model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_humidity]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_1) == list:
            failure_stress_1 = np.array(failure_stress_1)
        if type(failure_stress_1) != np.ndarray:
            raise TypeError(
                "failure_stress_1 must be a list or array of failure_stress data"
            )
        if type(failure_stress_2) == list:
            failure_stress_2 = np.array(failure_stress_2)
        if type(failure_stress_2) != np.ndarray:
            raise TypeError(
                "failure_stress_2 must be a list or array of failure_stress data"
            )
        if len(failure_stress_1) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_1"
            )
        if len(failure_stress_2) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_2"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_1) == list:
                right_censored_stress_1 = np.array(right_censored_stress_1)
            if type(right_censored_stress_1) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_1 must be a list or array of right censored failure_stress data"
                )
            if type(right_censored_stress_2) == list:
                right_censored_stress_2 = np.array(right_censored_stress_2)
            if type(right_censored_stress_2) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_2 must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress_1) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_1"
                )
            if len(right_censored_stress_2) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_2"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __dual_expon(stress, a, b, c):
            T = stress[0]
            H = stress[1]
            return c * np.exp(a / T + b / H)

        xdata = np.array(list(zip(failure_stress_1, failure_stress_2))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__dual_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, b, c]")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Lognormal
        ALT_fit_1 = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_1,
            right_censored_stress=right_censored_stress_1,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_2,
            right_censored_stress=right_censored_stress_2,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Lognormal_Dual_Exponential.LL),
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
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Lognormal-Dual-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Lognormal_Dual_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.sigma = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_Dual_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.c_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "b", "c", "sigma"],
            "Point Estimate": [self.a, self.b, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            use_life = self.c * np.exp(
                self.a / use_level_stress[0] + self.b / use_level_stress[1]
            )
            self.mu_at_use_stress = np.log(use_life)
            self.mean_life = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack([failure_stress_1, right_censored_stress_1])
            STRESS_2 = np.hstack([failure_stress_2, right_censored_stress_2])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_1 and right_censored_stress_2 arrays contains pairs of values that are not found in the failure_stress_1 and failure_stress_2 arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Lognormal_probability_plot is expecting to receive
                class __make_fitted_dist_params_lognormal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma
                        self2.gamma = 0

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * np.exp(self.a / stress_1 + self.b / stress_2)
                fitted_dist_params = __make_fitted_dist_params_lognormal(
                    mu=np.log(life), sigma=self.sigma
                )
                original_fit = Fit_Lognormal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress_pair), xvals=xvals
                )
                Lognormal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Lognormal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Lognormal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="     Stress 1 , Stress 2")
            leg._legend_box.align = "left"
            plt.title("Lognormal-Dual-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="lognormal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, b, c, sigma):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, a, b, c, sigma):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Power_Exponential:
    """
    Fit_Lognormal_Power_Exponential

    This function will Fit the Lognormal-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_thermal, stress_nonthermal]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power-Exponential model
    c - fitted parameter from the Power-Exponential model
    n - fitted parameter from the Power-Exponential model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_thermal=None,
        failure_stress_nonthermal=None,
        right_censored=None,
        right_censored_stress_thermal=None,
        right_censored_stress_nonthermal=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_voltage]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_thermal) == list:
            failure_stress_thermal = np.array(failure_stress_thermal)
        if type(failure_stress_thermal) != np.ndarray:
            raise TypeError(
                "failure_stress_thermal must be a list or array of thermal failure_stress data"
            )
        if type(failure_stress_nonthermal) == list:
            failure_stress_nonthermal = np.array(failure_stress_nonthermal)
        if type(failure_stress_nonthermal) != np.ndarray:
            raise TypeError(
                "failure_stress_nonthermal must be a list or array of nonthermal failure_stress data"
            )
        if len(failure_stress_thermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_thermal"
            )
        if len(failure_stress_nonthermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_nonthermal"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_thermal) == list:
                right_censored_stress_thermal = np.array(right_censored_stress_thermal)
            if type(right_censored_stress_thermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_thermal must be a list or array of right censored thermal failure_stress data"
                )
            if type(right_censored_stress_nonthermal) == list:
                right_censored_stress_nonthermal = np.array(
                    right_censored_stress_nonthermal
                )
            if type(right_censored_stress_nonthermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_nonthermal must be a list or array of right censored nonthermal failure_stress data"
                )
            if len(right_censored_stress_thermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_thermal"
                )
            if len(right_censored_stress_nonthermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_nonthermal"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power_expon(stress, a, c, n):
            T = stress[0]
            S = stress[1]
            return c * S ** n * np.exp(a / T)

        xdata = np.array(list(zip(failure_stress_thermal, failure_stress_nonthermal))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__power_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, c, n]")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Lognormal
        ALT_fit_1 = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_thermal,
            right_censored_stress=right_censored_stress_thermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Lognormal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_nonthermal,
            right_censored_stress=right_censored_stress_nonthermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_thermal = []
            right_censored_stress_nonthermal = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Lognormal_Power_Exponential.LL),
                guess,
                args=(
                    failures,
                    right_censored,
                    failure_stress_thermal,
                    failure_stress_nonthermal,
                    right_censored_stress_thermal,
                    right_censored_stress_nonthermal,
                ),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Lognormal-Power-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,c,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Lognormal_Power_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_thermal,
                failure_stress_nonthermal,
                right_censored_stress_thermal,
                right_censored_stress_nonthermal,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.n = params[2]
        self.sigma = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_Power_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_thermal)),
            np.array(tuple(failure_stress_nonthermal)),
            np.array(tuple(right_censored_stress_thermal)),
            np.array(tuple(right_censored_stress_nonthermal)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "c", "n", "sigma"],
            "Point Estimate": [self.a, self.c, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            use_life = (
                self.c
                * (use_level_stress[1]) ** self.n
                * np.exp(self.a / use_level_stress[0])
            )
            self.mu_at_use_stress = np.log(use_life)
            self.mean_life = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack(
                [failure_stress_thermal, right_censored_stress_thermal]
            )
            STRESS_2 = np.hstack(
                [failure_stress_nonthermal, right_censored_stress_nonthermal]
            )
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_thermal and right_censored_stress_nonthermal arrays contains pairs of values that are not found in the failure_stress_thermal and failure_stress_nonthermal arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Lognormal_probability_plot is expecting to receive
                class __make_fitted_dist_params_lognormal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma
                        self2.gamma = 0

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * stress_2 ** self.n * np.exp(self.a / stress_1)
                fitted_dist_params = __make_fitted_dist_params_lognormal(
                    mu=np.log(life), sigma=self.sigma
                )
                original_fit = Fit_Lognormal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress_pair), xvals=xvals
                )
                Lognormal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Lognormal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Lognormal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="Thermal stress , Non-thermal stress")
            leg._legend_box.align = "left"
            plt.title("Lognormal-Power-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="lognormal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, c, n, sigma):  # Log PDF
        life = c * S2 ** n * anp.exp(a / S1)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, a, c, n, sigma):  # Log SF
        life = c * S2 ** n * anp.exp(a / S1)
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Lognormal_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_Exponential:
    """
    Fit_Normal_Exponential

    This function will Fit the Normal-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.
    If you are using this model for the Arrhenius equation, a = Ea/K_B. When results are printed Ea will be provided in eV.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Exponential model
    b - fitted parameter from the Exponential model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __expon(stress, a, b):
            return b * np.exp(a / stress)

        if initial_guess is None:
            initial_guess, _ = curve_fit(__expon, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, b]")

        # this gets the common shape for the initial guess using the functions already built into ALT_probability_plot_Normal
        ALT_fit = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Normal_Exponential.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Normal-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Normal_Exponential.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Normal_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "b", "sigma"],
            "Point Estimate": [self.a, self.b, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            self.mu_at_use_stress = self.b * np.exp(self.a / use_level_stress)
            self.mean_life = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Normal_Exponential (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV",
            )
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common shape parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Normal_probability_plot is expecting to receive
                class __make_fitted_dist_params_normal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma

                life = self.b * np.exp(self.a / stress)
                fitted_dist_params = __make_fitted_dist_params_normal(
                    mu=life, sigma=self.sigma
                )
                original_fit = Fit_Normal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Normal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Normal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(xmin, xmax)
            plt.legend(title="Stress")
            plt.title("Normal-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="normal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, b, sigma):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, b, sigma):  # Log SF
        life = b * anp.exp(a / T)
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_Eyring:
    """
    Fit_Normal_Eyring

    This function will Fit the Normal-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Eyring model
    c - fitted parameter from the Eyring model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __eyring(stress, a, c):
            return 1 / stress * np.exp(-(c - a / stress))

        if initial_guess is None:
            initial_guess, _ = curve_fit(__eyring, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, c]")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Normal
        ALT_fit = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Normal_Eyring.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Normal-Eyring model. Try a better initial guess by specifying the parameter initial_guess = [a,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Normal_Eyring.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Normal_Eyring.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and c can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "c", "sigma"],
            "Point Estimate": [self.a, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            self.mu_at_use_stress = (
                1 / use_level_stress * np.exp(-(self.c - self.a / use_level_stress))
            )
            self.mean_life = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Normal_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Normal_probability_plot is expecting to receive
                class __make_fitted_dist_params_normal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma

                life = 1 / stress * np.exp(-(self.c - self.a / stress))
                fitted_dist_params = __make_fitted_dist_params_normal(
                    mu=life, sigma=self.sigma
                )
                original_fit = Fit_Normal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Normal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Normal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(xmin, xmax)
            plt.legend(title="Stress")
            plt.title("Normal-Eyring Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="normal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, c, sigma):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, c, sigma):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Eyring.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_Power:
    """
    Fit_Normal_Power

    This function will Fit the Normal-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power model
    n - fitted parameter from the Power model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power(stress, a, n):
            return a * stress ** n

        if initial_guess is None:
            initial_guess, _ = curve_fit(__power, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, n].")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Normal
        ALT_fit = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress,
            right_censored_stress=right_censored_stress,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape = ALT_fit.common_shape

        guess = [initial_guess[0], initial_guess[1], common_shape]
        all_data = np.hstack([failures, right_censored])
        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Normal_Power.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Normal-Power model. Try a better initial guess by specifying the parameter initial_guess = [a,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Normal_Power.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.n = params[1]
        self.sigma = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Normal_Power.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and n can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "n", "sigma"],
            "Point Estimate": [self.a, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            self.mu_at_use_stress = self.a * use_level_stress ** self.n
            self.mean_life = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str("Results from Fit_Normal_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Normal_probability_plot is expecting to receive
                class __make_fitted_dist_params_normal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma

                life = self.a * stress ** self.n
                fitted_dist_params = __make_fitted_dist_params_normal(
                    mu=life, sigma=self.sigma
                )
                original_fit = Fit_Normal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), xvals=xvals
                )
                Normal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Normal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(xmin, xmax)
            plt.legend(title="Stress")
            plt.title("Normal-Power Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="normal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, n, sigma):  # Log PDF
        life = a * T ** n
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, T, a, n, sigma):  # Log SF
        life = a * T ** n
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Power.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Exponential:
    """
    Fit_Normal_Dual_Exponential

    This function will Fit the Normal-Dual-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature-humidity. It is recommended that you ensure your temperature data are in Kelvin and humidity data range from 0 to 1.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as humidity) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as humidity) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Dual-Exponential model
    b - fitted parameter from the Dual-Exponential model
    c - fitted parameter from the Dual-Exponential model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_humidity]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_1) == list:
            failure_stress_1 = np.array(failure_stress_1)
        if type(failure_stress_1) != np.ndarray:
            raise TypeError(
                "failure_stress_1 must be a list or array of failure_stress data"
            )
        if type(failure_stress_2) == list:
            failure_stress_2 = np.array(failure_stress_2)
        if type(failure_stress_2) != np.ndarray:
            raise TypeError(
                "failure_stress_2 must be a list or array of failure_stress data"
            )
        if len(failure_stress_1) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_1"
            )
        if len(failure_stress_2) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_2"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_1) == list:
                right_censored_stress_1 = np.array(right_censored_stress_1)
            if type(right_censored_stress_1) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_1 must be a list or array of right censored failure_stress data"
                )
            if type(right_censored_stress_2) == list:
                right_censored_stress_2 = np.array(right_censored_stress_2)
            if type(right_censored_stress_2) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_2 must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress_1) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_1"
                )
            if len(right_censored_stress_2) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_2"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __dual_expon(stress, a, b, c):
            T = stress[0]
            H = stress[1]
            return c * np.exp(a / T + b / H)

        xdata = np.array(list(zip(failure_stress_1, failure_stress_2))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__dual_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, b, c].")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Normal
        ALT_fit_1 = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_1,
            right_censored_stress=right_censored_stress_1,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_2,
            right_censored_stress=right_censored_stress_2,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Normal_Dual_Exponential.LL),
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
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Normal-Dual-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Normal_Dual_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.sigma = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Normal_Dual_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.c_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "b", "c", "sigma"],
            "Point Estimate": [self.a, self.b, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            self.mu_at_use_stress = self.c * np.exp(
                self.a / use_level_stress[0] + self.b / use_level_stress[1]
            )
            self.mean_life = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Normal_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack([failure_stress_1, right_censored_stress_1])
            STRESS_2 = np.hstack([failure_stress_2, right_censored_stress_2])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_1 and right_censored_stress_2 arrays contains pairs of values that are not found in the failure_stress_1 and failure_stress_2 arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Normal_probability_plot is expecting to receive
                class __make_fitted_dist_params_normal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * np.exp(self.a / stress_1 + self.b / stress_2)
                fitted_dist_params = __make_fitted_dist_params_normal(
                    mu=life, sigma=self.sigma
                )
                original_fit = Fit_Normal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress_pair), xvals=xvals
                )
                Normal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Normal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(xmin, xmax)
            leg = plt.legend(title="     Stress 1 , Stress 2")
            leg._legend_box.align = "left"
            plt.title("Normal-Dual-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="normal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, b, c, sigma):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, a, b, c, sigma):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_Power_Exponential:
    """
    Fit_Normal_Power_Exponential

    This function will Fit the Normal-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_thermal, stress_nonthermal]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power-Exponential model
    c - fitted parameter from the Power-Exponential model
    n - fitted parameter from the Power-Exponential model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_thermal=None,
        failure_stress_nonthermal=None,
        right_censored=None,
        right_censored_stress_thermal=None,
        right_censored_stress_nonthermal=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 2:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_voltage]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_thermal) == list:
            failure_stress_thermal = np.array(failure_stress_thermal)
        if type(failure_stress_thermal) != np.ndarray:
            raise TypeError(
                "failure_stress_thermal must be a list or array of thermal failure_stress data"
            )
        if type(failure_stress_nonthermal) == list:
            failure_stress_nonthermal = np.array(failure_stress_nonthermal)
        if type(failure_stress_nonthermal) != np.ndarray:
            raise TypeError(
                "failure_stress_nonthermal must be a list or array of nonthermal failure_stress data"
            )
        if len(failure_stress_thermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_thermal"
            )
        if len(failure_stress_nonthermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_nonthermal"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_thermal) == list:
                right_censored_stress_thermal = np.array(right_censored_stress_thermal)
            if type(right_censored_stress_thermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_thermal must be a list or array of right censored thermal failure_stress data"
                )
            if type(right_censored_stress_nonthermal) == list:
                right_censored_stress_nonthermal = np.array(
                    right_censored_stress_nonthermal
                )
            if type(right_censored_stress_nonthermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_nonthermal must be a list or array of right censored nonthermal failure_stress data"
                )
            if len(right_censored_stress_thermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_thermal"
                )
            if len(right_censored_stress_nonthermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_nonthermal"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power_expon(stress, a, c, n):
            T = stress[0]
            S = stress[1]
            return c * S ** n * np.exp(a / T)

        xdata = np.array(list(zip(failure_stress_thermal, failure_stress_nonthermal))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__power_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, c, n].")

        # this gets the common sigma for the initial guess using the functions already built into ALT_probability_plot_Normal
        ALT_fit_1 = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_thermal,
            right_censored_stress=right_censored_stress_thermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_1 = ALT_fit_1.common_shape
        ALT_fit_2 = ALT_probability_plot_Normal(
            failures=failures,
            right_censored=right_censored,
            failure_stress=failure_stress_nonthermal,
            right_censored_stress=right_censored_stress_nonthermal,
            print_results=False,
            show_plot=False,
            common_shape_method="average",
        )
        common_shape_2 = ALT_fit_2.common_shape
        common_shape = np.average([common_shape_1, common_shape_2])

        guess = [initial_guess[0], initial_guess[1], initial_guess[2], common_shape]
        all_data = np.hstack([failures, right_censored])
        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_thermal = []
            right_censored_stress_nonthermal = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Normal_Power_Exponential.LL),
                guess,
                args=(
                    failures,
                    right_censored,
                    failure_stress_thermal,
                    failure_stress_nonthermal,
                    right_censored_stress_thermal,
                    right_censored_stress_nonthermal,
                ),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Normal-Power-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,c,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2], params[3]]
            LL2 = 2 * Fit_Normal_Power_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_thermal,
                failure_stress_nonthermal,
                right_censored_stress_thermal,
                right_censored_stress_nonthermal,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.n = params[2]
        self.sigma = params[3]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Normal_Power_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_thermal)),
            np.array(tuple(failure_stress_nonthermal)),
            np.array(tuple(right_censored_stress_thermal)),
            np.array(tuple(right_censored_stress_nonthermal)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        self.sigma_upper = self.sigma * (
            np.exp(Z * (self.sigma_SE / self.sigma))
        )  # a and b can be +- but sigma is strictly + so the formulas here are different for sigma
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        Data = {
            "Parameter": ["a", "c", "n", "sigma"],
            "Point Estimate": [self.a, self.c, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.sigma_upper],
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

        if use_level_stress is not None:
            self.mu_at_use_stress = (
                self.c
                * (use_level_stress[1]) ** self.n
                * np.exp(self.a / use_level_stress[0])
            )
            self.mean_life = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Normal_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack(
                [failure_stress_thermal, right_censored_stress_thermal]
            )
            STRESS_2 = np.hstack(
                [failure_stress_nonthermal, right_censored_stress_nonthermal]
            )
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_thermal and right_censored_stress_nonthermal arrays contains pairs of values that are not found in the failure_stress_thermal and failure_stress_nonthermal arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Normal_probability_plot is expecting to receive
                class __make_fitted_dist_params_normal:
                    def __init__(self2, mu, sigma):
                        self2.mu = mu
                        self2.sigma = sigma

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * stress_2 ** self.n * np.exp(self.a / stress_1)
                fitted_dist_params = __make_fitted_dist_params_normal(
                    mu=life, sigma=self.sigma
                )
                original_fit = Fit_Normal_2P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress_pair), xvals=xvals
                )
                Normal_probability_plot(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Normal_Distribution(
                            mu=self.mu_at_use_stress, sigma=self.sigma
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(xmin, xmax)
            leg = plt.legend(title="Thermal stress , Non-thermal stress")
            leg._legend_box.align = "left"
            plt.title("Normal-Power-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="normal", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, c, n, sigma):  # Log PDF
        life = c * S2 ** n * anp.exp(a / S1)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, a, c, n, sigma):  # Log SF
        life = c * S2 ** n * anp.exp(a / S1)
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_Exponential:
    """
    Fit_Exponential_Exponential

    This function will Fit the Exponential-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.
    If you are using this model for the Arrhenius equation, a = Ea/K_B. When results are printed Ea will be provided in eV.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Exponential model
    b - fitted parameter from the Exponential model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 1:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failure to calculate Exponential parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __expon(stress, a, b):
            return b * np.exp(a / stress)

        if initial_guess is None:
            initial_guess, _ = curve_fit(__expon, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, b].")

        guess = [initial_guess[0], initial_guess[1]]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Exponential_Exponential.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Exponential-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1]]
            LL2 = 2 * Fit_Exponential_Exponential.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Exponential_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)

        Data = {
            "Parameter": ["a", "b"],
            "Point Estimate": [self.a, self.b],
            "Standard Error": [self.a_SE, self.b_SE],
            "Lower CI": [self.a_lower, self.b_lower],
            "Upper CI": [self.a_upper, self.b_upper],
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

        if use_level_stress is not None:
            use_life = self.b * np.exp(self.a / use_level_stress)
            self.Lambda_at_use_stress = 1 / use_life
            self.mean_life = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Exponential_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV",
            )
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Exponential_probability_plot_Weibull_Scale is expecting to receive
                class __make_fitted_dist_params_expon:
                    def __init__(self2, Lambda):
                        self2.Lambda = Lambda
                        self2.gamma = 0
                        self2.Lambda_SE = None

                life = self.b * np.exp(self.a / stress)
                fitted_dist_params = __make_fitted_dist_params_expon(Lambda=1 / life)
                original_fit = Fit_Exponential_1P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Exponential_probability_plot_Weibull_Scale(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Exponential_Distribution(Lambda=self.Lambda_at_use_stress).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Exponential_Distribution(
                            Lambda=self.Lambda_at_use_stress
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Exponential-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, b):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, b):  # Log SF
        life = b * anp.exp(a / T)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_Exponential.logf(
            t_f, T_f, params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_Exponential.logR(
            t_rc, T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_Eyring:
    """
    Fit_Exponential_Eyring

    This function will Fit the Exponential-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Eyring model
    c - fitted parameter from the Eyring model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 1:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failure to calculate Exponential parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __eyring(stress, a, c):
            return 1 / stress * np.exp(-(c - a / stress))

        if initial_guess is None:
            initial_guess, _ = curve_fit(__eyring, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, c].")

        guess = [initial_guess[0], initial_guess[1]]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Exponential_Eyring.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Exponential-Eyring model. Try a better initial guess by specifying the parameter initial_guess = [a,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1]]
            LL2 = 2 * Fit_Exponential_Eyring.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Exponential_Eyring.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)

        Data = {
            "Parameter": ["a", "c"],
            "Point Estimate": [self.a, self.c],
            "Standard Error": [self.a_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.c_upper],
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

        if use_level_stress is not None:
            use_life = (
                1 / use_level_stress * np.exp(-(self.c - self.a / use_level_stress))
            )
            self.Lambda_at_use_stress = 1 / use_life
            self.mean_life = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Exponential_Eyring (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Exponential_probability_plot_Weibull_Scale is expecting to receive
                class __make_fitted_dist_params_expon:
                    def __init__(self2, Lambda):
                        self2.Lambda = Lambda
                        self2.gamma = 0
                        self2.Lambda_SE = None

                life = 1 / stress * np.exp(-(self.c - self.a / stress))
                fitted_dist_params = __make_fitted_dist_params_expon(Lambda=1 / life)
                original_fit = Fit_Exponential_1P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Exponential_probability_plot_Weibull_Scale(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Exponential_Distribution(Lambda=self.Lambda_at_use_stress).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Exponential_Distribution(
                            Lambda=self.Lambda_at_use_stress
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Exponential-Eyring Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, c):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, c):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_Eyring.logf(
            t_f, T_f, params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_Eyring.logR(
            t_rc, T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_Power:
    """
    Fit_Exponential_Power

    This function will Fit the Exponential-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power model
    n - fitted parameter from the Power model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress=None,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 1:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failure to calculate Exponential parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress) == list:
            failure_stress = np.array(failure_stress)
        if type(failure_stress) != np.ndarray:
            raise TypeError(
                "failure_stress must be a list or array of failure_stress data"
            )
        if len(failure_stress) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress) == list:
                right_censored_stress = np.array(right_censored_stress)
            if type(right_censored_stress) != np.ndarray:
                raise TypeError(
                    "right_censored_stress must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power(stress, a, n):
            return a * stress ** n

        if initial_guess is None:
            initial_guess, _ = curve_fit(__power, failure_stress, failures)
        if len(initial_guess) != 2:
            raise ValueError("initial_guess must have 2 elements: [a, n].")

        guess = [initial_guess[0], initial_guess[1]]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Exponential_Power.LL),
                guess,
                args=(failures, right_censored, failure_stress, right_censored_stress),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Exponential-Power model. Try a better initial guess by specifying the parameter initial_guess = [a,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1]]
            LL2 = 2 * Fit_Exponential_Power.LL(
                guess, failures, right_censored, failure_stress, right_censored_stress
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.n = params[1]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Exponential_Power.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)

        Data = {
            "Parameter": ["a", "n"],
            "Point Estimate": [self.a, self.n],
            "Standard Error": [self.a_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.n_upper],
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

        if use_level_stress is not None:
            use_life = self.a * use_level_stress ** self.n
            self.Lambda_at_use_stress = 1 / use_life
            self.mean_life = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Exponential_Power (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stress of",
                    use_level_stress,
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {"times": TIMES, "stress": STRESS, "cens_codes": CENS_CODES}
            df = pd.DataFrame(data, columns=["times", "stress", "cens_codes"])
            df_sorted = df.sort_values(by=["cens_codes", "stress", "times"])
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress
            for i, stress in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress"] == stress]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df["stress"] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Exponential_probability_plot_Weibull_Scale is expecting to receive
                class __make_fitted_dist_params_expon:
                    def __init__(self2, Lambda):
                        self2.Lambda = Lambda
                        self2.gamma = 0
                        self2.Lambda_SE = None

                life = self.a * stress ** self.n
                fitted_dist_params = __make_fitted_dist_params_expon(Lambda=1 / life)
                original_fit = Fit_Exponential_1P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i], label=str(stress), plot_CI=False, xvals=xvals
                )
                Exponential_probability_plot_Weibull_Scale(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(str(use_level_stress) + " (use level)")
                Exponential_Distribution(Lambda=self.Lambda_at_use_stress).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Exponential_Distribution(
                            Lambda=self.Lambda_at_use_stress
                        ).quantile(max(ALT_fit.y_array)),
                        ALT_fit.x_array,
                    ]
                )
            else:
                x_array = ALT_fit.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            plt.legend(title="Stress")
            plt.title("Exponential-Power Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, T, a, n):  # Log PDF
        life = a * T ** n
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, n):  # Log SF
        life = a * T ** n
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_Power.logf(
            t_f, T_f, params[0], params[1]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_Power.logR(
            t_rc, T_rc, params[0], params[1]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Exponential:
    """
    Fit_Exponential_Dual_Exponential

    This function will Fit the Exponential-Dual-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature-humidity. It is recommended that you ensure your temperature data are in Kelvin and humidity data range from 0 to 1.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as humidity) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as humidity) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,b,c]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Dual-Exponential model
    b - fitted parameter from the Dual-Exponential model
    c - fitted parameter from the Dual-Exponential model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    b_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    b_upper - the upper CI estimate of the parameter
    b_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 1:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failure to calculate Exponential parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_humidity]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_1) == list:
            failure_stress_1 = np.array(failure_stress_1)
        if type(failure_stress_1) != np.ndarray:
            raise TypeError(
                "failure_stress_1 must be a list or array of failure_stress data"
            )
        if type(failure_stress_2) == list:
            failure_stress_2 = np.array(failure_stress_2)
        if type(failure_stress_2) != np.ndarray:
            raise TypeError(
                "failure_stress_2 must be a list or array of failure_stress data"
            )
        if len(failure_stress_1) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_1"
            )
        if len(failure_stress_2) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_2"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_1) == list:
                right_censored_stress_1 = np.array(right_censored_stress_1)
            if type(right_censored_stress_1) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_1 must be a list or array of right censored failure_stress data"
                )
            if type(right_censored_stress_2) == list:
                right_censored_stress_2 = np.array(right_censored_stress_2)
            if type(right_censored_stress_2) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_2 must be a list or array of right censored failure_stress data"
                )
            if len(right_censored_stress_1) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_1"
                )
            if len(right_censored_stress_2) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_2"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __dual_expon(stress, a, b, c):
            T = stress[0]
            H = stress[1]
            return c * np.exp(a / T + b / H)

        xdata = np.array(list(zip(failure_stress_1, failure_stress_2))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__dual_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, b, c].")

        guess = [initial_guess[0], initial_guess[1], initial_guess[2]]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Exponential_Dual_Exponential.LL),
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
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Exponential-Dual-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,b,c]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Exponential_Dual_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Exponential_Dual_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.c_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)

        Data = {
            "Parameter": ["a", "b", "c"],
            "Point Estimate": [self.a, self.b, self.c],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper],
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

        if use_level_stress is not None:
            use_life = self.c * np.exp(
                self.a / use_level_stress[0] + self.b / use_level_stress[1]
            )
            self.Lambda_at_use_stress = 1 / use_life
            self.mean_life = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Exponential_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack([failure_stress_1, right_censored_stress_1])
            STRESS_2 = np.hstack([failure_stress_2, right_censored_stress_2])
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_1 and right_censored_stress_2 arrays contains pairs of values that are not found in the failure_stress_1 and failure_stress_2 arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Exponential_probability_plot_Weibull_Scale is expecting to receive
                class __make_fitted_dist_params_expon:
                    def __init__(self2, Lambda):
                        self2.Lambda = Lambda
                        self2.gamma = 0
                        self2.Lambda_SE = None

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * np.exp(self.a / stress_1 + self.b / stress_2)
                fitted_dist_params = __make_fitted_dist_params_expon(Lambda=1 / life)
                original_fit = Fit_Exponential_1P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i],
                    label=str(stress_pair),
                    plot_CI=False,
                    xvals=xvals,
                )
                Exponential_probability_plot_Weibull_Scale(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Exponential_Distribution(Lambda=self.Lambda_at_use_stress).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Exponential_Distribution(
                            Lambda=self.Lambda_at_use_stress
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="     Stress 1 , Stress 2")
            leg._legend_box.align = "left"
            plt.title("Exponential-Dual-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, b, c):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, a, b, c):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Exponential_Power_Exponential:
    """
    Fit_Exponential_Power_Exponential

    This function will Fit the Exponential-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_thermal - an array or list of the corresponding thermal stress (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_nonthermal - an array or list of the corresponding non-thermal stress (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_thermal, stress_nonthermal]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    initial_guess - starting values for [a,c,n]. Default is calculated using a curvefit to failure data. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

    Outputs:
    a - fitted parameter from the Power-Exponential model
    c - fitted parameter from the Power-Exponential model
    n - fitted parameter from the Power-Exponential model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    a_SE - the standard error (sqrt(variance)) of the parameter
    c_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    a_upper - the upper CI estimate of the parameter
    a_lower - the lower CI estimate of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures=None,
        failure_stress_thermal=None,
        failure_stress_nonthermal=None,
        right_censored=None,
        right_censored_stress_thermal=None,
        right_censored_stress_nonthermal=None,
        use_level_stress=None,
        show_plot=True,
        print_results=True,
        CI=0.95,
        initial_guess=None,
    ):
        if failures is None or len(failures) < 1:
            raise ValueError(
                "Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failure to calculate Exponential parameters."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )
        if use_level_stress is not None:
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be a 2 element list or array. eg. [use_temperature, use_voltage]"
                )
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError("failures must be a list or array of failure data")
        if type(failure_stress_thermal) == list:
            failure_stress_thermal = np.array(failure_stress_thermal)
        if type(failure_stress_thermal) != np.ndarray:
            raise TypeError(
                "failure_stress_thermal must be a list or array of thermal failure_stress data"
            )
        if type(failure_stress_nonthermal) == list:
            failure_stress_nonthermal = np.array(failure_stress_nonthermal)
        if type(failure_stress_nonthermal) != np.ndarray:
            raise TypeError(
                "failure_stress_nonthermal must be a list or array of nonthermal failure_stress data"
            )
        if len(failure_stress_thermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_thermal"
            )
        if len(failure_stress_nonthermal) != len(failures):
            raise ValueError(
                "The number of elements in failures does not match the number of elements in failure_stress_nonthermal"
            )
        if right_censored is not None:
            if type(right_censored) == list:
                right_censored = np.array(right_censored)
            if type(right_censored) != np.ndarray:
                raise TypeError(
                    "right_censored must be a list or array of right censored failure data"
                )
            if type(right_censored_stress_thermal) == list:
                right_censored_stress_thermal = np.array(right_censored_stress_thermal)
            if type(right_censored_stress_thermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_thermal must be a list or array of right censored thermal failure_stress data"
                )
            if type(right_censored_stress_nonthermal) == list:
                right_censored_stress_nonthermal = np.array(
                    right_censored_stress_nonthermal
                )
            if type(right_censored_stress_nonthermal) != np.ndarray:
                raise TypeError(
                    "right_censored_stress_nonthermal must be a list or array of right censored nonthermal failure_stress data"
                )
            if len(right_censored_stress_thermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_thermal"
                )
            if len(right_censored_stress_nonthermal) != len(right_censored):
                raise ValueError(
                    "The number of elements in right_censored does not match the number of elements in right_censored_stress_nonthermal"
                )

        # obtain a rough estimate for the initial guess using curvefit of failure data
        def __power_expon(stress, a, c, n):
            T = stress[0]
            S = stress[1]
            return c * S ** n * np.exp(a / T)

        xdata = np.array(list(zip(failure_stress_thermal, failure_stress_nonthermal))).T
        if initial_guess is None:
            initial_guess, _ = curve_fit(__power_expon, xdata, failures)
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must have 3 elements: [a, c, n].")

        guess = [initial_guess[0], initial_guess[1], initial_guess[2]]
        all_data = np.hstack([failures, right_censored])
        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin - 1, xmax + 1, 100)
        k = len(guess)
        n = len(all_data)

        # new format for right_censored is required of the LL function
        if right_censored is None:
            right_censored = []
            right_censored_stress_thermal = []
            right_censored_stress_nonthermal = []
        warnings.filterwarnings(
            "ignore"
        )  # necessary to suppress the warning about the jacobian when using the nelder-mead optimizer
        # this additional loop is used to make a bad initial guess much better. It works differently to changing tol within the minimize function. It will only run 2 or 3 times until the BIC is no longer changing
        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        tols = [1e-1, 1e-3, 1e-6]
        while delta_BIC > 0.001:
            if runs < 3:
                tol = tols[runs]
            else:
                tol = 1e-6
            runs += 1
            result = minimize(
                value_and_grad(Fit_Exponential_Power_Exponential.LL),
                guess,
                args=(
                    failures,
                    right_censored,
                    failure_stress_thermal,
                    failure_stress_nonthermal,
                    right_censored_stress_thermal,
                    right_censored_stress_nonthermal,
                ),
                jac=True,
                tol=tol,
                method="nelder-mead",
                options={"maxiter": 5000},
            )
            if result.success is False:
                raise RuntimeError(
                    "Fitting using Autograd FAILED for the Exponential-Power-Exponential model. Try a better initial guess by specifying the parameter initial_guess = [a,c,n]. Alternatively, try another life-stress model."
                )
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Exponential_Power_Exponential.LL(
                guess,
                failures,
                right_censored,
                failure_stress_thermal,
                failure_stress_nonthermal,
                right_censored_stress_thermal,
                right_censored_stress_nonthermal,
            )
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        self.a = params[0]
        self.c = params[1]
        self.n = params[2]
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = BIC_array[-1]

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Exponential_Power_Exponential.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_thermal)),
            np.array(tuple(failure_stress_nonthermal)),
            np.array(tuple(right_censored_stress_thermal)),
            np.array(tuple(right_censored_stress_nonthermal)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)

        Data = {
            "Parameter": ["a", "c", "n"],
            "Point Estimate": [self.a, self.c, self.n],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper],
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

        if use_level_stress is not None:
            use_life = (
                self.c
                * (use_level_stress[1]) ** self.n
                * np.exp(self.a / use_level_stress[0])
            )
            self.Lambda_at_use_stress = 1 / use_life
            self.mean_life = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            ).mean

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Results from Fit_Exponential_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")
            if use_level_stress is not None:
                print(
                    "At the use level stresses of",
                    use_level_stress[0],
                    "and",
                    use_level_stress[1],
                    ", the mean life is",
                    round(self.mean_life, 5),
                )

        if show_plot is True:
            TIMES = np.hstack([failures, right_censored])
            STRESS_1 = np.hstack(
                [failure_stress_thermal, right_censored_stress_thermal]
            )
            STRESS_2 = np.hstack(
                [failure_stress_nonthermal, right_censored_stress_nonthermal]
            )
            CENS_CODES = np.hstack(
                [np.ones_like(failures), np.zeros_like(right_censored)]
            )

            data = {
                "times": TIMES,
                "stress_1": STRESS_1,
                "stress_2": STRESS_2,
                "cens_codes": CENS_CODES,
            }
            df = pd.DataFrame(
                data, columns=["times", "stress_1", "stress_2", "cens_codes"]
            )
            df["stress_pair"] = (
                df["stress_1"].map(str) + " , " + df["stress_2"].map(str)
            )  # this combines each stress to make a "stress pair" which is treated as a unique stress combination
            df_sorted = df.sort_values(
                by=["cens_codes", "stress_1", "stress_2", "times"]
            )
            is_failure = df_sorted["cens_codes"] == 1
            is_right_cens = df_sorted["cens_codes"] == 0
            f_df = df_sorted[is_failure]
            rc_df = df_sorted[is_right_cens]
            unique_stresses_f = f_df.stress_pair.unique()
            if right_censored is not []:
                unique_stresses_rc = rc_df.stress_pair.unique()
                for (
                    item
                ) in (
                    unique_stresses_rc
                ):  # check that there are no unique right_censored stresses that are not also in failure stresses
                    if item not in unique_stresses_f:
                        raise ValueError(
                            "The right_censored_stress_thermal and right_censored_stress_nonthermal arrays contains pairs of values that are not found in the failure_stress_thermal and failure_stress_nonthermal arrays. This is equivalent to trying to fit a distribution to only censored data and cannot be done."
                        )
            # within this loop, each list of failures and right censored values will be unpacked for each unique stress
            for i, stress_pair in enumerate(unique_stresses_f):
                failure_current_stress_df = f_df[f_df["stress_pair"] == stress_pair]
                FAILURES = failure_current_stress_df["times"].values
                if right_censored is not []:
                    if stress_pair in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[
                            rc_df["stress_pair"] == stress_pair
                        ]
                        RIGHT_CENSORED = right_cens_current_stress_df["times"].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None

                # this is necessary to create the correct class structure that Exponential_probability_plot_Weibull_Scale is expecting to receive
                class __make_fitted_dist_params_expon:
                    def __init__(self2, Lambda):
                        self2.Lambda = Lambda
                        self2.gamma = 0
                        self2.Lambda_SE = None

                pair = stress_pair.split(" , ")
                stress_1 = float(pair[0])
                stress_2 = float(pair[1])
                life = self.c * stress_2 ** self.n * np.exp(self.a / stress_1)
                fitted_dist_params = __make_fitted_dist_params_expon(Lambda=1 / life)
                original_fit = Fit_Exponential_1P(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    show_probability_plot=False,
                    print_results=False,
                )
                original_fit.distribution.CDF(
                    color=color_list[i],
                    label=str(stress_pair),
                    plot_CI=False,
                    xvals=xvals,
                )
                Exponential_probability_plot_Weibull_Scale(
                    failures=FAILURES,
                    right_censored=RIGHT_CENSORED,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_list[i],
                    linestyle="--",
                    label="",
                )
            if use_level_stress is not None:
                use_label_str = str(
                    str(float(use_level_stress[0]))
                    + " , "
                    + str(float(use_level_stress[1]))
                    + " (use level)"
                )
                Exponential_Distribution(Lambda=self.Lambda_at_use_stress).CDF(
                    label=use_label_str, color=color_list[i + 1], linestyle="--"
                )
                x_array = np.hstack(
                    [
                        Exponential_Distribution(
                            Lambda=self.Lambda_at_use_stress
                        ).quantile(max(ALT_fit_1.y_array)),
                        ALT_fit_1.x_array,
                    ]
                )
            else:
                x_array = ALT_fit_1.x_array
            plt.xlim(10 ** xmin, 10 ** xmax)
            leg = plt.legend(title="Thermal stress , Non-thermal stress")
            leg._legend_box.align = "left"
            plt.title("Exponential-Power-Exponential Model")
            probability_plot_xyticks()
            probability_plot_xylims(
                x=x_array, y=ALT_fit_1.y_array, dist="weibull", spacing=0.1
            )
            plt.tight_layout()

    @staticmethod
    def logf(t, S1, S2, a, c, n):  # Log PDF
        life = c * S2 ** n * anp.exp(a / S1)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, a, c, n):  # Log SF
        life = c * S2 ** n * anp.exp(a / S1)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Exponential_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()  # failure times
        LL_rc += Fit_Exponential_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)
