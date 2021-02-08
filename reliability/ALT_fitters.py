import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import autograd.numpy as anp
from autograd.scipy.special import erf
from autograd.differential_operators import hessian
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import (
    Weibull_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
    Exponential_Distribution,
)
from reliability.Utils import (
    colorprint,
    round_to_decimals,
    ALT_fitters_input_checking,
    ALT_least_squares,
    ALT_MLE_optimisation,
    life_stress_plot,
    ALT_prob_plot,
)

pd.set_option("display.width", 200)  # prevents wrapping after default 80 characters
pd.set_option("display.max_columns", 9)  # shows the dataframe without ... truncation
shape_change_threshold = 0.5


class Fit_Weibull_Exponential:
    """
    Fit_Weibull_Exponential

    This function will Fit the Weibull-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.
    If you are using this model for the Arrhenius equation, a = Ea/K_B. When results are printed Ea will be provided in eV.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Exponential",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Exponential", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1], common_beta]  # a, b, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Exponential",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.beta]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b is strictly positive
        self.b_upper = self.b * (np.exp(Z * (self.b_SE / self.b)))
        self.b_lower = self.b * (np.exp(-Z * (self.b_SE / self.b)))
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "beta"],
            "Point Estimate": [self.a, self.b, self.beta],
            "Standard Error": [self.a_SE, self.b_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.b * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Weibull_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV\n",
            )

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Weibull_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Weibull_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Eyring:
    """
    Fit_Weibull_Eyring

    This function will Fit the Weibull-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Eyring",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Eyring.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Eyring", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1], common_beta]  # a, c, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Eyring",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.beta]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c can be positive or negative
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "beta"],
            "Point Estimate": [self.a, self.c, self.beta],
            "Standard Error": [self.a_SE, self.c_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return 1 / S1 * np.exp(-(self.c - self.a / S1))

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Eyring",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Eyring",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Weibull_Eyring.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc += Fit_Weibull_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Power:
    """
    Fit_Weibull_Power

    This function will Fit the Weibull-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Power",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1], common_beta]  # a, n, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.n = MLE_results.n
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n, self.beta]
        hessian_matrix = hessian(LL_func)(
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
        # a is strictly positive
        self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
        self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "n", "beta"],
            "Point Estimate": [self.a, self.n, self.beta],
            "Standard Error": [self.a_SE, self.n_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.a * S1 ** self.n

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Weibull_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc += Fit_Weibull_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
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
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Dual-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Dual_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_beta,
        ]  # a, b, c, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Exponential",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.c = MLE_results.c
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c, self.beta]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b can be positive or negative
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c", "beta"],
            "Point Estimate": [self.a, self.b, self.c, self.beta],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * np.exp(self.a / S1 + self.b / S2)

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups)) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Weibull_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Dual-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Dual-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Weibull_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc += Fit_Weibull_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Power_Exponential:
    """
    Fit_Weibull_Power_Exponential

    This function will Fit the Weibull-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (non-thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (non-thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC).
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level.
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided).
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided).
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Power-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Power_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_beta,
        ]  # a, c, n, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power-Exponential",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n, self.beta]
        hessian_matrix = hessian(LL_func)(
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
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_SE = abs(covariance_matrix[3][3]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n", "beta"],
            "Point Estimate": [self.a, self.c, self.n, self.beta],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.beta_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.beta_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S2 ** self.n) * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups)) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Weibull_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Power-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Power-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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


class Fit_Weibull_Dual_Power:
    """
    Fit_Weibull_Dual_Power

    This function will Fit the Weibull-Dual-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with two non-thermal stresses such as voltage and load.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual-Power model
    n - fitted parameter from the Dual-Power model
    m - fitted parameter from the Dual-Power model
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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Weibull",
            life_stress_model="Dual-Power",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Weibull_Dual_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Power",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        if len(betas) > 0:
            common_beta = float(np.average(betas))
        else:
            common_beta = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_beta,
        ]  # c, n, m, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Power",
            dist="Weibull",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.m = MLE_results.m
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.n, self.m, self.beta]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.c_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.m_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "n", "m", "beta"],
            "Point Estimate": [self.c, self.n, self.m, self.beta],
            "Standard Error": [self.c_SE, self.n_SE, self.m_SE, self.beta_SE],
            "Lower CI": [self.c_lower, self.n_lower, self.m_lower, self.beta_lower],
            "Upper CI": [self.c_upper, self.n_upper, self.m_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S1 ** self.n) * (S2 ** self.m)

        # use level stress calculations
        if use_level_stress is not None:
            self.alpha_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Weibull_Distribution(
                alpha=self.alpha_at_use_stress, beta=self.beta
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups)) * self.beta
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original alpha": alphas_for_change_df,
                "original beta": betas_for_change_df,
                "new alpha": new_alphas,
                "common beta": common_betas,
                "beta change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Weibull_Dual_Power (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Weibull distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Weibull",
                model="Dual-Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.beta,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Weibull",
                model="Dual-Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

    @staticmethod
    def logf(t, S1, S2, c, n, m, beta):  # Log PDF
        life = c * (S1 ** n) * (S2 ** m)
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, S1, S2, c, n, m, beta):  # Log SF
        life = c * (S1 ** n) * (S2 ** m)
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        # failure times
        LL_f += Fit_Weibull_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc += Fit_Weibull_Dual_Power.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
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
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Exponential",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Exponential", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, b, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Exponential",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b is strictly positive
        self.b_upper = self.b * (np.exp(Z * (self.b_SE / self.b)))
        self.b_lower = self.b * (np.exp(-Z * (self.b_SE / self.b)))
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "sigma"],
            "Point Estimate": [self.a, self.b, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.b * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(life_func(S1=use_level_stress))
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress)))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV\n",
            )

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Lognormal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Lognormal_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_Eyring:
    """
    Fit_Lognormal_Eyring

    This function will Fit the Lognormal-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Eyring",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Eyring.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Eyring", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Eyring",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c can be positive or negative
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "sigma"],
            "Point Estimate": [self.a, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return 1 / S1 * np.exp(-(self.c - self.a / S1))

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(life_func(S1=use_level_stress))
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress)))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Eyring",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Eyring",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Lognormal_Eyring.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Lognormal_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_Power:
    """
    Fit_Lognormal_Power

    This function will Fit the Lognormal-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Power",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a is strictly positive
        self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
        self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "n", "sigma"],
            "Point Estimate": [self.a, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.a * S1 ** self.n

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(life_func(S1=use_level_stress))
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress)))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Lognormal_Power.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Lognormal_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
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
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Dual-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Dual_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, b, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Exponential",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b can be positive or negative
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c", "sigma"],
            "Point Estimate": [self.a, self.b, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * np.exp(self.a / S1 + self.b / S2)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(
                life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            )
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress[0], S2=stress[1])))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Dual-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Dual-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
    failure_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (non-thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (non-thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC).
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level.
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided).
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided).
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Power-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Power_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, c, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power-Exponential",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n", "sigma"],
            "Point Estimate": [self.a, self.c, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S2 ** self.n) * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(
                life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            )
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress[0], S2=stress[1])))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Power-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Power-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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


class Fit_Lognormal_Dual_Power:
    """
    Fit_Lognormal_Dual_Power

    This function will Fit the Lognormal-Dual-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with two non-thermal stresses such as voltage and load.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual-Power model
    n - fitted parameter from the Dual-Power model
    m - fitted parameter from the Dual-Power model
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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Lognormal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Lognormal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Lognormal",
            life_stress_model="Dual-Power",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Lognormal_Dual_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Power",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Lognormal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # c, n, m, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Power",
            dist="Lognormal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.m = MLE_results.m
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.n, self.m, self.sigma]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.c_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.m_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "n", "m", "sigma"],
            "Point Estimate": [self.c, self.n, self.m, self.sigma],
            "Standard Error": [self.c_SE, self.n_SE, self.m_SE, self.sigma_SE],
            "Lower CI": [self.c_lower, self.n_lower, self.m_lower, self.sigma_lower],
            "Upper CI": [self.c_upper, self.n_upper, self.m_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S1 ** self.n) * (S2 ** self.m)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = np.log(
                life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            )
            self.distribution_at_use_stress = Lognormal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(np.log(life_func(S1=stress[0], S2=stress[1])))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Lognormal_Dual_Power ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Lognormal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Lognormal",
                model="Dual-Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Lognormal",
                model="Dual-Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

    @staticmethod
    def logf(t, S1, S2, c, n, m, sigma):  # Log PDF
        life = c * (S1 ** n) * (S2 ** m)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, c, n, m, sigma):  # Log SF
        life = c * (S1 ** n) * (S2 ** m)
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        # failure times
        LL_f += Fit_Lognormal_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc += Fit_Lognormal_Dual_Power.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
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
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.


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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Exponential",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Exponential", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, b, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b is strictly positive
        self.b_upper = self.b * (np.exp(Z * (self.b_SE / self.b)))
        self.b_lower = self.b * (np.exp(-Z * (self.b_SE / self.b)))
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "sigma"],
            "Point Estimate": [self.a, self.b, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.b * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Normal_Exponential (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV\n",
            )

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Normal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Normal_Exponential.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Eyring:
    """
    Fit_Normal_Eyring

    This function will Fit the Normal-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Eyring",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Eyring.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Eyring", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Eyring",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c can be positive or negative
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "sigma"],
            "Point Estimate": [self.a, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return 1 / S1 * np.exp(-(self.c - self.a / S1))

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Normal_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Eyring",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Eyring",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Normal_Eyring.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc += Fit_Normal_Eyring.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Power:
    """
    Fit_Normal_Power

    This function will Fit the Normal-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Power",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a is strictly positive
        self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
        self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "n", "sigma"],
            "Point Estimate": [self.a, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.a * S1 ** self.n

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Normal_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Normal_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc += Fit_Normal_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
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
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Dual-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Dual_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, b, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b can be positive or negative
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c", "sigma"],
            "Point Estimate": [self.a, self.b, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * np.exp(self.a / S1 + self.b / S2)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Normal_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Dual-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Dual-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Normal_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc += Fit_Normal_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Power_Exponential:
    """
    Fit_Normal_Power_Exponential

    This function will Fit the Normal-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (non-thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (non-thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC).
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level.
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided).
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided).
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Power-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Power_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, c, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power-Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(
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
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n", "sigma"],
            "Point Estimate": [self.a, self.c, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S2 ** self.n) * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Normal_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Power-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Power-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Normal_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc += Fit_Normal_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Power:
    """
    Fit_Normal_Dual_Power

    This function will Fit the Normal-Dual-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with two non-thermal stresses such as voltage and load.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual-Power model
    n - fitted parameter from the Dual-Power model
    m - fitted parameter from the Dual-Power model
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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters (mu and sigma) at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    mu_at_use_stress - the equivalent Normal mu parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Normal distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Normal",
            life_stress_model="Dual-Power",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Dual_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Power",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        if len(sigmas) > 0:
            common_sigma = float(np.average(sigmas))
        else:
            common_sigma = 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # c, n, m, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Power",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.m = MLE_results.m
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.n, self.m, self.sigma]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.c_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.m_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "n", "m", "sigma"],
            "Point Estimate": [self.c, self.n, self.m, self.sigma],
            "Standard Error": [self.c_SE, self.n_SE, self.m_SE, self.sigma_SE],
            "Lower CI": [self.c_lower, self.n_lower, self.m_lower, self.sigma_lower],
            "Upper CI": [self.c_upper, self.n_upper, self.m_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S1 ** self.n) * (S2 ** self.m)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Normal_Distribution(
                mu=self.mu_at_use_stress, sigma=self.sigma
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (
                    common_sigmas[i] - sigmas_for_change_df[i]
                ) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(
                        str("+" + str(round(sigma_diff * 100, 2)) + "%")
                    )
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Normal_Dual_Power (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The sigma parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Normal",
                model="Dual-Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=self.sigma,
                scale_for_change_df=mus_for_change_df,
                shape_for_change_df=sigmas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Normal",
                model="Dual-Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

    @staticmethod
    def logf(t, S1, S2, c, n, m, sigma):  # Log PDF
        life = c * (S1 ** n) * (S2 ** m)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, c, n, m, sigma):  # Log SF
        life = c * (S1 ** n) * (S2 ** m)
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc += Fit_Normal_Dual_Power.logR(
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
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.


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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Exponential",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Exponential", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []  # weibull betas
        betas_for_change_df = []
        alphas_for_change_df = []  # weibull alphas
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, b

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Exponential",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b is strictly positive
        self.b_upper = self.b * (np.exp(Z * (self.b_SE / self.b)))
        self.b_lower = self.b * (np.exp(-Z * (self.b_SE / self.b)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b"],
            "Point Estimate": [self.a, self.b],
            "Standard Error": [self.a_SE, self.b_SE],
            "Lower CI": [self.a_lower, self.b_lower],
            "Upper CI": [self.a_upper, self.b_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.b * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))  # 1/lambda
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10 ** -5, 5),
                "eV\n",
            )

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Exponential_Exponential.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc += Fit_Exponential_Exponential.logR(
            t_rc, T_rc, params[0], params[1]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Eyring:
    """
    Fit_Exponential_Eyring

    This function will Fit the Exponential-Eyring life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with temperature. It is recommended that you ensure your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Eyring",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Eyring.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Eyring", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, c

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Eyring",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c can be positive or negative
        self.c_upper = self.c + (Z * self.c_SE)
        self.c_lower = self.c + (-Z * self.c_SE)

        # results dataframe
        results_data = {
            "Parameter": ["a", "c"],
            "Point Estimate": [self.a, self.c],
            "Standard Error": [self.a_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.c_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return 1 / S1 * np.exp(-(self.c - self.a / S1))

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Eyring (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Eyring",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Eyring",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Exponential_Eyring.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc += Fit_Exponential_Eyring.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Power:
    """
    Fit_Exponential_Power

    This function will Fit the Exponential-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with non-thermal stresses (typically in fatigue applications).

    Inputs:
    failures - an array or list of the failure times.
    failure_stress - an array or list of the corresponding stresses (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times.
    right_censored_stress - an array or list of the corresponding stresses (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Power",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power", failures=failures, stress_1_array=failure_stress
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.n = MLE_results.n
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        # a is strictly positive
        self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
        self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)

        # results dataframe
        results_data = {
            "Parameter": ["a", "n"],
            "Point Estimate": [self.a, self.n],
            "Standard Error": [self.a_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.n_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params, failures, right_censored, failure_stress, right_censored_stress
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1):
            return self.a * S1 ** self.n

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Power (" + str(CI_rounded) + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Exponential_Power.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc += Fit_Exponential_Power.logR(t_rc, T_rc, params[0], params[1]).sum()
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
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Dual-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Dual_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # a, b, c

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Exponential",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.c = MLE_results.c
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c]
        hessian_matrix = hessian(LL_func)(
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
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # b can be positive or negative
        self.b_upper = self.b + (Z * self.b_SE)
        self.b_lower = self.b + (-Z * self.b_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c"],
            "Point Estimate": [self.a, self.b, self.c],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * np.exp(self.a / S1 + self.b / S2)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Dual_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Dual-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Dual-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Exponential_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Exponential_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Power_Exponential:
    """
    Fit_Exponential_Power_Exponential

    This function will Fit the Exponential-Power-Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with thermal and non-thermal stresses. It is essential that you ensure your thermal stress is stress_thermal and your non-thermal stress is stress_nonthermal.
    Also ensure that your temperature data are in Kelvin.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (non-thermal stress) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (non-thermal stress) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC).
    change_of_parameters - a dataframe showing the change of the parameters at each stress level.
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided).
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided).
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided).
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Power-Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Power_Exponential.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power-Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # a, c, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Power-Exponential",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n]
        hessian_matrix = hessian(LL_func)(
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
        self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        # a can be positive or negative
        self.a_upper = self.a + (Z * self.a_SE)
        self.a_lower = self.a + (-Z * self.a_SE)
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n"],
            "Point Estimate": [self.a, self.c, self.n],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S2 ** self.n) * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Power_Exponential ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Power-Exponential",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Power-Exponential",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

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
        # failure times
        LL_f += Fit_Exponential_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Exponential_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Power:
    """
    Fit_Exponential_Dual_Power

    This function will Fit the Exponential-Dual-Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
    This model is most appropriate to model a life-stress relationship with two non-thermal stresses such as voltage and load.

    Inputs:
    failures - an array or list of the failure times.
    failure_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stress 2 (such as load) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as load) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True
    show_life_stress_plot - True/False. Default is True
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual-Power model
    n - fitted parameter from the Dual-Power model
    m - fitted parameter from the Dual-Power model
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
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):

        inputs = ALT_fitters_input_checking(
            dist="Exponential",
            life_stress_model="Dual-Power",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Exponential_Dual_Power.LL

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual-Power",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            if right_censored_groups is None:
                rc = None
            else:
                rc = right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # c, n, m

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual-Power",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.m = MLE_results.m
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.n, self.m]
        hessian_matrix = hessian(LL_func)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.c_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.m_SE = abs(covariance_matrix[2][2]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)

        # results dataframe
        results_data = {
            "Parameter": ["c", "n", "m"],
            "Point Estimate": [self.c, self.n, self.m],
            "Standard Error": [self.c_SE, self.n_SE, self.m_SE],
            "Lower CI": [self.c_lower, self.n_lower, self.m_lower],
            "Upper CI": [self.c_upper, self.n_upper, self.m_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(
            data=GoF_data, columns=["Goodness of fit", "Value"]
        )

        def life_func(S1, S2):
            return self.c * (S1 ** self.n) * (S2 ** self.m)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(
                S1=use_level_stress[0], S2=use_level_stress[1]
            )
            self.distribution_at_use_stress = Exponential_Distribution(
                Lambda=self.Lambda_at_use_stress
            )
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(
                str(
                    str(round_to_decimals(stress[0]))
                    + ", "
                    + str(round_to_decimals(stress[1]))
                )
            )
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / life_func(S1=stress[0], S2=stress[1])
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (
                    common_betas[i] - betas_for_change_df[i]
                ) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(
                        str("+" + str(round(beta_diff * 100, 2)) + "%")
                    )
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            if right_censored is None:
                n = len(failures)
            else:
                n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = round_to_decimals(len(right_censored) / n * 100)
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str(
                    "Results from Fit_Exponential_Dual_Power ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate."
                    )
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n"
                    )
                )

        if show_probability_plot is True:
            self.probability_plot = ALT_prob_plot(
                dist="Exponential",
                model="Dual-Power",
                stresses_for_groups=stresses_for_groups,
                failure_groups=failure_groups,
                right_censored_groups=right_censored_groups,
                life_func=life_func,
                shape=None,
                scale_for_change_df=alphas_for_change_df,
                shape_for_change_df=betas_for_change_df,
                use_level_stress=use_level_stress,
            )

        if show_life_stress_plot is True:
            self.life_stress_plot = life_stress_plot(
                dist="Exponential",
                model="Dual-Power",
                life_func=life_func,
                failure_groups=failure_groups,
                stresses_for_groups=stresses_for_groups,
                use_level_stress=use_level_stress,
            )

    @staticmethod
    def logf(t, S1, S2, c, n, m):  # Log PDF
        life = c * (S1 ** n) * (S2 ** m)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, c, n, m):  # Log SF
        life = c * (S1 ** n) * (S2 ** m)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = 0
        LL_rc = 0
        # failure times
        LL_f += Fit_Exponential_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc += Fit_Exponential_Dual_Power.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)
