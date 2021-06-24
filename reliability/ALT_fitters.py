import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
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


class Fit_Everything_ALT:
    """
    Fit_Everything_ALT
    This function will fit all available ALT models for the data you enter, which may include right censored data.

    ALT models are either single stress (Exponential, Eyring, Power) or dual stress (Dual_Exponential, Power_Exponential, Dual_Power).
    Depending on the data you enter (ie. whether failure_stress_2 is provided), the applicable set of ALT models will be fitted.

    Inputs:
    failures - an array or list of the failure times (this does not need to be sorted).
    failure_stress_1 - an array or list of the corresponding stresses (such as temperature or voltage) at which each failure occurred.
        This must match the length of failures as each failure is tied to a failure stress.
    failure_stress_2 - an array or list of the corresponding stresses (such as temperature or voltage) at which each failure occurred.
        This must match the length of failures as each failure is tied to a failure stress.
        Optional input. Providing this will trigger the use of dual stress models.
        Leaving this empty will trigger the use of single stress models.
    right_censored - an array or list of the right failure times (this does not need to be sorted). Optional Input.
    right_censored_stress_1 - an array or list of the corresponding stresses (such as temperature or voltage) at which each right_censored data point was obtained.
        This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    right_censored_stress_2 - an array or list of the corresponding stresses (such as temperature or voltage) at which each right_censored data point was obtained.
        This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
        Conditionally optional input. This must be provided if failure_stress_2 is provided.
    use_level_stress - The use level stress at which you want to know the mean life. Optional input.
        This must be a list [stress_1,stress_2] if failure_stress_2 is provided.
    print_results - True/False. Default is True
    show_probability_plot - True/False. Default is True. Provides a probability plot of each of the fitted ALT model.
    show_best_distribution_probability_plot - True/False. Defaults to True. Provides a probability plot in a new figure of the best ALT model.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.
    sort_by - goodness of fit test to sort results by. Must be 'BIC','AICc', or 'Log-likelihood'. Default is BIC.
    exclude - list or array of strings specifying which distributions to exclude. Default is None. Options are:
        Weibull_Exponential
        Weibull_Eyring
        Weibull_Power
        Weibull_Dual_Exponential
        Weibull_Power_Exponential
        Weibull_Dual_Power
        Lognormal_Exponential
        Lognormal_Eyring
        Lognormal_Power
        Lognormal_Dual_Exponential
        Lognormal_Power_Exponential
        Lognormal_Dual_Power
        Normal_Exponential
        Normal_Eyring
        Normal_Power
        Normal_Dual_Exponential
        Normal_Power_Exponential
        Normal_Dual_Power
        Exponential_Exponential
        Exponential_Eyring
        Exponential_Power
        Exponential_Dual_Exponential
        Exponential_Power_Exponential
        Exponential_Dual_Power

    Outputs:
    results - the dataframe of results. Fitted parameters in this dataframe may be accessed by name. See below example.
    best_model_name - the name of the best fitting ALT model. E.g. 'Weibull_Exponential'. See above list for exclude.
    best_model_at_use_stress - a distribution object created based on the parameters of the best fitting ALT model at the use stress.
        This is only provided if the use_level_stress is provided. This is because use_level_stress is required to find the scale parameter.
    parameters and goodness of fit results for each fitted model. For example, the Weibull_Exponential model values are:
        Weibull_Exponential_a
        Weibull_Exponential_b
        Weibull_Exponential_beta
        Weibull_Exponential_BIC
        Weibull_Exponential_AICc
        Weibull_Exponential_loglik
    excluded_models - a list of the models which were excluded. This will always include at least half the models since only single stress OR dual stress can be fitted depending on the data.

    From the results, the models are sorted based on their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the model to the data.

    Example Usage:
    failures = [619, 417, 173, 161, 1016, 512, 999, 1131, 1883, 2413, 3105, 2492]
    failure_stresses = [500, 500, 500, 500, 400, 400, 400, 400, 350, 350, 350, 350]
    right_censored = [29, 180, 1341]
    right_censored_stresses = [500, 400, 350]
    use_level_stress = 300
    output = Fit_Everything_ALT(failures=failures,failure_stress_1=failure_stresses,right_censored=right_censored, right_censored_stress_1=right_censored_stresses, use_level_stress=use_level_stress)

    To extract the parameters of the Weibull_Exponential model from the results dataframe, you may access the parameters by name:
    print('Weibull Exponential beta =',output.Weibull_Exponential_beta)
    >>> Weibull Exponential beta = 3.0807072337386123
    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_best_distribution_probability_plot=True,
        print_results=True,
        exclude=None,
        sort_by="BIC",
    ):

        # these are only here for code formatting reasons. They get redefined later
        self._Fit_Everything_ALT__Weibull_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Eyring_params = None
        self._Fit_Everything_ALT__Weibull_Power_params = None
        self._Fit_Everything_ALT__Lognormal_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Eyring_params = None
        self._Fit_Everything_ALT__Lognormal_Power_params = None
        self._Fit_Everything_ALT__Normal_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Eyring_params = None
        self._Fit_Everything_ALT__Normal_Power_params = None
        self._Fit_Everything_ALT__Exponential_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Eyring_params = None
        self._Fit_Everything_ALT__Exponential_Power_params = None
        self._Fit_Everything_ALT__Weibull_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Power_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Dual_Power_params = None
        self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Power_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Dual_Power_params = None
        self._Fit_Everything_ALT__Normal_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Power_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Dual_Power_params = None
        self._Fit_Everything_ALT__Exponential_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Power_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Dual_Power_params = None

        # for passing to the probability plot
        self.__use_level_stress = use_level_stress

        if print_results not in [True, False]:
            raise ValueError(
                "print_results must be either True or False. Defaults is True."
            )
        if show_probability_plot not in [True, False]:
            raise ValueError(
                "show_probability_plot must be either True or False. Default is True."
            )
        if show_best_distribution_probability_plot not in [True, False]:
            raise ValueError(
                "show_best_distribution_probability_plot must be either True or False. Default is True."
            )

        single_stress_ALT_models_list = [
            "Weibull_Exponential",
            "Weibull_Eyring",
            "Weibull_Power",
            "Lognormal_Exponential",
            "Lognormal_Eyring",
            "Lognormal_Power",
            "Normal_Exponential",
            "Normal_Eyring",
            "Normal_Power",
            "Exponential_Exponential",
            "Exponential_Eyring",
            "Exponential_Power",
        ]

        dual_stress_ALT_models_list = [
            "Weibull_Dual_Exponential",
            "Weibull_Power_Exponential",
            "Weibull_Dual_Power",
            "Lognormal_Dual_Exponential",
            "Lognormal_Power_Exponential",
            "Lognormal_Dual_Power",
            "Normal_Dual_Exponential",
            "Normal_Power_Exponential",
            "Normal_Dual_Power",
            "Exponential_Dual_Exponential",
            "Exponential_Power_Exponential",
            "Exponential_Dual_Power",
        ]
        all_ALT_models_list = (
            single_stress_ALT_models_list + dual_stress_ALT_models_list
        )

        excluded_models = []
        unknown_exclusions = []
        if exclude is not None:
            for item in exclude:
                if item.title() in all_ALT_models_list:
                    excluded_models.append(item.title)
                else:
                    unknown_exclusions.append(item)
            if len(unknown_exclusions) > 0:
                colorprint(
                    str(
                        "WARNING: The following items were not recognised ALT models to exclude: "
                        + str(unknown_exclusions)
                    ),
                    text_color="red",
                )
                colorprint("Available ALT models to exclude are:", text_color="red")
                for item in all_ALT_models_list:
                    colorprint(item, text_color="red")

        if failure_stress_2 is not None:
            dual_stress = True
            excluded_models.extend(single_stress_ALT_models_list)
        else:
            dual_stress = False
            excluded_models.extend(dual_stress_ALT_models_list)
        self.excluded_models = excluded_models

        # create an empty dataframe to append the data from the fitted distributions
        if dual_stress is True:
            df = pd.DataFrame(
                columns=[
                    "ALT_model",
                    "a",
                    "b",
                    "c",
                    "m",
                    "n",
                    "beta",
                    "sigma",
                    "Log-likelihood",
                    "AICc",
                    "BIC",
                ]
            )
        else:  # same df but without column m
            df = pd.DataFrame(
                columns=[
                    "ALT_model",
                    "a",
                    "b",
                    "c",
                    "n",
                    "beta",
                    "sigma",
                    "Log-likelihood",
                    "AICc",
                    "BIC",
                ]
            )

        # Fit the parametric models and extract the fitted parameters
        if "Weibull_Exponential" not in self.excluded_models:
            self.__Weibull_Exponential_params = Fit_Weibull_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Exponential_a = self.__Weibull_Exponential_params.a
            self.Weibull_Exponential_b = self.__Weibull_Exponential_params.b
            self.Weibull_Exponential_beta = self.__Weibull_Exponential_params.beta
            self.Weibull_Exponential_loglik = self.__Weibull_Exponential_params.loglik
            self.Weibull_Exponential_BIC = self.__Weibull_Exponential_params.BIC
            self.Weibull_Exponential_AICc = self.__Weibull_Exponential_params.AICc

            df = df.append(
                {
                    "ALT_model": "Weibull_Exponential",
                    "a": self.Weibull_Exponential_a,
                    "b": self.Weibull_Exponential_b,
                    "c": "",
                    "n": "",
                    "beta": self.Weibull_Exponential_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Exponential_loglik,
                    "AICc": self.Weibull_Exponential_AICc,
                    "BIC": self.Weibull_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Weibull_Eyring" not in self.excluded_models:
            self.__Weibull_Eyring_params = Fit_Weibull_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Eyring_a = self.__Weibull_Eyring_params.a
            self.Weibull_Eyring_c = self.__Weibull_Eyring_params.c
            self.Weibull_Eyring_beta = self.__Weibull_Eyring_params.beta
            self.Weibull_Eyring_loglik = self.__Weibull_Eyring_params.loglik
            self.Weibull_Eyring_BIC = self.__Weibull_Eyring_params.BIC
            self.Weibull_Eyring_AICc = self.__Weibull_Eyring_params.AICc

            df = df.append(
                {
                    "ALT_model": "Weibull_Eyring",
                    "a": self.Weibull_Eyring_a,
                    "b": "",
                    "c": self.Weibull_Eyring_c,
                    "n": "",
                    "beta": self.Weibull_Eyring_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Eyring_loglik,
                    "AICc": self.Weibull_Eyring_AICc,
                    "BIC": self.Weibull_Eyring_BIC,
                },
                ignore_index=True,
            )

        if "Weibull_Power" not in self.excluded_models:
            self.__Weibull_Power_params = Fit_Weibull_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Power_a = self.__Weibull_Power_params.a
            self.Weibull_Power_n = self.__Weibull_Power_params.n
            self.Weibull_Power_beta = self.__Weibull_Power_params.beta
            self.Weibull_Power_loglik = self.__Weibull_Power_params.loglik
            self.Weibull_Power_BIC = self.__Weibull_Power_params.BIC
            self.Weibull_Power_AICc = self.__Weibull_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Weibull_Power",
                    "a": self.Weibull_Power_a,
                    "b": "",
                    "c": "",
                    "n": self.Weibull_Power_n,
                    "beta": self.Weibull_Power_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Power_loglik,
                    "AICc": self.Weibull_Power_AICc,
                    "BIC": self.Weibull_Power_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Exponential" not in self.excluded_models:
            self.__Lognormal_Exponential_params = Fit_Lognormal_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Exponential_a = self.__Lognormal_Exponential_params.a
            self.Lognormal_Exponential_b = self.__Lognormal_Exponential_params.b
            self.Lognormal_Exponential_sigma = self.__Lognormal_Exponential_params.sigma
            self.Lognormal_Exponential_loglik = (
                self.__Lognormal_Exponential_params.loglik
            )
            self.Lognormal_Exponential_BIC = self.__Lognormal_Exponential_params.BIC
            self.Lognormal_Exponential_AICc = self.__Lognormal_Exponential_params.AICc

            df = df.append(
                {
                    "ALT_model": "Lognormal_Exponential",
                    "a": self.Lognormal_Exponential_a,
                    "b": self.Lognormal_Exponential_b,
                    "c": "",
                    "n": "",
                    "beta": "",
                    "sigma": self.Lognormal_Exponential_sigma,
                    "Log-likelihood": self.Lognormal_Exponential_loglik,
                    "AICc": self.Lognormal_Exponential_AICc,
                    "BIC": self.Lognormal_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Eyring" not in self.excluded_models:
            self.__Lognormal_Eyring_params = Fit_Lognormal_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Eyring_a = self.__Lognormal_Eyring_params.a
            self.Lognormal_Eyring_c = self.__Lognormal_Eyring_params.c
            self.Lognormal_Eyring_sigma = self.__Lognormal_Eyring_params.sigma
            self.Lognormal_Eyring_loglik = self.__Lognormal_Eyring_params.loglik
            self.Lognormal_Eyring_BIC = self.__Lognormal_Eyring_params.BIC
            self.Lognormal_Eyring_AICc = self.__Lognormal_Eyring_params.AICc

            df = df.append(
                {
                    "ALT_model": "Lognormal_Eyring",
                    "a": self.Lognormal_Eyring_a,
                    "b": "",
                    "c": self.Lognormal_Eyring_c,
                    "n": "",
                    "beta": "",
                    "sigma": self.Lognormal_Eyring_sigma,
                    "Log-likelihood": self.Lognormal_Eyring_loglik,
                    "AICc": self.Lognormal_Eyring_AICc,
                    "BIC": self.Lognormal_Eyring_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Power" not in self.excluded_models:
            self.__Lognormal_Power_params = Fit_Lognormal_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Power_a = self.__Lognormal_Power_params.a
            self.Lognormal_Power_n = self.__Lognormal_Power_params.n
            self.Lognormal_Power_sigma = self.__Lognormal_Power_params.sigma
            self.Lognormal_Power_loglik = self.__Lognormal_Power_params.loglik
            self.Lognormal_Power_BIC = self.__Lognormal_Power_params.BIC
            self.Lognormal_Power_AICc = self.__Lognormal_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Lognormal_Power",
                    "a": self.Lognormal_Power_a,
                    "b": "",
                    "c": "",
                    "n": self.Lognormal_Power_n,
                    "beta": "",
                    "sigma": self.Lognormal_Power_sigma,
                    "Log-likelihood": self.Lognormal_Power_loglik,
                    "AICc": self.Lognormal_Power_AICc,
                    "BIC": self.Lognormal_Power_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Exponential" not in self.excluded_models:
            self.__Normal_Exponential_params = Fit_Normal_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Exponential_a = self.__Normal_Exponential_params.a
            self.Normal_Exponential_b = self.__Normal_Exponential_params.b
            self.Normal_Exponential_sigma = self.__Normal_Exponential_params.sigma
            self.Normal_Exponential_loglik = self.__Normal_Exponential_params.loglik
            self.Normal_Exponential_BIC = self.__Normal_Exponential_params.BIC
            self.Normal_Exponential_AICc = self.__Normal_Exponential_params.AICc

            df = df.append(
                {
                    "ALT_model": "Normal_Exponential",
                    "a": self.Normal_Exponential_a,
                    "b": self.Normal_Exponential_b,
                    "c": "",
                    "n": "",
                    "beta": "",
                    "sigma": self.Normal_Exponential_sigma,
                    "Log-likelihood": self.Normal_Exponential_loglik,
                    "AICc": self.Normal_Exponential_AICc,
                    "BIC": self.Normal_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Eyring" not in self.excluded_models:
            self.__Normal_Eyring_params = Fit_Normal_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Eyring_a = self.__Normal_Eyring_params.a
            self.Normal_Eyring_c = self.__Normal_Eyring_params.c
            self.Normal_Eyring_sigma = self.__Normal_Eyring_params.sigma
            self.Normal_Eyring_loglik = self.__Normal_Eyring_params.loglik
            self.Normal_Eyring_BIC = self.__Normal_Eyring_params.BIC
            self.Normal_Eyring_AICc = self.__Normal_Eyring_params.AICc

            df = df.append(
                {
                    "ALT_model": "Normal_Eyring",
                    "a": self.Normal_Eyring_a,
                    "b": "",
                    "c": self.Normal_Eyring_c,
                    "n": "",
                    "beta": "",
                    "sigma": self.Normal_Eyring_sigma,
                    "Log-likelihood": self.Normal_Eyring_loglik,
                    "AICc": self.Normal_Eyring_AICc,
                    "BIC": self.Normal_Eyring_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Power" not in self.excluded_models:
            self.__Normal_Power_params = Fit_Normal_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Power_a = self.__Normal_Power_params.a
            self.Normal_Power_n = self.__Normal_Power_params.n
            self.Normal_Power_sigma = self.__Normal_Power_params.sigma
            self.Normal_Power_loglik = self.__Normal_Power_params.loglik
            self.Normal_Power_BIC = self.__Normal_Power_params.BIC
            self.Normal_Power_AICc = self.__Normal_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Normal_Power",
                    "a": self.Normal_Power_a,
                    "b": "",
                    "c": "",
                    "n": self.Normal_Power_n,
                    "beta": "",
                    "sigma": self.Normal_Power_sigma,
                    "Log-likelihood": self.Normal_Power_loglik,
                    "AICc": self.Normal_Power_AICc,
                    "BIC": self.Normal_Power_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Exponential" not in self.excluded_models:
            self.__Exponential_Exponential_params = Fit_Exponential_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Exponential_a = self.__Exponential_Exponential_params.a
            self.Exponential_Exponential_b = self.__Exponential_Exponential_params.b
            self.Exponential_Exponential_loglik = (
                self.__Exponential_Exponential_params.loglik
            )
            self.Exponential_Exponential_BIC = self.__Exponential_Exponential_params.BIC
            self.Exponential_Exponential_AICc = (
                self.__Exponential_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Exponential_Exponential",
                    "a": self.Exponential_Exponential_a,
                    "b": self.Exponential_Exponential_b,
                    "c": "",
                    "n": "",
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Exponential_loglik,
                    "AICc": self.Exponential_Exponential_AICc,
                    "BIC": self.Exponential_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Eyring" not in self.excluded_models:
            self.__Exponential_Eyring_params = Fit_Exponential_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Eyring_a = self.__Exponential_Eyring_params.a
            self.Exponential_Eyring_c = self.__Exponential_Eyring_params.c
            self.Exponential_Eyring_loglik = self.__Exponential_Eyring_params.loglik
            self.Exponential_Eyring_BIC = self.__Exponential_Eyring_params.BIC
            self.Exponential_Eyring_AICc = self.__Exponential_Eyring_params.AICc

            df = df.append(
                {
                    "ALT_model": "Exponential_Eyring",
                    "a": self.Exponential_Eyring_a,
                    "b": "",
                    "c": self.Exponential_Eyring_c,
                    "n": "",
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Eyring_loglik,
                    "AICc": self.Exponential_Eyring_AICc,
                    "BIC": self.Exponential_Eyring_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Power" not in self.excluded_models:
            self.__Exponential_Power_params = Fit_Exponential_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Power_a = self.__Exponential_Power_params.a
            self.Exponential_Power_n = self.__Exponential_Power_params.n
            self.Exponential_Power_loglik = self.__Exponential_Power_params.loglik
            self.Exponential_Power_BIC = self.__Exponential_Power_params.BIC
            self.Exponential_Power_AICc = self.__Exponential_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Exponential_Power",
                    "a": self.Exponential_Power_a,
                    "b": "",
                    "c": "",
                    "n": self.Exponential_Power_n,
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Power_loglik,
                    "AICc": self.Exponential_Power_AICc,
                    "BIC": self.Exponential_Power_BIC,
                },
                ignore_index=True,
            )

        if "Weibull_Dual_Exponential" not in self.excluded_models:
            self.__Weibull_Dual_Exponential_params = Fit_Weibull_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Dual_Exponential_a = self.__Weibull_Dual_Exponential_params.a
            self.Weibull_Dual_Exponential_b = self.__Weibull_Dual_Exponential_params.b
            self.Weibull_Dual_Exponential_c = self.__Weibull_Dual_Exponential_params.c
            self.Weibull_Dual_Exponential_beta = (
                self.__Weibull_Dual_Exponential_params.beta
            )
            self.Weibull_Dual_Exponential_loglik = (
                self.__Weibull_Dual_Exponential_params.loglik
            )
            self.Weibull_Dual_Exponential_BIC = (
                self.__Weibull_Dual_Exponential_params.BIC
            )
            self.Weibull_Dual_Exponential_AICc = (
                self.__Weibull_Dual_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Weibull_Dual_Exponential",
                    "a": self.Weibull_Dual_Exponential_a,
                    "b": self.Weibull_Dual_Exponential_b,
                    "c": self.Weibull_Dual_Exponential_c,
                    "m": "",
                    "n": "",
                    "beta": self.Weibull_Dual_Exponential_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Dual_Exponential_loglik,
                    "AICc": self.Weibull_Dual_Exponential_AICc,
                    "BIC": self.Weibull_Dual_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Weibull_Power_Exponential" not in self.excluded_models:
            self.__Weibull_Power_Exponential_params = Fit_Weibull_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Power_Exponential_a = self.__Weibull_Power_Exponential_params.a
            self.Weibull_Power_Exponential_c = self.__Weibull_Power_Exponential_params.c
            self.Weibull_Power_Exponential_n = self.__Weibull_Power_Exponential_params.n
            self.Weibull_Power_Exponential_beta = (
                self.__Weibull_Power_Exponential_params.beta
            )
            self.Weibull_Power_Exponential_loglik = (
                self.__Weibull_Power_Exponential_params.loglik
            )
            self.Weibull_Power_Exponential_BIC = (
                self.__Weibull_Power_Exponential_params.BIC
            )
            self.Weibull_Power_Exponential_AICc = (
                self.__Weibull_Power_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Weibull_Power_Exponential",
                    "a": self.Weibull_Power_Exponential_a,
                    "b": "",
                    "c": self.Weibull_Power_Exponential_c,
                    "m": "",
                    "n": self.Weibull_Power_Exponential_n,
                    "beta": self.Weibull_Power_Exponential_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Power_Exponential_loglik,
                    "AICc": self.Weibull_Power_Exponential_AICc,
                    "BIC": self.Weibull_Power_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Weibull_Dual_Power" not in self.excluded_models:
            self.__Weibull_Dual_Power_params = Fit_Weibull_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Dual_Power_c = self.__Weibull_Dual_Power_params.c
            self.Weibull_Dual_Power_m = self.__Weibull_Dual_Power_params.m
            self.Weibull_Dual_Power_n = self.__Weibull_Dual_Power_params.n
            self.Weibull_Dual_Power_beta = self.__Weibull_Dual_Power_params.beta
            self.Weibull_Dual_Power_loglik = self.__Weibull_Dual_Power_params.loglik
            self.Weibull_Dual_Power_BIC = self.__Weibull_Dual_Power_params.BIC
            self.Weibull_Dual_Power_AICc = self.__Weibull_Dual_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Weibull_Dual_Power",
                    "a": "",
                    "b": "",
                    "c": self.Weibull_Dual_Power_c,
                    "m": self.Weibull_Dual_Power_m,
                    "n": self.Weibull_Dual_Power_n,
                    "beta": self.Weibull_Dual_Power_beta,
                    "sigma": "",
                    "Log-likelihood": self.Weibull_Dual_Power_loglik,
                    "AICc": self.Weibull_Dual_Power_AICc,
                    "BIC": self.Weibull_Dual_Power_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Dual_Exponential" not in self.excluded_models:
            self.__Lognormal_Dual_Exponential_params = Fit_Lognormal_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Dual_Exponential_a = (
                self.__Lognormal_Dual_Exponential_params.a
            )
            self.Lognormal_Dual_Exponential_b = (
                self.__Lognormal_Dual_Exponential_params.b
            )
            self.Lognormal_Dual_Exponential_c = (
                self.__Lognormal_Dual_Exponential_params.c
            )
            self.Lognormal_Dual_Exponential_sigma = (
                self.__Lognormal_Dual_Exponential_params.sigma
            )
            self.Lognormal_Dual_Exponential_loglik = (
                self.__Lognormal_Dual_Exponential_params.loglik
            )
            self.Lognormal_Dual_Exponential_BIC = (
                self.__Lognormal_Dual_Exponential_params.BIC
            )
            self.Lognormal_Dual_Exponential_AICc = (
                self.__Lognormal_Dual_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Lognormal_Dual_Exponential",
                    "a": self.Lognormal_Dual_Exponential_a,
                    "b": self.Lognormal_Dual_Exponential_b,
                    "c": self.Lognormal_Dual_Exponential_c,
                    "m": "",
                    "n": "",
                    "beta": "",
                    "sigma": self.Lognormal_Dual_Exponential_sigma,
                    "Log-likelihood": self.Lognormal_Dual_Exponential_loglik,
                    "AICc": self.Lognormal_Dual_Exponential_AICc,
                    "BIC": self.Lognormal_Dual_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Power_Exponential" not in self.excluded_models:
            self.__Lognormal_Power_Exponential_params = Fit_Lognormal_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Power_Exponential_a = (
                self.__Lognormal_Power_Exponential_params.a
            )
            self.Lognormal_Power_Exponential_c = (
                self.__Lognormal_Power_Exponential_params.c
            )
            self.Lognormal_Power_Exponential_n = (
                self.__Lognormal_Power_Exponential_params.n
            )
            self.Lognormal_Power_Exponential_sigma = (
                self.__Lognormal_Power_Exponential_params.sigma
            )
            self.Lognormal_Power_Exponential_loglik = (
                self.__Lognormal_Power_Exponential_params.loglik
            )
            self.Lognormal_Power_Exponential_BIC = (
                self.__Lognormal_Power_Exponential_params.BIC
            )
            self.Lognormal_Power_Exponential_AICc = (
                self.__Lognormal_Power_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Lognormal_Power_Exponential",
                    "a": self.Lognormal_Power_Exponential_a,
                    "b": "",
                    "c": self.Lognormal_Power_Exponential_c,
                    "m": "",
                    "n": self.Lognormal_Power_Exponential_n,
                    "beta": "",
                    "sigma": self.Lognormal_Power_Exponential_sigma,
                    "Log-likelihood": self.Lognormal_Power_Exponential_loglik,
                    "AICc": self.Lognormal_Power_Exponential_AICc,
                    "BIC": self.Lognormal_Power_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Lognormal_Dual_Power" not in self.excluded_models:
            self.__Lognormal_Dual_Power_params = Fit_Lognormal_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Dual_Power_c = self.__Lognormal_Dual_Power_params.c
            self.Lognormal_Dual_Power_m = self.__Lognormal_Dual_Power_params.m
            self.Lognormal_Dual_Power_n = self.__Lognormal_Dual_Power_params.n
            self.Lognormal_Dual_Power_sigma = self.__Lognormal_Dual_Power_params.sigma
            self.Lognormal_Dual_Power_loglik = self.__Lognormal_Dual_Power_params.loglik
            self.Lognormal_Dual_Power_BIC = self.__Lognormal_Dual_Power_params.BIC
            self.Lognormal_Dual_Power_AICc = self.__Lognormal_Dual_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Lognormal_Dual_Power",
                    "a": "",
                    "b": "",
                    "c": self.Lognormal_Dual_Power_c,
                    "m": self.Lognormal_Dual_Power_m,
                    "n": self.Lognormal_Dual_Power_n,
                    "beta": "",
                    "sigma": self.Lognormal_Dual_Power_sigma,
                    "Log-likelihood": self.Lognormal_Dual_Power_loglik,
                    "AICc": self.Lognormal_Dual_Power_AICc,
                    "BIC": self.Lognormal_Dual_Power_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Dual_Exponential" not in self.excluded_models:
            self.__Normal_Dual_Exponential_params = Fit_Normal_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Dual_Exponential_a = self.__Normal_Dual_Exponential_params.a
            self.Normal_Dual_Exponential_b = self.__Normal_Dual_Exponential_params.b
            self.Normal_Dual_Exponential_c = self.__Normal_Dual_Exponential_params.c
            self.Normal_Dual_Exponential_sigma = (
                self.__Normal_Dual_Exponential_params.sigma
            )
            self.Normal_Dual_Exponential_loglik = (
                self.__Normal_Dual_Exponential_params.loglik
            )
            self.Normal_Dual_Exponential_BIC = self.__Normal_Dual_Exponential_params.BIC
            self.Normal_Dual_Exponential_AICc = (
                self.__Normal_Dual_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Normal_Dual_Exponential",
                    "a": self.Normal_Dual_Exponential_a,
                    "b": self.Normal_Dual_Exponential_b,
                    "c": self.Normal_Dual_Exponential_c,
                    "m": "",
                    "n": "",
                    "beta": "",
                    "sigma": self.Normal_Dual_Exponential_sigma,
                    "Log-likelihood": self.Normal_Dual_Exponential_loglik,
                    "AICc": self.Normal_Dual_Exponential_AICc,
                    "BIC": self.Normal_Dual_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Power_Exponential" not in self.excluded_models:
            self.__Normal_Power_Exponential_params = Fit_Normal_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Power_Exponential_a = self.__Normal_Power_Exponential_params.a
            self.Normal_Power_Exponential_c = self.__Normal_Power_Exponential_params.c
            self.Normal_Power_Exponential_n = self.__Normal_Power_Exponential_params.n
            self.Normal_Power_Exponential_sigma = (
                self.__Normal_Power_Exponential_params.sigma
            )
            self.Normal_Power_Exponential_loglik = (
                self.__Normal_Power_Exponential_params.loglik
            )
            self.Normal_Power_Exponential_BIC = (
                self.__Normal_Power_Exponential_params.BIC
            )
            self.Normal_Power_Exponential_AICc = (
                self.__Normal_Power_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Normal_Power_Exponential",
                    "a": self.Normal_Power_Exponential_a,
                    "b": "",
                    "c": self.Normal_Power_Exponential_c,
                    "m": "",
                    "n": self.Normal_Power_Exponential_n,
                    "beta": "",
                    "sigma": self.Normal_Power_Exponential_sigma,
                    "Log-likelihood": self.Normal_Power_Exponential_loglik,
                    "AICc": self.Normal_Power_Exponential_AICc,
                    "BIC": self.Normal_Power_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Normal_Dual_Power" not in self.excluded_models:
            self.__Normal_Dual_Power_params = Fit_Normal_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Dual_Power_c = self.__Normal_Dual_Power_params.c
            self.Normal_Dual_Power_m = self.__Normal_Dual_Power_params.m
            self.Normal_Dual_Power_n = self.__Normal_Dual_Power_params.n
            self.Normal_Dual_Power_sigma = self.__Normal_Dual_Power_params.sigma
            self.Normal_Dual_Power_loglik = self.__Normal_Dual_Power_params.loglik
            self.Normal_Dual_Power_BIC = self.__Normal_Dual_Power_params.BIC
            self.Normal_Dual_Power_AICc = self.__Normal_Dual_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Normal_Dual_Power",
                    "a": "",
                    "b": "",
                    "c": self.Normal_Dual_Power_c,
                    "m": self.Normal_Dual_Power_m,
                    "n": self.Normal_Dual_Power_n,
                    "beta": "",
                    "sigma": self.Normal_Dual_Power_sigma,
                    "Log-likelihood": self.Normal_Dual_Power_loglik,
                    "AICc": self.Normal_Dual_Power_AICc,
                    "BIC": self.Normal_Dual_Power_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Dual_Exponential" not in self.excluded_models:
            self.__Exponential_Dual_Exponential_params = (
                Fit_Exponential_Dual_Exponential(
                    failures=failures,
                    failure_stress_1=failure_stress_1,
                    failure_stress_2=failure_stress_2,
                    right_censored=right_censored,
                    right_censored_stress_1=right_censored_stress_1,
                    right_censored_stress_2=right_censored_stress_2,
                    use_level_stress=use_level_stress,
                    CI=CI,
                    optimizer=optimizer,
                    show_probability_plot=False,
                    show_life_stress_plot=False,
                    print_results=False,
                )
            )
            self.Exponential_Dual_Exponential_a = (
                self.__Exponential_Dual_Exponential_params.a
            )
            self.Exponential_Dual_Exponential_b = (
                self.__Exponential_Dual_Exponential_params.b
            )
            self.Exponential_Dual_Exponential_c = (
                self.__Exponential_Dual_Exponential_params.c
            )
            self.Exponential_Dual_Exponential_loglik = (
                self.__Exponential_Dual_Exponential_params.loglik
            )
            self.Exponential_Dual_Exponential_BIC = (
                self.__Exponential_Dual_Exponential_params.BIC
            )
            self.Exponential_Dual_Exponential_AICc = (
                self.__Exponential_Dual_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Exponential_Dual_Exponential",
                    "a": self.Exponential_Dual_Exponential_a,
                    "b": self.Exponential_Dual_Exponential_b,
                    "c": self.Exponential_Dual_Exponential_c,
                    "m": "",
                    "n": "",
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Dual_Exponential_loglik,
                    "AICc": self.Exponential_Dual_Exponential_AICc,
                    "BIC": self.Exponential_Dual_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Power_Exponential" not in self.excluded_models:
            self.__Exponential_Power_Exponential_params = (
                Fit_Exponential_Power_Exponential(
                    failures=failures,
                    failure_stress_1=failure_stress_1,
                    failure_stress_2=failure_stress_2,
                    right_censored=right_censored,
                    right_censored_stress_1=right_censored_stress_1,
                    right_censored_stress_2=right_censored_stress_2,
                    use_level_stress=use_level_stress,
                    CI=CI,
                    optimizer=optimizer,
                    show_probability_plot=False,
                    show_life_stress_plot=False,
                    print_results=False,
                )
            )
            self.Exponential_Power_Exponential_a = (
                self.__Exponential_Power_Exponential_params.a
            )
            self.Exponential_Power_Exponential_c = (
                self.__Exponential_Power_Exponential_params.c
            )
            self.Exponential_Power_Exponential_n = (
                self.__Exponential_Power_Exponential_params.n
            )
            self.Exponential_Power_Exponential_loglik = (
                self.__Exponential_Power_Exponential_params.loglik
            )
            self.Exponential_Power_Exponential_BIC = (
                self.__Exponential_Power_Exponential_params.BIC
            )
            self.Exponential_Power_Exponential_AICc = (
                self.__Exponential_Power_Exponential_params.AICc
            )

            df = df.append(
                {
                    "ALT_model": "Exponential_Power_Exponential",
                    "a": self.Exponential_Power_Exponential_a,
                    "b": "",
                    "c": self.Exponential_Power_Exponential_c,
                    "m": "",
                    "n": self.Exponential_Power_Exponential_n,
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Power_Exponential_loglik,
                    "AICc": self.Exponential_Power_Exponential_AICc,
                    "BIC": self.Exponential_Power_Exponential_BIC,
                },
                ignore_index=True,
            )

        if "Exponential_Dual_Power" not in self.excluded_models:
            self.__Exponential_Dual_Power_params = Fit_Exponential_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Dual_Power_c = self.__Exponential_Dual_Power_params.c
            self.Exponential_Dual_Power_m = self.__Exponential_Dual_Power_params.m
            self.Exponential_Dual_Power_n = self.__Exponential_Dual_Power_params.n
            self.Exponential_Dual_Power_loglik = (
                self.__Exponential_Dual_Power_params.loglik
            )
            self.Exponential_Dual_Power_BIC = self.__Exponential_Dual_Power_params.BIC
            self.Exponential_Dual_Power_AICc = self.__Exponential_Dual_Power_params.AICc

            df = df.append(
                {
                    "ALT_model": "Exponential_Dual_Power",
                    "a": "",
                    "b": "",
                    "c": self.Exponential_Dual_Power_c,
                    "m": self.Exponential_Dual_Power_m,
                    "n": self.Exponential_Dual_Power_n,
                    "beta": "",
                    "sigma": "",
                    "Log-likelihood": self.Exponential_Dual_Power_loglik,
                    "AICc": self.Exponential_Dual_Power_AICc,
                    "BIC": self.Exponential_Dual_Power_BIC,
                },
                ignore_index=True,
            )

        # change to sorting by BIC if there is insufficient data to get the AICc for everything that was fitted
        if (
            sort_by.upper() in ["AIC", "AICC"]
            and "Insufficient data" in df["AICc"].values
        ):
            sort_by = "BIC"
        # sort the dataframe by BIC, AICc, or log-likelihood. Smallest AICc, BIC, log-likelihood is better fit
        if type(sort_by) != str:
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', or 'Log-likelihood'. Default is 'BIC'."
            )
        if sort_by.upper() == "BIC":
            df2 = df.reindex(df.BIC.sort_values().index)
        elif sort_by.upper() in ["AICC", "AIC"]:
            df2 = df.reindex(df.AICc.sort_values().index)
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
                "Invalid input to sort_by. Options are 'BIC', 'AICc', or 'Log-likelihood'. Default is 'BIC'."
            )
        if len(df2.index.values) == 0:
            raise ValueError("You have excluded all available ALT models")
        self.results = df2

        # creates a distribution object of the best fitting distribution and assigns its name
        best_model = self.results["ALT_model"].values[0]
        self.best_model_name = best_model
        if use_level_stress is not None:
            if best_model == "Weibull_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Exponential_b
                    * np.exp(self.Weibull_Exponential_a / use_level_stress),
                    beta=self.Weibull_Exponential_beta,
                )
            elif best_model == "Weibull_Eyring":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=(1 / use_level_stress)
                    * np.exp(
                        -(
                            self.Weibull_Eyring_c
                            - self.Weibull_Eyring_a / use_level_stress
                        )
                    ),
                    beta=self.Weibull_Eyring_beta,
                )
            elif best_model == "Weibull_Power":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Power_a
                    * use_level_stress ** self.Weibull_Power_n,
                    beta=self.Weibull_Power_beta,
                )
            elif best_model == "Lognormal_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(
                        self.Lognormal_Exponential_b
                        * np.exp(self.Lognormal_Exponential_a / use_level_stress)
                    ),
                    sigma=self.Lognormal_Exponential_sigma,
                )
            elif best_model == "Lognormal_Eyring":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(
                        (1 / use_level_stress)
                        * np.exp(
                            -(
                                self.Lognormal_Eyring_c
                                - self.Lognormal_Eyring_a / use_level_stress
                            )
                        )
                    ),
                    sigma=self.Lognormal_Eyring_sigma,
                )
            elif best_model == "Lognormal_Power":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(
                        self.Lognormal_Power_a
                        * use_level_stress ** self.Lognormal_Power_n
                    ),
                    sigma=self.Lognormal_Power_sigma,
                )
            elif best_model == "Normal_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Exponential_b
                    * np.exp(self.Normal_Exponential_a / use_level_stress),
                    sigma=self.Normal_Exponential_sigma,
                )
            elif best_model == "Normal_Eyring":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=(1 / use_level_stress)
                    * np.exp(
                        -(
                            self.Normal_Eyring_c
                            - self.Normal_Eyring_a / use_level_stress
                        )
                    ),
                    sigma=self.Normal_Eyring_sigma,
                )
            elif best_model == "Normal_Power":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Power_a * use_level_stress ** self.Normal_Power_n,
                    sigma=self.Normal_Power_sigma,
                )
            elif best_model == "Exponential_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1
                    / (
                        self.Exponential_Exponential_b
                        * np.exp(self.Exponential_Exponential_a / use_level_stress)
                    )
                )
            elif best_model == "Exponential_Eyring":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1
                    / (
                        (1 / use_level_stress)
                        * np.exp(
                            -(
                                self.Exponential_Eyring_c
                                - self.Exponential_Eyring_a / use_level_stress
                            )
                        )
                    )
                )
            elif best_model == "Exponential_Power":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1
                    / (
                        self.Exponential_Power_a
                        * use_level_stress ** self.Exponential_Power_n
                    )
                )
            elif best_model == "Weibull_Dual_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Dual_Exponential_c
                    * np.exp(
                        self.Weibull_Dual_Exponential_a / use_level_stress[0]
                        + self.Weibull_Dual_Exponential_b / use_level_stress[1]
                    ),
                    beta=self.Weibull_Dual_Exponential_beta,
                )
            elif best_model == "Weibull_Power_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Power_Exponential_c
                    * use_level_stress[1] ** self.Weibull_Power_Exponential_n
                    * np.exp(self.Weibull_Power_Exponential_a / use_level_stress[0]),
                    beta=self.Weibull_Power_Exponential_beta,
                )
            elif best_model == "Weibull_Dual_Power":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Dual_Power_c
                    * use_level_stress[0] ** self.Weibull_Dual_Power_m
                    * use_level_stress[1] ** self.Weibull_Dual_Power_n,
                    beta=self.Weibull_Dual_Power_beta,
                )
            elif best_model == "Lognormal_Dual_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Dual_Exponential_c
                        * np.exp(
                            self.Lognormal_Dual_Exponential_a / use_level_stress[0]
                            + self.Lognormal_Dual_Exponential_b / use_level_stress[1]
                        )
                    ),
                    sigma=self.Lognormal_Dual_Exponential_sigma,
                )
            elif best_model == "Lognormal_Power_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Power_Exponential_c
                        * use_level_stress[1] ** self.Lognormal_Power_Exponential_n
                        * np.exp(
                            self.Lognormal_Power_Exponential_a / use_level_stress[0]
                        )
                    ),
                    sigma=self.Lognormal_Power_Exponential_sigma,
                )
            elif best_model == "Lognormal_Dual_Power":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Dual_Power_c
                        * use_level_stress[0] ** self.Lognormal_Dual_Power_m
                        * use_level_stress[1] ** self.Lognormal_Dual_Power_n
                    ),
                    sigma=self.Lognormal_Dual_Power_sigma,
                )
            elif best_model == "Normal_Dual_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Dual_Exponential_c
                    * np.exp(
                        self.Normal_Dual_Exponential_a / use_level_stress[0]
                        + self.Normal_Dual_Exponential_b / use_level_stress[1]
                    ),
                    sigma=self.Normal_Dual_Exponential_sigma,
                )
            elif best_model == "Normal_Power_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Power_Exponential_c
                    * use_level_stress[1] ** self.Normal_Power_Exponential_n
                    * np.exp(self.Normal_Power_Exponential_a / use_level_stress[0]),
                    sigma=self.Normal_Power_Exponential_sigma,
                )
            elif best_model == "Normal_Dual_Power":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Dual_Power_c
                    * use_level_stress[0] ** self.Normal_Dual_Power_m
                    * use_level_stress[1] ** self.Normal_Dual_Power_n,
                    sigma=self.Normal_Dual_Power_sigma,
                )
            elif best_model == "Exponential_Dual_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Dual_Exponential_c
                    * np.exp(
                        self.Exponential_Dual_Exponential_a / use_level_stress[0]
                        + self.Exponential_Dual_Exponential_b / use_level_stress[1]
                    )
                )
            elif best_model == "Exponential_Power_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Power_Exponential_c
                    * use_level_stress[1] ** self.Exponential_Power_Exponential_n
                    * np.exp(self.Exponential_Power_Exponential_a / use_level_stress[0])
                )
            elif best_model == "Exponential_Dual_Power":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Dual_Power_c
                    * use_level_stress[0] ** self.Exponential_Dual_Power_m
                    * use_level_stress[1] ** self.Exponential_Dual_Power_n
                )

        # print the results
        if print_results is True:  # printing occurs by default
            if right_censored is not None:
                frac_cens = (
                    len(right_censored) / (len(failures) + len(right_censored))
                ) * 100
            else:
                frac_cens = 0
                right_censored = []
            if frac_cens % 1 < 1e-10:
                frac_cens = int(frac_cens)
            colorprint("Results from Fit_Everything_ALT:", bold=True, underline=True)
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_cens) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")

            if use_level_stress is not None:
                if type(use_level_stress) not in [list, np.ndarray]:
                    use_level_stress_str = str(round_to_decimals(use_level_stress))
                else:
                    use_level_stress_str = str(
                        str(round_to_decimals(use_level_stress[0]))
                        + ", "
                        + str(round_to_decimals(use_level_stress[1]))
                    )
                print(
                    str(
                        "At the use level stress of "
                        + use_level_stress_str
                        + ", the "
                        + self.best_model_name
                        + " model has a mean life of "
                        + str(round_to_decimals(self.best_model_at_use_stress.mean))
                    )
                )

        if show_probability_plot is True:
            # plotting occurs by default
            Fit_Everything_ALT.probability_plot(self)

        if show_best_distribution_probability_plot is True:
            Fit_Everything_ALT.probability_plot(self, best_only=True)

        if (
            show_probability_plot is True
            or show_best_distribution_probability_plot is True
        ):
            plt.show()

    def probplot_layout(self):
        items = len(self.results.index.values)  # number of items that were fitted
        if items in [10, 11, 12]:  # --- w , h
            cols, rows, figsize = 4, 3, (15, 8)
        elif items in [7, 8, 9]:
            cols, rows, figsize = 3, 3, (12.5, 8)
        elif items in [5, 6]:
            cols, rows, figsize = 3, 2, (12.5, 6)
        elif items == 4:
            cols, rows, figsize = 2, 2, (10, 6)
        elif items == 3:
            cols, rows, figsize = 3, 1, (12.5, 5)
        elif items == 2:
            cols, rows, figsize = 2, 1, (10, 4)
        elif items == 1:
            cols, rows, figsize = 1, 1, (7.5, 4)
        return cols, rows, figsize

    def probability_plot(self, best_only=False):
        from reliability.Utils import ALT_prob_plot

        use_level_stress = self.__use_level_stress
        plt.figure()
        if best_only is False:
            cols, rows, figsize = Fit_Everything_ALT.probplot_layout(self)
            # this is the order to plot to match the results dataframe
            plotting_order = self.results["ALT_model"].values
            plt.suptitle("Probability plots of each fitted ALT model\n\n")
            subplot_counter = 1
        else:
            # plots the best model only
            plotting_order = [self.results["ALT_model"].values[0]]
            rows, cols, subplot_counter = 1, 1, 1

        for item in plotting_order:
            ax = plt.subplot(rows, cols, subplot_counter)

            if item == "Weibull_Exponential":

                def life_func(S1):
                    return self.Weibull_Exponential_b * np.exp(
                        self.Weibull_Exponential_a / S1
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Weibull_Eyring":

                def life_func(S1):
                    return (
                        1
                        / S1
                        * np.exp(-(self.Weibull_Eyring_c - self.Weibull_Eyring_a / S1))
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Eyring_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Weibull_Power":

                def life_func(S1):
                    return self.Weibull_Power_a * S1 ** self.Weibull_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Power_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Exponential":

                def life_func(S1):
                    return self.Lognormal_Exponential_b * np.exp(
                        self.Lognormal_Exponential_a / S1
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Eyring":

                def life_func(S1):
                    return (
                        1
                        / S1
                        * np.exp(
                            -(self.Lognormal_Eyring_c - self.Lognormal_Eyring_a / S1)
                        )
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Eyring_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Power":

                def life_func(S1):
                    return self.Lognormal_Power_a * S1 ** self.Lognormal_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Exponential":

                def life_func(S1):
                    return self.Normal_Exponential_b * np.exp(
                        self.Normal_Exponential_a / S1
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Eyring":

                def life_func(S1):
                    return (
                        1
                        / S1
                        * np.exp(-(self.Normal_Eyring_c - self.Normal_Eyring_a / S1))
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Eyring_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Power":

                def life_func(S1):
                    return self.Normal_Power_a * S1 ** self.Normal_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Exponential":

                def life_func(S1):
                    return self.Exponential_Exponential_b * np.exp(
                        self.Exponential_Exponential_a / S1
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Eyring":

                def life_func(S1):
                    return (
                        1
                        / S1
                        * np.exp(
                            -(
                                self.Exponential_Eyring_c
                                - self.Exponential_Eyring_a / S1
                            )
                        )
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Power":

                def life_func(S1):
                    return self.Exponential_Power_a * S1 ** self.Exponential_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Weibull_Dual_Exponential_c * np.exp(
                        self.Weibull_Dual_Exponential_a / S1
                        + self.Weibull_Dual_Exponential_b / S2
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Dual_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Weibull_Power_Exponential_c
                        * (S2 ** self.Weibull_Power_Exponential_n)
                        * np.exp(self.Weibull_Power_Exponential_a / S1)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Power_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Weibull_Dual_Power_c
                        * (S1 ** self.Weibull_Dual_Power_m)
                        * (S2 ** self.Weibull_Dual_Power_n)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Dual_Power_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Lognormal_Dual_Exponential_c * np.exp(
                        self.Lognormal_Dual_Exponential_a / S1
                        + self.Lognormal_Dual_Exponential_b / S2
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Dual_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Lognormal_Power_Exponential_c
                        * (S2 ** self.Lognormal_Power_Exponential_n)
                        * np.exp(self.Lognormal_Power_Exponential_a / S1)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Power_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Lognormal_Dual_Power_c
                        * (S1 ** self.Lognormal_Dual_Power_m)
                        * (S2 ** self.Lognormal_Dual_Power_n)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Dual_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Normal_Dual_Exponential_c * np.exp(
                        self.Normal_Dual_Exponential_a / S1
                        + self.Normal_Dual_Exponential_b / S2
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Dual_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Normal_Power_Exponential_c
                        * (S2 ** self.Normal_Power_Exponential_n)
                        * np.exp(self.Normal_Power_Exponential_a / S1)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Power_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Normal_Dual_Power_c
                        * (S1 ** self.Normal_Dual_Power_m)
                        * (S2 ** self.Normal_Dual_Power_n)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Dual_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Exponential_Dual_Exponential_c * np.exp(
                        self.Exponential_Dual_Exponential_a / S1
                        + self.Exponential_Dual_Exponential_b / S2
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Exponential_Power_Exponential_c
                        * (S2 ** self.Exponential_Power_Exponential_n)
                        * np.exp(self.Exponential_Power_Exponential_a / S1)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Exponential_Dual_Power_c
                        * (S1 ** self.Exponential_Dual_Power_m)
                        * (S2 ** self.Exponential_Dual_Power_n)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            else:
                raise ValueError("unknown item was fitted")

            if best_only is False:
                plt.title(item)
                ax.set_yticklabels([], minor=False)
                ax.set_xticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.set_xticklabels([], minor=True)
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.get_legend().remove()
                subplot_counter += 1
            else:
                plt.title("Probability plot of best model\n" + item)
        if best_only is False:
            plt.tight_layout()
            plt.gcf().set_size_inches(figsize)


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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Weibull_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Weibull_Exponential.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Eyring",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Weibull_Eyring.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Weibull_Eyring.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Weibull_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Weibull_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Dual_Exponential:
    """
    Fit_Weibull_Dual_Exponential

    This function will Fit the Weibull_Dual_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Dual_Exponential model
    b - fitted parameter from the Dual_Exponential model
    c - fitted parameter from the Dual_Exponential model
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
            life_stress_model="Dual_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Exponential",
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
            model="Dual_Exponential",
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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Weibull",
            model="Dual_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.beta,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Dual_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Weibull_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc = Fit_Weibull_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Power_Exponential:
    """
    Fit_Weibull_Power_Exponential

    This function will Fit the Weibull_Power_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Power_Exponential model
    c - fitted parameter from the Power_Exponential model
    n - fitted parameter from the Power_Exponential model
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
            life_stress_model="Power_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power_Exponential",
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
            model="Power_Exponential",
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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Weibull",
            model="Power_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.beta,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Power_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        LL_f = Fit_Weibull_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc = Fit_Weibull_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Dual_Power:
    """
    Fit_Weibull_Dual_Power

    This function will Fit the Weibull_Dual_Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual_Power model
    n - fitted parameter from the Dual_Power model
    m - fitted parameter from the Dual_Power model
    beta - the fitted Weibull_2P beta
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    c_SE - the standard error (sqrt(variance)) of the parameter
    m_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    m_upper - the upper CI estimate of the parameter
    m_lower - the lower CI estimate of the parameter
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Power",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Power",
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
        ]  # c, m, n, beta

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual_Power",
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
        self.m = MLE_results.m
        self.n = MLE_results.n
        self.beta = MLE_results.beta
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n, self.beta]
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
        self.m_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.beta_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # beta is strictly positive
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n", "beta"],
            "Point Estimate": [self.c, self.m, self.n, self.beta],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE, self.beta_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower, self.beta_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper, self.beta_upper],
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
            return self.c * (S1 ** self.m) * (S2 ** self.n)

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Weibull",
            model="Dual_Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.beta,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Weibull",
            model="Dual_Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, c, m, n, beta):  # Log PDF
        life = c * (S1 ** m) * (S2 ** n)
        return (
            (beta - 1) * anp.log(t / life) + anp.log(beta / life) - (t / life) ** beta
        )

    @staticmethod
    def logR(t, S1, S2, c, m, n, beta):  # Log SF
        life = c * (S1 ** m) * (S2 ** n)
        return -((t / life) ** beta)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Weibull_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc = Fit_Weibull_Dual_Power.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Lognormal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Lognormal_Exponential.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Eyring",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Lognormal_Eyring.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Lognormal_Eyring.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Lognormal_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Lognormal_Power.logR(
            t_rc, T_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_Dual_Exponential:
    """
    Fit_Lognormal_Dual_Exponential

    This function will Fit the Lognormal_Dual_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Dual_Exponential model
    b - fitted parameter from the Dual_Exponential model
    c - fitted parameter from the Dual_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Exponential",
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
            model="Dual_Exponential",
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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Lognormal",
            model="Dual_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Dual_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        LL_f = Fit_Lognormal_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc = Fit_Lognormal_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Power_Exponential:
    """
    Fit_Lognormal_Power_Exponential

    This function will Fit the Lognormal_Power_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Power_Exponential model
    c - fitted parameter from the Power_Exponential model
    n - fitted parameter from the Power_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Power_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power_Exponential",
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
            model="Power_Exponential",
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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Lognormal",
            model="Power_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Power_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        LL_f = Fit_Lognormal_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc = Fit_Lognormal_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_Dual_Power:
    """
    Fit_Lognormal_Dual_Power

    This function will Fit the Lognormal_Dual_Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual_Power model
    m - fitted parameter from the Dual_Power model
    n - fitted parameter from the Dual_Power model
    sigma - the fitted Lognormal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    c_SE - the standard error (sqrt(variance)) of the parameter
    m_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    m_upper - the upper CI estimate of the parameter
    m_lower - the lower CI estimate of the parameter
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Power",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Power",
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
        ]  # c, m, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual_Power",
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
        self.m = MLE_results.m
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n, self.sigma]
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
        self.m_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n", "sigma"],
            "Point Estimate": [self.c, self.m, self.n, self.sigma],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper, self.sigma_upper],
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
            return self.c * (S1 ** self.m) * (S2 ** self.n)

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Lognormal",
            model="Dual_Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Lognormal",
            model="Dual_Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, c, m, n, sigma):  # Log PDF
        life = c * (S1 ** m) * (S2 ** n)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t) - anp.log(life)) / sigma) ** 2))
            / (t * sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, c, m, n, sigma):  # Log SF
        life = c * (S1 ** m) * (S2 ** n)
        return anp.log(
            0.5 - 0.5 * erf((anp.log(t) - anp.log(life)) / (sigma * 2 ** 0.5))
        )

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Lognormal_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc = Fit_Lognormal_Dual_Power.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.


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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Normal_Exponential.logf(
            t_f, T_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Normal_Exponential.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Eyring",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Normal_Eyring.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Normal_Eyring.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Normal_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Normal_Power.logR(t_rc, T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Exponential:
    """
    Fit_Normal_Dual_Exponential

    This function will Fit the Normal_Dual_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Dual_Exponential model
    b - fitted parameter from the Dual_Exponential model
    c - fitted parameter from the Dual_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Exponential",
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
            model="Dual_Exponential",
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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Normal",
            model="Dual_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Dual_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Normal_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc = Fit_Normal_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Power_Exponential:
    """
    Fit_Normal_Power_Exponential

    This function will Fit the Normal_Power_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Power_Exponential model
    c - fitted parameter from the Power_Exponential model
    n - fitted parameter from the Power_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Power_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power_Exponential",
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
            model="Power_Exponential",
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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Normal",
            model="Power_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Power_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Normal_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()
        # right censored times
        LL_rc = Fit_Normal_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Power:
    """
    Fit_Normal_Dual_Power

    This function will Fit the Normal_Dual_Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual_Power model
    m - fitted parameter from the Dual_Power model
    n - fitted parameter from the Dual_Power model
    sigma - the fitted Normal_2P sigma
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    c_SE - the standard error (sqrt(variance)) of the parameter
    m_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    m_upper - the upper CI estimate of the parameter
    m_lower - the lower CI estimate of the parameter
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Power",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Power",
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
        ]  # c, m, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual_Power",
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
        self.m = MLE_results.m
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n, self.sigma]
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
        self.m_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)
        # sigma is strictly positive
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n", "sigma"],
            "Point Estimate": [self.c, self.m, self.n, self.sigma],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper, self.sigma_upper],
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
            return self.c * (S1 ** self.m) * (S2 ** self.n)

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
        self.__scale_for_change_df = mus_for_change_df
        self.__shape_for_change_df = sigmas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Normal",
            model="Dual_Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=self.sigma,
            scale_for_change_df=mus_for_change_df,
            shape_for_change_df=sigmas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Normal",
            model="Dual_Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, c, m, n, sigma):  # Log PDF
        life = c * (S1 ** m) * (S2 ** n)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(
            (sigma * (2 * anp.pi) ** 0.5)
        )

    @staticmethod
    def logR(t, S1, S2, c, m, n, sigma):  # Log SF
        life = c * (S1 ** m) * (S2 ** n)
        return anp.log((1 + erf(((life - t) / sigma) / 2 ** 0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = Fit_Normal_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]
        ).sum()  # failure times
        LL_rc = Fit_Normal_Dual_Power.logR(
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.


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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Exponential_Exponential.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Exponential.logR(t_rc, T_rc, params[0], params[1]).sum()
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Eyring",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Exponential_Eyring.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Eyring.logR(t_rc, T_rc, params[0], params[1]).sum()
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Exponential_Power.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Power.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Exponential:
    """
    Fit_Exponential_Dual_Exponential

    This function will Fit the Exponential_Dual_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Dual_Exponential model
    b - fitted parameter from the Dual_Exponential model
    c - fitted parameter from the Dual_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Exponential",
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
            model="Dual_Exponential",
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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Dual_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Dual_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Exponential_Dual_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Exponential_Dual_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Power_Exponential:
    """
    Fit_Exponential_Power_Exponential

    This function will Fit the Exponential_Power_Exponential life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    a - fitted parameter from the Power_Exponential model
    c - fitted parameter from the Power_Exponential model
    n - fitted parameter from the Power_Exponential model
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
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Power_Exponential",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power_Exponential",
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
            model="Power_Exponential",
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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Power_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Power_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
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
        # failure times
        LL_f = Fit_Exponential_Power_Exponential.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Exponential_Power_Exponential.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Power:
    """
    Fit_Exponential_Dual_Power

    This function will Fit the Exponential_Dual_Power life-stress model to the data provided. Please see the online documentation for the equations of this model.
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
    show_probability_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    show_life_stress_plot - True/False/axes. Default is True. If an axes object is passed it will be used.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

    Outputs:
    c - fitted parameter from the Dual_Power model
    m - fitted parameter from the Dual_Power model
    n - fitted parameter from the Dual_Power model
    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
    loglik - Log Likelihood (as used in Minitab and Reliasoft)
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    c_SE - the standard error (sqrt(variance)) of the parameter
    m_SE - the standard error (sqrt(variance)) of the parameter
    n_SE - the standard error (sqrt(variance)) of the parameter
    c_upper - the upper CI estimate of the parameter
    c_lower - the lower CI estimate of the parameter
    m_upper - the upper CI estimate of the parameter
    m_lower - the lower CI estimate of the parameter
    n_upper - the upper CI estimate of the parameter
    n_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters - a dataframe showing the change of the parameters at each stress level
    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
    Lambda_at_use_stress - the equivalent Exponential Lambda parameter at the use level stress (only provided if use_level_stress is provided)
    distribution_at_use_stress - the Exponential distribution at the use level stress (only provided if use_level_stress is provided)
    probability_plot - the axes handles for the figure object from the probability plot (only provided if show_probability_plot is True)
    life_stress_plot - the axes handles for the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
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
            life_stress_model="Dual_Power",
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
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Power",
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
        ]  # c, m, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimisation(
            model="Dual_Power",
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
        self.m = MLE_results.m
        self.n = MLE_results.n
        self.success = MLE_results.success

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n]
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
        self.m_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
        # c is strictly positive
        self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
        self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        # m can be positive or negative
        self.m_upper = self.m + (Z * self.m_SE)
        self.m_lower = self.m + (-Z * self.m_SE)
        # n can be positive or negative
        self.n_upper = self.n + (Z * self.n_SE)
        self.n_lower = self.n + (-Z * self.n_SE)

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n"],
            "Point Estimate": [self.c, self.m, self.n],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper],
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
            return self.c * (S1 ** self.m) * (S2 ** self.n)

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
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

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

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Dual_Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Dual_Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, c, m, n):  # Log PDF
        life = c * (S1 ** m) * (S2 ** n)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, c, m, n):  # Log SF
        life = c * (S1 ** m) * (S2 ** n)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Dual_Power.logf(
            t_f, S1_f, S2_f, params[0], params[1], params[2]
        ).sum()
        # right censored times
        LL_rc = Fit_Exponential_Dual_Power.logR(
            t_rc, S1_rc, S2_rc, params[0], params[1], params[2]
        ).sum()
        return -(LL_f + LL_rc)
