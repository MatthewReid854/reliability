'''
Accelerated Life Testing

Within the module ALT, are the following functions:
- acceleration_factor - Given T_use and two out of the three values for AF, T_acc, Ea, it will find the third value.
- ALT_probability_plot_Weibull - produces an ALT probability plot by fitting a Weibull distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Lognormal - produces an ALT probability plot by fitting a Lognormal distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Normal - produces an ALT probability plot by fitting a Normal distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Gamma - produces an ALT probability plot by fitting a Gamma distribution to each unique stress and then finding a common shape parameter
'''

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from reliability import Probability_plotting
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Gamma_2P, Fit_Normal_2P
import pandas as pd
from scipy.optimize import minimize


class acceleration_factor:
    '''
    The Arrhenius model for Acceleration factor due to higher temperature is:
    AF = exp(Ea/K(1/T_use-1/T_acc))
    This function accepts T_use as a mandotory input and the user may specify any two of the three other variables, and the third variable will be found.

    Inputs:
    T_use - Temp of usage in Celsius
    T_acc - Temp of acceleration in Celsius (optional input)
    Ea - Activation energy in eV (optional input)
    AF - Acceleration factor (optional input)
    Two of the three optional inputs must be specified and the third one will be found.
    print_results - True/False. Default is True

    Returns
    Results will be printed to console if print_results is True
    AF - Acceleration Factor
    T_acc - Accelerated temperature
    T_use - Use temperature
    Ea - Activation energy (in eV)
    '''

    def __init__(self, AF=None, T_use=None, T_acc=None, Ea=None, print_results=True):
        if T_use is None:
            raise ValueError('T_use must be specified')
        args = [AF, T_acc, Ea]
        nonecounter = 0
        for item in args:
            if item is None:
                nonecounter += 1
        if nonecounter > 1:
            raise ValueError('You must specify two out of three of the optional inputs (T_acc, AF, Ea) and the third one will be found.')

        if AF is None:
            a = Ea / (8.617333262145 * 10 ** -5)
            AF = np.exp(a / (T_use + 273.15) - a / (T_acc + 273.15))
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if Ea is None:
            Ea = np.log(AF) * (8.617333262145 * 10 ** -5) / (1 / (T_use + 273.15) - 1 / (T_acc + 273.15))
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if T_acc is None:
            T_acc = (1 / (1 / (T_use + 273.15) - np.log(AF) * (8.617333262145 * 10 ** -5) / Ea)) - 273.15
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if print_results is True:
            print('Acceleration Factor:', self.AF)
            print('Use Temperature:', self.T_use, '°C')
            print('Accelerated Temperature:', self.T_acc, '°C')
            print('Activation Energy (eV):', self.Ea, 'eV')


class ALT_probability_plot_Weibull:
    '''
    ALT_probability_plot_Weibull

    This function produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level, as well as a refitted distribution assuming a common shape parameter (beta).

    Inputs:
    failures - an array or list of all the failure times
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    common_beta_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_beta parameter. 'BIC' will find the common_beta that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in beta
    common_beta - the common beta parameter
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_beta
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_beta

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_beta_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_beta_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_beta_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_beta_method must be either BIC, weighted_average, or average. Default is BIC.')
        if len(failures) != len(failure_stress):
            raise ValueError('The length of failures does not match the length of failure_stress')
        if type(failures) is list:
            failures = np.array(failures)
        elif type(failures) is np.ndarray:
            pass
        else:
            raise ValueError('failures must be an array or list')
        if type(failure_stress) is list:
            failure_stress = np.array(failure_stress)
        elif type(failure_stress) is np.ndarray:
            pass
        else:
            raise ValueError('failure_stress must be an array or list')
        if right_censored is not None:
            if len(right_censored) != len(right_censored_stress):
                raise ValueError('The length of right_censored does not match the length of right_censored_stress')
            if type(right_censored) is list:
                right_censored = np.array(right_censored)
            elif type(right_censored) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored must be an array or list')
            if type(right_censored_stress) is list:
                right_censored_stress = np.array(right_censored_stress)
            elif type(right_censored_stress) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored_stress must be an array or list')

        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin, xmax, 100)

        if right_censored is not None:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack([np.ones_like(failures), np.zeros_like(right_censored)])
        else:
            TIMES = failures
            STRESS = failure_stress
            CENS_CODES = np.ones_like(failures)

        data = {'times': TIMES, 'stress': STRESS, 'cens_codes': CENS_CODES}
        df = pd.DataFrame(data, columns=['times', 'stress', 'cens_codes'])
        df_sorted = df.sort_values(by=['cens_codes', 'stress', 'times'])
        is_failure = df_sorted['cens_codes'] == 1
        is_right_cens = df_sorted['cens_codes'] == 0
        f_df = df_sorted[is_failure]
        rc_df = df_sorted[is_right_cens]
        unique_stresses_f = f_df.stress.unique()
        if right_censored is not None:
            unique_stresses_rc = rc_df.stress.unique()
            for item in unique_stresses_rc:  # check that there are no unique right_censored stresses that are not also in failure stresses
                if item not in unique_stresses_f:
                    raise ValueError('The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done.')

        weibull_fit_alpha_array = []
        weibull_fit_beta_array = []
        weibull_fit_alpha_array_common_beta = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        weights_array = []
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
        for stress in unique_stresses_f:
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            len_f = len(FAILURES)
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    len_rc = len(RIGHT_CENSORED)
                else:
                    RIGHT_CENSORED = None
                    len_rc = 0
            else:
                RIGHT_CENSORED = None
                len_rc = 0

            weights_array.append(len_f + len_rc)
            weibull_fit = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            weibull_fit_alpha_array.append(weibull_fit.alpha)
            weibull_fit_beta_array.append(weibull_fit.beta)
        common_beta_guess = np.average(weibull_fit_beta_array)

        def __BIC_minimizer(common_beta_X):
            '''
            __BIC_minimizer is used by the minimize function to get the beta which gives the lowest overall BIC
            '''
            BIC_tot = 0
            for stress in unique_stresses_f:
                failure_current_stress_df = f_df[f_df['stress'] == stress]
                FAILURES = failure_current_stress_df['times'].values
                if right_censored is not None:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None
                weibull_fit_common_beta = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_beta_X)
                BIC_tot += weibull_fit_common_beta.BIC
            return BIC_tot

        if common_beta_method == 'BIC':
            optimized_beta_results = minimize(__BIC_minimizer, x0=common_beta_guess, method='nelder-mead')
            common_beta = optimized_beta_results.x[0]
        elif common_beta_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_beta = sum(weights * np.array(weibull_fit_beta_array))
        elif common_beta_method == 'average':
            common_beta = common_beta_guess  # this was just the numerical average obtained above
        self.common_beta = common_beta

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common beta plot
        AICc_total = 0
        BIC_total = 0
        AICc = True
        for i, stress in enumerate(unique_stresses_f):
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                else:
                    RIGHT_CENSORED = None
            else:
                RIGHT_CENSORED = None
            weibull_fit_common_beta = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_beta)
            weibull_fit_alpha_array_common_beta.append(weibull_fit_common_beta.alpha)
            if type(weibull_fit_common_beta.AICc) == str:
                AICc = False
            else:
                AICc_total += weibull_fit_common_beta.AICc
            BIC_total += weibull_fit_common_beta.BIC
            if show_plot is True:
                weibull_fit_common_beta.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Weibull_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
                plt.legend(title='Stress')
                plt.xlim(10 ** (xmin + 1), 10 ** (xmax - 1))
                if common_beta_method == 'BIC':
                    plt.title(str('ALT Weibull Probability Plot\nOptimal BIC ' + r'$\beta$ = ' + str(round(common_beta, 4))))
                elif common_beta_method == 'weighted_average':
                    plt.title(str('ALT Weibull Probability Plot\nWeighted average ' + r'$\beta$ = ' + str(round(common_beta, 4))))
                elif common_beta_method == 'average':
                    plt.title(str('ALT Weibull Probability Plot\nAverage ' + r'$\beta$ = ' + str(round(common_beta, 4))))
        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        beta_difs = (common_beta - np.array(weibull_fit_beta_array)) / np.array(weibull_fit_beta_array)
        beta_differences = []
        for item in beta_difs:
            if item > 0:
                beta_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                beta_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original alpha': weibull_fit_alpha_array, 'original beta': weibull_fit_beta_array, 'new alpha': weibull_fit_alpha_array_common_beta, 'common beta': np.ones_like(unique_stresses_f) * common_beta, 'beta change': beta_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original alpha', 'original beta', 'new alpha', 'common beta', 'beta change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Weibull probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)


class ALT_probability_plot_Lognormal:
    '''
    ALT_probability_plot_Lognormal

    This function produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level, as well as a refitted distribution assuming a common shape parameter (sigma).

    Inputs:
    failures - an array or list of all the failure times
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    common_sigma_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_sigma parameter. 'BIC' will find the common_sigma that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_sigma
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_sigma

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in sigma
    common_sigma - the common sigma parameter

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_sigma_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_sigma_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_sigma_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_sigma_method must be either BIC, weighted_average, or average. Default is BIC.')
        if len(failures) != len(failure_stress):
            raise ValueError('The length of failures does not match the length of failure_stress')
        if type(failures) is list:
            failures = np.array(failures)
        elif type(failures) is np.ndarray:
            pass
        else:
            raise ValueError('failures must be an array or list')
        if type(failure_stress) is list:
            failure_stress = np.array(failure_stress)
        elif type(failure_stress) is np.ndarray:
            pass
        else:
            raise ValueError('failure_stress must be an array or list')
        if right_censored is not None:
            if len(right_censored) != len(right_censored_stress):
                raise ValueError('The length of right_censored does not match the length of right_censored_stress')
            if type(right_censored) is list:
                right_censored = np.array(right_censored)
            elif type(right_censored) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored must be an array or list')
            if type(right_censored_stress) is list:
                right_censored_stress = np.array(right_censored_stress)
            elif type(right_censored_stress) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored_stress must be an array or list')

        xmin = np.floor(np.log10(min(failures))) - 1
        xmax = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin, xmax, 100)

        if right_censored is not None:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack([np.ones_like(failures), np.zeros_like(right_censored)])
        else:
            TIMES = failures
            STRESS = failure_stress
            CENS_CODES = np.ones_like(failures)

        data = {'times': TIMES, 'stress': STRESS, 'cens_codes': CENS_CODES}
        df = pd.DataFrame(data, columns=['times', 'stress', 'cens_codes'])
        df_sorted = df.sort_values(by=['cens_codes', 'stress', 'times'])
        is_failure = df_sorted['cens_codes'] == 1
        is_right_cens = df_sorted['cens_codes'] == 0
        f_df = df_sorted[is_failure]
        rc_df = df_sorted[is_right_cens]
        unique_stresses_f = f_df.stress.unique()
        if right_censored is not None:
            unique_stresses_rc = rc_df.stress.unique()
            for item in unique_stresses_rc:  # check that there are no unique right_censored stresses that are not also in failure stresses
                if item not in unique_stresses_f:
                    raise ValueError('The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done.')

        lognormal_fit_mu_array = []
        lognormal_fit_sigma_array = []
        lognormal_fit_mu_array_common_sigma = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        weights_array = []
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
        for stress in unique_stresses_f:
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            len_f = len(FAILURES)
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    len_rc = len(RIGHT_CENSORED)
                else:
                    RIGHT_CENSORED = None
                    len_rc = 0
            else:
                RIGHT_CENSORED = None
                len_rc = 0

            weights_array.append(len_f + len_rc)
            lognormal_fit = Fit_Lognormal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            lognormal_fit_mu_array.append(lognormal_fit.mu)
            lognormal_fit_sigma_array.append(lognormal_fit.sigma)
        common_sigma_guess = np.average(lognormal_fit_sigma_array)

        def __BIC_minimizer(common_sigma_X):
            '''
            __BIC_minimizer is used by the minimize function to get the sigma which gives the lowest overall BIC
            '''
            BIC_tot = 0
            for stress in unique_stresses_f:
                failure_current_stress_df = f_df[f_df['stress'] == stress]
                FAILURES = failure_current_stress_df['times'].values
                if right_censored is not None:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None
                lognormal_fit_common_sigma = Fit_Lognormal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_sigma_X)
                BIC_tot += lognormal_fit_common_sigma.BIC
            return BIC_tot

        if common_sigma_method == 'BIC':
            optimized_sigma_results = minimize(__BIC_minimizer, x0=common_sigma_guess, method='nelder-mead')
            common_sigma = optimized_sigma_results.x[0]
        elif common_sigma_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_sigma = sum(weights * np.array(lognormal_fit_sigma_array))
        elif common_sigma_method == 'average':
            common_sigma = common_sigma_guess  # this was just the numerical average obtained above
        self.common_sigma = common_sigma

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common sigma plot
        AICc_total = 0
        BIC_total = 0
        AICc = True
        for i, stress in enumerate(unique_stresses_f):
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                else:
                    RIGHT_CENSORED = None
            else:
                RIGHT_CENSORED = None
            lognormal_fit_common_sigma = Fit_Lognormal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_sigma)
            lognormal_fit_mu_array_common_sigma.append(lognormal_fit_common_sigma.mu)
            if type(lognormal_fit_common_sigma.AICc) == str:
                AICc = False
            else:
                AICc_total += lognormal_fit_common_sigma.AICc
            BIC_total += lognormal_fit_common_sigma.BIC
            if show_plot is True:
                lognormal_fit_common_sigma.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Lognormal_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
                plt.legend(title='Stress')
                plt.xlim(10 ** (xmin + 1), 10 ** (xmax - 1))
                if common_sigma_method == 'BIC':
                    plt.title(str('ALT Lognormal Probability Plot\nOptimal BIC ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))
                elif common_sigma_method == 'weighted_average':
                    plt.title(str('ALT Lognormal Probability Plot\nWeighted average ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))
                elif common_sigma_method == 'average':
                    plt.title(str('ALT Lognormal Probability Plot\nAverage ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        sigma_difs = (common_sigma - np.array(lognormal_fit_sigma_array)) / np.array(lognormal_fit_sigma_array)
        sigma_differences = []
        for item in sigma_difs:
            if item > 0:
                sigma_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                sigma_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original mu': lognormal_fit_mu_array, 'original sigma': lognormal_fit_sigma_array, 'new mu': lognormal_fit_mu_array_common_sigma, 'common sigma': np.ones_like(unique_stresses_f) * common_sigma, 'sigma change': sigma_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original mu', 'original sigma', 'new mu', 'common sigma', 'sigma change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Lognormal probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)


class ALT_probability_plot_Normal:
    '''
    ALT_probability_plot_Normal

    This function produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level, as well as a refitted distribution assuming a common shape parameter (sigma).

    Inputs:
    failures - an array or list of all the failure times
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    common_sigma_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_sigma parameter. 'BIC' will find the common_sigma that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_sigma
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_sigma

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in sigma
    common_sigma - the common sigma parameter

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_sigma_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_sigma_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_sigma_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_sigma_method must be either BIC, weighted_average, or average. Default is BIC.')
        if len(failures) != len(failure_stress):
            raise ValueError('The length of failures does not match the length of failure_stress')
        if type(failures) is list:
            failures = np.array(failures)
        elif type(failures) is np.ndarray:
            pass
        else:
            raise ValueError('failures must be an array or list')
        if type(failure_stress) is list:
            failure_stress = np.array(failure_stress)
        elif type(failure_stress) is np.ndarray:
            pass
        else:
            raise ValueError('failure_stress must be an array or list')
        if right_censored is not None:
            if len(right_censored) != len(right_censored_stress):
                raise ValueError('The length of right_censored does not match the length of right_censored_stress')
            if type(right_censored) is list:
                right_censored = np.array(right_censored)
            elif type(right_censored) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored must be an array or list')
            if type(right_censored_stress) is list:
                right_censored_stress = np.array(right_censored_stress)
            elif type(right_censored_stress) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored_stress must be an array or list')

        delta = max(failures) - min(failures)
        xmin = min(failures) - delta * 0.2
        xmax = max(failures) + delta * 0.2
        xvals = np.linspace(xmin, xmax, 100)

        if right_censored is not None:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack([np.ones_like(failures), np.zeros_like(right_censored)])
        else:
            TIMES = failures
            STRESS = failure_stress
            CENS_CODES = np.ones_like(failures)

        data = {'times': TIMES, 'stress': STRESS, 'cens_codes': CENS_CODES}
        df = pd.DataFrame(data, columns=['times', 'stress', 'cens_codes'])
        df_sorted = df.sort_values(by=['cens_codes', 'stress', 'times'])
        is_failure = df_sorted['cens_codes'] == 1
        is_right_cens = df_sorted['cens_codes'] == 0
        f_df = df_sorted[is_failure]
        rc_df = df_sorted[is_right_cens]
        unique_stresses_f = f_df.stress.unique()
        if right_censored is not None:
            unique_stresses_rc = rc_df.stress.unique()
            for item in unique_stresses_rc:  # check that there are no unique right_censored stresses that are not also in failure stresses
                if item not in unique_stresses_f:
                    raise ValueError('The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done.')

        normal_fit_mu_array = []
        normal_fit_sigma_array = []
        normal_fit_mu_array_common_sigma = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        weights_array = []
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common sigma parameter
        for stress in unique_stresses_f:
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            len_f = len(FAILURES)
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    len_rc = len(RIGHT_CENSORED)
                else:
                    RIGHT_CENSORED = None
                    len_rc = 0
            else:
                RIGHT_CENSORED = None
                len_rc = 0

            weights_array.append(len_f + len_rc)
            normal_fit = Fit_Normal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            normal_fit_mu_array.append(normal_fit.mu)
            normal_fit_sigma_array.append(normal_fit.sigma)
        common_sigma_guess = np.average(normal_fit_sigma_array)

        def __BIC_minimizer(common_sigma_X):
            '''
            __BIC_minimizer is used by the minimize function to get the sigma which gives the lowest overall BIC
            '''
            BIC_tot = 0
            for stress in unique_stresses_f:
                failure_current_stress_df = f_df[f_df['stress'] == stress]
                FAILURES = failure_current_stress_df['times'].values
                if right_censored is not None:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None
                normal_fit_common_sigma = Fit_Normal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_sigma_X)
                BIC_tot += normal_fit_common_sigma.BIC
            return BIC_tot

        if common_sigma_method == 'BIC':
            optimized_sigma_results = minimize(__BIC_minimizer, x0=common_sigma_guess, method='nelder-mead')
            common_sigma = optimized_sigma_results.x[0]
        elif common_sigma_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_sigma = sum(weights * np.array(normal_fit_sigma_array))
        elif common_sigma_method == 'average':
            common_sigma = common_sigma_guess  # this was just the numerical average obtained above
        self.common_sigma = common_sigma

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common sigma plot
        AICc_total = 0
        BIC_total = 0
        AICc = True
        for i, stress in enumerate(unique_stresses_f):
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                else:
                    RIGHT_CENSORED = None
            else:
                RIGHT_CENSORED = None
            normal_fit_common_sigma = Fit_Normal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_sigma)
            normal_fit_mu_array_common_sigma.append(normal_fit_common_sigma.mu)
            if type(normal_fit_common_sigma.AICc) == str:
                AICc = False
            else:
                AICc_total += normal_fit_common_sigma.AICc
            BIC_total += normal_fit_common_sigma.BIC
            if show_plot is True:
                normal_fit_common_sigma.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Normal_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
                plt.legend(title='Stress')
                plt.xlim(xmin, xmax)
                if common_sigma_method == 'BIC':
                    plt.title(str('ALT Normal Probability Plot\nOptimal BIC ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))
                elif common_sigma_method == 'weighted_average':
                    plt.title(str('ALT Normal Probability Plot\nWeighted average ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))
                elif common_sigma_method == 'average':
                    plt.title(str('ALT Normal Probability Plot\nAverage ' + r'$\sigma$ = ' + str(round(common_sigma, 4))))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        sigma_difs = (common_sigma - np.array(normal_fit_sigma_array)) / np.array(normal_fit_sigma_array)
        sigma_differences = []
        for item in sigma_difs:
            if item > 0:
                sigma_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                sigma_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original mu': normal_fit_mu_array, 'original sigma': normal_fit_sigma_array, 'new mu': normal_fit_mu_array_common_sigma, 'common sigma': np.ones_like(unique_stresses_f) * common_sigma, 'sigma change': sigma_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original mu', 'original sigma', 'new mu', 'common sigma', 'sigma change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Normal probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)


class ALT_probability_plot_Gamma:
    '''
    ALT_probability_plot_Gamma

    This function produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level, as well as a refitted distribution assuming a common shape parameter (beta).

    Inputs:
    failures - an array or list of all the failure times
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True
    common_beta_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_beta parameter. 'BIC' will find the common_beta that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_beta
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_beta

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in beta
    common_beta - the common beta parameter

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_beta_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_beta_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_beta_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_beta_method must be either BIC, weighted_average, or average. Default is BIC.')
        if len(failures) != len(failure_stress):
            raise ValueError('The length of failures does not match the length of failure_stress')
        if type(failures) is list:
            failures = np.array(failures)
        elif type(failures) is np.ndarray:
            pass
        else:
            raise ValueError('failures must be an array or list')
        if type(failure_stress) is list:
            failure_stress = np.array(failure_stress)
        elif type(failure_stress) is np.ndarray:
            pass
        else:
            raise ValueError('failure_stress must be an array or list')
        if right_censored is not None:
            if len(right_censored) != len(right_censored_stress):
                raise ValueError('The length of right_censored does not match the length of right_censored_stress')
            if type(right_censored) is list:
                right_censored = np.array(right_censored)
            elif type(right_censored) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored must be an array or list')
            if type(right_censored_stress) is list:
                right_censored_stress = np.array(right_censored_stress)
            elif type(right_censored_stress) is np.ndarray:
                pass
            else:
                raise ValueError('right_censored_stress must be an array or list')

        xvals = np.linspace(0, max(failures) * 1.2, 100)

        if right_censored is not None:
            TIMES = np.hstack([failures, right_censored])
            STRESS = np.hstack([failure_stress, right_censored_stress])
            CENS_CODES = np.hstack([np.ones_like(failures), np.zeros_like(right_censored)])
        else:
            TIMES = failures
            STRESS = failure_stress
            CENS_CODES = np.ones_like(failures)

        data = {'times': TIMES, 'stress': STRESS, 'cens_codes': CENS_CODES}
        df = pd.DataFrame(data, columns=['times', 'stress', 'cens_codes'])
        df_sorted = df.sort_values(by=['cens_codes', 'stress', 'times'])
        is_failure = df_sorted['cens_codes'] == 1
        is_right_cens = df_sorted['cens_codes'] == 0
        f_df = df_sorted[is_failure]
        rc_df = df_sorted[is_right_cens]
        unique_stresses_f = f_df.stress.unique()
        if right_censored is not None:
            unique_stresses_rc = rc_df.stress.unique()
            for item in unique_stresses_rc:  # check that there are no unique right_censored stresses that are not also in failure stresses
                if item not in unique_stresses_f:
                    raise ValueError('The right_censored_stress array contains values that are not in the failure_stress array. This is equivalent to trying to fit a distribution to only censored data and cannot be done.')

        gamma_fit_alpha_array = []
        gamma_fit_beta_array = []
        gamma_fit_alpha_array_common_beta = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        weights_array = []
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
        for stress in unique_stresses_f:
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            len_f = len(FAILURES)
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    len_rc = len(RIGHT_CENSORED)
                else:
                    RIGHT_CENSORED = None
                    len_rc = 0
            else:
                RIGHT_CENSORED = None
                len_rc = 0

            weights_array.append(len_f + len_rc)
            gamma_fit = Fit_Gamma_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            gamma_fit_alpha_array.append(gamma_fit.alpha)
            gamma_fit_beta_array.append(gamma_fit.beta)
        common_beta_guess = np.average(gamma_fit_beta_array)

        def __BIC_minimizer(common_beta_X):
            '''
            __BIC_minimizer is used by the minimize function to get the beta which gives the lowest overall BIC
            '''
            BIC_tot = 0
            for stress in unique_stresses_f:
                failure_current_stress_df = f_df[f_df['stress'] == stress]
                FAILURES = failure_current_stress_df['times'].values
                if right_censored is not None:
                    if stress in unique_stresses_rc:
                        right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                        RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                    else:
                        RIGHT_CENSORED = None
                else:
                    RIGHT_CENSORED = None
                gamma_fit_common_beta = Fit_Gamma_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_beta_X)
                BIC_tot += gamma_fit_common_beta.BIC
            return BIC_tot

        if common_beta_method == 'BIC':
            optimized_beta_results = minimize(__BIC_minimizer, x0=common_beta_guess, method='nelder-mead')
            common_beta = optimized_beta_results.x[0]
        elif common_beta_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_beta = sum(weights * np.array(gamma_fit_beta_array))
        elif common_beta_method == 'average':
            common_beta = common_beta_guess  # this was just the numerical average obtained above
        self.common_beta = common_beta

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common beta plot
        AICc_total = 0
        BIC_total = 0
        AICc = True
        for i, stress in enumerate(unique_stresses_f):
            failure_current_stress_df = f_df[f_df['stress'] == stress]
            FAILURES = failure_current_stress_df['times'].values
            if right_censored is not None:
                if stress in unique_stresses_rc:
                    right_cens_current_stress_df = rc_df[rc_df['stress'] == stress]
                    RIGHT_CENSORED = right_cens_current_stress_df['times'].values
                else:
                    RIGHT_CENSORED = None
            else:
                RIGHT_CENSORED = None
            gamma_fit_common_beta = Fit_Gamma_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_beta)
            gamma_fit_alpha_array_common_beta.append(gamma_fit_common_beta.alpha)
            if type(gamma_fit_common_beta.AICc) == str:
                AICc = False
            else:
                AICc_total += gamma_fit_common_beta.AICc
            BIC_total += gamma_fit_common_beta.BIC
            if show_plot is True:
                gamma_fit_common_beta.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Gamma_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
                plt.legend(title='Stress')
                plt.xlim([0, max(failures) * 1.2])
                if common_beta_method == 'BIC':
                    plt.title(str('ALT Gamma Probability Plot\nOptimal BIC ' + r'$\beta$ = ' + str(round(common_beta, 4))))
                elif common_beta_method == 'weighted_average':
                    plt.title(str('ALT Gamma Probability Plot\nWeighted average ' + r'$\beta$ = ' + str(round(common_beta, 4))))
                elif common_beta_method == 'average':
                    plt.title(str('ALT Gamma Probability Plot\nAverage ' + r'$\beta$ = ' + str(round(common_beta, 4))))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        beta_difs = (common_beta - np.array(gamma_fit_beta_array)) / np.array(gamma_fit_beta_array)
        beta_differences = []
        for item in beta_difs:
            if item > 0:
                beta_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                beta_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original alpha': gamma_fit_alpha_array, 'original beta': gamma_fit_beta_array, 'new alpha': gamma_fit_alpha_array_common_beta, 'common beta': np.ones_like(unique_stresses_f) * common_beta, 'beta change': beta_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original alpha', 'original beta', 'new alpha', 'common beta', 'beta change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Gamma probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)

