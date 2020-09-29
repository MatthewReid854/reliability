'''
ALT_probability_plotting

Probability plots for accelerated life testing (ALT) models

Within the module ALT_probability_plotting, are the following functions:
- ALT_probability_plot_Weibull - produces an ALT probability plot by fitting a Weibull distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Lognormal - produces an ALT probability plot by fitting a Lognormal distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Normal - produces an ALT probability plot by fitting a Normal distribution to each unique stress and then finding a common shape parameter
- ALT_probability_plot_Exponential - produces an ALT probability plot by fitting an Weibull distribution to each unique stress and then fitting an Exponential distribution (equivalent to forcing the common shape parameter to be 1)
'''

import numpy as np
import matplotlib.pyplot as plt
from reliability import Probability_plotting
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Normal_2P, Fit_Expon_1P
from reliability.Utils import probability_plot_xylims, probability_plot_xyticks
import pandas as pd
from scipy.optimize import minimize


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
    common_shape_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_shape parameter. 'BIC' will find the common_shape that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in shape
    common_shape - the common shape (beta) parameter
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_shape
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_shape

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_shape_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_shape_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_shape_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_shape_method must be either BIC, weighted_average, or average. Default is BIC.')
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

        xmin_log = np.floor(np.log10(min(failures))) - 1
        xmax_log = np.ceil(np.log10(max(failures))) + 1
        xvals = np.logspace(xmin_log, xmax_log, 100)

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
        weibull_fit_alpha_array_common_shape = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        weights_array = []
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common shape parameter
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
        common_shape_guess = np.average(weibull_fit_beta_array)

        def __BIC_minimizer(common_shape_X):  # lgtm [py/similar-function]
            '''
            __BIC_minimizer is used by the minimize function to get the shape which gives the lowest overall BIC
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
                weibull_fit_common_shape = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_shape_X)
                BIC_tot += weibull_fit_common_shape.BIC
            return BIC_tot

        if common_shape_method == 'BIC':
            optimized_beta_results = minimize(__BIC_minimizer, x0=common_shape_guess, method='nelder-mead')
            common_shape = optimized_beta_results.x[0]
        elif common_shape_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_shape = sum(weights * np.array(weibull_fit_beta_array))
        elif common_shape_method == 'average':
            common_shape = common_shape_guess  # this was just the numerical average obtained above
        self.common_shape = common_shape

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common beta plot
        AICc_total = 0
        BIC_total = 0
        x_array = np.array([])
        y_array = np.array([])
        AICc = True  # default flag that gets changed if AICc is a string
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
            weibull_fit_common_shape = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=common_shape)
            weibull_fit_alpha_array_common_shape.append(weibull_fit_common_shape.alpha)
            if type(weibull_fit_common_shape.AICc) == str:
                AICc = False
            else:
                AICc_total += weibull_fit_common_shape.AICc
            BIC_total += weibull_fit_common_shape.BIC
            if show_plot is True:
                weibull_fit_common_shape.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals, plot_CI=False)  # plotting of the confidence intervals has been turned off
                Probability_plotting.Weibull_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, plot_CI=False, color=color_list[i], label=str(stress))
            x, y = Probability_plotting.plotting_positions(failures=FAILURES, right_censored=RIGHT_CENSORED)
            x_array = np.append(x_array, np.array(x))
            y_array = np.append(y_array, np.array(y))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        beta_difs = (common_shape - np.array(weibull_fit_beta_array)) / np.array(weibull_fit_beta_array)
        beta_differences = []
        for item in beta_difs:
            if item > 0:
                beta_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                beta_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original alpha': weibull_fit_alpha_array, 'original beta': weibull_fit_beta_array, 'new alpha': weibull_fit_alpha_array_common_shape, 'common beta': np.ones_like(unique_stresses_f) * common_shape, 'beta change': beta_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original alpha', 'original beta', 'new alpha', 'common beta', 'beta change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        self.x_array = x_array
        self.y_array = y_array

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Weibull probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)

        if show_plot is True:
            plt.legend(title='Stress')
            if common_shape_method == 'BIC':
                plt.title(str('ALT Weibull Probability Plot\nOptimal BIC ' + r'$\beta$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'weighted_average':
                plt.title(str('ALT Weibull Probability Plot\nWeighted average ' + r'$\beta$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'average':
                plt.title(str('ALT Weibull Probability Plot\nAverage ' + r'$\beta$ = ' + str(round(common_shape, 4))))
            probability_plot_xylims(x=x_array, y=y_array, dist='weibull', spacing=0.05)
            probability_plot_xyticks()
            plt.tight_layout()


class ALT_probability_plot_Exponential:
    '''
    ALT_probability_plot_Exponential

    This function produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level (using Weibull to check beta), as well as a refitted distribution assuming an Exponential distribution (same as Weibull with beta = 1).
    This is essentially the same as ALT_probability_plot_Weibull, but the beta parameter is forced to be 1 to make it Exponential and the alpha parameter becomes the Lambda parameter since alpha=1/Lambda.
    Why do this instead of just a Weibull plot? For the same reason that we use the exponential distribution instead of the Weibull. It is a more simple distribution leading to less overfitting. The AICc and BIC will also show this effect as the number of parameters is 1 instead of 2.
    Note that the parameter 'common_shape_method' is not available in this plot as beta is forced to be 1.
    A comparison between the Weibull and Exponential distributions is included in this function which will raise a notification if the BIC of Weibull is lower than the BIC of Exponential.

    Inputs:
    failures - an array or list of all the failure times
    failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
    right_censored - an array or list of all the right censored failure times
    right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
    print_results - True/False. Default is True
    show_plot - True/False. Default is True

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in shape parameter
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the Exponential distribution
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the Exponential distribution
    BIC_sum_weibull - the sum of the BIC for each of the distributions when fitted using the Weibull distribution with common_shape_method = 'average'
    AICc_sum_weibull - the sum of the AICc for each of the distributions when fitted using the Weibull distribution with common_shape_method = 'average'

    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True):

        # input type checking and converting to arrays in preperation for creation of dataframe
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
        expon_fit_lambda_array = []
        color_list = ['steelblue', 'darkorange', 'red', 'green', 'purple', 'blue', 'grey', 'deeppink', 'cyan', 'chocolate']
        # within this loop, each list of failures and right censored values will be unpacked for each unique stress to find the common beta parameter
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

            weibull_fit = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            weibull_fit_alpha_array.append(weibull_fit.alpha)
            weibull_fit_beta_array.append(weibull_fit.beta)

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common beta plot
        AICc_total = 0
        BIC_total = 0
        AICc_total_weib = 0
        BIC_total_weib = 0
        AICc = True
        AICc_weib = True
        x_array = np.array([])
        y_array = np.array([])
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
            expon_fit = Fit_Expon_1P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False)
            weib_fit = Fit_Weibull_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_beta=np.average(weibull_fit_beta_array))
            expon_fit_lambda_array.append(expon_fit.Lambda)
            if type(expon_fit.AICc) == str:
                AICc = False
            else:
                AICc_total += expon_fit.AICc
            if type(weib_fit.AICc) == str:
                AICc_weib = False
            else:
                AICc_total_weib += weib_fit.AICc
            BIC_total += expon_fit.BIC
            BIC_total_weib += weib_fit.BIC
            if show_plot is True:
                expon_fit.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals, plot_CI=False)  # plotting of the confidence intervals has been turned off
                Probability_plotting.Weibull_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, plot_CI=False, color=color_list[i], label=str(stress))
            x, y = Probability_plotting.plotting_positions(failures=FAILURES, right_censored=RIGHT_CENSORED)
            x_array = np.append(x_array, np.array(x))
            y_array = np.append(y_array, np.array(y))

        self.BIC_sum = np.sum(BIC_total)
        self.BIC_sum_weibull = np.sum(BIC_total_weib)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        if AICc_weib is True:
            self.AICc_sum_weibull = np.sum(AICc_total_weib)
        else:
            self.AICc_sum_weibull = 'Insufficient Data'
        beta_difs = (1 - np.array(weibull_fit_beta_array)) / np.array(weibull_fit_beta_array)
        beta_differences = []
        for item in beta_difs:
            if item > 0:
                beta_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                beta_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'weibull alpha': weibull_fit_alpha_array, 'weibull beta': weibull_fit_beta_array, 'new 1/Lambda': 1 / np.array(expon_fit_lambda_array), 'common shape': np.ones_like(unique_stresses_f), 'shape change': beta_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'weibull alpha', 'weibull beta', 'new 1/Lambda', 'common shape', 'shape change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        self.x_array = x_array
        self.y_array = y_array

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Exponential probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)
            print('Total AICc (weibull):', self.AICc_sum_weibull)
            print('Total BIC (weibull):', self.BIC_sum_weibull)
        if self.BIC_sum > self.BIC_sum_weibull:
            print('WARNING: The Weibull distribution would be a more appropriate fit for this data set as it has a lower BIC (using the average method to obtain BIC) than the Exponential distribution.')

        if show_plot is True:
            plt.legend(title='Stress')
            plt.title('ALT Exponential Probability Plot')
            probability_plot_xylims(x=x_array, y=y_array, dist='weibull', spacing=0.05)
            probability_plot_xyticks()
            plt.tight_layout()


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
    common_shape_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_shape parameter. 'BIC' will find the common_shape that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_shape
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_shape

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in sigma
    common_shape - the common sigma parameter

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_shape_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_shape_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_shape_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_shape_method must be either BIC, weighted_average, or average. Default is BIC.')
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
        lognormal_fit_mu_array_common_shape = []
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
        common_shape_guess = np.average(lognormal_fit_sigma_array)

        def __BIC_minimizer(common_shape_X):  # lgtm [py/similar-function]
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
                lognormal_fit_common_shape = Fit_Lognormal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_shape_X)
                BIC_tot += lognormal_fit_common_shape.BIC
            return BIC_tot

        if common_shape_method == 'BIC':
            optimized_sigma_results = minimize(__BIC_minimizer, x0=common_shape_guess, method='nelder-mead')
            common_shape = optimized_sigma_results.x[0]
        elif common_shape_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_shape = sum(weights * np.array(lognormal_fit_sigma_array))
        elif common_shape_method == 'average':
            common_shape = common_shape_guess  # this was just the numerical average obtained above
        self.common_shape = common_shape

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common sigma plot
        AICc_total = 0
        BIC_total = 0
        x_array = np.array([])
        y_array = np.array([])
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
            lognormal_fit_common_shape = Fit_Lognormal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_shape)
            lognormal_fit_mu_array_common_shape.append(lognormal_fit_common_shape.mu)
            if type(lognormal_fit_common_shape.AICc) == str:
                AICc = False
            else:
                AICc_total += lognormal_fit_common_shape.AICc
            BIC_total += lognormal_fit_common_shape.BIC
            if show_plot is True:
                lognormal_fit_common_shape.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Lognormal_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
            x, y = Probability_plotting.plotting_positions(failures=FAILURES, right_censored=RIGHT_CENSORED)
            x_array = np.append(x_array, np.array(x))
            y_array = np.append(y_array, np.array(y))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        sigma_difs = (common_shape - np.array(lognormal_fit_sigma_array)) / np.array(lognormal_fit_sigma_array)
        sigma_differences = []
        for item in sigma_difs:
            if item > 0:
                sigma_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                sigma_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original mu': lognormal_fit_mu_array, 'original sigma': lognormal_fit_sigma_array, 'new mu': lognormal_fit_mu_array_common_shape, 'common sigma': np.ones_like(unique_stresses_f) * common_shape, 'sigma change': sigma_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original mu', 'original sigma', 'new mu', 'common sigma', 'sigma change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        self.x_array = x_array
        self.y_array = y_array

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Lognormal probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)

        if show_plot is True:
            plt.legend(title='Stress')
            if common_shape_method == 'BIC':
                plt.title(str('ALT Lognormal Probability Plot\nOptimal BIC ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'weighted_average':
                plt.title(str('ALT Lognormal Probability Plot\nWeighted average ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'average':
                plt.title(str('ALT Lognormal Probability Plot\nAverage ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            probability_plot_xylims(x=x_array, y=y_array, dist='lognormal', spacing=0.05)
            probability_plot_xyticks()
            plt.tight_layout()


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
    common_shape_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_shape parameter. 'BIC' will find the common_shape that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average.
    BIC_sum - the sum of the BIC for each of the distributions when fitted using the common_shape
    AICc_sum - the sum of the AICc for each of the distributions when fitted using the common_shape

    Outputs:
    The plot will be produced if show_plot is True
    A dataframe of the fitted distributions parameters will be printed if print_results is True
    results - a dataframe of the fitted distributions parameters and change in sigma
    common_shape - the common sigma parameter

    Note that the time to run the function will be a few seconds if you have a large amount of data and the common_shape_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer.
    With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient.
    '''

    def __init__(self, failures, failure_stress, right_censored=None, right_censored_stress=None, print_results=True, show_plot=True, common_shape_method='BIC'):

        # input type checking and converting to arrays in preperation for creation of dataframe
        if common_shape_method not in ['BIC', 'weighted_average', 'average']:
            raise ValueError('common_shape_method must be either BIC, weighted_average, or average. Default is BIC.')
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
        normal_fit_mu_array_common_shape = []
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
        common_shape_guess = np.average(normal_fit_sigma_array)

        def __BIC_minimizer(common_shape_X):  # lgtm [py/similar-function]
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
                normal_fit_common_shape = Fit_Normal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_shape_X)
                BIC_tot += normal_fit_common_shape.BIC
            return BIC_tot

        if common_shape_method == 'BIC':
            optimized_sigma_results = minimize(__BIC_minimizer, x0=common_shape_guess, method='nelder-mead')
            common_shape = optimized_sigma_results.x[0]
        elif common_shape_method == 'weighted_average':
            total_data = sum(weights_array)
            weights = np.array(weights_array) / total_data
            common_shape = sum(weights * np.array(normal_fit_sigma_array))
        elif common_shape_method == 'average':
            common_shape = common_shape_guess  # this was just the numerical average obtained above
        self.common_shape = common_shape

        # within this loop, each list of failures and right censored values will be unpacked for each unique stress and plotted as a probability plot as well as the CDF of the common sigma plot
        AICc_total = 0
        BIC_total = 0
        x_array = np.array([])
        y_array = np.array([])
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
            normal_fit_common_shape = Fit_Normal_2P(failures=FAILURES, right_censored=RIGHT_CENSORED, show_probability_plot=False, print_results=False, force_sigma=common_shape)
            normal_fit_mu_array_common_shape.append(normal_fit_common_shape.mu[0])
            if type(normal_fit_common_shape.AICc) == str:
                AICc = False
            else:
                AICc_total += normal_fit_common_shape.AICc
            BIC_total += normal_fit_common_shape.BIC
            if show_plot is True:
                normal_fit_common_shape.distribution.CDF(linestyle='--', color=color_list[i], xvals=xvals)
                Probability_plotting.Normal_probability_plot(failures=FAILURES, right_censored=RIGHT_CENSORED, color=color_list[i], label=str(stress))
            x, y = Probability_plotting.plotting_positions(failures=FAILURES, right_censored=RIGHT_CENSORED)
            x_array = np.append(x_array, np.array(x))
            y_array = np.append(y_array, np.array(y))

        self.BIC_sum = np.sum(BIC_total)
        if AICc is True:
            self.AICc_sum = np.sum(AICc_total)
        else:
            self.AICc_sum = 'Insufficient Data'
        sigma_difs = (common_shape - np.array(normal_fit_sigma_array)) / np.array(normal_fit_sigma_array)
        sigma_differences = []
        for item in sigma_difs:
            if item > 0:
                sigma_differences.append(str('+' + str(round(item * 100, 2)) + '%'))
            else:
                sigma_differences.append(str(str(round(item * 100, 2)) + '%'))
        results = {'stress': unique_stresses_f, 'original mu': normal_fit_mu_array, 'original sigma': normal_fit_sigma_array, 'new mu': normal_fit_mu_array_common_shape, 'common sigma': np.ones_like(unique_stresses_f) * common_shape, 'sigma change': sigma_differences}
        results_df = pd.DataFrame(results, columns=['stress', 'original mu', 'original sigma', 'new mu', 'common sigma', 'sigma change'])
        blankIndex = [''] * len(results_df)
        results_df.index = blankIndex
        self.results = results_df
        self.x_array = x_array
        self.y_array = y_array

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print('\nALT Normal probability plot results:')
            print(self.results)
            print('Total AICc:', self.AICc_sum)
            print('Total BIC:', self.BIC_sum)

        if show_plot is True:
            plt.legend(title='Stress')
            if common_shape_method == 'BIC':
                plt.title(str('ALT Normal Probability Plot\nOptimal BIC ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'weighted_average':
                plt.title(str('ALT Normal Probability Plot\nWeighted average ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            elif common_shape_method == 'average':
                plt.title(str('ALT Normal Probability Plot\nAverage ' + r'$\sigma$ = ' + str(round(common_shape, 4))))
            probability_plot_xylims(x=x_array, y=y_array, dist='normal', spacing=0.05)
            probability_plot_xyticks()
            plt.tight_layout()
