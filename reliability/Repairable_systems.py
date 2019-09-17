'''
Repairable systems
This is a collection of functions used for repairable systems.

Currently included functions are:
reliability_growth - uses the Duane method to find the instantaneous MTBF and produce a reliability growth plot.
optimal_replacement_time - Calculates the cost model to determine how cost varies with replacement time. The cost model may be NHPP (as good as old) or HPP (as good as new). Default is HPP.
ROCOF - rate of occurrence of failures. Uses the Laplace test to determine if there is a trend in the failure times.
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import integrate
import scipy.stats as ss


class reliability_growth:
    def __init__(self, times=None, xmax=None, target_MTBF=None, show_plot=True, print_results=True, **kwargs):
        '''
        Uses the Duane method to find the instantaneous MTBF and produce a reliability growth plot.

        Inputs:
        times - array of list of failure times
        xmax - xlim to plot up to. Default is 1.5*max(times)
        target_MTBF - specify the target MTBF to obtain the total time on test required to reach it.
        show_plot - True/False. Defaults to true. Other keyword arguments are passed to the plot for style
        print_results - True/False. Defaults to True.

        Outputs:
        If print_results is True it will print a summary of the results
        Lambda - the lambda parameter from the Duane model
        Beta - the beta parameter from the Duane model
        time_to_target - Time to target is only returned if target_MTBF is specified.
        If show_plot is True, it will plot the reliability growth. Use plt.show() to show the plot.
        '''
        if times is None:
            raise ValueError('times must be an array or list of failure times')
        if type(times) == list:
            times = np.sort(np.array(times))
        elif type(times) == np.ndarray:
            times = np.sort(times)
        else:
            raise ValueError('times must be an array or list of failure times')
        if min(times) < 0:
            raise ValueError('failure times cannot be negative. times must be an array or list of failure times')
        if xmax == None:
            xmax = int(max(times) * 1.5)
        if 'color' in kwargs:
            c = kwargs.pop('color')
        else:
            c = 'steelblue'

        N = np.arange(1, len(times) + 1)
        theta_c = times / N
        ln_t = np.log(times)
        ln_theta_c = np.log(theta_c)
        z = np.polyfit(ln_t, ln_theta_c, 1)  # fit a straight line to the data to get the parameters lambda and beta
        beta = 1 - z[0]
        Lambda = np.exp(-z[1])
        xvals = np.linspace(0, xmax, 1000)
        theta_i = (xvals ** (1 - beta)) / (Lambda * beta)  # the smooth line
        theta_i_points = (times ** (1 - beta)) / (Lambda * beta)  # the failure times highlighted along the line
        self.Lambda = Lambda
        self.Beta = beta

        if print_results == True:
            print('Reliability growth model parameters:\nlambda:', Lambda, '\nbeta:', beta)

        if target_MTBF is not None:
            t_target = (target_MTBF * Lambda * beta) ** (1 / (1 - beta))
            self.time_to_target = t_target
            print('Time to reach target MTBF:', t_target)
        else:
            self.time_to_target = 'specify a target to obtain the time_to_target'

        if show_plot == True:
            plt.plot(xvals, theta_i, color=c, **kwargs)
            plt.plot(times, theta_i_points, 'o', color=c, alpha=0.5)
            if target_MTBF is not None:
                plt.plot([0, t_target, t_target], [target_MTBF, target_MTBF, 0], 'red', label='Reliability target', linewidth=1)
            plt.title('Reliability Growth')
            plt.xlabel('Total time on test')
            plt.ylabel('Instantaneous MTBF')
            plt.xlim([0, max(xvals)])
            plt.ylim([0, max(theta_i) * 1.2])


class optimal_replacement_time:
    '''
    Calculates the cost model to determine how cost varies with replacement time.
    The cost model may be HPP (good as new replacement) or NHPP (as good as old replacement). Default is HPP.

    Inputs:
    Cost_PM - cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM - cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha - scale parameter of the underlying Weibull distribution
    weibull_beta - shape parameter of the underlying Weibull distribution. Should be greater than 1 otherwise conducting PM is not economical.
    q - restoration factor. q=1 is Power Law NHPP (as good as old), q=0 is HPP (as good as new). Default is q=0 (as good as new).
    show_plot - True/False. Defaults to True. Other plotting keywords are also accepted and used.
    print_results - True/False. Defaults to True

    Outputs:
    ORT - the optimal replacement time
    min_cost - the minimum cost per unit time
    Plot of cost model if show_plot is set to True. Use plt.show() to display it.
    Printed results if print_results is set to True.
    '''

    def __init__(self, cost_PM, cost_CM, weibull_alpha, weibull_beta, show_plot=True, print_results=True, q=0, **kwargs):
        if 'color' in kwargs:
            c = kwargs.pop('color')
        else:
            c = 'steelblue'
        if cost_PM > cost_CM:
            raise ValueError('Cost_PM must be less than Cost_CM otherwise preventative maintenance should not be conducted.')
        if weibull_beta < 1:
            warnings.warn('weibull_beta is < 1 so the hazard rate is decreasing, therefore preventative maintenance should not be conducted.')

        if q == 1:  # as good as old
            alpha_multiple = 4  # just used for plot limits
            t = np.linspace(1, weibull_alpha * alpha_multiple, 100000)
            CPUT = ((cost_PM * (t / weibull_alpha) ** weibull_beta) + cost_CM) / t
            ORT = weibull_alpha * ((cost_CM / (cost_PM * (weibull_beta - 1))) ** (1 / weibull_beta))
            min_cost = ((cost_PM * (ORT / weibull_alpha) ** weibull_beta) + cost_CM) / ORT
        elif q == 0:  # as good as new
            alpha_multiple = 3
            t = np.linspace(1, weibull_alpha * alpha_multiple, 10000)
            CPUT = []  # cost per unit time
            R = lambda x: np.exp(-((x / weibull_alpha) ** weibull_beta))
            for T in t:
                SF = np.exp(-((T / weibull_alpha) ** weibull_beta))
                integral_R, error = integrate.quad(R, 0, T)
                CPUT.append((cost_PM * SF + cost_CM * (1 - SF)) / integral_R)
                idx = np.argmin(CPUT)
                min_cost = CPUT[idx]  # minimum cost per unit time
                ORT = t[idx]  # optimal replacement time
        else:
            raise ValueError('q must be 0 or 1. Default is 0. Use 0 for "as good as new" and use 1 for "as good as old".')
        self.ORT = ORT
        self.min_cost = min_cost

        if min_cost < 1:
            min_cost_rounded = round(min_cost, -int(np.floor(np.log10(abs(min_cost)))) + 1)  # this rounds to exactly 2 sigfigs no matter the number of preceding zeros
        else:
            min_cost_rounded = round(min_cost, 2)
        ORT_rounded = round(ORT, 2)

        if print_results == True:
            if q == 0:
                print('Cost model assuming as good as new replacement (q=0):')
            else:
                print('Cost model assuming as good as old replacement (q=1):')
            print('The minimum cost per unit time is', min_cost_rounded, '\nThe optimal replacement time is', ORT_rounded)

        if show_plot == True:
            plt.plot(t, CPUT, color=c, **kwargs)
            plt.plot(ORT, min_cost, 'o', color=c)
            text_str = str('\nMinimum cost per unit time is ' + str(min_cost_rounded) + '\nOptimal replacement time is ' + str(ORT_rounded))
            plt.text(ORT, min_cost, text_str, verticalalignment='top')
            plt.xlabel('Replacement time')
            plt.ylabel('Cost per unit time')
            plt.title('Optimal replacement time estimation')
            plt.ylim([0, min_cost * 2])
            plt.xlim([0, weibull_alpha * alpha_multiple])


class ROCOF:
    '''
    Uses the failure times or failure interarrival times to determine if there is a trend in those times.
    The test for statistical significance is the Laplace test which compares the Laplace test statistic (U) with the z value (z_crit) from the standard normal distribution
    If there is a statistically significant trend, the parameters of the model (Lambda_hat and Beta_hat) are calculated.
    By default the results are printed and a plot of the times and MTBF is plotted.

    Inputs:
    times_between_failures - these are the failure interarrival times.
    failure_times - these are the actual failure times.
        Note 1: You can specify either times_between_failures OR failure_times but not both. Both options are provided for convenience so the conversion between the two is done internally. failure_times should be the same as np.cumsum(times_between_failures).
        Note 2: The repair time is assumed to be negligible. If the repair times are not negligibly small then you will need to manually adjust your input to factor in the repair times.
    test_end - use this to specify the end of the test if the test did not end at the time of the last failure.
    CI - the confidence interval for the Laplace test. Default is 0.95 for 95% CI.
    show_plot - True/False. Default is True. Plotting keywords are also accepted (eg. color, linestyle).
    print_results - True/False. Default is True

    Outputs:
    U - The Laplace test statistic
    z_crit - (lower,upper) bound on z value. This is based on the CI.
    trend - 'improving','worsening','constant'. This is based on the comparison of U with z_crit
    Beta_hat - the Beta parameter for the NHPP Power Law model. Only calculated if the trend is not constant.
    Lambda_hat - the Lambda parameter for the NHPP Power Law model. Only calculated if the trend is not constant.
    ROCOF - the Rate of OCcurrence Of Failures. Only calculated if the trend is constant. If trend is not constant then ROCOF changes over time in accordance with Beta_hat and Lambda_hat.
    printed results. Only printed if print_results is True.
    plotted results. Only plotted of plot_results is True. Use plt.show() to display it.
    '''

    def __init__(self, times_between_failures=None, failure_times=None, CI=0.95, test_end=None, show_plot=True, print_results=True, **kwargs):
        if times_between_failures is not None and failure_times is not None:
            raise ValueError('You have specified both times_between_failures and failure times. You can specify one but not both. Use times_between_failures for failure interarrival times, and failure_times for the actual failure times. failure_times should be the same as np.cumsum(times_between_failures)')
        if times_between_failures is not None:
            if any(t <= 0 for t in times_between_failures):
                raise ValueError('times_between_failures cannot be less than zero')
            if type(times_between_failures) == list:
                ti = times_between_failures
            elif type(times_between_failures) == np.ndarray:
                ti = list(times_between_failures)
            else:
                raise ValueError('times_between_failures must be a list or array')
        if failure_times is not None:
            if any(t <= 0 for t in failure_times):
                raise ValueError('failure_times cannot be less than zero')
            if type(failure_times) == list:
                failure_times = np.sort(np.array(failure_times))
            elif type(failure_times) == np.ndarray:
                failure_times = np.sort(failure_times)
            else:
                raise ValueError('failure_times must be a list or array')
            failure_times[1:] -= failure_times[:-1].copy()  # this is the opposite of np.cumsum
            ti = list(failure_times)
        if test_end is not None and type(test_end) not in [float, int]:
            raise ValueError('test_end should be a float or int. Use test_end to specify the end time of a test which was not failure terminated.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% confidence interval.')
        if test_end is None:
            tn = sum(ti)
            n = len(ti) - 1
        else:
            tn = test_end
            n = len(ti)
            if tn < sum(ti):
                raise ValueError('test_end cannot be less than the final test time')

        if 'linestyle' in kwargs:
            ls = kwargs.pop('linestyle')
        else:
            ls = '--'
        if 'label' in kwargs:
            label_1 = kwargs.pop('label')
        else:
            label_1 = 'Failure interarrival times'

        tc = np.cumsum(ti[0:n])
        sum_tc = sum(tc)
        z_crit = ss.norm.ppf((1 - CI) / 2)  # z statistic based on CI
        U = (sum_tc / n - tn / 2) / (tn * (1 / (12 * n)) ** 0.5)
        self.U = U
        self.z_crit = (z_crit, -z_crit)
        results_str = str('Laplace test results: U = ' + str(round(U, 3)) + ', z_crit = (' + str(round(z_crit, 2)) + ',+' + str(round(-z_crit, 2)) + ')')
        if print_results == True:
            print(results_str)

        x = np.arange(1, len(ti) + 1)
        if U < z_crit:
            if print_results == True:
                print('At', int(CI * 100), '% confidence level the ROCOF is IMPROVING. Assume NHPP.')
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = 'improving'
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = 'ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate.'
            if L < 1:
                L_rounded = round(L, -int(np.floor(np.log10(abs(L)))) + 3)  # this rounds to exactly 4 sigfigs no matter the number of preceding zeros
            else:
                L_rounded = round(L, 2)
            if print_results == True:
                print('ROCOF assuming NHPP has parameters: Beta_hat =', round(B, 3), ', Lambda_hat =', L_rounded)
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        elif U > -z_crit:
            if print_results == True:
                print('At', int(CI * 100), '% confidence level the ROCOF is WORSENING. Assume NHPP.')
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = 'worsening'
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = 'ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate.'
            if L < 1:
                L_rounded = round(L, -int(np.floor(np.log10(abs(L)))) + 3)  # this rounds to exactly 4 sigfigs no matter the number of preceding zeros
            else:
                L_rounded = round(L, 2)
            if print_results == True:
                print('ROCOF assuming NHPP has parameters: Beta_hat =', round(B, 3), ', Lambda_hat =', L_rounded)
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        else:
            if print_results == True:
                print('At', int(CI * 100), '% confidence level the ROCOF is CONSTANT. Assume HPP.')
            rocof = n / sum(ti)
            self.trend = 'constant'
            self.ROCOF = rocof
            self.Beta_hat = 'not calculated when trend is constant'
            self.Lambda_hat = 'not calculated when trend is constant'
            x_to_plot = x
            MTBF = np.ones_like(x_to_plot) / rocof
            if rocof < 1:
                rocof_rounded = round(rocof, -int(np.floor(np.log10(abs(rocof)))) + 1)  # this rounds to exactly 2 sigfigs no matter the number of preceding zeros
            else:
                rocof_rounded = round(rocof, 2)
            if print_results == True:
                print('ROCOF assuming HPP is', rocof_rounded, 'failures per unit time.')

        if show_plot == True:
            plt.plot(x_to_plot, MTBF, linestyle=ls, label='MTBF')
            plt.scatter(x, ti, label=label_1, **kwargs)
            plt.ylabel('Times between failures')
            plt.xlabel('Failure number')
            title_str = str('Failure interarrival times vs failure number\nAt ' + str(int(CI * 100)) + '% confidence level the ROCOF is ' + self.trend)
            plt.title(title_str)
            plt.legend()
