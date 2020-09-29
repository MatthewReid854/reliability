'''
Reliability_testing

This is a collection of several statistical tests and reliability test planners.
Included functions are:
one_sample_proportion - Calculates the upper and lower bounds of reliability for a given number of trials and successes.
two_proportion_test - Calculates whether the difference in test results between two samples is statistically significant.
sample_size_no_failures - used to determine the sample size required for a test in which no failures are expected, and the desired outcome is the lower bound on the reliability based on the sample size and desired confidence interval.
sequential_sampling_chart - plots the accept/reject boundaries for a given set of quality and risk levels. If supplied, the test results are also plotted on the chart.
reliability_test_planner - Finds the lower confidence bound on MTBF for a given test duration, number of failures, and specified confidence interval.
reliability_test_duration - Finds the duration of a reliability test based on producers and consumers risk, and the MTBF design and MTBF required.
chi2test - performs the chi-squared goodness of fit test to determine if we can accept or reject the hypothesis that data is from a distribution.
KStest - performs the Kolmogorov-Smirnov goodness of fit test to determine if we can accept or reject the hypothesis that data is from a distribution.
'''

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
from reliability.Distributions import Normal_Distribution, Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution


def one_sample_proportion(trials=None, successes=None, CI=0.95):
    '''
    Calculates the upper and lower bounds of reliability for a given number of trials and successes.

    inputs:
    trials - the number of trials which were conducted
    successes - the number of trials which were successful
    CI - the desired confidence interval. Defaults to 0.95 for 95% CI.

    returns: lower, upper - Confidence interval limits.
        will return nan for lower or upper if only one sided CI is calculated (ie. when successes=0 or successes=trials).
    '''
    if trials is None or successes is None:
        raise ValueError('You must specify the number of trials and successes.')
    if successes > trials:
        raise ValueError('successes cannot be greater than trials')
    if successes == 0 or successes == trials:  # calculate 1 sided CI in these cases
        n = 1
    else:
        n = 2
    V1_lower = 2 * successes
    V2_lower = 2 * (trials - successes + 1)
    alpha_lower = (1 - CI) / n
    F_lower = ss.f.ppf(alpha_lower, V1_lower, V2_lower)
    LOWER_LIM = (V1_lower * F_lower) / (V2_lower + V1_lower * F_lower)

    V1_upper = 2 * (successes + 1)
    V2_upper = 2 * (trials - successes)
    alpha_upper = 1 - alpha_lower
    F_upper = ss.f.ppf(alpha_upper, V1_upper, V2_upper)
    UPPER_LIM = (V1_upper * F_upper) / (V2_upper + V1_upper * F_upper)

    return LOWER_LIM, UPPER_LIM  # will return nan for lower or upper if only one sided CI is calculated (ie. when successes=0 or successes=trials).


def two_proportion_test(sample_1_trials=None, sample_1_successes=None, sample_2_trials=None, sample_2_successes=None, CI=0.95):
    '''
    Calculates whether the difference in test results between two samples is statistically significant. For example, assume we have
    a poll of respondents in which 27/40 people agreed, and another poll in which 42/80 agreed. This test will determine if the difference
    is statistically significant for the given sample sizes at the specified confidence level.

    inputs:
    sample_1_trials - number of trials in the first sample
    sample_1_successes - number of successes in the first sample
    sample_2_trials - number of trials in the second sample
    sample_2_successes - number of successes in the second sample
    CI - desired confidence interval. Defaults to 0.95 for 95% CI.

    returns:
    lower,upper,result - lower and upper are bounds on the difference. If the bounds do not include 0 then it is a statistically significant difference.
    '''
    if CI < 0.5 or CI >= 1:
        raise ValueError('CI must be between 0.5 and 1. Default is 0.95')
    if sample_1_trials is None or sample_1_successes is None or sample_2_trials is None or sample_2_successes is None:
        raise ValueError('You must specify the number of trials and successes for both samples.')
    if sample_1_successes > sample_1_trials or sample_2_successes > sample_2_trials:
        raise ValueError('successes cannot be greater than trials')
    p1 = sample_1_successes / sample_1_trials
    p2 = sample_2_successes / sample_2_trials
    diff = p1 - p2
    Z = ss.norm.ppf(1 - ((1 - CI) / 2))
    k = Z * ((p1 * (1 - p1) / sample_1_trials) + (p2 * (1 - p2) / sample_2_trials)) ** 0.5
    lower = diff - k
    upper = diff + k
    if lower < 0 and upper > 0:
        result = 'non-significant'
    else:
        result = 'significant'
    return lower, upper, result


def sample_size_no_failures(reliability, CI=0.95, lifetimes=1, weibull_shape=1):
    '''
    This is used to determine the sample size required for a test in which no failures are expected, and the desired
    outcome is the lower bound on the reliability based on the sample size and desired confidence interval.

    inputs:
    reliability - lower bound on product reliability (between 0 and 1)
    CI - confidence interval of result (between 0.5 and 1). Defaults to 0.95 for 95% CI.
    lifetimes - if testing the product for multiple lifetimes then more failures are expected so a smaller sample
        size will be required to demonstrate the desired reliability (assuming no failures). Conversely, if testing for
        less than one full lifetime then a larger sample size will be required. Default is 1.
    weibull_shape - if the weibull shape (beta) of the failure mode is known, specify it here. Otherwise leave the
        default of 1 for the exponential distribution.

    returns:
    number of items required in the test. This will always be an integer (rounded up).
    '''
    if CI < 0.5 or CI >= 1:
        raise ValueError('CI must be between 0.5 and 1')
    if reliability <= 0 or reliability >= 1:
        raise ValueError('Reliability must be between 0 and 1')
    if weibull_shape < 0:
        raise ValueError('Weibull shape must be greater than 0. Default (exponential distribution) is 1. If unknown then use 1.')
    if lifetimes > 5:
        print('Testing for greater than 5 lifetimes is highly unlikely to result in zero failures.')
    if lifetimes <= 0:
        raise ValueError('lifetimes must be >0. Default is 1. No more than 5 is recommended due to test feasibility.')
    n = int(np.ceil((np.log(1 - CI)) / (lifetimes ** weibull_shape * np.log(reliability))))  # rounds up to nearest integer
    return n


def sequential_samling_chart(p1, p2, alpha, beta, show_plot=True, print_results=True, test_results=None, max_samples=100):
    '''
    This function plots the accept/reject boundaries for a given set of quality and risk levels. If supplied, the test results are also
    plotted on the chart.

    inputs:
    p1 - producer_quality. The acceptable failure rate for the producer (typical around 0.01)
    p2 - consumer_quality. The acceptable failure rate for the consumer (typical around 0.1)
    alpha - producer_risk. Producer's CI = 1-alpha (typically 0.05)
    beta - consumer_risk. Consumer's CI = 1-beta (typically 0.1)
    test_results - array or list of binary test results. eg. [0,0,0,1] for 3 successes and 1 failure. Default=None
    show_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True.
    max_samples - the x_lim of the plot. optional input. Default=100.

    outputs:
    The sequential sampling chart - A plot of sequential sampling chart with decision boundaries. test_results are only plotted on the chart
    if provided as an input.
    results - a dataframe of tabulated decision results.

    '''
    if type(test_results) == list:
        F = np.array(test_results)
    elif type(test_results) == np.ndarray:
        F = test_results
    elif test_results is None:
        F = None
    else:
        raise ValueError('test_results must be a binary array or list with 1 as failures and 0 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')

    a = 1 - alpha
    b = 1 - beta
    d = np.log(p2 / p1) + np.log((1 - p1) / (1 - p2))
    h1 = np.log((1 - a) / b) / d
    h2 = np.log((1 - b) / a) / d
    s = np.log((1 - p1) / (1 - p2)) / d

    xvals = np.arange(max_samples + 1)
    rejection_line = s * xvals - h1
    acceptance_line = s * xvals + h2
    acceptance_line[acceptance_line < 0] = 0

    upper_line = np.ones_like(xvals) * (s * max_samples - h1)
    lower_line_range = np.linspace(-h2 / s, max_samples, max_samples + 1)
    acceptance_line2 = s * lower_line_range + h2  # this is the visible part of the line that starts beyond x=0

    acceptance_array = np.asarray(np.floor(s * xvals + h2), dtype=int)
    rejection_array = np.asarray(np.ceil(s * xvals - h1), dtype=int)
    for i, x in enumerate(xvals):  # this replaces cases where the criteria exceeds the number of samples
        if rejection_array[i] > x:
            rejection_array[i] = -1

    data = {'Samples': xvals, 'Failures to accept': acceptance_array, 'Failures to reject': rejection_array}
    df = pd.DataFrame(data, columns=['Samples', 'Failures to accept', 'Failures to reject'])
    df.set_index('Samples', inplace=True)
    df.loc[df['Failures to accept'] < 0, 'Failures to accept'] = 'x'
    df.loc[df['Failures to reject'] < 0, 'Failures to reject'] = 'x'

    if print_results is True:
        print(df)

    if show_plot is True:
        # plots the results of tests if they are specified
        if type(F) == np.ndarray:
            if all(F) not in [0, 1]:
                raise ValueError('test_results must be a binary array or list with 0 as failures and 1 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')
            nx = []
            ny = []
            failure_count = 0
            sample_count = 0
            for f in F:
                if f == 0:
                    sample_count += 1
                    nx.append(sample_count)
                    ny.append(failure_count)
                elif f == 1:
                    sample_count += 1
                    nx.append(sample_count)
                    ny.append(failure_count)
                    failure_count += 1
                    nx.append(sample_count)
                    ny.append(failure_count)
                else:
                    raise ValueError('test_results must be a binary array or list with 0 as failures and 1 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')
            plt.plot(nx, ny, label='test results')

        # plots the decision boundaries and shades the areas red and green
        plt.plot(lower_line_range, acceptance_line2, linestyle='--', color='green')
        plt.plot(xvals, rejection_line, linestyle='--', color='red')
        plt.fill_between(xvals, rejection_line, upper_line, color='red', alpha=0.3, label='Reject sample')
        plt.fill_between(xvals, acceptance_line, rejection_line, color='gray', alpha=0.1, label='Keep Testing')
        plt.fill_between(lower_line_range, 0, acceptance_line2, color='green', alpha=0.3, label='Accept Sample')
        plt.ylim([0, max(rejection_line)])
        plt.xlim([0, max(xvals)])
        plt.xlabel('Number of samples tested')
        plt.ylabel('Number of failures from samples tested')
        plt.title('Sequential sampling decision boundaries')
        plt.legend()
        plt.show()
    return df


class reliability_test_planner:
    '''
    reliability_test_planner

    Solves for unknown test planner variables, given known variables.
    The Chi-squared distribution is used to find the lower confidence bound on MTBF for a given test duration, number of failures, and specified confidence interval.
    The equation for time-terminated tests is: MTBF = (2*test_duration)/(chisquared_inverse(CI, 2*number_of_failures+2))
    The equation for failure-terminated tests is: MTBF = (2*test_duration)/(chisquared_inverse(CI, 2*number_of_failures))
    This equation can be rearranged to solve for any of the 4 variables. For example, you may want to know how many failures you are allowed to have in a given test duration to achieve a particular MTBF.
    The user must specify any 3 out of the 4 variables (not including one_sided, print_results, or time_terminated) and the remaining variable will be calculated.

    Inputs:
    MTBF - mean time between failures. This is the lower confidence bound on the MTBF. Units given in same units as the test_duration.
    number_of_failures - the number of failures recorded (or allowed) to achieve the MTBF. Must be an integer.
    test_duration - the amount of time on test required (or performed) to achieve the MTBF. May also be distance, rounds fires, cycles, etc. Units given in same units as MTBF.
    CI - the confidence interval at which the lower confidence bound on the MTBF is given. Must be between 0.5 and 1. For example, specify 0.95 for 95% confidence interval.
    print_results - True/False. Default is True.
    one_sided - True/False. Default is True. If set to False, the two sided confidence interval will be returned.
    time_terminated - True/False. Default is True. If set to False, the formula for the failure-terminated test will be used.

    Outputs:
    If print_results is True, all the variables will be printed.
    An output object is also returned with the same values as the inputs and the remaining value also calculated.

    Examples:
    reliability_test_planner(test_duration=19520,CI=0.8,number_of_failures=7)
        Reliability Test Planner results for time-terminated test
        Solving for MTBF
        Test duration: 19520
        MTBF (lower confidence bound): 1658.3248534993454
        Number of failures: 7
        Confidence interval (2 sided):0.8

    output = reliability_test_planner(number_of_failures=6,test_duration=10000,CI=0.8, print_results=False)
    print(output.MTBF)
        949.4807763260345
    '''

    def __init__(self, MTBF=None, number_of_failures=None, CI=None, test_duration=None, one_sided=True, time_terminated=True, print_results=True):

        print_CI_warn = False  # used later if the CI is calculated
        if CI is not None:
            if CI < 0.5 or CI >= 1:
                raise ValueError('CI must be between 0.5 and 1. For example, specify CI=0.95 for 95% confidence interval')
            if one_sided is True:
                CI_adj = CI
            else:
                CI_adj = 1 - ((1 - CI) / 2)

        if time_terminated is True:
            p = 2
        elif time_terminated is False:
            p = 0
        else:
            raise ValueError('time_terminated must be True or False. Default is True for the time terminated test (a test stopped after a set time rather than after a set number of failures).')

        if one_sided is True:
            sides = 1
        elif one_sided is False:
            sides = 2
        else:
            raise ValueError('one_sided must be True or False. Default is True for the one sided confidence interval.')

        if print_results not in [True, False]:
            raise ValueError('print_results must be True or False. Default is True.')

        if number_of_failures is not None:
            if number_of_failures % 1 != 0 or number_of_failures < 0:
                raise ValueError('number_of_failures must be a positive integer')

        if MTBF is None and number_of_failures is not None and CI is not None and test_duration is not None:
            soln_type = 'MTBF'
            MTBF = (2 * test_duration) / ss.chi2.ppf(CI_adj, 2 * number_of_failures + p)

        elif MTBF is not None and number_of_failures is None and CI is not None and test_duration is not None:
            soln_type = 'failures'
            number_of_failures = 0
            while True:  # this requires an iterative search. Begins at 0 and increments by 1 until the solution is found
                result = (2 * test_duration) / ss.chi2.ppf(CI_adj, 2 * number_of_failures + p) - MTBF
                if result < 0:  # solution is found when result returns a negative number (indicating too many failures)
                    break
                number_of_failures += 1

            MTBF_check = (2 * test_duration) / ss.chi2.ppf(CI_adj, 2 * 0 + p)  # checks that the maximum possible MTBF (when there are 0 failures) is within the test_duration
            if MTBF_check < MTBF:
                raise ValueError('The specified MTBF is not possible given the specified test_duration. You must increase your test_duration or decrease your MTBF.')

        elif MTBF is not None and number_of_failures is not None and CI is None and test_duration is not None:
            soln_type = 'CI'
            CI_calc = ss.chi2.cdf(test_duration / (MTBF * 0.5), 2 * number_of_failures + p)
            if one_sided is True:
                CI = CI_calc
            else:
                CI = 1 - (2 * (1 - CI_calc))  # this can give negative numbers, but only when the inputs result in an impossible CI.
            if CI < 0.5:
                print_CI_warn = True

        elif MTBF is not None and number_of_failures is not None and CI is not None and test_duration is None:
            soln_type = 'test_duration'
            test_duration = ss.chi2.ppf(CI_adj, 2 * number_of_failures + p) * MTBF / 2

        elif MTBF is not None and number_of_failures is not None and CI is not None and test_duration is not None:
            raise ValueError('All inputs were specified. Nothing to calculate.')

        else:
            raise ValueError('More than one input was not specified. You must specify any 3 out of the 4 inputs (not including one_sided or print_results) and the remaining input will be calculated.')

        self.test_duration = test_duration
        self.MTBF = MTBF
        self.number_of_failures = number_of_failures
        self.CI = CI
        if print_results is True:
            if time_terminated is True:
                print('\nReliability Test Planner results for time-terminated test')
            else:
                print('\nReliability Test Planner results for failure-terminated test')
            if soln_type == 'MTBF':
                print('Solving for MTBF')
            elif soln_type == 'failures':
                print('Solving for number_of_failures')
            elif soln_type == 'CI':
                print('Solving for CI')
            else:
                print('Solving for test_duration')
            print('Test duration:', self.test_duration)
            print('MTBF (lower confidence bound):', self.MTBF)
            print('Number of failures:', self.number_of_failures)
            print(str('Confidence interval (' + str(sides) + ' sided): ' + str(self.CI)))
            if print_CI_warn is True:
                print('WARNING: The calculated CI is less than 0.5. This indicates that the desired MTBF is unachievable for the specified test_duration and number_of_failures.')


def reliability_test_duration(MTBF_required, MTBF_design, consumer_risk, producer_risk, one_sided=True, time_terminated=True, show_plot=True, print_results=True):
    '''
    reliability_test_duration

    Calculates the required duration for a reliability test to achieve the specified producers and consumers risks.
    This is done based on the specified MTBF required and MTBF design.

    Inputs:
    MTBF_required - the required MTBF that the equipment must demonstrate during the test
    MTBF_design - the design target for the MTBF that the producer aims to achieve
    consumer_risk - the risk the consumer is accepting. This is the probability that a bad product will be accepted as a good product by the consumer.
    producer_risk - the risk the producer is accepting. This is the probability that a good product will be rejected as a bad product by the consumer.
    one_sided - default is True. The risk is analogous to the confidence interval, and the confidence interval can be one sided or two sided.
    time_terminated - default is True. whether the test is time terminated or failure terminated. Typically it will be time terminated if the required test duration is sought.
    show_plot - True/False. Default is True. This will create a plot of the risk vs test duration. Use plt.show() to show it.
    print_results - True/False. Default is True. This will print the results to the console.

    Returns:
    test_duration
    '''

    if consumer_risk <= 0 or consumer_risk > 0.5:
        raise ValueError('consumer_risk must be between 0 and 0.5')
    if producer_risk <= 0 or producer_risk > 0.5:
        raise ValueError('producer_risk must be between 0 and 0.5')
    if MTBF_design <= MTBF_required:
        raise ValueError('MTBF_design must exceed MTBF_required')
    if one_sided not in [True, False]:
        raise ValueError('one_sided must be True or False. Default is True')
    if time_terminated not in [True, False]:
        raise ValueError('time_terminated must be True or False. Default is True')
    if show_plot not in [True, False]:
        raise ValueError('show_plot must be True or False. Default is True')
    if print_results not in [True, False]:
        raise ValueError('print_results must be True or False. Default is True')

    duration_array = []
    producer_risk_array = []
    failures = 0  # initial counter. Incremented each iteration
    solution_index = False  # initial vlue to be updated later
    max_failures = 1e10  # initial value to be updated later
    event_check = False
    time_start = time.time()
    time_out = 10  # seconds until first warning about long runtime
    while True:
        result1 = reliability_test_planner(number_of_failures=failures, CI=1 - consumer_risk, MTBF=MTBF_required, one_sided=one_sided, time_terminated=time_terminated, print_results=False)  # finds the test duration based on MTBF required and consumer risk
        result2 = reliability_test_planner(MTBF=MTBF_design, test_duration=result1.test_duration, number_of_failures=failures, one_sided=one_sided, time_terminated=time_terminated, print_results=False)  # finds the producer risk based on test duration and MTBR of design
        duration_array.append(result1.test_duration)
        producer_risk_array.append(result2.CI)
        if producer_risk_array[-1] < producer_risk and event_check is False:  # check whether the number of failures resulted in the correct producer risk
            solution_index = failures - 1  # we have exceeded the target so need to go back one to find the point it was below, and one more to find the point it was above
            max_failures = solution_index * 1.5
            event_check = True
        if failures > max_failures:
            break
        failures += 1  # increment failures
        if time.time() - time_start > time_out:
            print('WARNING: The algorithm is taking a long time to find the solution. This is probably because MTBF_required is too close to MTBF_design so the item struggles to pass the test. --- Current runtime:', int(round(time.time() - time_start, 0)), 'seconds')
            time_out += 10

    duration_solution = duration_array[solution_index]
    if print_results is True:
        if time_terminated is True:
            print('\nReliability Test Duration Solver for time-terminated test')
        else:
            print('\nReliability Test Duration Solver for failure-terminated test')
        print('Required test duration:', duration_solution)
        print("Specified consumer's risk:", consumer_risk)
        print("Specified producer's risk:", producer_risk)
        print('Specified MTBF required by the consumer:', MTBF_required)
        print('Specified MTBF designed to by the producer:', MTBF_design)

    if show_plot is True:
        consumer_risk_array = np.ones_like(duration_array) * consumer_risk
        plt.plot(duration_array, producer_risk_array, label="Producer's risk")
        plt.plot(duration_array, consumer_risk_array, label="Consumer's risk")
        plt.scatter(duration_array, producer_risk_array, color='k', marker='.', label='Failure events')

        plt.xlabel('Test duration')
        plt.ylabel('Risk')
        plt.legend(loc='upper right')
        plt.xlim(min(duration_array), max(duration_array))
        plt.axvline(x=duration_solution, color='k', linestyle='--', linewidth=1)
        plt.title("Test duration vs Producer's and Consumer's Risk")
        plt.text(x=duration_solution, y=plt.ylim()[0], s=str(' Test duration\n ' + str(int(math.ceil(duration_solution)))), va='bottom')
    return duration_solution


class chi2test:
    '''
    chi2test

    Performs the Chi-squared test for goodness of fit to determine whether we can accept or reject the hypothesis that the data is from the specified distribution at the specified level of significance.
    This method is not a means of comparing distributions (which can be done with AICc, BIC, and AD), but instead allows us to accept or reject a hypothesis that data come from a distribution.
    Note that the result is sensitive to the bins. For this reason, it is recommended to leave bins as the default value.

    Inputs:
    distribution - a distribution object created using the reliability.Distributions module
    data - an array or list of data that are hypothesised to come from the distribution
    significance - This is the complement of confidence. 0.05 significance is the same as 95% confidence. Must be between 0 and 0.5. Default is 0.05.
    bins - an array or list of the bin edges from which to group the data OR a string for the bin edge method from numpy. String options are auto, fd, doane, scott, stone, rice, sturges, or sqrt. For more information see the numpy documentation on numpy.histogram_bin_edges. Default is auto.
    print_results - if True the results will be printed. Default is True
    show_plot - if True a plot of the distribution and histogram will be shown. Default is True.

    Outputs:
    chisquared_statistic - the chi-squared statistic
    chisquared_critical_value - the chi-squared critical value
    hypothesis - 'ACCEPT' or 'REJECT'. If chisquared_statistic < chisquared_critical_value then we can accept the hypothesis that the data is from the specified distribution
    bin_edges - the bin edges used. If bins is a list or array then bin_edges = bins. If bins is a string then you can find the bin_edges that were calculated using this output.
    '''

    def __init__(self, distribution, data, significance=0.05, bins=None, print_results=True, show_plot=True):

        # ensure the input is a distribution object
        if type(distribution) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution]:
            raise ValueError('distribution must be a probability distribution object from the reliability.Distributions module. First define the distribution using Reliability.Distributions.___')

        # ensure data is a list or array
        if type(data) not in [list, np.ndarray]:
            raise ValueError('data must be a list or array')
        if min(data) < 0 and type(distribution) != Normal_Distribution:
            raise ValueError('data contains values below 0 which is not appropriate when the distribution is not a Normal Distribution')

        if significance <= 0 or significance > 0.5:
            raise ValueError('significance should be between 0 and 0.5. Default is 0.05 which gives 95% confidence')

        if bins is None:
            bins = 'auto'
        if type(bins) not in [str, list, np.ndarray]:
            raise ValueError('bins must be a list or array of the bin edges OR a string for the bin edge method from numpy. String options are auto, fd, doane, scott, stone, rice, sturges, or sqrt. For more information see the numpy documentation on numpy.histogram_bin_edges')

        observed, bin_edges = np.histogram(data, bins=bins, normed=False)  # get a histogram of the data to find the observed values

        if sum(observed) != len(data):
            print('Warning: the bins do not encompass all of the data')
            print('data range:', min(data), 'to', max(data))
            print('bins range:', min(bin_edges), 'to', max(bin_edges))
            observed, bin_edges = np.histogram(data, bins='auto', normed=False)
            print('bins has been reset to "auto".')
            print('The new bins are:', bin_edges, '\n')

        if min(bin_edges < 0) and type(distribution) != Normal_Distribution:
            observed, bin_edges = np.histogram(data, bins='auto', normed=False)  # error will result if bins contains values below 0 for anything but the Normal Distribution
            print('Warning: The specified bins contained values below 0. This is not appropriate when the distribution is not a Normal Distribution. bins has been reset to "auto".')
            print('The new bins are:', bin_edges)

        cdf = distribution.CDF(xvals=bin_edges, show_plot=False)
        cdf_diff = np.diff(cdf) / sum(np.diff(cdf))  # this ensures the sum is 1
        expected = len(data) * cdf_diff

        n = len(observed)
        parameters = distribution.parameters
        if parameters[-1] == 0:  # if the gamma parameter is 0 then adjust the number of parameters to ignore gamma
            k = len(parameters) - 1
        else:
            k = len(parameters)
        if n - k - 1 <= 0:
            raise ValueError(str('The length of bins is insufficient. Using a ' + str(distribution.name2) + ' distribution, the minimum acceptable length of bins is ' + str(k + 2)))

        self.bin_edges = bin_edges
        self.chisquared_statistic, _ = ss.chisquare(f_obs=observed, f_exp=expected, ddof=k)
        self.chisquared_critical_value = ss.chi2.ppf(1 - significance, df=n - k - 1)
        if self.chisquared_statistic < self.chisquared_critical_value:
            self.hypothesis = 'ACCEPT'
        else:
            self.hypothesis = 'REJECT'

        if print_results is True:
            print('Chi-squared statistic:', self.chisquared_statistic)
            print('Chi-squared critical value:', self.chisquared_critical_value)
            print('At the', significance, 'significance level, we can', self.hypothesis, 'the hypothesis that the data comes from a', distribution.param_title_long)

        if show_plot is True:
            plt.figure('Chi-squared test')
            bin_edges_to_plot = np.nan_to_num(x=bin_edges, posinf=max(data) * 1000, neginf=min(data))
            plt.hist(x=data, bins=bin_edges_to_plot, density=True, cumulative=True, color='lightgrey', edgecolor='k', linewidth=0.5, label='Cumulative Histogram')
            distribution.CDF(label=distribution.param_title_long)
            plt.title('Chi-squared test\nHypothesised distribution CDF vs cumulative histogram of data')
            xmax = max(distribution.quantile(0.9999), max(data))
            xmin = min(distribution.quantile(0.0001), min(data))
            if xmin > 0 and xmin / (xmax - xmin) < 0.05:  # if xmin is near zero then set it to zero
                xmin = 0
            plt.xlim(xmin, xmax)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.show()


class KStest:
    '''
    KStest

    Performs the Kolmogorov-Smirnov goodness of fit test to determine whether we can accept or reject the hypothesis that the data is from the specified distribution at the specified level of significance.
    This method is not a means of comparing distributions (which can be done with AICc, BIC, and AD), but instead allows us to accept or reject a hypothesis that data come from a distribution.

    Inputs:
    distribution - a distribution object created using the reliability.Distributions module
    data - an array or list of data that are hypothesised to come from the distribution
    significance - This is the complement of confidence. 0.05 significance is the same as 95% confidence. Must be between 0 and 0.5. Default is 0.05.
    print_results - if True the results will be printed. Default is True
    show_plot - if True a plot of the distribution CDF and empirical CDF will be shown. Default is True.

    Outputs:
    KS_statistic - the Kolmogorov-Smirnov statistic
    KS_critical_value - the Kolmogorov-Smirnov critical value
    hypothesis - 'ACCEPT' or 'REJECT'. If KS_statistic < KS_critical_value then we can accept the hypothesis that the data is from the specified distribution
    '''

    def __init__(self, distribution, data, significance=0.05, print_results=True, show_plot=True):

        # ensure the input is a distribution object
        if type(distribution) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution]:
            raise ValueError('distribution must be a probability distribution object from the reliability.Distributions module. First define the distribution using Reliability.Distributions.___')

        if min(data) < 0 and type(distribution) != Normal_Distribution:
            raise ValueError('data contains values below 0 which is not appropriate when the distribution is not a Normal Distribution')

        if significance <= 0 or significance > 0.5:
            raise ValueError('significance should be between 0 and 0.5. Default is 0.05 which gives 95% confidence')

        # need to sort data to ensure it is ascending
        if type(data) is list:
            data = np.sort(np.array(data))
        elif type(data) is np.ndarray:
            data = np.sort(data)
        else:
            raise ValueError('data must be an array or list')

        n = len(data)
        fitted_cdf = distribution.CDF(xvals=data, show_plot=False)

        i_array = np.arange(1, n + 1)  # array of 1 to n
        Sn = i_array / n  # empirical cdf 1
        Sn_1 = (i_array - 1) / n  # empirical cdf 2
        self.KS_statistic = max(np.hstack([abs(fitted_cdf - Sn), abs(fitted_cdf - Sn_1)]))  # Kolmogorov-Smirnov test statistic
        self.KS_critical_value = ss.kstwo.ppf(q=1 - significance, n=n)

        if self.KS_statistic < self.KS_critical_value:
            self.hypothesis = 'ACCEPT'
        else:
            self.hypothesis = 'REJECT'

        if print_results is True:
            print('Kolmogorov-Smirnov statistic:', self.KS_statistic)
            print('Kolmogorov-Smirnov critical value:', self.KS_critical_value)
            print('At the', significance, 'significance level, we can', self.hypothesis, 'the hypothesis that the data comes from a', distribution.param_title_long)

        if show_plot is True:
            plt.figure('Kolmogorov-Smirnov test')
            Sn_all = np.hstack([Sn_1, 1])
            SN_plot_x = [0]
            SN_plot_y = [0]
            for idx in np.arange(n):  # build the step plot
                SN_plot_x.extend((data[idx], data[idx]))
                SN_plot_y.extend((Sn_all[idx], Sn_all[idx + 1]))
            SN_plot_x.append(max(data) * 1000)
            SN_plot_y.append(1)
            distribution.CDF(label=distribution.param_title_long)
            plt.plot(SN_plot_x, SN_plot_y, label='Empirical CDF')
            xmax = max(distribution.quantile(0.9999), max(data))
            xmin = min(distribution.quantile(0.0001), min(data))
            if xmin > 0 and xmin / (xmax - xmin) < 0.05:  # if xmin is near zero then set it to zero
                xmin = 0
            plt.xlim(xmin, xmax)
            plt.ylim(0, 1.1)
            plt.title('Kolmogorov-Smirnov test\nHypothesised distribution CDF vs empirical CDF of data')
            plt.legend()
            plt.show()
