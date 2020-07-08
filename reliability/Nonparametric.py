'''
Non-parametric estimates of survival function, cumulative distribution function, and cumulative hazard function
Two estimation methods are implemented:
- Kaplan-Meier
- Nelson-Aalen
These methods arrive at very similar results but are distinctly different in their approach. Kaplan-Meier is more popular.
Both methods support failures and right censored data.
Confidence intervals are provided using the Greenwood formula with Normal approximation (as implemented in Minitab).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


class KaplanMeier:
    '''
    Kaplan-Meier

    Uses the Kaplan-Meier estimation method to calculate the reliability from failure data.
    Right censoring is supported and confidence bounds are provided.
    The confidence bounds are calculated using the Greenwood formula with Normal approximation, which is the same as
    featured in Minitab.
    The Kaplan-Meier method provides the SF. With a little algebra, the CDF and CHF are also obtained from the SF.
    It is not possible to obtain a useful version of the PDF or HF as the derivative of a stepwise function produces very spikey functions.

    Inputs:
    failures - an array or list of failure times. Sorting is automatic so times do not need to be provided in any order.
    right_censored - an array or list of right censored failure times. Defaults to None.
    show_plot - True/False. Default is True. Plots the CDF, SF, or CHF as specified by plot_type.
    plot_type - SF, CDF, or CHF. Default is SF.
    print_results - True/False. Default is True. Will display a pandas dataframe in the console.
    plot_CI - shades the upper and lower confidence interval
    CI - confidence interval between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    results - dataframe of results for the SF
    KM - list of Kaplan-Meier column from results dataframe. This column is the non parametric estimate of the Survival Function (reliability function).
    xvals - the x-values to plot the stepwise plot as seen when show_plot=True
    SF - survival function stepwise values (these differ from the KM values as there are extra values added in to make the plot into a step plot)
    CDF - cumulative distribution function stepwise values
    CHF - cumulative hazard function stepwise values
    SF_lower - survival function stepwise values for lower CI
    SF_upper - survival function stepwise values for upper CI
    CDF_lower - cumulative distribution function stepwise values for lower CI
    CDF_upper - cumulative distribution function stepwise values for upper CI
    CHF_lower - cumulative hazard function stepwise values for lower CI
    CHF_upper - cumulative hazard function stepwise values for upper CI

    Example Usage:
    f = [5248,7454,16890,17200,38700,45000,49390,69040,72280,131900]
    rc = [3961,4007,4734,6054,7298,10190,23060,27160,28690,37100,40060,45670,53000,67000,69630,77350,78470,91680,105700,106300,150400]
    KaplanMeier(failures = f, right_censored = rc)
    '''

    def __init__(self, failures=None, right_censored=None, show_plot=True, print_results=True, plot_CI=True, CI=0.95, plot_type='SF', **kwargs):
        np.seterr(divide='ignore')  # divide by zero occurs if last detapoint is a failure so risk set is zero

        if failures is None:
            raise ValueError('failures must be provided to calculate non-parametric estimates.')
        if right_censored is None:
            right_censored = []  # create empty array so it can be added in hstack
        if plot_type not in ['CDF', 'SF', 'CHF', 'cdf', 'sf', 'chf']:
            raise ValueError('plot_type must be CDF, SF, or CHF. Default is SF.')

        # turn the failures and right censored times into a two lists of times and censoring codes
        times = np.hstack([failures, right_censored])
        F = np.ones_like(failures)
        RC = np.zeros_like(right_censored)  # censored values are given the code of 0
        cens_code = np.hstack([F, RC])
        Data = {'times': times, 'cens_code': cens_code}
        df = pd.DataFrame(Data, columns=['times', 'cens_code'])
        df2 = df.sort_values(by='times')
        d = df2['times'].values
        c = df2['cens_code'].values

        self.data = d
        self.censor_codes = c

        if CI < 0 or CI > 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals.')

        n = len(d)  # number of items
        failures_array = np.arange(1, n + 1)  # array of number of items (1 to n)
        remaining_array = failures_array[::-1]  # items remaining (n to 1)
        KM = []  # R(t)
        KM_upper = []  # upper CI
        KM_lower = []  # lower CI
        z = ss.norm.ppf(1 - (1 - CI) / 2)
        frac = []
        delta = 0
        for i in failures_array:
            if i == 1:
                KM.append((remaining_array[i - 1] - c[i - 1]) / remaining_array[i - 1])
            else:
                KM.append(((remaining_array[i - 1] - c[i - 1]) / remaining_array[i - 1]) * KM[i - 2])
            # greenwood confidence interval calculations. Uses Normal approximation (same method as in Minitab)
            if c[i - 1] == 1:
                risk_set = n - i + 1
                frac.append(1 / ((risk_set) * (risk_set - 1)))
                sumfrac = sum(frac)
                R2 = KM[i - 1] ** 2
                if R2 > 0:  # required if the last piece of data is a failure
                    delta = ((sumfrac * R2) ** 0.5) * z
                else:
                    delta = 0
            KM_upper.append(KM[i - 1] + delta)
            KM_lower.append(KM[i - 1] - delta)
        KM_lower = np.array(KM_lower)
        KM_upper = np.array(KM_upper)
        KM_upper[KM_upper > 1] = 1
        KM_lower[KM_lower < 0] = 0

        # assemble the pandas dataframe for the output
        DATA = {'Failure times': d,
                'Censoring code (censored=0)': c,
                'Items remaining': remaining_array,
                'Kaplan-Meier Estimate': KM,
                'Lower CI bound': KM_lower,
                'Upper CI bound': KM_upper}
        dfx = pd.DataFrame(DATA, columns=['Failure times', 'Censoring code (censored=0)', 'Items remaining', 'Kaplan-Meier Estimate', 'Lower CI bound', 'Upper CI bound'])
        dfy = dfx.set_index('Failure times')
        pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
        pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
        self.results = dfy
        self.KM = KM

        KM_x = [0]
        KM_y = [1]  # adds a start point for 100% reliability at 0 time
        KM_y_upper = []
        KM_y_lower = []

        for i in failures_array:
            if i == 1:
                if c[i - 1] == 0:  # if the first item is censored
                    KM_x.append(d[i - 1])
                    KM_y.append(1)
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
                else:  # if the first item is a failure
                    KM_x.append(d[i - 1])
                    KM_x.append(d[i - 1])
                    KM_y.append(1)
                    KM_y.append(KM[i - 1])
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
            else:
                if KM[i - 2] == KM[i - 1]:  # if the next item is censored
                    KM_x.append(d[i - 1])
                    KM_y.append(KM[i - 1])
                    KM_y_lower.append(KM_lower[i - 2])
                    KM_y_upper.append(KM_upper[i - 2])
                else:  # if the next item is a failure
                    KM_x.append(d[i - 1])
                    KM_y.append(KM[i - 2])
                    KM_y_lower.append(KM_lower[i - 2])
                    KM_y_upper.append(KM_upper[i - 2])
                    KM_x.append(d[i - 1])
                    KM_y.append(KM[i - 1])
                    KM_y_lower.append(KM_lower[i - 2])
                    KM_y_upper.append(KM_upper[i - 2])
        KM_y_lower.append(KM_y_lower[-1])
        KM_y_upper.append(KM_y_upper[-1])
        self.xvals = np.array(KM_x)
        self.SF = np.array(KM_y)
        self.SF_lower = np.array(KM_y_lower)
        self.SF_upper = np.array(KM_y_upper)
        self.CDF = 1 - self.SF
        self.CDF_lower = 1 - self.SF_upper
        self.CDF_upper = 1 - self.SF_lower
        self.CHF = -np.log(self.SF)
        self.CHF_lower = -np.log(self.SF_upper)
        self.CHF_upper = -np.log(self.SF_lower)

        if print_results is True:
            print(dfy)  # this will print the pandas dataframe
        # plotting section
        if show_plot is True:
            # extract certain keyword arguments or specify them if they are not set. We cannot pass all kwargs to CI plots as some are not appropriate (eg. label)

            if plot_type == 'SF':
                p = plt.plot(self.xvals, self.SF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Kaplan-Meier SF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.SF_lower, self.SF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Kaplan-Meier estimate of Survival Function'
                plt.xlabel('Failure units')
                plt.ylabel('Reliability')
                plt.title(title_text)
                plt.xlim([0, max(KM_x)])
                plt.ylim([0, 1.1])
            if plot_type == 'CDF':
                p = plt.plot(self.xvals, self.CDF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Kaplan-Meier CDF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.CDF_lower, self.CDF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Kaplan-Meier estimate of Cumulative Density Function'
                plt.xlabel('Failure units')
                plt.ylabel('Fraction Failing')
                plt.title(title_text)
                plt.xlim([0, max(KM_x)])
                plt.ylim([0, 1.1])
            if plot_type == 'CHF':
                p = plt.plot(self.xvals, self.CHF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Kaplan-Meier CHF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.CHF_lower, self.CHF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Kaplan-Meier estimate of Cumulative Hazard Function'
                plt.xlabel('Failure units')
                plt.ylabel('Cumulative Hazard')
                plt.title(title_text)
                plt.xlim([0, max(KM_x)])


class NelsonAalen:
    '''
    Nelson-Aalen

    Uses the Nelson-Aalen estimation method to calculate the reliability from failure data.
    Right censoring is supported and confidence bounds are provided.
    The confidence bounds are calculated using the Greenwood formula with Normal approximation.
    The Nelson-Aalen method provides the CHF. With a little algebra, the CDF and SF are also obtained from the CHF.
    It is not possible to obtain a useful version of the PDF or HF as the derivative of a stepwise function produces very spikey functions.
    Nelson-Aalen does obtain the HF directly which is then used to obtain the CHF, but this function is not smooth and is of little use

    Inputs:
    failure - an array or list of failure times. Sorting is automatic so times do not need to be provided in any order.
    right_censored - an array or list of right censored failure times. Defaults to None.
    show_plot - True/False. Default is True. Plots the SF.
    print_results - True/False. Default is True. Will display a pandas dataframe in the console.
    plot_CI - shades the upper and lower confidence interval
    CI - confidence interval between 0 and 1. Default is 0.95 for 95% CI.
    plot_type - SF, CDF, or CHF. Default is SF.

    Outputs:
    results - dataframe of results
    NA - list of Nelson-Aalen column from results dataframe. This column is the non parametric estimate of the Survival Function (reliability function).
    xvals - the x-values to plot the stepwise plot as seen when show_plot=True
    SF - survival function stepwise values (these differ from the NA values as there are extra values added in to make the plot into a step plot)
    CDF - cumulative distribution function stepwise values
    CHF - cumulative hazard function stepwise values
    SF_lower - survival function stepwise values for lower CI
    SF_upper - survival function stepwise values for upper CI
    CDF_lower - cumulative distribution function stepwise values for lower CI
    CDF_upper - cumulative distribution function stepwise values for upper CI
    CHF_lower - cumulative hazard function stepwise values for lower CI
    CHF_upper - cumulative hazard function stepwise values for upper CI

    Example Usage:
    f = [5248,7454,16890,17200,38700,45000,49390,69040,72280,131900]
    rc = [3961,4007,4734,6054,7298,10190,23060,27160,28690,37100,40060,45670,53000,67000,69630,77350,78470,91680,105700,106300,150400]
    NelsonAalen(failures = f, right_censored = rc)
    '''

    def __init__(self, failures=None, right_censored=None, show_plot=True, print_results=True, plot_CI=True, CI=0.95, plot_type='SF', **kwargs):
        np.seterr(divide='ignore')  # divide by zero occurs if last detapoint is a failure so risk set is zero

        if failures is None:
            raise ValueError('failures must be provided to calculate non-parametric estimates.')
        if right_censored is None:
            right_censored = []  # create empty array so it can be added in hstack
        if plot_type not in ['CDF', 'SF', 'CHF', 'cdf', 'sf', 'chf']:
            raise ValueError('plot_type must be CDF, SF, or CHF. Default is SF.')

        # turn the failures and right censored times into a two lists of times and censoring codes
        times = np.hstack([failures, right_censored])
        F = np.ones_like(failures)
        RC = np.zeros_like(right_censored)  # censored values are given the code of 0
        cens_code = np.hstack([F, RC])
        Data = {'times': times, 'cens_code': cens_code}
        df = pd.DataFrame(Data, columns=['times', 'cens_code'])
        df2 = df.sort_values(by='times')
        d = df2['times'].values
        c = df2['cens_code'].values

        self.data = d
        self.censor_codes = c

        if CI < 0 or CI > 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals.')

        n = len(d)  # number of items
        failures_array = np.arange(1, n + 1)  # array of number of items (1 to n)
        remaining_array = failures_array[::-1]  # items remaining (n to 1)
        h = []
        H = []
        NA = []  # R(t)
        NA_upper = []  # upper CI
        NA_lower = []  # lower CI
        z = ss.norm.ppf(1 - (1 - CI) / 2)
        frac = []
        delta = 0
        for i in failures_array:
            h.append((c[i - 1]) / remaining_array[i - 1])  # obtain HF
            H.append(sum(h))  # obtain CHF
            NA.append(np.exp(-H[-1]))

            # greenwood confidence interval calculations. Uses Normal approximation
            if c[i - 1] == 1:
                risk_set = n - i + 1
                frac.append(1 / ((risk_set) * (risk_set - 1)))
                sumfrac = sum(frac)
                R2 = NA[i - 1] ** 2
                if R2 > 0:  # required if the last piece of data is a failure
                    delta = ((sumfrac * R2) ** 0.5) * z
                else:
                    delta = 0
            NA_upper.append(NA[i - 1] + delta)
            NA_lower.append(NA[i - 1] - delta)
        NA_lower = np.array(NA_lower)
        NA_upper = np.array(NA_upper)
        NA_upper[NA_upper > 1] = 1
        NA_lower[NA_lower < 0] = 0

        # assemble the pandas dataframe for the output
        DATA = {'Failure times': d,
                'Censoring code (censored=0)': c,
                'Items remaining': remaining_array,
                'Nelson-Aalen Estimate': NA,
                'Lower CI bound': NA_lower,
                'Upper CI bound': NA_upper}
        dfx = pd.DataFrame(DATA, columns=['Failure times', 'Censoring code (censored=0)', 'Items remaining', 'Nelson-Aalen Estimate', 'Lower CI bound', 'Upper CI bound'])
        dfy = dfx.set_index('Failure times')
        pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
        pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
        self.results = dfy
        self.NA = NA

        NA_x = [0]
        NA_y = [1]  # adds a start point for 100% reliability at 0 time
        NA_y_upper = []
        NA_y_lower = []

        for i in failures_array:
            if i == 1:
                if c[i - 1] == 0:  # if the first item is censored
                    NA_x.append(d[i - 1])
                    NA_y.append(1)
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
                else:  # if the first item is a failure
                    NA_x.append(d[i - 1])
                    NA_x.append(d[i - 1])
                    NA_y.append(1)
                    NA_y.append(NA[i - 1])
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
            else:
                if NA[i - 2] == NA[i - 1]:  # if the next item is censored
                    NA_x.append(d[i - 1])
                    NA_y.append(NA[i - 1])
                    NA_y_lower.append(NA_lower[i - 2])
                    NA_y_upper.append(NA_upper[i - 2])
                else:  # if the next item is a failure
                    NA_x.append(d[i - 1])
                    NA_y.append(NA[i - 2])
                    NA_y_lower.append(NA_lower[i - 2])
                    NA_y_upper.append(NA_upper[i - 2])
                    NA_x.append(d[i - 1])
                    NA_y.append(NA[i - 1])
                    NA_y_lower.append(NA_lower[i - 2])
                    NA_y_upper.append(NA_upper[i - 2])
        NA_y_lower.append(NA_y_lower[-1])
        NA_y_upper.append(NA_y_upper[-1])
        self.xvals = np.array(NA_x)
        self.SF = np.array(NA_y)
        self.SF_lower = np.array(NA_y_lower)
        self.SF_upper = np.array(NA_y_upper)
        self.CDF = 1 - self.SF
        self.CDF_lower = 1 - self.SF_upper
        self.CDF_upper = 1 - self.SF_lower
        self.CHF = -np.log(self.SF)
        self.CHF_lower = -np.log(self.SF_upper)
        self.CHF_upper = -np.log(self.SF_lower)

        if print_results is True:
            print(dfy)  # this will print the pandas dataframe
        # plotting section
        if show_plot is True:
            # extract certain keyword arguments or specify them if they are not set. We cannot pass all kwargs to CI plots as some are not appropriate (eg. label)
            if plot_type == 'SF':
                p = plt.plot(self.xvals, self.SF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Nelson-Aalen SF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.SF_lower, self.SF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Nelson-Aalen estimate of Survival Function'
                plt.xlabel('Failure units')
                plt.ylabel('Reliability')
                plt.title(title_text)
                plt.xlim([0, max(NA_x)])
                plt.ylim([0, 1.1])
            if plot_type == 'CDF':
                p = plt.plot(self.xvals, self.CDF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Nelson-Aalen CDF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.CDF_lower, self.CDF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Nelson-Aalen estimate of Cumulative Density Function'
                plt.xlabel('Failure units')
                plt.ylabel('Fraction Failing')
                plt.title(title_text)
                plt.xlim([0, max(NA_x)])
                plt.ylim([0, 1.1])
            if plot_type == 'CHF':
                p = plt.plot(self.xvals, self.CHF, **kwargs)
                if plot_CI is True:  # plots the confidence bounds
                    title_text = str('Nelson-Aalen CHF estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                    plt.fill_between(self.xvals, self.CHF_lower, self.CHF_upper, color=p[0].get_color(), alpha=0.3)
                else:
                    title_text = 'Nelson-Aalen estimate of Cumulative Hazard Function'
                plt.xlabel('Failure units')
                plt.ylabel('Cumulative Hazard')
                plt.title(title_text)
                plt.xlim([0, max(NA_x)])
