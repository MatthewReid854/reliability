'''
Fitters
This module contains custom fitting functions for parametric distributions which support both complete and right censored data.
The supported distributions are:
Weibull_2P
Weibull_3P
Exponential_1P
Exponential_2P
Gamma_2P
Gamma_3P
Lognormal_2P
Lognormal_3P
Normal_2P
Beta_2P
Weibull_Mixture

Note that the Beta distribution is only for data in the range 0-1.
There is also a Fit_Everything function which will fit all distributions except the Weibull mixture model and will provide plots and a table of values.
All functions in this module work using autograd to find the derivative of the log-likelihood function. In this way, the code only needs to specify
the log PDF and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained
using scipy. If the distribution is an extremely bad fit or is heavily censored (>99%) then these guesses may be poor and the fit might not be successful.
Generally the fit achieved by autograd is highly successful.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as ss
import warnings
from reliability.Distributions import Weibull_Distribution, Gamma_Distribution, Beta_Distribution, Exponential_Distribution, Normal_Distribution, Lognormal_Distribution
from reliability.Nonparametric import KaplanMeier
import autograd.numpy as anp
from autograd import value_and_grad
from autograd.scipy.special import gamma as agamma
from autograd.scipy.special import beta as abeta
from autograd.differential_operators import hessian
from autograd_gamma import betainc
from autograd.scipy.special import erf
from autograd_gamma import gammaincc

anp.seterr('ignore')


class Fit_Everything:
    '''
    Fit_Everything
    This function will fit all available distributions for the data you enter, which may include right censored data.

    Inputs:
    failures - an array or list of the failure times (this does not need to be sorted).
    right_censored - an array or list of the right failure times (this does not need to be sorted).
    sort_by - goodness of fit test to sort results by. Must be either 'BIC' or 'AICc'. Default is BIC.
    print_results - True/False. Defaults to True. Will show the results of the fitted parameters and the goodness of fit
        tests in a dataframe.
    show_histogram_plot - True/False. Defaults to True. Will show a histogram (scaled to account for censored data) with
        the PDF and CDF of each fitted distribution
    show_PP_plot - True/False. Defaults to True.
        Provides a comparison of parametric vs non-parametric fit using Probability-Probability (PP) plot.
    show_probability_plot - True/False. Defaults to True. Provides a probability plot of each of the fitted distributions.

    Outputs:
    results - the dataframe of results. Fitted parameters in this dataframe may be accessed by name. See below example.
        In displaying these results, the pandas dataframe is designed to use the common greek letter parametrisations
        rather than the scale, shape, location , threshold parametrisations which can become confusing for some
        distributions.
    best_distribution - a distribution object created based on the parameters of the best fitting distribution
    best_distribution_name - the name of the best fitting distribution. E.g. 'Weibull_3P'
    parameters and goodness of fit tests for each fitted distribution. For example, the Weibull_3P distribution values are:
        Weibull_3P_alpha
        Weibull_3P_beta
        Weibull_3P_gamma
        Weibull_3P_BIC
        Weibull_3P_AICc
    All parametric models have the number of parameters in the name. For example, Weibull_2P used alpha and beta, whereas Weibull_3P
    uses alpha, beta, and gamma. This is applied even for Normal_2P for consistency in naming conventions.
    If plot_results is True, the plot will show the PDF and CDF of all fitted distributions plotted with a histogram of the data.
    From the results, the distributions are sorted based on their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the distribution to the data.
    Confidence intervals for each of the fitted parameters are not supported. This feature will be incorporated in
    future releases in 2020. See the python library "lifelines" or JMP Pro software if this is required.
    Whilst Minitab uses the Anderson-Darling statistic for the goodness of fit, it is generally recognised that AICc and BIC
    are more accurate measures as they take into account the number of parameters in the distribution.

    Example Usage:
    X = [0.95892,1.43249,1.04221,0.67583,3.28411,1.03072,0.05826,1.81387,2.06383,0.59762,5.99005,1.92145,1.35179,0.50391]
    output = Fit_Everything(X)
    To extract the parameters of the Weibull distribution from the results dataframe, you may access the parameters by name:
    print('Weibull Alpha =',output.Weibull_2P_alpha,'\nWeibull Beta =',output.Weibull_2P_beta)
    '''

    def __init__(self, failures=None, right_censored=None, sort_by='BIC', print_results=True, show_histogram_plot=True, show_PP_plot=None, show_probability_plot=True):
        if failures is None or len(failures) < 3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate 3 parameter models.')
        if sort_by not in ['AICc', 'BIC']:
            raise ValueError('sort_by must be either AICc or BIC. Defaults to BIC')
        if show_histogram_plot not in [True, False]:
            raise ValueError('show_histogram_plot must be either True or False. Defaults to True.')
        if print_results not in [True, False]:
            raise ValueError('print_results must be either True or False. Defaults to True.')
        if show_PP_plot not in [True, False, None]:
            raise ValueError('show_PP_plot must be either True or False. Defaults to True.')
        if show_probability_plot not in [True, False]:
            raise ValueError('show_probability_plot must be either True or False. Defaults to True.')

        self.failures = failures
        self.right_censored = right_censored
        if show_PP_plot is None:
            show_PP_plot = True
        if right_censored is None:
            RC = []
        else:
            RC = right_censored
        self._all_data = np.hstack([failures, RC])
        if min(self._all_data) <= 0:
            raise ValueError('All failure and censoring times must be greater than zero.')

        # These are all used for scaling the histogram when there is censored data
        self._frac_fail = len(failures) / len(self._all_data)

        # Kaplan-Meier estimate of quantiles. Used in P-P plot.
        d = sorted(self._all_data)  # sorting the failure data is necessary for plotting quantiles in order
        nonparametric = KaplanMeier(failures=failures, right_censored=right_censored, print_results=False, show_plot=False)
        self._nonparametric_CDF = 1 - np.array(nonparametric.KM)  # change SF into CDF

        # parametric models
        self.__Weibull_3P_params = Fit_Weibull_3P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Weibull_3P_alpha = self.__Weibull_3P_params.alpha
        self.Weibull_3P_beta = self.__Weibull_3P_params.beta
        self.Weibull_3P_gamma = self.__Weibull_3P_params.gamma
        self.Weibull_3P_BIC = self.__Weibull_3P_params.BIC
        self.Weibull_3P_AICc = self.__Weibull_3P_params.AICc
        self._parametric_CDF_Weibull_3P = self.__Weibull_3P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Gamma_3P_params = Fit_Gamma_3P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Gamma_3P_alpha = self.__Gamma_3P_params.alpha
        self.Gamma_3P_beta = self.__Gamma_3P_params.beta
        self.Gamma_3P_gamma = self.__Gamma_3P_params.gamma
        self.Gamma_3P_BIC = self.__Gamma_3P_params.BIC
        self.Gamma_3P_AICc = self.__Gamma_3P_params.AICc
        self._parametric_CDF_Gamma_3P = self.__Gamma_3P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Expon_2P_params = Fit_Expon_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Expon_2P_lambda = self.__Expon_2P_params.Lambda
        self.Expon_2P_gamma = self.__Expon_2P_params.gamma
        self.Expon_2P_BIC = self.__Expon_2P_params.BIC
        self.Expon_2P_AICc = self.__Expon_2P_params.AICc
        self._parametric_CDF_Exponential_2P = self.__Expon_2P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Lognormal_3P_params = Fit_Lognormal_3P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Lognormal_3P_mu = self.__Lognormal_3P_params.mu
        self.Lognormal_3P_sigma = self.__Lognormal_3P_params.sigma
        self.Lognormal_3P_gamma = self.__Lognormal_3P_params.gamma
        self.Lognormal_3P_BIC = self.__Lognormal_3P_params.BIC
        self.Lognormal_3P_AICc = self.__Lognormal_3P_params.AICc
        self._parametric_CDF_Lognormal_3P = self.__Lognormal_3P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Normal_2P_params = Fit_Normal_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Normal_2P_mu = self.__Normal_2P_params.mu
        self.Normal_2P_sigma = self.__Normal_2P_params.sigma
        self.Normal_2P_BIC = self.__Normal_2P_params.BIC
        self.Normal_2P_AICc = self.__Normal_2P_params.AICc
        self._parametric_CDF_Normal_2P = self.__Normal_2P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Lognormal_2P_params = Fit_Lognormal_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Lognormal_2P_mu = self.__Lognormal_2P_params.mu
        self.Lognormal_2P_sigma = self.__Lognormal_2P_params.sigma
        self.Lognormal_2P_BIC = self.__Lognormal_2P_params.BIC
        self.Lognormal_2P_AICc = self.__Lognormal_2P_params.AICc
        self._parametric_CDF_Lognormal_2P = self.__Lognormal_2P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Weibull_2P_params = Fit_Weibull_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Weibull_2P_alpha = self.__Weibull_2P_params.alpha
        self.Weibull_2P_beta = self.__Weibull_2P_params.beta
        self.Weibull_2P_BIC = self.__Weibull_2P_params.BIC
        self.Weibull_2P_AICc = self.__Weibull_2P_params.AICc
        self._parametric_CDF_Weibull_2P = self.__Weibull_2P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Gamma_2P_params = Fit_Gamma_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Gamma_2P_alpha = self.__Gamma_2P_params.alpha
        self.Gamma_2P_beta = self.__Gamma_2P_params.beta
        self.Gamma_2P_BIC = self.__Gamma_2P_params.BIC
        self.Gamma_2P_AICc = self.__Gamma_2P_params.AICc
        self._parametric_CDF_Gamma_2P = self.__Gamma_2P_params.distribution.CDF(xvals=d, show_plot=False)

        self.__Expon_1P_params = Fit_Expon_1P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
        self.Expon_1P_lambda = self.__Expon_1P_params.Lambda
        self.Expon_1P_BIC = self.__Expon_1P_params.BIC
        self.Expon_1P_AICc = self.__Expon_1P_params.AICc
        self._parametric_CDF_Exponential_1P = self.__Expon_1P_params.distribution.CDF(xvals=d, show_plot=False)

        if max(failures) <= 1:
            self.__Beta_2P_params = Fit_Beta_2P(failures=failures, right_censored=right_censored, show_probability_plot=False, print_results=False)
            self.Beta_2P_alpha = self.__Beta_2P_params.alpha
            self.Beta_2P_beta = self.__Beta_2P_params.beta
            self.Beta_2P_BIC = self.__Beta_2P_params.BIC
            self.Beta_2P_AICc = self.__Beta_2P_params.AICc
            self._parametric_CDF_Beta_2P = self.__Beta_2P_params.distribution.CDF(xvals=d, show_plot=False)
        else:
            self.Beta_2P_alpha = 0
            self.Beta_2P_beta = 0
            self.Beta_2P_BIC = 0
            self.Beta_2P_AICc = 0

        # assemble the output dataframe
        DATA = {'Distribution': ['Weibull_3P', 'Weibull_2P', 'Normal_2P', 'Exponential_1P', 'Exponential_2P', 'Lognormal_2P', 'Lognormal_3P', 'Gamma_2P', 'Gamma_3P', 'Beta_2P'],
                'Alpha': [self.Weibull_3P_alpha, self.Weibull_2P_alpha, '', '', '', '', '', self.Gamma_2P_alpha, self.Gamma_3P_alpha, self.Beta_2P_alpha],
                'Beta': [self.Weibull_3P_beta, self.Weibull_2P_beta, '', '', '', '', '', self.Gamma_2P_beta, self.Gamma_3P_beta, self.Beta_2P_beta],
                'Gamma': [self.Weibull_3P_gamma, '', '', '', self.Expon_2P_gamma, '', self.Lognormal_3P_gamma, '', self.Gamma_3P_gamma, ''],
                'Mu': ['', '', self.Normal_2P_mu, '', '', self.Lognormal_2P_mu, self.Lognormal_3P_mu, '', '', ''],
                'Sigma': ['', '', self.Normal_2P_sigma, '', '', self.Lognormal_2P_sigma, self.Lognormal_3P_sigma, '', '', ''],
                'Lambda': ['', '', '', self.Expon_1P_lambda, self.Expon_2P_lambda, '', '', '', '', ''],
                'AICc': [self.Weibull_3P_AICc, self.Weibull_2P_AICc, self.Normal_2P_AICc, self.Expon_1P_AICc, self.Expon_2P_AICc, self.Lognormal_2P_AICc, self.Lognormal_3P_AICc, self.Gamma_2P_AICc, self.Gamma_3P_AICc, self.Beta_2P_AICc],
                'BIC': [self.Weibull_3P_BIC, self.Weibull_2P_BIC, self.Normal_2P_BIC, self.Expon_1P_BIC, self.Expon_2P_BIC, self.Lognormal_2P_BIC, self.Lognormal_2P_BIC, self.Gamma_2P_BIC, self.Gamma_3P_BIC, self.Beta_2P_BIC]}

        df = pd.DataFrame(DATA, columns=['Distribution', 'Alpha', 'Beta', 'Gamma', 'Mu', 'Sigma', 'Lambda', 'AICc', 'BIC'])
        # sort the dataframe by BIC or AICc and replace na and 0 values with spaces. Smallest AICc or BIC is better fit
        if sort_by in ['BIC', 'bic']:
            df2 = df.reindex(df.BIC.sort_values().index)
        elif sort_by in ['AICc', 'AIC', 'aic', 'aicc']:
            df2 = df.reindex(df.AICc.sort_values().index)
        else:
            raise ValueError('Invalid input to sort_by. Options are BIC or AICc. Default is BIC')
        df3 = df2.set_index('Distribution').fillna('')
        if self.Beta_2P_BIC == 0:  # remove beta if it was not fitted (due to data being outside of 0 to 1 range)
            df3 = df3.drop('Beta_2P', axis=0)
        self.results = df3

        # creates a distribution object of the best fitting distribution and assigns its name
        best_dist = df3.index.values[0]
        self.best_distribution_name = best_dist
        if best_dist == 'Weibull_2P':
            self.best_distribution = Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta)
        elif best_dist == 'Weibull_3P':
            self.best_distribution = Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta, gamma=self.Weibull_3P_gamma)
        elif best_dist == 'Gamma_2P':
            self.best_distribution = Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta)
        elif best_dist == 'Gamma_3P':
            self.best_distribution = Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta, gamma=self.Gamma_3P_gamma)
        elif best_dist == 'Lognormal_2P':
            self.best_distribution = Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma)
        elif best_dist == 'Lognormal_3P':
            self.best_distribution = Lognormal_Distribution(mu=self.Lognormal_3P_mu, sigma=self.Lognormal_3P_sigma, gamma=self.Lognormal_3P_gamma)
        elif best_dist == 'Exponential_1P':
            self.best_distribution = Exponential_Distribution(Lambda=self.Expon_1P_lambda)
        elif best_dist == 'Exponential_2P':
            self.best_distribution = Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma)
        elif best_dist == 'Normal_2P':
            self.best_distribution = Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma)
        elif best_dist == 'Beta_2P':
            self.best_distribution = Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta)

        # print the results
        if print_results is True:  # printing occurs by default
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(self.results)

        if show_histogram_plot is True:
            Fit_Everything.histogram_plot(self)  # plotting occurs by default

        if show_PP_plot is True:
            Fit_Everything.P_P_plot(self)  # plotting occurs by default

        if show_probability_plot is True:
            Fit_Everything.probability_plot(self)  # plotting occurs by default

        if show_histogram_plot is True or show_PP_plot is True or show_probability_plot is True:
            plt.show()

    def histogram_plot(self):
        X = self.failures
        # define plotting limits
        delta = max(X) - min(X)
        xmin = 0
        xmax = max(X) + delta
        xvals = np.linspace(xmin, xmax, 1000)

        plt.figure(figsize=(14, 6))
        plt.subplot(121)  # PDF

        # make this histogram. Can't use plt.hist due to need to scale the heights when there's censored data
        num_bins = min(int(len(X) / 2), 30)
        hist, bins = np.histogram(X, bins=num_bins, density=True)
        hist_cumulative = np.cumsum(hist) / sum(hist)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist * self._frac_fail, align='center', width=width, alpha=0.2, color='k', edgecolor='k')

        Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).PDF(xvals=xvals, label=r'Weibull ($\alpha , \beta$)')
        Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta, gamma=self.Weibull_3P_gamma).PDF(xvals=xvals, label=r'Weibull ($\alpha , \beta , \gamma$)')
        Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).PDF(xvals=xvals, label=r'Gamma ($\alpha , \beta$)')
        Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta, gamma=self.Gamma_3P_gamma).PDF(xvals=xvals, label=r'Gamma ($\alpha , \beta , \gamma$)')
        Exponential_Distribution(Lambda=self.Expon_1P_lambda).PDF(xvals=xvals, label=r'Exponential ($\lambda$)')
        Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).PDF(xvals=xvals, label=r'Exponential ($\lambda , \gamma$)')
        Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).PDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma$)')
        Lognormal_Distribution(mu=self.Lognormal_3P_mu, sigma=self.Lognormal_3P_sigma, gamma=self.Lognormal_3P_gamma).PDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma , \gamma$)')
        Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).PDF(xvals=xvals, label=r'Normal ($\mu , \sigma$)')
        if max(X) <= 1:  # condition for Beta Dist to be fitted
            Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).PDF(xvals=xvals, label=r'Beta ($\alpha , \beta$)')
        plt.legend()
        plt.xlim([xmin, xmax])
        plt.title('Probability Density Function')
        plt.xlabel('Data')
        plt.ylabel('Probability density')
        plt.legend()

        plt.subplot(122)  # CDF
        plt.bar(center, hist_cumulative * self._frac_fail, align='center', width=width, alpha=0.2, color='k', edgecolor='k')
        Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).CDF(xvals=xvals, label=r'Weibull ($\alpha , \beta$)')
        Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta, gamma=self.Weibull_3P_gamma).CDF(xvals=xvals, label=r'Weibull ($\alpha , \beta , \gamma$)')
        Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).CDF(xvals=xvals, label=r'Gamma ($\alpha , \beta$)')
        Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta, gamma=self.Gamma_3P_gamma).CDF(xvals=xvals, label=r'Gamma ($\alpha , \beta , \gamma$)')
        Exponential_Distribution(Lambda=self.Expon_1P_lambda).CDF(xvals=xvals, label=r'Exponential ($\lambda$)')
        Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).CDF(xvals=xvals, label=r'Exponential ($\lambda , \gamma$)')
        Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).CDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma$)')
        Lognormal_Distribution(mu=self.Lognormal_3P_mu, sigma=self.Lognormal_3P_sigma, gamma=self.Lognormal_3P_gamma).CDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma , \gamma$)')
        Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).CDF(xvals=xvals, label=r'Normal ($\mu , \sigma$)')
        if max(X) <= 1:  # condition for Beta Dist to be fitted
            Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).CDF(xvals=xvals, label=r'Beta ($\alpha , \beta$)')
        plt.legend()
        plt.xlim([xmin, xmax])
        plt.title('Cumulative Distribution Function')
        plt.xlabel('Data')
        plt.ylabel('Cumulative probability density')
        plt.suptitle('Histogram plot of each fitted distribution')
        plt.legend()

    def P_P_plot(self):  # probability-probability plot of parametric vs non-parametric
        # plot each of the results
        if max(self.failures) <= 1:
            cols = 6  # this is for when the beta distribution was fitted
            size = (11, 4.65)
        else:
            cols = 5
            size = (10, 4.65)
        plt.figure(figsize=size)
        plt.suptitle('Semi-parametric Probability-Probability plots of each fitted distribution\nParametric (x-axis) vs Non-Parametric (y-axis)')

        plt.subplot(2, cols, 1)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Exponential_1P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Exponential_1P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Exponential_1P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, 2)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Weibull_2P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Weibull_2P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Weibull_2P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, 3)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Gamma_2P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Gamma_2P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Gamma_2P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, 4)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Lognormal_2P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Lognormal_2P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Lognormal_2P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, 5)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Normal_2P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Normal_2P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Normal_2P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, cols + 1)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Exponential_2P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Exponential_2P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Exponential_2P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, cols + 2)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Weibull_3P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Weibull_3P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Weibull_3P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, cols + 3)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Gamma_3P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Gamma_3P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Gamma_3P')
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, cols, cols + 4)
        xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Lognormal_3P]))
        plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Lognormal_3P, marker='.', color='k')
        plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
        plt.axis('square')
        plt.title('Lognormal_3P')
        plt.yticks([])
        plt.xticks([])

        if max(self.failures) <= 1:
            plt.subplot(2, 6, 6)
            xlim = max(np.hstack([self._nonparametric_CDF, self._parametric_CDF_Beta_2P]))
            plt.scatter(self._nonparametric_CDF, self._parametric_CDF_Beta_2P, marker='.', color='k')
            plt.plot([0, xlim], [0, xlim], 'r', alpha=0.7)
            plt.axis('square')
            plt.title('Beta_2P')
            plt.yticks([])
            plt.xticks([])
        plt.subplots_adjust(left=0.04, bottom=0.07, right=0.96, top=0.87)

    def probability_plot(self):
        from reliability.Probability_plotting import Weibull_probability_plot, Normal_probability_plot, Gamma_probability_plot, Exponential_probability_plot, Beta_probability_plot, Lognormal_probability_plot
        rows = 2
        if max(self.failures) <= 1:
            cols = 6  # this is for when the beta distribution was fitted
        else:
            cols = 5

        plt.figure()
        plt.subplot(rows, cols, 1)
        Exponential_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Expon_1P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Exponential_1P')

        plt.subplot(rows, cols, 2)
        Weibull_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Weibull_2P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Weibull_2P')

        plt.subplot(rows, cols, 3)
        Gamma_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Gamma_2P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Gamma_2P')

        plt.subplot(rows, cols, 4)
        Lognormal_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Lognormal_2P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Lognormal_2P')

        plt.subplot(rows, cols, 5)
        Normal_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Normal_2P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Normal_2P')

        plt.subplot(2, cols, cols + 1)
        Exponential_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Expon_2P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Exponential_2P')

        plt.subplot(2, cols, cols + 2)
        Weibull_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Weibull_3P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Weibull_3P')

        plt.subplot(2, cols, cols + 3)
        Gamma_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Gamma_3P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Gamma_3P')

        plt.subplot(2, cols, cols + 4)
        Lognormal_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Lognormal_3P_params)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.get_legend().remove()
        plt.title('Lognormal_3P')

        if max(self.failures) <= 1:
            plt.subplot(rows, 6, 6)
            Beta_probability_plot(failures=self.failures, right_censored=self.right_censored, __fitted_dist_params=self.__Beta_2P_params)
            ax = plt.gca()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.get_legend().remove()
            plt.title('Beta_2P')

        if max(self.failures) <= 1:
            plt.gcf().set_size_inches(11, 5)  # this is for when the beta distribution was fitted
        else:
            plt.gcf().set_size_inches(10, 5)

        plt.suptitle('Probability plots of each fitted distribution\n')
        plt.subplots_adjust(left=0.04, bottom=0.09, right=0.96, top=0.86, wspace=0.2, hspace=0.32)
        plt.show()


class Fit_Weibull_2P:
    '''
    Fit_Weibull_2P

    Fits a 2-parameter Weibull distribution (alpha,beta) to the data provided.

    inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    force_beta - Use this to specify the beta value if you need to force beta to be a certain value. Used in ALT probability plotting. Optional input.
    kwargs are accepted for the fitted line (eg. linestyle, label, color)

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Weibull_2P alpha parameter
    beta - the fitted Weibull_2P beta parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Weibull_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95, force_beta=None, **kwargs):
        if force_beta is not None and (failures is None or len(failures) < 1):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failures to calculate Weibull parameters when force_beta is specified.')
        elif force_beta is None and (failures is None or len(failures) < 2):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # solve it
        self.gamma = 0
        sp = ss.weibull_min.fit(all_data, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        warnings.filterwarnings('ignore')  # necessary to supress the warning about the jacobian when using the nelder-mead optimizer

        if force_beta is None:
            guess = [sp[2], sp[0]]
            result = minimize(value_and_grad(Fit_Weibull_2P.LL), guess, args=(failures, right_censored), jac=True, tol=1e-6, method='nelder-mead')
        else:
            guess = [sp[2]]
            result = minimize(value_and_grad(Fit_Weibull_2P.LL_fb), guess, args=(failures, right_censored, force_beta), jac=True, tol=1e-6, method='nelder-mead')

        if result.success is True:
            params = result.x
            self.success = True
            if force_beta is None:
                self.alpha = params[0]
                self.beta = params[1]
            else:
                self.alpha = params * 1  # the *1 converts ndarray to float64
                self.beta = force_beta
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Weibull_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[2]
            self.beta = sp[0]

        params = [self.alpha, self.beta]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Weibull_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Weibull_Distribution(alpha=self.alpha, beta=self.beta)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P.LL_fb)(np.array(tuple([self.alpha])), np.array(tuple(failures)), np.array(tuple(right_censored)), np.array(tuple([force_beta])))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = ''
            self.Cov_alpha_beta = ''
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = ''
            self.beta_lower = ''

        Data = {'Parameter': ['Alpha', 'Beta'],
                'Point Estimate': [self.alpha, self.beta],
                'Standard Error': [self.alpha_SE, self.beta_SE],
                'Lower CI': [self.alpha_lower, self.beta_lower],
                'Upper CI': [self.alpha_upper, self.beta_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Weibull_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self, **kwargs)

    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Weibull_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)

    def LL_fb(params, T_f, T_rc, force_beta):  # log likelihood function (2 parameter weibull) FORCED BETA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], force_beta).sum()  # failure times
        LL_rc += Fit_Weibull_2P.logR(T_rc, params[0], force_beta).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_3P:
    '''
    Fit_Weibull_3P
    Fits a 3-parameter Weibull distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Weibull_3P alpha parameter
    beta - the fitted Weibull_3P beta parameter
    gamma - the fitted Weibull_3P gamma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Weibull_Distribution object with the parameters of the fitted distribution
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
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        if failures is None or len(failures) < 3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate Weibull parameters.')
        if right_censored is None:
            right_censored = []  # fill with empty list if not specified
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # get a quick guess for gamma by setting it as the minimum of the data
        offset = 0.0001  # this is to ensure the upper bound for gamma is not equal to min(data) which would result in inf log-likelihood. This small offset fixes that issue
        gamma_initial_guess = min(all_data) - offset
        self.gamma = gamma_initial_guess

        # obtain the initial guess for alpha and beta
        data_shifted = all_data - self.gamma
        sp = ss.weibull_min.fit(data_shifted, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0], self.gamma]
        self.initial_guess = guess
        k = len(guess)
        n = len(all_data)

        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0

        gamma_lower_bound = 0.95 * gamma_initial_guess  # 0.95 is found to be the optimal point to minimise the error while also not causing autograd to fail
        bnds = [(0, None), (0, None), (gamma_lower_bound, min(all_data) - offset)]  # bounds on the solution. Helps a lot with stability
        while delta_BIC > 0.001 and runs < 5:  # exits after BIC convergence or 5 iterations
            runs += 1
            result = minimize(value_and_grad(Fit_Weibull_3P.LL), guess, args=(failures, right_censored), jac=True, method='L-BFGS-B', bounds=bnds)
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Weibull_3P.LL(guess, failures, right_censored)
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        if result.success is True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
            self.gamma = params[2]
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Weibull_3P. The fit from Scipy was used instead so the results may not be accurate.')
            sp = ss.weibull_min.fit(all_data, optimizer='powell')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

        params = [self.alpha, self.beta, self.gamma]
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Weibull_Distribution(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_3P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.gamma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        self.gamma_upper = self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
        self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

        Data = {'Parameter': ['Alpha', 'Beta', 'Gamma'],
                'Point Estimate': [self.alpha, self.beta, self.gamma],
                'Standard Error': [self.alpha_SE, self.beta_SE, self.gamma_SE],
                'Lower CI': [self.alpha_lower, self.beta_lower, self.gamma_lower],
                'Upper CI': [self.alpha_upper, self.beta_upper, self.gamma_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Weibull_3P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Weibull_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, a, b, g):  # Log PDF (3 parameter Weibull)
        return (b - 1) * anp.log((t - g) / a) + anp.log(b / a) - ((t - g) / a) ** b

    def logR(t, a, b, g):  # Log SF (3 parameter Weibull)
        return -(((t - g) / a) ** b)

    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_3P.logf(T_f, params[0], params[1], params[2]).sum()  # failure times
        LL_rc += Fit_Weibull_3P.logR(T_rc, params[0], params[1], params[2]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Weibull_Mixture:
    '''
    Fit_Weibull_Mixture
    Fits a mixture of 2 x Weibull_2P distributions (this does not fit the gamma parameter).
    Right censoring is supported, though care should be taken to ensure that there still appears to be two groups when plotting only the failure data.
    A second group cannot be made from a mostly or totally censored set of samples.
    Use this model when you think there are multiple failure modes acting to create the failure data.
    Whilst some failure modes may not be fitted as well by a Weibull distribution as they may be by another distribution, it
    is unlikely that a mixture of data from two distributions (particularly if they are overlapping) will be fitted
    noticeably better by other types of mixtures than would be achieved by a Weibull mixture. For this reason, other types
    of mixtures are not implemented.

    Inputs:
    failures - an array or list of the failure data. There must be at least 4 failures, but it is highly recommended to use another model if you have
        less than 20 failures.
    right_censored - an array or list of right censored data
    print_results - True/False. This will print results to console. Default is True
    show_plot - True/False. This will show the PDF and CDF of the Weibull mixture with a histogram of the data. Default is True.

    Outputs:
    alpha_1 - the fitted Weibull_2P alpha parameter for the first (left) group
    beta_1 - the fitted Weibull_2P beta parameter for the first (left) group
    alpha_2 - the fitted Weibull_2P alpha parameter for the second (right) group
    beta_2 - the fitted Weibull_2P beta parameter for the second (right) group
    proportion_1 - the fitted proportion of the first (left) group
    proportion_2 - the fitted proportion of the second (right) group. Same as 1-proportion_1
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    '''

    def __init__(self, failures=None, right_censored=None, show_plot=True, print_results=True):
        if failures is None or len(failures) < 4:  # it is possible to fit a mixture model with as few as 4 samples but it is inappropriate to do so. You should have at least 10, and preferably a lot more (>20) samples before using a mixture model.
            raise ValueError('The minimum number of failures to fit a mixture model is 4 (2 failures for each weibull). It is highly recommended that a mixture model is only used when sufficient data (>10 samples) is available.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])
        if min(all_data) <= 0:
            raise ValueError('All failure and censoring times must be greater than zero.')

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
                batch = hist_counts[0:i + batch_width]
            elif i > batch_width and i > len(hist_counts - batch_width):
                batch = hist_counts[i - batch_width:len(hist_counts)]
            else:
                batch = hist_counts[i - batch_width:i + batch_width]  # the histogram counts are batched (actual batch size = 2 x batch_width)
            if max(batch) == x:  # if the current point is higher than the rest of the batch then it is counted as a peak
                peaks_x.append(midbins[i])
                peaks_y.append(x)
        if len(peaks_x) > 2:  # if there are more than 2 peaks, the mean is moved based on the height of the peaks. Higher peaks will attract the mean towards them more than smaller peaks.
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
        group_1_estimates = Fit_Weibull_2P(failures=GROUP_1_failures, right_censored=GROUP_1_right_cens, show_probability_plot=False, print_results=False)
        group_2_estimates = Fit_Weibull_2P(failures=GROUP_2_failures, right_censored=GROUP_2_right_cens, show_probability_plot=False, print_results=False)
        p_guess = (len(GROUP_1_failures) + len(GROUP_1_right_cens)) / len(all_data)  # proportion guess
        guess = [group_1_estimates.alpha, group_1_estimates.beta, group_2_estimates.alpha, group_2_estimates.beta, p_guess]  # A1,B1,A2,B2,P

        # solve it
        bnds = [(0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None), (0.0001, 0.9999)]  # bounds of solution
        result = minimize(value_and_grad(Fit_Weibull_Mixture.LL), guess, args=(failures, right_censored), jac=True, bounds=bnds, tol=1e-6)
        params = result.x
        self.alpha_1 = params[0]
        self.beta_1 = params[1]
        self.alpha_2 = params[2]
        self.beta_2 = params[3]
        self.proportion_1 = params[4]
        self.proportion_2 = 1 - params[4]

        params = [self.alpha_1, self.beta_1, self.alpha_2, self.beta_2, self.proportion_1]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Weibull_Mixture.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2

        if print_results is True:
            print('Parameters:', '\nAlpha 1:', self.alpha_1, '\nBeta 1:', self.beta_1, '\nAlpha 2:', self.alpha_2, '\nBeta 2:', self.beta_2, '\nProportion 1:', self.proportion_1)
        if show_plot is True:
            xvals = np.linspace(0, max(failures) * 1.05, 1000)
            plt.figure(figsize=(14, 6))
            plt.subplot(121)
            # make the histogram. Can't use plt.hist due to need to scale the heights when there's censored data
            num_bins = min(int(len(failures) / 2), 30)
            hist, bins = np.histogram(failures, bins=num_bins, density=True)
            hist_cumulative = np.cumsum(hist) / sum(hist)
            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2
            frac_failing = len(failures) / len(all_data)
            plt.bar(center, hist * frac_failing, align='center', width=width, alpha=0.2, color='k', edgecolor='k')

            yvals_p1_pdf = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1).PDF(xvals=xvals, show_plot=False)
            yvals_p2_pdf = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2).PDF(xvals=xvals, show_plot=False)
            plt.plot(xvals, yvals_p1_pdf * self.proportion_1)
            plt.plot(xvals, yvals_p2_pdf * self.proportion_2)
            plt.title('Weibull Mixture PDF')
            plt.xlabel('Failure Times')
            plt.ylabel('Probability Density')

            plt.subplot(122)
            # make the histogram. Can't use plt.hist due to need to scale the heights when there's censored data
            plt.bar(center, hist_cumulative * frac_failing, align='center', width=width, alpha=0.2, color='k', edgecolor='k')
            yvals_p1_cdf = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1).CDF(xvals=xvals, show_plot=False)
            yvals_p2_cdf = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2).CDF(xvals=xvals, show_plot=False)
            y_mixture = yvals_p1_cdf * self.proportion_1 + yvals_p2_cdf * self.proportion_2
            plt.plot(xvals, y_mixture)
            plt.title('Weibull Mixture CDF')
            plt.xlabel('Failure Times')
            plt.ylabel('Cumulative Probability Density')
            plt.show()

    def logf(t, a1, b1, a2, b2, p):  # Log Mixture PDF (2 parameter Weibull)
        return anp.log(p * ((b1 * t ** (b1 - 1)) / (a1 ** b1)) * anp.exp(-((t / a1) ** b1)) + (1 - p) * ((b2 * t ** (b2 - 1)) / (a2 ** b2)) * anp.exp(-((t / a2) ** b2)))

    def logR(t, a1, b1, a2, b2, p):  # Log Mixture SF (2 parameter Weibull)
        return anp.log(p * anp.exp(-((t / a1) ** b1)) + (1 - p) * anp.exp(-((t / a2) ** b2)))

    def LL(params, T_f, T_rc):  # Log Mixture Likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_Mixture.logf(T_f, params[0], params[1], params[2], params[3], params[4]).sum()  # failure times
        LL_rc += Fit_Weibull_Mixture.logR(T_rc, params[0], params[1], params[2], params[3], params[4]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Expon_1P:
    '''
    Fit_Expon_1P
    Fits a 1-parameter Exponential distribution (Lambda) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    Lambda - the fitted Expon_1P lambda parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - an Exponential_Distribution object with the parameters of the fitted distribution
    Lambda_SE - the standard error (sqrt(variance)) of the parameter
    Lambda_upper - the upper CI estimate of the parameter
    Lambda_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for the parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        if failures is None or len(failures) < 1:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least one failure to calculate Exponential parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty list if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # solve it
        self.gamma = 0
        sp = ss.expon.fit(all_data, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [1 / sp[1]]
        warnings.filterwarnings('ignore')  # necessary to supress the warning about the jacobian when using the nelder-mead optimizer
        result = minimize(value_and_grad(Fit_Expon_1P.LL), guess, args=(failures, right_censored), jac=True, tol=1e-6, method='nelder-mead')

        if result.success is True:
            params = result.x
            self.success = True
            self.Lambda = params[0]
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Expon_1P. The fit from Scipy was used instead so results may not be accurate.')
            self.Lambda = 1 / sp[1]

        params = [self.Lambda]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Expon_1P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Exponential_Distribution(Lambda=self.Lambda)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Expon_1P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
        self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
        SE_inv = abs(1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z)
        Data = {'Parameter': ['Lambda', '1/Lambda'],
                'Point Estimate': [self.Lambda, 1 / self.Lambda],
                'Standard Error': [self.Lambda_SE, SE_inv],
                'Lower CI': [self.Lambda_lower, 1 / self.Lambda_upper],
                'Upper CI': [self.Lambda_upper, 1 / self.Lambda_lower]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Expon_1P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Exponential_probability_plot_Weibull_Scale
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Exponential_probability_plot_Weibull_Scale(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, L):  # Log PDF (1 parameter Expon)
        return anp.log(L) - L * t

    def logR(t, L):  # Log SF (1 parameter Expon)
        return -(L * t)

    def LL(params, T_f, T_rc):  # log likelihood function (1 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Expon_1P.logf(T_f, params[0]).sum()  # failure times
        LL_rc += Fit_Expon_1P.logR(T_rc, params[0]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Expon_2P:
    '''
    Fit_Expon_2P
    Fits a 2-parameter Exponential distribution (Lambda,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    Lambda - the fitted Expon_2P lambda parameter
    Lambda_inv - the inverse of the Lambda parameter (1/Lambda)
    gamma - the fitted Expon_2P gamma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
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

    *Note that this is a 2 parameter distribution but Lambda_inv is also provided as some programs (such as minitab and scipy.stats) use this instead of Lambda
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        # Regarding the confidence intervals of the parameters, the gamma parameter is estimated by optimizing the log-likelihood function but
        # it is assumed as fixed because the variance-covariance matrix of the estimated parameters cannot be determined numerically. By assuming
        # the standard error in gamma is zero, we can use Expon_1P to obtain the confidence intervals for Lambda. This is the same procedure
        # performed by both Reliasoft and Minitab. You may find the results are slightly different to Minitab and this is because the optimisation
        # of gamma is done more efficiently than Minitab does it. This is evidenced by comparing the log-likelihood for the same data input.
        if failures is None or len(failures) < 2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failure to calculate Exponential parameters.')
        if right_censored is None:
            right_censored = []  # fill with empty lists if not specified
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # get a quick initial guess for gamma by setting gamma as the minimum of all data
        offset = 0.001  # this is to ensure the upper bound for gamma is not equal to min(data) which would result in inf log-likelihood. This small offset fixes that issue
        self.gamma = min(all_data) - offset

        # get an initial guess for Lambda
        data_shifted = all_data - self.gamma
        sp = ss.expon.fit(data_shifted, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[1], self.gamma]  # this uses the inverted form given by scipy
        self.initial_guess = guess
        k = len(guess)
        n = len(all_data)

        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        bnds2 = [(0, None), (0, min(all_data) - offset)]  # bounds on the solution. Helps a lot with stability
        inv = True  # try the inverted form first
        # The reason for having an inverted and non-inverted cases is due to the gradient being too shallow in some cases. If Lambda<1 we invert it so it's bigger. This prevents the gradient getting too shallow for the optimizer to find the correct minimum.
        while delta_BIC > 0.001 and runs < 5:  # exits after BIC convergence or 5 iterations
            runs += 1
            if inv is True:
                result = minimize(value_and_grad(Fit_Expon_2P.LL_inv), guess, args=(failures, right_censored), jac=True, method='L-BFGS-B', bounds=bnds2)
            if result.success is False or inv is False:
                if runs == 1:
                    guess = [1 / sp[1], self.gamma]  # fix the guess to be the non-inverted form
                    self.initial_guess = guess
                result = minimize(value_and_grad(Fit_Expon_2P.LL), guess, args=(failures, right_censored), jac=True, method='L-BFGS-B', bounds=bnds2)
                inv = False  # inversion status changed for subsequent loops

            params = result.x
            guess = [params[0], params[1]]
            if inv is False:
                LL2 = 2 * Fit_Expon_2P.LL(guess, failures, right_censored)
            else:
                LL2 = 2 * Fit_Expon_2P.LL_inv(guess, failures, right_censored)
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        if result.success is True:
            params = result.x
            self.success = True
            if inv is False:
                self.Lambda = params[0]
            else:
                self.Lambda = 1 / params[0]
            self.gamma = params[1]
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Expon_2P. The fit from Scipy was used instead so results may not be accurate.')
            sp = ss.expon.fit(all_data, optimizer='powell')
            self.Lambda = sp[1]
            self.gamma = sp[0]

        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Exponential_Distribution(Lambda=self.Lambda, gamma=self.gamma)

        # confidence interval estimates of parameters. Uses Expon_1P because gamma (while optimized) cannot be used in the MLE solution as the solution is unbounded
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Expon_1P.LL)(np.array(tuple([self.Lambda])), np.array(tuple(failures - self.gamma)), np.array(tuple(right_censored - self.gamma)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.gamma_SE = 0
        self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
        self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
        self.gamma_upper = self.gamma
        self.gamma_lower = self.gamma
        self.Lambda_inv = 1 / self.Lambda
        self.Lambda_SE_inv = abs(1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z)
        self.Lambda_lower_inv = 1 / self.Lambda_upper
        self.Lambda_upper_inv = 1 / self.Lambda_lower

        Data = {'Parameter': ['Lambda', '1/Lambda', 'Gamma'],
                'Point Estimate': [self.Lambda, self.Lambda_inv, self.gamma],
                'Standard Error': [self.Lambda_SE, self.Lambda_SE_inv, self.gamma_SE],
                'Lower CI': [self.Lambda_lower, self.Lambda_lower_inv, self.gamma_lower],
                'Upper CI': [self.Lambda_upper, self.Lambda_upper_inv, self.gamma_upper]}

        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Expon_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Exponential_probability_plot_Weibull_Scale
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Exponential_probability_plot_Weibull_Scale(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, L, g):  # Log PDF (2 parameter Expon)
        return anp.log(L) - L * (t - g)

    def logR(t, L, g):  # Log SF (2 parameter Expon)
        return -(L * (t - g))

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Expon_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Expon_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)

    # #this is the inverted forms of the above functions. It simply changes Lambda to be 1/Lambda which is necessary when Lambda<<1
    def LL_inv(params, T_f, T_rc):  # log likelihood function (2 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Expon_2P.logf(T_f, 1 / params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Expon_2P.logR(T_rc, 1 / params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Normal_2P:
    '''
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
    force_sigma - Use this to specify the sigma value if you need to force sigma to be a certain value. Used in ALT probability plotting. Optional input.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    mu - the fitted Normal_2P mu parameter
    sigma - the fitted Normal_2P sigma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Normal_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma - the covariance between the parameters
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95, force_sigma=None):
        if force_sigma is not None and (failures is None or len(failures) < 1):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failures to calculate Normal parameters when force_sigma is specified.')
        elif force_sigma is None and (failures is None or len(failures) < 2):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # solve it
        sp = ss.norm.fit(all_data, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        warnings.filterwarnings('ignore')  # necessary to supress the warning about the jacobian when using the Powell optimizer
        if force_sigma is None:
            guess = [sp[0], sp[1]]
            result = minimize(value_and_grad(Fit_Normal_2P.LL), guess, args=(failures, right_censored), jac=True, method='powell', tol=1e-6)
        else:
            guess = [sp[0]]
            result = minimize(value_and_grad(Fit_Normal_2P.LL_fs), guess, args=(failures, right_censored, force_sigma), jac=True, method='powell', tol=1e-6)

        if result.success is True:
            params = result.x
            self.success = True
            if force_sigma is None:
                self.mu = params[0]
                self.sigma = params[1]
            else:
                self.mu = params * 1  # the *-1 converts ndarray to float64
                self.sigma = force_sigma
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Normal_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.mu = sp[0]
            self.sigma = sp[1]

        params = [self.mu, self.sigma]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Normal_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Normal_Distribution(mu=self.mu, sigma=self.sigma)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Normal_2P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = abs(covariance_matrix[0][1])
            self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        else:
            hessian_matrix = hessian(Fit_Normal_2P.LL_fs)(np.array(tuple([self.mu])), np.array(tuple(failures)), np.array(tuple(right_censored)), np.array(tuple([force_sigma])))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = ''
            self.Cov_mu_sigma = ''
            self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = ''
            self.sigma_lower = ''

        Data = {'Parameter': ['Mu', 'Sigma'],
                'Point Estimate': [self.mu, self.sigma],
                'Standard Error': [self.mu_SE, self.sigma_SE],
                'Lower CI': [self.mu_lower, self.sigma_lower],
                'Upper CI': [self.mu_upper, self.sigma_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Normal_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Normal_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Normal_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, mu, sigma):  # Log PDF (Normal)
        return anp.log(anp.exp(-0.5 * (((t - mu) / sigma) ** 2))) - anp.log((sigma * (2 * anp.pi) ** 0.5))

    def logR(t, mu, sigma):  # Log SF (Normal)
        return anp.log((1 + erf(((mu - t) / sigma) / 2 ** 0.5)) / 2)

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Normal_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)

    def LL_fs(params, T_f, T_rc, force_sigma):  # log likelihood function (2 parameter weibull) FORCED SIGMA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Normal_2P.logf(T_f, params[0], force_sigma).sum()  # failure times
        LL_rc += Fit_Normal_2P.logR(T_rc, params[0], force_sigma).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_2P:
    '''
    Fit_Lognormal_2P
    Fits a 2-parameter Lognormal distribution (mu,sigma) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    force_sigma - Use this to specify the sigma value if you need to force sigma to be a certain value. Used in ALT probability plotting. Optional input.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    mu - the fitted Lognormal_2P mu parameter
    sigma - the fitted Lognormal_2P sigma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Lognormal_Distribution object with the parameters of the fitted distribution
    mu_SE - the standard error (sqrt(variance)) of the parameter
    sigma_SE - the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma - the covariance between the parameters
    mu_upper - the upper CI estimate of the parameter
    mu_lower - the lower CI estimate of the parameter
    sigma_upper - the upper CI estimate of the parameter
    sigma_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95, force_sigma=None):
        if force_sigma is not None and (failures is None or len(failures) < 1):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failures to calculate Lognormal parameters when force_sigma is specified.')
        elif force_sigma is None and (failures is None or len(failures) < 2):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')

        self.gamma = 0
        all_data = np.hstack([failures, right_censored])

        # solve it
        sp = ss.lognorm.fit(all_data, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        if force_sigma is None:
            bnds = [(0.0001, None), (0.0001, None)]  # bounds of solution
            guess = [np.log(sp[2]), sp[0]]
            result = minimize(value_and_grad(Fit_Lognormal_2P.LL), guess, args=(failures, right_censored), jac=True, bounds=bnds, tol=1e-6)
        else:
            bnds = [(0.0001, None)]  # bounds of solution
            guess = [np.log(sp[2])]
            result = minimize(value_and_grad(Fit_Lognormal_2P.LL_fs), guess, args=(failures, right_censored, force_sigma), jac=True, bounds=bnds, tol=1e-6)

        if result.success is True:
            params = result.x
            self.success = True
            if force_sigma is None:
                self.mu = params[0]
                self.sigma = params[1]
            else:
                self.mu = params[0]
                self.sigma = force_sigma

        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Lognormal_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.mu = np.log(sp[2])
            self.sigma = sp[0]

        params = [self.mu, self.sigma]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Lognormal_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Lognormal_Distribution(mu=self.mu, sigma=self.sigma)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = abs(covariance_matrix[0][1])
            self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        else:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL_fs)(np.array(tuple([self.mu])), np.array(tuple(failures)), np.array(tuple(right_censored)), np.array(tuple([force_sigma])))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = ''
            self.Cov_mu_sigma = ''
            self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = ''
            self.sigma_lower = ''

        Data = {'Parameter': ['Mu', 'Sigma'],
                'Point Estimate': [self.mu, self.sigma],
                'Standard Error': [self.mu_SE, self.sigma_SE],
                'Lower CI': [self.mu_lower, self.sigma_lower],
                'Upper CI': [self.mu_upper, self.sigma_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Lognormal_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Lognormal_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, mu, sigma):  # Log PDF (Lognormal)
        return anp.log(anp.exp(-0.5 * (((anp.log(t) - mu) / sigma) ** 2)) / (t * sigma * (2 * anp.pi) ** 0.5))

    def logR(t, mu, sigma):  # Log SF (Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t) - mu) / (sigma * 2 ** 0.5)))

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter lognormal)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Lognormal_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)

    def LL_fs(params, T_f, T_rc, force_sigma):  # log likelihood function (2 parameter lognormal) FORCED SIGMA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_2P.logf(T_f, params[0], force_sigma).sum()  # failure times
        LL_rc += Fit_Lognormal_2P.logR(T_rc, params[0], force_sigma).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Lognormal_3P:
    '''
    Fit_Lognormal_3P
    Fits a 3-parameter Lognormal distribution (mu,sigma,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    mu - the fitted Lognormal_3P mu parameter
    sigma - the fitted Lognormal_3P sigma parameter
    gamma - the fitted Lognormal_3P gamma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
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
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        if failures is None or len(failures) < 3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate Lognormal parameters.')
        if right_censored is None:
            right_censored = []  # fill with empty list if not specified
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # this tries two methods to get the guess for gamma. If the fast way fails (which is about 1 in 1000 chance) then it will do the slower more reliable way.
        success = False
        iterations = 0
        offset = 0.0001  # this is to ensure the upper bound for gamma is not equal to min(data) which would result in inf log-likelihood. This small offset fixes that issue
        while success is False:
            iterations += 1
            if iterations == 1:
                # get a quick initial guess using the minimum of the data
                if min(all_data) <= np.e:
                    self.gamma = 0
                else:
                    self.gamma = np.log(min(all_data))
                gamma_initial_guess = self.gamma
            else:
                # get a better guess for gamma by optimizing the LL of a shifted distribution. This will only be run if the first attempt didn't work
                gamma_initial_guess = min(all_data) - offset
                bnds1 = [(0, min(all_data) - offset)]  # bounds on the solution. Helps a lot with stability
                gamma_res = minimize(Fit_Lognormal_3P.gamma_optimizer, gamma_initial_guess, args=(failures, right_censored), method='L-BFGS-B', bounds=bnds1)
                self.gamma = gamma_res.x[0]

            # obtain the initial guess for mu and sigma
            data_shifted = all_data - self.gamma
            sp = ss.lognorm.fit(data_shifted, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
            guess = [np.log(sp[2]), sp[0], self.gamma]
            self.initial_guess = guess
            k = len(guess)
            n = len(all_data)

            delta_BIC = 1
            BIC_array = [1000000]
            runs = 0

            gamma_lower_bound = 0.85 * gamma_initial_guess  # 0.85 is found to be the optimal point to minimise the error while also not causing autograd to fail
            bnds2 = [(-10, None), (0, None), (gamma_lower_bound, min(all_data) - offset)]  # bounds on the solution. Helps a lot with stability
            while delta_BIC > 0.001 and runs < 5:  # exits after BIC convergence or 5 iterations
                runs += 1
                result = minimize(value_and_grad(Fit_Lognormal_3P.LL), guess, args=(failures, right_censored), jac=True, method='L-BFGS-B', bounds=bnds2)
                params = result.x
                guess = [params[0], params[1], params[2]]
                LL2 = 2 * Fit_Lognormal_3P.LL(guess, failures, right_censored)
                BIC_array.append(np.log(n) * k + LL2)
                delta_BIC = abs(BIC_array[-1] - BIC_array[-2])
            success = result.success

        if result.success is True:
            params = result.x
            self.success = True
            self.mu = params[0]
            self.sigma = params[1]
            self.gamma = params[2]
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Lognormal_3P. The fit from Scipy was used instead so the results may not be accurate.')
            sp = ss.lognorm.fit(all_data, optimizer='powell')
            self.mu = np.log(sp[2])
            self.sigma = sp[0]
            self.gamma = sp[1]

        params = [self.mu, self.sigma, self.gamma]
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Lognormal_Distribution(mu=self.mu, sigma=self.sigma, gamma=self.gamma)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Lognormal_3P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.gamma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.mu_upper = self.mu + (Z * self.mu_SE)  # Mu can be positive or negative.
        self.mu_lower = self.mu + (-Z * self.mu_SE)
        self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))  # sigma is strictly positive
        self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        self.gamma_upper = self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
        self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

        Data = {'Parameter': ['Mu', 'Sigma', 'Gamma'],
                'Point Estimate': [self.mu, self.sigma, self.gamma],
                'Standard Error': [self.mu_SE, self.sigma_SE, self.gamma_SE],
                'Lower CI': [self.mu_lower, self.sigma_lower, self.gamma_lower],
                'Upper CI': [self.mu_upper, self.sigma_upper, self.gamma_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Lognormal_3P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Lognormal_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def gamma_optimizer(gamma_guess, failures, right_censored):
        failures_shifted = failures - gamma_guess[0]
        right_censored_shifted = right_censored - gamma_guess[0]
        all_data_shifted = np.hstack([failures_shifted, right_censored_shifted])
        sp = ss.lognorm.fit(all_data_shifted, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [np.log(sp[2]), sp[0]]
        warnings.filterwarnings('ignore')  # necessary to supress the warning about the jacobian when using the nelder-mead optimizer
        result = minimize(value_and_grad(Fit_Lognormal_2P.LL), guess, args=(failures_shifted, right_censored_shifted), jac=True, tol=1e-2, method='nelder-mead')

        if result.success is True:
            params = result.x
            mu = params[0]
            sigma = params[1]
        else:
            print('WARNING: Fitting using Autograd FAILED for the gamma optimisation section of Lognormal_3P. The fit from Scipy was used instead so results may not be accurate.')
            mu = sp[2]
            sigma = sp[0]

        LL2 = 2 * Fit_Lognormal_2P.LL([mu, sigma], failures_shifted, right_censored_shifted)
        return LL2

    def logf(t, mu, sigma, gamma):  # Log PDF (3 parameter Lognormal)
        return anp.log(anp.exp(-0.5 * (((anp.log(t - gamma) - mu) / sigma) ** 2)) / ((t - gamma) * sigma * (2 * anp.pi) ** 0.5))

    def logR(t, mu, sigma, gamma):  # Log SF (3 parameter Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t - gamma) - mu) / (sigma * 2 ** 0.5)))

    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Lognormal)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Lognormal_3P.logf(T_f, params[0], params[1], params[2]).sum()  # failure times
        LL_rc += Fit_Lognormal_3P.logR(T_rc, params[0], params[1], params[2]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Gamma_2P:
    '''
    Fit_Gamma_2P
    Fits a 2-parameter Gamma distribution (alpha,beta) to the data provided.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
    force_beta - Use this to specify the beta value if you need to force beta to be a certain value. Used in ALT probability plotting. Optional input.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Gamma_2P alpha parameter
    beta - the fitted Gamma_2P beta parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Gamma_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95, force_beta=None):
        if force_beta is not None and (failures is None or len(failures) < 1):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least 1 failures to calculate Gamma parameters when force_sigma is specified.')
        elif force_beta is None and (failures is None or len(failures) < 2):
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Gamma parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # solve it
        self.gamma = 0
        sp = ss.gamma.fit(all_data, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        warnings.filterwarnings('ignore')
        if force_beta is None:
            guess = [sp[2], sp[0]]
            result = minimize(value_and_grad(Fit_Gamma_2P.LL), guess, args=(failures, right_censored), jac=True, method='nelder-mead', tol=1e-10)
        else:
            guess = [sp[2]]
            result = minimize(value_and_grad(Fit_Gamma_2P.LL_fb), guess, args=(failures, right_censored, force_beta), jac=True, method='nelder-mead', tol=1e-10)

        if result.success is True:
            params = result.x
            self.success = True
            if force_beta is None:
                self.alpha = params[0]
                self.beta = params[1]
            else:
                self.alpha = params[0]
                self.beta = force_beta
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Gamma_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

        params = [self.alpha, self.beta]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Gamma_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Gamma_Distribution(alpha=self.alpha, beta=self.beta)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        if force_beta is None:
            hessian_matrix = hessian(Fit_Gamma_2P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = abs(covariance_matrix[0][1])
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        else:
            hessian_matrix = hessian(Fit_Gamma_2P.LL_fb)(np.array(tuple([self.alpha])), np.array(tuple(failures)), np.array(tuple(right_censored)), np.array(tuple([force_beta])))
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = ''
            self.Cov_alpha_beta = ''
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = ''
            self.beta_lower = ''
        Data = {'Parameter': ['Alpha', 'Beta'],
                'Point Estimate': [self.alpha, self.beta],
                'Standard Error': [self.alpha_SE, self.beta_SE],
                'Lower CI': [self.alpha_lower, self.beta_lower],
                'Upper CI': [self.alpha_upper, self.beta_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Gamma_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Gamma_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, a, b):  # Log PDF (2 parameter Gamma)
        return anp.log(t ** (b - 1)) - anp.log((a ** b) * agamma(b)) - (t / a)

    def logR(t, a, b):  # Log SF (2 parameter Gamma)
        return anp.log(gammaincc(b, t / a))

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Gamma_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)

    def LL_fb(params, T_f, T_rc, force_beta):  # log likelihood function (2 parameter Gamma) FORCED BETA
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_2P.logf(T_f, params[0], force_beta).sum()  # failure times
        LL_rc += Fit_Gamma_2P.logR(T_rc, params[0], force_beta).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Gamma_3P:
    '''
    Fit_Gamma_3P
    Fits a 3-parameter Gamma distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Gamma_3P alpha parameter
    beta - the fitted Gamma_3P beta parameter
    gamma - the fitted Gamma_3P gamma parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
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
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        if failures is None or len(failures) < 3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate Gamma parameters.')
        if right_censored is None:
            right_censored = []  # fill with empty list if not specified
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])

        # get a quick guess for gamma by setting it as the minimum of all the data.
        offset = 0.0001  # this is to ensure the upper bound for gamma is not equal to min(data) which would result in inf log-likelihood. This small offset fixes that issue
        self.gamma = min(all_data) - offset

        # obtain the initial guess for alpha and beta
        data_shifted = all_data - self.gamma
        sp = ss.gamma.fit(data_shifted, floc=0, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0], self.gamma]
        self.initial_guess = guess
        k = len(guess)
        n = len(all_data)

        delta_BIC = 1
        BIC_array = [1000000]
        runs = 0
        bnds = [(0, None), (0, None), (0, min(all_data) - offset)]  # bounds on the solution. Helps a lot with stability
        while delta_BIC > 0.001 and runs < 5:  # exits after BIC convergence or 5 iterations
            runs += 1
            result = minimize(value_and_grad(Fit_Gamma_3P.LL), guess, args=(failures, right_censored), jac=True, method='L-BFGS-B', bounds=bnds)
            params = result.x
            guess = [params[0], params[1], params[2]]
            LL2 = 2 * Fit_Gamma_3P.LL(guess, failures, right_censored)
            BIC_array.append(np.log(n) * k + LL2)
            delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

        if result.success is True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
            self.gamma = params[2]
        else:
            self.success = False
            print('WARNING: Fitting using Autograd FAILED for Gamma_3P. The fit from Scipy was used instead so the results may not be accurate.')
            sp = ss.gamma.fit(all_data, optimizer='powell')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

        params = [self.alpha, self.beta, self.gamma]
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Gamma_Distribution(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Gamma_3P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.gamma_SE = abs(covariance_matrix[2][2]) ** 0.5
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        self.gamma_upper = self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer.
        self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))

        Data = {'Parameter': ['Alpha', 'Beta', 'Gamma'],
                'Point Estimate': [self.alpha, self.beta, self.gamma],
                'Standard Error': [self.alpha_SE, self.beta_SE, self.gamma_SE],
                'Lower CI': [self.alpha_lower, self.beta_lower, self.gamma_lower],
                'Upper CI': [self.alpha_upper, self.beta_upper, self.gamma_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Gamma_3P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Gamma_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, a, b, g):  # Log PDF (3 parameter Gamma)
        return anp.log((t - g) ** (b - 1)) - anp.log((a ** b) * agamma(b)) - ((t - g) / a)

    def logR(t, a, b, g):  # Log SF (3 parameter Gamma)
        return anp.log(gammaincc(b, (t - g) / a))

    def LL(params, T_f, T_rc):  # log likelihood function (3 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_3P.logf(T_f, params[0], params[1], params[2]).sum()  # failure times
        LL_rc += Fit_Gamma_3P.logR(T_rc, params[0], params[1], params[2]).sum()  # right censored times
        return -(LL_f + LL_rc)


class Fit_Beta_2P:
    '''
    Fit_Beta_2P
    Fits a 2-parameter Beta distribution (alpha,beta) to the data provided.
    All data must be in the range 0-1.

    Inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data
    show_probability_plot - True/False. Defaults to True.
    print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

    Outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Beta_2P alpha parameter
    beta - the fitted Beta_2P beta parameter
    loglik2 - LogLikelihood*-2
    AICc - Akaike Information Criterion
    BIC - Bayesian Information Criterion
    distribution - a Beta_Distribution object with the parameters of the fitted distribution
    alpha_SE - the standard error (sqrt(variance)) of the parameter
    beta_SE - the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta - the covariance between the parameters
    alpha_upper - the upper CI estimate of the parameter
    alpha_lower - the lower CI estimate of the parameter
    beta_upper - the upper CI estimate of the parameter
    beta_lower - the lower CI estimate of the parameter
    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
    '''

    def __init__(self, failures=None, right_censored=None, show_probability_plot=True, print_results=True, CI=0.95):
        if failures is None or len(failures) < 2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Beta parameters.')
        if CI <= 0 or CI >= 1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored = []

        # adjust inputs to be arrays
        if type(failures) == list:
            failures = np.array(failures)
        if type(failures) != np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures, right_censored])
        if min(all_data) < 0 or max(all_data) > 1:
            raise ValueError('All data must be between 0 and 1 to use the beta distribution.')
        bnds = [(0.0001, None), (0.0001, None)]  # bounds of solution

        # solve it
        self.gamma = 0
        sp = ss.beta.fit(all_data, floc=0, fscale=1, optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[0], sp[1]]
        result = minimize(value_and_grad(Fit_Beta_2P.LL), guess, args=(failures, right_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success is True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Beta_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[0]
            self.beta = sp[1]

        params = [self.alpha, self.beta]
        k = len(params)
        n = len(all_data)
        LL2 = 2 * Fit_Beta_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k ** 2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = 'Insufficient data'
        self.BIC = np.log(n) * k + LL2
        self.distribution = Beta_Distribution(alpha=self.alpha, beta=self.beta)

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Beta_2P.LL)(np.array(tuple(params)), np.array(tuple(failures)), np.array(tuple(right_censored)))
        covariance_matrix = np.linalg.inv(hessian_matrix)
        self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
        self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
        self.Cov_alpha_beta = abs(covariance_matrix[0][1])
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        Data = {'Parameter': ['Alpha', 'Beta'],
                'Point Estimate': [self.alpha, self.beta],
                'Standard Error': [self.alpha_SE, self.beta_SE],
                'Lower CI': [self.alpha_lower, self.beta_lower],
                'Upper CI': [self.alpha_upper, self.beta_upper]}
        df = pd.DataFrame(Data, columns=['Parameter', 'Point Estimate', 'Standard Error', 'Lower CI', 'Upper CI'])
        self.results = df.set_index('Parameter')

        if print_results is True:
            pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
            print(str('Results from Fit_Beta_2P (' + str(int(CI * 100)) + '% CI):'))
            print(self.results)

        if show_probability_plot is True:
            from reliability.Probability_plotting import Beta_probability_plot
            if len(right_censored) == 0:
                rc = None
            else:
                rc = right_censored
            Beta_probability_plot(failures=failures, right_censored=rc, __fitted_dist_params=self)

    def logf(t, a, b):  # Log PDF (2 parameter Beta)
        return anp.log(((t ** (a - 1)) * ((1 - t) ** (b - 1)))) - anp.log(abeta(a, b))

    def logR(t, a, b):  # Log SF (2 parameter Beta)
        return anp.log(1 - betainc(a, b, t))

    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter beta)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Beta_2P.logf(T_f, params[0], params[1]).sum()  # failure times
        LL_rc += Fit_Beta_2P.logR(T_rc, params[0], params[1]).sum()  # right censored times
        return -(LL_f + LL_rc)
