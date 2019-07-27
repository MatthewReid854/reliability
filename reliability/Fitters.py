'''
Fitters

This module contains custom fitting functions for parametric distributions which support left or right censored data (but not both together).
The supported distributions for failures and right censored data are:
Weibull_2P
Weibull_3P
Exponential_1P
Exponential_2P
Gamma_2P
Gamma_3P
Lognormal_2P
Normal_2P
Beta_2P
Weibull_Mixture

The supported distributions for failures and left censored data are:
Weibull_2P
Exponential_1P
Gamma_2P
Lognormal_2P
Normal_2P
Beta_2P
Weibull_Mixture

Note that the Beta distribution is only for data in the range 0-1.
There is also a Fit_Everything function which will fit all distributions except the Weibull mixture model and will provide plots and a table of values.

All functions in this module work using autograd to find the derivative of the log-likelihood function. In this way, the code only needs to specify
the log PDF, log CDF, and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained
using scipy. If the distribution is an extremely bad fit or is heavily censored then these guesses may be poor and the fit might not be successful.
Generally the fit achieved by autograd is highly successful.

A special thanks goes to Cameron Davidson-Pilon (author of the Python library "lifelines" and website "dataorigami.net") for providing help with getting
autograd to work, and for writing the python library "autograd-gamma", without which it would be impossible to fit the Beta or Gamma distributions using
autograd.
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
from autograd_gamma import betainc
from autograd.scipy.special import erf
from autograd_gamma import gammainc, gammaincc
anp.seterr('ignore')

class Fit_Everything:
    '''
    Fit_Everything

    This function will fit all available distributions for the data you enter. You may also specify left or right censored data (either but not both).

    inputs:
    failures - an array or list of the failure times (this does not need to be sorted).
    left_censored - an array or list of the left failure times (this does not need to be sorted).
    right_censored - an array or list of the right failure times (this does not need to be sorted).
    sort_by - goodness of fit test to sort results by. Must be either 'BIC' or 'AIC'. Default is BIC.
    show_plot - True/False. Defaults to True
    print_results - True/False. Defaults to True. Will show the results of the fitted parameters and the goodness of fit
        tests in a dataframe.
    show_quantile_plot - True/False. Defaults to True unless there is left censored data in which case Kaplan Meier cannot be applied.
        Provides a comparison of parametric vs non-parametric fit.

    outputs:
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
        Weibull_3P_AIC
        Weibull_3P_SQD

    All parametric models have the number of parameters in the name. For example, Weibull_2P used alpha and beta, whereas Weibull_3P
    uses alpha, beta, and gamma. This is applied even for Normal_2P for consistency in naming conventions.

    If plot_results is True, the plot will show the PDF and CDF of all fitted distributions plotted with a histogram of the data.
    From the results, the distributions are sorted based on their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the distribution to the data.

    Confidence intervals for each of the fitted parameters are not supported. This feature may be incorporated in
    future releases, however, the need has not been identified. See the python library "lifelines" or JMP Pro software if this is required.
    Whilst Minitab uses the Anderson-Darling statistic for the goodness of fit, it is generally recognised that AICc and BIC
    are more accurate measures as they take into account the number of parameters in the distribution.

    Example Usage:
    X = [0.95892,1.43249,1.04221,0.67583,3.28411,1.03072,0.05826,1.81387,2.06383,0.59762,5.99005,1.92145,1.35179,0.50391]
    output = Fit_Everything(X)

    To extract the parameters of the Weibull distribution from the results dataframe, you may access the parameters by name:
    print('Weibull Alpha =',output.Weibull_2P_alpha,'\nWeibull Beta =',output.Weibull_2P_beta)

    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None, sort_by='BIC', show_plot=True, print_results=True,show_quantile_plot=True):
        if failures is None or len(failures)<3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate 3 parameter models.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')
        if sort_by not in ['AIC','BIC']:
            raise ValueError('sort_by must be either AIC or BIC. Defaults to BIC')
        if show_plot not in [True,False]:
            raise  ValueError('show_plot must be either True of False. Defaults to True.')
        if print_results not in [True, False]:
            raise ValueError('print_results must be either True of False. Defaults to True.')
        if show_quantile_plot not in [True, False]:
            raise ValueError('show_quantile_plot must be either True of False. Defaults to True.')
        if min(failures)<=0:
            raise ValueError('All failure times must be greater than zero.')

        self.failures = failures
        if left_censored is None:
            LC = []
        else:
            LC = left_censored
            show_quantile_plot = False #can't do Kaplan-Meier estimates with left censored data
        if right_censored is None:
            RC = []
        else:
            RC = right_censored
        self._all_data = np.hstack([failures, LC, RC])

        #These are all used for scaling the histogram when there is censored data
        self._frac_fail = len(failures) / len(self._all_data)
        self._frac_fail_L = len(failures) / len(np.hstack([failures, LC]))
        # self._left_cens_ratio = len(LC) / len(failures)

        # Kaplan-Meier estimate of quantiles. Used in Q-Q plot.
        d = sorted(self._all_data)  # sorting the failure data is necessary for plotting quantiles in order
        nonparametric = KaplanMeier(failures=failures,right_censored=right_censored, print_results=False, show_plot=False)
        self._nonparametric_CDF = 1 - np.array(nonparametric.KM)  # change SF into CDF

        #parametric models
        AIC_BIC_metric = self._all_data
        if left_censored is None: #This section includes location shifted (gamma>0) distributions. We can only fit 3P distributions when there is no left censored data
            self._left_cens = False
            #used by plot distributions to tell what parameters were calculated
            Weibull_3P_params = Fit_Weibull_3P(failures=failures, right_censored=right_censored)
            self.Weibull_3P_alpha = Weibull_3P_params.alpha
            self.Weibull_3P_beta = Weibull_3P_params.beta
            self.Weibull_3P_gamma = Weibull_3P_params.gamma
            self.Weibull_3P_BIC = Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta,gamma=self.Weibull_3P_gamma).BIC(AIC_BIC_metric)
            self.Weibull_3P_AIC = Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta,gamma=self.Weibull_3P_gamma).AICc(AIC_BIC_metric)
            self._parametric_CDF_Weibull_3P = Weibull_Distribution(alpha=self.Weibull_3P_alpha, beta=self.Weibull_3P_beta, gamma=self.Weibull_3P_gamma).CDF(xvals=d, show_plot=False)


            Gamma_3P_params = Fit_Gamma_3P(failures=failures, right_censored=right_censored)
            self.Gamma_3P_alpha = Gamma_3P_params.alpha
            self.Gamma_3P_beta = Gamma_3P_params.beta
            self.Gamma_3P_gamma = Gamma_3P_params.gamma
            self.Gamma_3P_BIC = Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta,gamma=self.Gamma_3P_gamma).BIC(AIC_BIC_metric)
            self.Gamma_3P_AIC = Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta,gamma=self.Gamma_3P_gamma).AICc(AIC_BIC_metric)
            self._parametric_CDF_Gamma_3P = Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta, gamma=self.Gamma_3P_gamma).CDF(xvals=d, show_plot=False)


            Expon_2P_params = Fit_Expon_2P(failures=failures,right_censored=right_censored)
            self.Expon_2P_lambda = Expon_2P_params.Lambda
            self.Expon_2P_gamma = Expon_2P_params.gamma
            self.Expon_2P_BIC = Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).BIC(AIC_BIC_metric)
            self.Expon_2P_AIC = Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).AICc(AIC_BIC_metric)
            self._parametric_CDF_Exponential_2P = Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).CDF(xvals=d, show_plot=False)


        else: #fills the non-calculated distributions with zeros so we don't get an error when these values are entered in the dataframe
            self._left_cens = True  # used by plot distributions to tell what parameters were calculated
            self.Weibull_3P_alpha = 0
            self.Weibull_3P_beta = 0
            self.Weibull_3P_gamma = 0
            self.Weibull_3P_BIC = 0
            self.Weibull_3P_AIC = 0
            self.Gamma_3P_alpha = 0
            self.Gamma_3P_beta = 0
            self.Gamma_3P_gamma = 0
            self.Gamma_3P_BIC = 0
            self.Gamma_3P_AIC = 0
            self.Expon_2P_lambda = 0
            self.Expon_2P_gamma = 0
            self.Expon_2P_BIC = 0
            self.Expon_2P_AIC = 0

        Normal_2P_params = Fit_Normal_2P(failures=failures, right_censored=right_censored,left_censored=left_censored)
        self.Normal_2P_mu = Normal_2P_params.mu
        self.Normal_2P_sigma = Normal_2P_params.sigma
        self.Normal_2P_BIC = Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).BIC(AIC_BIC_metric)
        self.Normal_2P_AIC = Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).AICc(AIC_BIC_metric)
        self._parametric_CDF_Normal_2P = Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).CDF(xvals=d, show_plot=False)

        Lognormal_2P_params = Fit_Lognormal_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        self.Lognormal_2P_mu = Lognormal_2P_params.mu
        self.Lognormal_2P_sigma = Lognormal_2P_params.sigma
        self.Lognormal_2P_BIC = Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).BIC(AIC_BIC_metric)
        self.Lognormal_2P_AIC = Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).AICc(AIC_BIC_metric)
        self._parametric_CDF_Lognormal_2P = Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).CDF(xvals=d, show_plot=False)

        Weibull_2P_params = Fit_Weibull_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        self.Weibull_2P_alpha = Weibull_2P_params.alpha
        self.Weibull_2P_beta = Weibull_2P_params.beta
        self.Weibull_2P_BIC = Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).BIC(AIC_BIC_metric)
        self.Weibull_2P_AIC = Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).AICc(AIC_BIC_metric)
        self._parametric_CDF_Weibull_2P = Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).CDF(xvals=d, show_plot=False)

        Gamma_2P_params = Fit_Gamma_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        self.Gamma_2P_alpha = Gamma_2P_params.alpha
        self.Gamma_2P_beta = Gamma_2P_params.beta
        self.Gamma_2P_BIC = Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).BIC(AIC_BIC_metric)
        self.Gamma_2P_AIC = Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).AICc(AIC_BIC_metric)
        self._parametric_CDF_Gamma_2P = Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).CDF(xvals=d, show_plot=False)

        Expon_1P_params = Fit_Expon_1P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        self.Expon_1P_lambda = Expon_1P_params.Lambda
        self.Expon_1P_BIC = Exponential_Distribution(Lambda=self.Expon_1P_lambda).BIC(AIC_BIC_metric)
        self.Expon_1P_AIC = Exponential_Distribution(Lambda=self.Expon_1P_lambda).AICc(AIC_BIC_metric)
        self._parametric_CDF_Exponential_1P = Exponential_Distribution(Lambda=self.Expon_1P_lambda).CDF(xvals=d, show_plot=False)

        if max(failures)<=1:
            Beta_2P_params = Fit_Beta_2P(failures=failures, right_censored=right_censored, left_censored=left_censored)
            self.Beta_2P_alpha = Beta_2P_params.alpha
            self.Beta_2P_beta = Beta_2P_params.beta
            self.Beta_2P_BIC = Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).BIC(AIC_BIC_metric)
            self.Beta_2P_AIC = Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).AICc(AIC_BIC_metric)
            self._parametric_CDF_Beta_2P = Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).CDF(xvals=d, show_plot=False)
        else:
            self.Beta_2P_alpha = 0
            self.Beta_2P_beta = 0
            self.Beta_2P_BIC = 0
            self.Beta_2P_AIC = 0

        #assemble the output dataframe
        DATA = {'Distribution': ['Weibull_3P','Weibull_2P','Normal_2P','Exponential_1P','Exponential_2P','Lognormal_2P','Gamma_2P','Gamma_3P','Beta_2P'],
                'Alpha':[self.Weibull_3P_alpha,self.Weibull_2P_alpha,'','','','',self.Gamma_2P_alpha,self.Gamma_3P_alpha,self.Beta_2P_alpha],
                'Beta':[self.Weibull_3P_beta,self.Weibull_2P_beta,'','','','',self.Gamma_2P_beta,self.Gamma_3P_beta,self.Beta_2P_beta],
                'Gamma':[self.Weibull_3P_gamma,'','','',self.Expon_2P_gamma,'','',self.Gamma_3P_gamma,''],
                'Mu':['','',self.Normal_2P_mu,'','',self.Lognormal_2P_mu,'','',''],
                'Sigma':['','',self.Normal_2P_sigma,'','',self.Lognormal_2P_sigma,'','',''],
                'Lambda':['','','',self.Expon_1P_lambda,self.Expon_2P_lambda,'','','',''],
                'AICc':[self.Weibull_3P_AIC, self.Weibull_2P_AIC, self.Normal_2P_AIC, self.Expon_1P_AIC, self.Expon_2P_AIC, self.Lognormal_2P_AIC, self.Gamma_2P_AIC, self.Gamma_3P_AIC, self.Beta_2P_AIC],
                'BIC':[self.Weibull_3P_BIC, self.Weibull_2P_BIC, self.Normal_2P_BIC, self.Expon_1P_BIC, self.Expon_2P_BIC, self.Lognormal_2P_BIC, self.Gamma_2P_BIC, self.Gamma_3P_BIC, self.Beta_2P_BIC]}

        df = pd.DataFrame(DATA,columns = ['Distribution','Alpha','Beta','Gamma','Mu','Sigma','Lambda','AICc','BIC'])
        #sort the dataframe by BIC of AICc and replace na and 0 values with spaces. Most negative AICc or BIC is better fit
        if sort_by in ['BIC','bic']:
            df2 = df.reindex(df.BIC.sort_values().index)
        elif sort_by in ['AICc','AIC','aic']:
            df2 = df.reindex(df.AICc.sort_values().index)
        else:
            raise ValueError('Invalid input to sort_by. Options are BIC or AIC. Default is BIC')
        df3 = df2.set_index('Distribution').fillna('').replace(to_replace=0,value='')
        if self.Beta_2P_BIC == 0: #remove beta if it was not fitted (due to data being outside of 0 to 1 range)
            df3 = df3.drop('Beta_2P',axis=0)
        if self._left_cens ==True: #remove all location shifted distributions if they were not fitted (due to having left censored data)
            df3 = df3.drop(['Weibull_3P','Exponential_2P','Gamma_3P'], axis=0)
        self.results = df3

        #creates a distribution object of the best fitting distribution and assigns its name
        best_dist = df3.index.values[0]
        self.best_distribution_name = best_dist
        if best_dist == 'Weibull_2P':
            self.best_distribution = Weibull_Distribution(alpha=self.Weibull_2P_alpha,beta=self.Weibull_2P_beta)
        elif best_dist == 'Weibull_3P':
            self.best_distribution = Weibull_Distribution(alpha=self.Weibull_3P_alpha,beta=self.Weibull_3P_beta,gamma=self.Weibull_3P_gamma)
        elif best_dist == 'Gamma_2P':
            self.best_distribution = Gamma_Distribution(alpha=self.Gamma_2P_alpha,beta=self.Gamma_2P_beta)
        elif best_dist == 'Gamma_3P':
            self.best_distribution = Gamma_Distribution(alpha=self.Gamma_3P_alpha,beta=self.Gamma_3P_beta,gamma=self.Gamma_3P_gamma)
        elif best_dist == 'Lognormal_2P':
            self.best_distribution = Lognormal_Distribution(mu=self.Lognormal_2P_mu,sigma=self.Lognormal_2P_sigma)
        elif best_dist == 'Exponential_1P':
            self.best_distribution = Exponential_Distribution(Lambda=self.Expon_1P_lambda)
        elif best_dist == 'Exponential_2P':
            self.best_distribution = Exponential_Distribution(Lambda=self.Expon_2P_lambda,gamma=self.Expon_2P_gamma)
        elif best_dist == 'Normal_2P':
            self.best_distribution = Normal_Distribution(mu=self.Normal_2P_mu,sigma=self.Normal_2P_sigma)
        elif best_dist == 'Beta_2P':
            self.best_distribution = Beta_Distribution(alpha=self.Beta_2P_alpha,beta=self.Beta_2P_beta)

        #print the results
        if print_results==True: #printing occurs by default
            pd.set_option('display.width', 200) #prevents wrapping after default 80 characters
            pd.set_option('display.max_columns', 9)#shows the dataframe without ... truncation
            print(self.results)

        if show_plot==True:
            Fit_Everything.plot_distributions(self) #plotting occurs by default

        if show_quantile_plot==True:
            Fit_Everything.Q_Q_plot(self) #plotting occurs by default unless there is left censored data

        if show_plot==True or show_quantile_plot==True:
            plt.show()

    def plot_distributions(self):
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
        hist, bins = np.histogram(X, bins=num_bins,density=True)
        hist_cumulative = np.cumsum(hist) / sum(hist)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        plt.bar(center, hist*self._frac_fail, align='center', width=width,alpha=0.2,color='k',edgecolor='k')

        if self._left_cens == False: #we can only plot the 3P distributions if they were fitted (in the case the left censored data was not provided)
            Weibull_Distribution(alpha=self.Weibull_3P_alpha,beta=self.Weibull_3P_beta,gamma=self.Weibull_3P_gamma).PDF(xvals=xvals,label=r'Weibull ($\alpha , \beta , \gamma$)')
            Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta,gamma=self.Gamma_3P_gamma).PDF(xvals=xvals,label=r'Gamma ($\alpha , \beta , \gamma$)')
            Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).PDF(xvals=xvals, label=r'Exponential ($\lambda , \gamma$)')
        Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta).PDF(xvals=xvals, label=r'Weibull ($\alpha , \beta$)')
        Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).PDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma$)')
        Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).PDF(xvals=xvals, label=r'Normal ($\mu , \sigma$)')
        Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).PDF(xvals=xvals,label=r'Gamma ($\alpha , \beta$)')
        Exponential_Distribution(Lambda=self.Expon_1P_lambda).PDF(xvals=xvals, label=r'Exponential ($\lambda$)')
        if max(X)<=1: #condition for Beta Dist to be fitted
            Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).PDF(xvals=xvals,label=r'Beta ($\alpha , \beta$)')
        plt.legend()
        plt.xlim([xmin, xmax])
        plt.title('PDF of fitted distributions')
        plt.xlabel('Data')
        plt.ylabel('Probability density')
        plt.legend()

        plt.subplot(122)  # CDF
        if self._left_cens == True:
            plt.bar(center, hist_cumulative*self._frac_fail+(1-self._frac_fail_L), align='center', width=width, alpha=0.2, color='k',edgecolor='k')
        else:
            plt.bar(center, hist_cumulative*self._frac_fail, align='center', width=width,alpha=0.2,color='k',edgecolor='k')
        if self._left_cens == False: #we can only plot the 3P distributions if they were fitted (in the case the left censored data was not provided)
            Weibull_Distribution(alpha=self.Weibull_3P_alpha,beta=self.Weibull_3P_beta,gamma=self.Weibull_3P_gamma).CDF(xvals=xvals,label=r'Weibull ($\alpha , \beta , \gamma$)')
            Gamma_Distribution(alpha=self.Gamma_3P_alpha, beta=self.Gamma_3P_beta,gamma=self.Gamma_3P_gamma).CDF(xvals=xvals,label=r'Gamma ($\alpha , \beta , \gamma$)')
            Exponential_Distribution(Lambda=self.Expon_2P_lambda, gamma=self.Expon_2P_gamma).CDF(xvals=xvals, label=r'Exponential ($\lambda , \gamma$)')
        Weibull_Distribution(alpha=self.Weibull_2P_alpha,beta=self.Weibull_2P_beta).CDF(xvals=xvals,label=r'Weibull ($\alpha , \beta$)')
        Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma).CDF(xvals=xvals, label=r'Lognormal ($\mu , \sigma$)')
        Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma).CDF(xvals=xvals, label=r'Normal ($\mu , \sigma$)')
        Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta).CDF(xvals=xvals,label=r'Gamma ($\alpha , \beta$)')
        Exponential_Distribution(Lambda=self.Expon_1P_lambda).CDF(xvals=xvals, label=r'Exponential ($\lambda$)')
        if max(X)<=1: #condition for Beta Dist to be fitted
            Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta).CDF(xvals=xvals,label=r'Beta ($\alpha , \beta$)')
        plt.legend()
        plt.xlim([xmin, xmax])
        plt.title('CDF of fitted distributions')
        plt.xlabel('Data')
        plt.ylabel('Probability density')
        plt.legend()

    def Q_Q_plot(self): #quantile-quantile plot of parametric vs non-parametric

        plot_id = 251 #set dimensions of plot
        fig_size = (10, 5)

        #plot each of the results
        plt.figure(figsize=fig_size)
        plt.suptitle('Quantile-Quantile plots of Parametric (x-axis)\nvs Non-Parametric (y-axis) for all distributions')
        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Exponential_1P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Exponential_1P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Exponential')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Weibull_2P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Weibull_2P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Weibull')
        plt.yticks([])
        plt.xticks([])
        plot_id += 1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Gamma_2P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Gamma_2P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Gamma')
        plt.yticks([])
        plt.xticks([])
        plot_id += 1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Normal_2P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Normal_2P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Normal')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Lognormal_2P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Lognormal_2P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Lognormal')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Exponential_2P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Exponential_2P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Exponential\n(2 parameter)')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Weibull_3P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Weibull_3P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Weibull\n(3 parameter)')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        plt.subplot(plot_id)
        xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Gamma_3P]))
        plt.plot(self._nonparametric_CDF,self._parametric_CDF_Gamma_3P,'b',alpha=0.8,linewidth=2)
        plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
        plt.axis('square')
        plt.title('Gamma\n(3 parameter)')
        plt.yticks([])
        plt.xticks([])
        plot_id+=1

        if max(self.failures)<=1:
            plt.subplot(plot_id)
            xlim = max(np.hstack([self._nonparametric_CDF,self._parametric_CDF_Beta_2P]))
            plt.plot(self._nonparametric_CDF,self._parametric_CDF_Beta_2P,'b',alpha=0.8,linewidth=2)
            plt.plot([0,xlim],[0,xlim],'r',alpha=0.7)
            plt.axis('square')
            plt.title('Beta')
            plt.yticks([])
            plt.xticks([])

class Fit_Weibull_2P:
    '''
    Fit_Weibull_2P

    Fits a 2-parameter Weibull distribution (alpha,beta) to the data provided.
    You may also enter left or right censored data (either but not both).

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Weibull_2P alpha parameter
    beta - the fitted Weibull_2P beta parameter
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Weibull parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        self.gamma = 0
        sp = ss.weibull_min.fit(all_data,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0]]
        result = minimize(value_and_grad(Fit_Weibull_2P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Weibull_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[2]
            self.beta = sp[0]

    def logf(t,a,b): #Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t/a) + anp.log(b/a) - (t / a) ** b

    def logF(t,a,b): #Log CDF (2 parameter Weibull)
        return anp.log(1-anp.exp((-((t/a)**b))))

    def logR(t,a,b): #Log SF (2 parameter Weibull)
        return -((t/a)**b)

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Weibull_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        LL_lc += Fit_Weibull_2P.logF(T_lc, params[0], params[1]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)


class Fit_Weibull_3P:
    '''
    Fit_Weibull_3P

    Fits a 3-parameter Weibull distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.
    Left censored data is not supported because of the way the gamma parameter is obtained. If you have left censored data, use
    Fit_Weibull_2P instead.

    inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Weibull_3P alpha parameter
    beta - the fitted Weibull_3P beta parameter
    gamma - the fitted Weibull_3P gamma parameter
    '''
    def __init__(self,failures=None,right_censored=None):
        if failures is None or len(failures)<3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate Weibull parameters.')
        if right_censored is None:
            right_censored=[]   # fill with empty list if not specified
        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures,right_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        shift = min(all_data)-0.01 #the 0.01 is to avoid taking the log of zero in logf and logF
        self.gamma = shift+0.001 #this adds 0.001 instead of 0.01 to avoid the min(failures) equalling gamma which would cause AIC and BIC to be inf. The difference it causes is negligible
        data_shifted = all_data-shift
        sp = ss.weibull_min.fit(data_shifted,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0]]
        result = minimize(value_and_grad(Fit_Weibull_3P.LL), guess, args=(failures - shift, right_censored - shift), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Weibull_3P. The fit from Scipy was used instead so results may not be accurate.')
            sp = ss.weibull_min.fit(all_data, optimizer='powell')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

    def logf(t,a,b): #Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t/a) + anp.log(b/a) - (t / a) ** b

    def logR(t,a,b): #Log SF (2 parameter Weibull)
        return -((t/a)**b)

    def LL(params,T_f,T_rc): #log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Weibull_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        return -(LL_f+LL_rc)

class Fit_Weibull_Mixture:
    '''
    Fit_Weibull_Mixture

    Fits a mixture of 2 x Weibull_2P distributions (this does not fit the gamma parameter).
    Censoring is supported (left or right but not both), though care should be taken to ensure that there still appears
    to be two groups when plotting only the failure data. A second group cannot be made from a mostly or totally censored
    set of samples.

    Use this model when you think there are multiple failure modes acting to create the failure data.
    Whilst some failure modes may not be fitted as well by a Weibull distribution as they may be by another distribution, it
    is unlikely that a mixture of data from two distributions (particularly if they are overlapping) will be fitted
    noticeably better by other types of mixtures than would be achieved by a Weibull mixture. For this reason, other types
    of mixtures are not implemented.

    inputs:
    failures - an array or list of the failure data. There must be at least 4 failures, but it is highly recommended to use another model if you have
        less than 20 failures.
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data
    print_results - True/False. This will print results to console. Default is False
    show_plot - True/False. This will show the PDF and CDF of the Weibull mixture with a histogram of the data. Default is False.

    outputs:
    alpha_1 - the fitted Weibull_2P alpha parameter for the first (left) group
    beta_1 - the fitted Weibull_2P beta parameter for the first (left) group
    alpha_2 - the fitted Weibull_2P alpha parameter for the second (right) group
    beta_2 - the fitted Weibull_2P beta parameter for the second (right) group
    proportion_1 - the fitted proportion of the first (left) group
    proportion_2 - the fitted proportion of the second (right) group. Same as 1-proportion_1
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None,show_plot=False,print_results=False):
        if failures is None or len(failures) < 4:  # it is possible to fit a mixture model with as few as 4 samples but it is inappropriate to do so. You should have at least 10, and preferably a lot more (>20) samples before using a mixture model.
            raise ValueError('The minimum number of failures to fit a mixture model is 4 (2 failures for each weibull). It is highly recommended that a mixture model is only used when sufficient data (>10 samples) is available.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        if min(all_data)<=0:
            raise ValueError('All failure and censoring times must be greater than zero.')

        #find the division line. This is to assign data to each group
        h = np.histogram(failures, bins=50, density=True)
        hist_counts = h[0]
        hist_bins = h[1]
        midbins = []
        for i in range(len(hist_bins)):
            if i>0 and i<len(hist_bins):
                midbins.append((hist_bins[i]+hist_bins[i-1])/2)
        peaks_x = []
        peaks_y = []
        batch_width = 8
        for i,x in enumerate(hist_counts):
            if i<batch_width:
                batch = hist_counts[0:i+batch_width]
            elif i>batch_width and i>len(hist_counts-batch_width):
                batch = hist_counts[i - batch_width:len(hist_counts)]
            else:
                batch = hist_counts[i - batch_width:i + batch_width] #the histogram counts are batched (actual batch size = 2 x batch_width)
            if max(batch)==x: #if the current point is higher than the rest of the batch then it is counted as a peak
                peaks_x.append(midbins[i])
                peaks_y.append(x)
        if len(peaks_x)>2: #if there are more than 2 peaks, the mean is moved based on the height of the peaks. Higher peaks will attract the mean towards them more than smaller peaks.
            yfracs = np.array(peaks_y)/sum(peaks_y)
            division_line = sum(peaks_x*yfracs)
        else:
            division_line = np.average(peaks_x)
        self.division_line = division_line
        #this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        for item in failures:
            if item<division_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        GROUP_1_left_cens = left_censored #group 1 cannot have right censored data otherwise there would be no group 2. The opposite is also true.
        GROUP_2_right_cens = right_censored

        # get inputs for the guess by fitting a weibull to each of the groups with their respective censored data
        group_1_estimates = Fit_Weibull_2P(failures=GROUP_1_failures,left_censored=GROUP_1_left_cens)
        group_2_estimates = Fit_Weibull_2P(failures=GROUP_2_failures,right_censored=GROUP_2_right_cens)
        p_guess = (len(GROUP_1_failures)+len(GROUP_1_left_cens)) / len(all_data) #proportion guess
        guess = [group_1_estimates.alpha,group_1_estimates.beta,group_2_estimates.alpha,group_2_estimates.beta,p_guess] # A1,B1,A2,B2,P

        # solve it
        bnds = [(0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None), (0.0001, 0.9999)]  # bounds of solution
        result = minimize(value_and_grad(Fit_Weibull_Mixture.LL), guess, args=(failures,right_censored,left_censored), jac=True, bounds=bnds, tol=1e-6)
        params = result.x
        self.alpha_1 = params[0]
        self.beta_1 = params[1]
        self.alpha_2 = params[2]
        self.beta_2 = params[3]
        self.proportion_1 = params[4]
        self.proportion_2 = 1-params[4]

        if print_results==True:
            print('Parameters:', '\nAlpha 1:', self.alpha_1, '\nBeta 1:', self.beta_1, '\nAlpha 2:', self.alpha_2, '\nBeta 2:', self.beta_2, '\nProportion 1:', self.proportion_1)
        if show_plot==True:
            xvals = np.linspace(0, max(failures) * 1.05, 1000)
            plt.figure(figsize=(14, 6))

            plt.subplot(121)

            # make the histogram. Can't use plt.hist due to need to scale the heights when there's censored data
            num_bins = min(int(len(failures) / 2), 30)
            hist, bins = np.histogram(failures, bins=num_bins, density=True)
            hist_cumulative = np.cumsum(hist) / sum(hist)
            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2
            frac_failing = len(failures)/len(all_data)
            frac_fail_L = len(failures) / len(np.hstack([failures, left_censored]))
            plt.bar(center, hist*frac_failing, align='center', width=width, alpha=0.2, color='k', edgecolor='k')

            yvals_p1_pdf = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1).PDF(xvals=xvals, show_plot=False)
            yvals_p2_pdf = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2).PDF(xvals=xvals, show_plot=False)
            plt.plot(xvals, yvals_p1_pdf * self.proportion_1)
            plt.plot(xvals, yvals_p2_pdf * self.proportion_2)
            plt.title('Weibull Mixture PDF')
            plt.xlabel('Failure Times')
            plt.ylabel('Probability Density')

            plt.subplot(122)
            # make the histogram. Can't use plt.hist due to need to scale the heights when there's censored data
            if len(left_censored)>0:
                plt.bar(center, hist_cumulative * frac_failing + (1 - frac_fail_L), align='center', width=width, alpha=0.2, color='k', edgecolor='k')
            else:
                plt.bar(center, hist_cumulative * frac_failing, align='center', width=width, alpha=0.2, color='k', edgecolor='k')

            yvals_p1_cdf = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1).CDF(xvals=xvals, show_plot=False)
            yvals_p2_cdf = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2).CDF(xvals=xvals, show_plot=False)
            y_mixture = yvals_p1_cdf * self.proportion_1 + yvals_p2_cdf * self.proportion_2
            plt.plot(xvals, y_mixture)
            plt.title('Weibull Mixture CDF')
            plt.xlabel('Failure Times')
            plt.ylabel('Cumulative Probability Density')
            plt.show()

    def logf(t,a1,b1,a2,b2,p): #Log Mixture PDF (2 parameter Weibull)
        return anp.log(p*((b1*t**(b1-1))/(a1**b1))*anp.exp(-((t/a1)**b1))+(1-p)*((b2*t**(b2-1))/(a2**b2))*anp.exp(-((t/a2)**b2)))

    def logF(t,a1,b1,a2,b2,p): #Log Mixture CDF (2 parameter Weibull)
        return anp.log(p*(1-anp.exp(-((t/a1)**b1)))+(1-p)*(1-anp.exp(-((t/a2)**b2))))

    def logR(t,a1,b1,a2,b2,p): #Log Mixture SF (2 parameter Weibull)
        return anp.log(p*anp.exp(-((t/a1)**b1))+(1-p)*anp.exp(-((t/a2)**b2)))

    def LL(params,T_f,T_rc,T_lc): #Log Mixture Likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Weibull_Mixture.logf(T_f, params[0], params[1], params[2], params[3], params[4]).sum() #failure times
        LL_rc += Fit_Weibull_Mixture.logR(T_rc, params[0], params[1], params[2], params[3], params[4]).sum() #right censored times
        LL_lc += Fit_Weibull_Mixture.logF(T_lc, params[0], params[1], params[2], params[3], params[4]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

class Fit_Expon_1P:
    '''
    Fit_Expon_1P

    Fits a 1-parameter Exponential distribution (Lambda) to the data provided.
    You may also enter left or right censored data (either but not both).

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    Lambda - the fitted Expon_1P lambda parameter
    '''

    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<1:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least one failure to calculate Exponential parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        bnds = [(0.0001, None)] #bounds of solution

        #solve it
        self.gamma = 0
        sp = ss.expon.fit(all_data,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [1/sp[1]]
        result = minimize(value_and_grad(Fit_Expon_1P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.Lambda = params[0]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Expon_1P. The fit from Scipy was used instead so results may not be accurate.')
            self.Lambda = 1/sp[1]

    def logf(t,L): #Log PDF (1 parameter Expon)
        return anp.log(L) - L*t

    def logF(t,L): #Log CDF (1 parameter Expon)
        return anp.log(1-anp.exp(-(L*t)))

    def logR(t,L): #Log SF (1 parameter Expon)
        return -(L*t)

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (1 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Expon_1P.logf(T_f, params[0]).sum() #failure times
        LL_rc += Fit_Expon_1P.logR(T_rc, params[0]).sum() #right censored times
        LL_lc += Fit_Expon_1P.logF(T_lc, params[0]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

class Fit_Expon_2P:
    '''
    Fit_Expon_2P

    Fits a 2-parameter Exponential distribution (Lambda,gamma) to the data provided.
    You may also enter right censored data.
    Left censored data is not supported because of the way the gamma parameter is obtained. If you have left censored data, use
    Fit_Expon_1P instead.

    inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    Lambda - the fitted Expon_2P lambda parameter
    gamma - the fitted Expon_2P gamma parameter
    '''
    def __init__(self,failures=None,right_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failure to calculate Exponential parameters.')
        if right_censored is None:
            right_censored=[]         # fill with empty lists if not specified
        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures,right_censored])
        bnds = [(0.0001, None)] #bounds of solution

        #solve it
        shift = min(all_data)-0.01 #the 0.01 is to avoid taking the log of zero in logF
        self.gamma = shift+0.01
        data_shifted = all_data-shift
        sp = ss.expon.fit(data_shifted,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [1/sp[1]]
        result = minimize(value_and_grad(Fit_Expon_2P.LL), guess, args=(failures - shift, right_censored - shift), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.Lambda = params[0]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Expon_2P. The fit from Scipy was used instead so results may not be accurate.')
            sp = ss.expon.fit(all_data, optimizer='powell')
            self.Lambda = 1/sp[1]
            self.gamma = sp[0]

    def logf(t,L): #Log PDF (1 parameter Expon)
        return anp.log(L) - L*t

    def logR(t,L): #Log SF (1 parameter Expon)
        return -(L*t)

    def LL(params,T_f,T_rc): #log likelihood function (1 parameter Expon)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Expon_1P.logf(T_f, params[0]).sum() #failure times
        LL_rc += Fit_Expon_1P.logR(T_rc, params[0]).sum() #right censored times
        return -(LL_f+LL_rc)

class Fit_Normal_2P:
    '''
    Fit_Normal_2P

    Fits a 2-parameter Normal distribution (mu,sigma) to the data provided.
    You may also enter left or right censored data (either but not both).

    Note that it will return a fit that may be partially in the negative domain (x<0).
    If you need an entirely positive distribution that is similar to Normal then consider using Weibull.

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    mu - the fitted Normal_2P mu parameter
    sigma - the fitted Normal_2P sigma parameter
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Normal parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        sp = ss.norm.fit(all_data,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[0],sp[1]]
        result = minimize(value_and_grad(Fit_Normal_2P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.mu = params[0]
            self.sigma = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Normal_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.mu = sp[0]
            self.sigma = sp[1]

    def logf(t,mu,sigma): #Log PDF (Normal)
        return anp.log(anp.exp(-0.5*(((t-mu)/sigma)**2)))-anp.log((sigma*(2*anp.pi)**0.5))

    def logF(t,mu,sigma): #Log CDF (Normal)
        return anp.log((1 + erf(((t-mu)/sigma) / 2**0.5)) / 2)

    def logR(t,mu,sigma): #Log SF (Normal)
        return anp.log((1 + erf(((mu - t) / sigma) / 2 ** 0.5)) / 2)

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (2 parameter weibull)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Normal_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Normal_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        LL_lc += Fit_Normal_2P.logF(T_lc, params[0], params[1]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

class Fit_Lognormal_2P:
    '''
    Fit_Lognormal_2P

    Fits a 2-parameter Lognormal distribution (mu,sigma) to the data provided.
    You may also enter left or right censored data (either but not both).

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    mu - the fitted Lognormal_2P mu parameter
    sigma - the fitted Lognormal_2P sigma parameter
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Lognormal parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')

        self.gamma = 0
        all_data = np.hstack([failures,right_censored,left_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        sp = ss.lognorm.fit(all_data,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [np.log(sp[2]),sp[0]]
        result = minimize(value_and_grad(Fit_Lognormal_2P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.mu = params[0]
            self.sigma = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Lognormal_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.mu = np.log(sp[2])
            self.sigma = sp[0]

    def logf(t,mu,sigma): #Log PDF (Lognormal)
        # return anp.log(anp.exp(-0.5*(((t-mu)/sigma)**2)))-anp.log((sigma*(2*anp.pi)**0.5))
        return anp.log(anp.exp(-0.5 * (((anp.log(t) - mu) / sigma) ** 2)) / (t * sigma * (2 * anp.pi) ** 0.5))

    def logF(t,mu,sigma): #Log CDF (Lognormal)
        # return anp.log((1 + erf(((t-mu)/sigma) / 2**0.5)) / 2)
        return anp.log(0.5 + 0.5 * erf((anp.log(t) - mu) / (sigma * 2 ** 0.5)))

    def logR(t,mu,sigma): #Log SF (Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t) - mu) / (sigma * 2 ** 0.5)))

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (2 parameter lognormal)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Lognormal_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Lognormal_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        LL_lc += Fit_Lognormal_2P.logF(T_lc, params[0], params[1]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

class Fit_Gamma_2P:
    '''
    Fit_Gamma_2P

    Fits a 2-parameter Gamma distribution (alpha,beta) to the data provided.
    You may also enter left or right censored data (either but not both).

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Gamma_2P alpha parameter
    beta - the fitted Gamma_2P beta parameter
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Gamma parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        self.gamma = 0
        sp = ss.gamma.fit(all_data,floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0]]
        result = minimize(value_and_grad(Fit_Gamma_2P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-10)

        if result.success == True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Gamma_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

    def logf(t,a,b): #Log PDF (2 parameter Gamma)
        return anp.log(t**(b-1)) -anp.log((a**b) * agamma(b)) - (t/a)

    def logF(t,a,b): #Log CDF (2 parameter Gamma)
        return anp.log(gammainc(b,t/a))

    def logR(t,a,b): #Log SF (2 parameter Gamma)
        return anp.log(gammaincc(b,t/a))

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (2 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Gamma_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Gamma_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        LL_lc += Fit_Gamma_2P.logF(T_lc, params[0], params[1]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

class Fit_Gamma_3P:
    '''
    Fit_Gamma_3P

    Fits a 3-parameter Gamma distribution (alpha,beta,gamma) to the data provided.
    You may also enter right censored data.
    Left censored data is not supported because of the way the gamma parameter is obtained. If you have left censored data, use
    Fit_Gamma_2P instead.

    inputs:
    failures - an array or list of failure data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Gamma_3P alpha parameter
    beta - the fitted Gamma_3P beta parameter
    gamma - the fitted Gamma_3P gamma parameter
    '''
    def __init__(self,failures=None,right_censored=None):
        if failures is None or len(failures)<3:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least three failures to calculate Gamma parameters.')
        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        all_data = np.hstack([failures,right_censored])
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        # solve it
        shift = min(all_data) - 0.01  # the 0.01 is to avoid taking the log of zero in logf
        self.gamma = shift+0.001 #this adds 0.001 instead of 0.01 to avoid the min(failures) equalling gamma which would cause AIC and BIC to be inf. The difference it causes is negligible
        data_shifted = all_data - shift
        sp = ss.gamma.fit(data_shifted, floc=0,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[2], sp[0]]
        result = minimize(value_and_grad(Fit_Gamma_3P.LL), guess, args=(failures-shift, right_censored-shift), jac=True, bounds=bnds, tol=1e-10)

        if result.success == True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Gamma_3P. The fit from Scipy was used instead so results may not be accurate.')
            sp = ss.gamma.fit(all_data, optimizer='powell')
            self.alpha = sp[2]
            self.beta = sp[0]
            self.gamma = sp[1]

    def logf(t,a,b): #Log PDF (2 parameter Gamma)
        return anp.log(t**(b-1)) -anp.log((a**b) * agamma(b)) - (t/a)

    def logR(t,a,b): #Log SF (2 parameter Gamma)
        return anp.log(gammaincc(b,t/a))

    def LL(params,T_f,T_rc): #log likelihood function (2 parameter Gamma)
        LL_f = 0
        LL_rc = 0
        LL_f += Fit_Gamma_3P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Gamma_3P.logR(T_rc, params[0], params[1]).sum() #right censored times
        return -(LL_f+LL_rc)

class Fit_Beta_2P:
    '''
    Fit_Beta_2P

    Fits a 2-parameter Beta distribution (alpha,beta) to the data provided.
    You may also enter left or right censored data (either but not both).
    All data must be in the range 0-1.

    inputs:
    failures - an array or list of failure data
    left_censored - an array or list of left censored data
    right_censored - an array or list of right censored data

    outputs:
    success - Whether the solution was found by autograd (True/False)
        if success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if
        there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and
        if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using
        another distribution.
    alpha - the fitted Beta_2P alpha parameter
    beta - the fitted Beta_2P beta parameter
    '''
    def __init__(self,failures=None,right_censored=None,left_censored=None):
        if failures is None or len(failures)<2:
            raise ValueError('Maximum likelihood estimates could not be calculated for these data. There must be at least two failures to calculate Beta parameters.')
        if right_censored is not None and left_censored is not None: #check that a mix of left and right censoring is not entered
            raise ValueError('You have specified both left and right censoring. You can specify either but not both.')

        # fill with empty lists if not specified
        if right_censored is None:
            right_censored=[]
        if left_censored is None:
            left_censored=[]

        # adjust inputs to be arrays
        if type(failures)==list:
            failures = np.array(failures)
        if type(failures)!= np.ndarray:
            raise TypeError('failures must be a list or array of failure data')
        if type(right_censored) == list:
            right_censored = np.array(right_censored)
        if type(right_censored) != np.ndarray:
            raise TypeError('right_censored must be a list or array of right censored failure data')
        if type(left_censored) == list:
            left_censored = np.array(left_censored)
        if type(left_censored) != np.ndarray:
            raise TypeError('left_censored must be a list or array of left censored failure data')
        all_data = np.hstack([failures,right_censored,left_censored])
        if min(all_data)<0 or max(all_data)>1:
            raise ValueError('All data must be between 0 and 1 to use the beta distribution.')
        bnds = [(0.0001, None), (0.0001, None)] #bounds of solution

        #solve it
        self.gamma = 0
        sp = ss.beta.fit(all_data,floc=0,fscale=1,optimizer='powell')  # scipy's answer is used as an initial guess. Scipy is only correct when there is no censored data
        guess = [sp[0], sp[1]]
        result = minimize(value_and_grad(Fit_Beta_2P.LL), guess, args=(failures, right_censored, left_censored), jac=True, bounds=bnds, tol=1e-6)

        if result.success == True:
            params = result.x
            self.success = True
            self.alpha = params[0]
            self.beta = params[1]
        else:
            self.success = False
            warnings.warn('Fitting using Autograd FAILED for Beta_2P. The fit from Scipy was used instead so results may not be accurate.')
            self.alpha = sp[0]
            self.beta = sp[1]

    def logf(t,a,b): #Log PDF (2 parameter Beta)
        return anp.log(((t ** (a - 1)) * ((1 - t) ** (b - 1)))) - anp.log(abeta(a, b))

    def logF(t,a,b): #Log CDF (2 parameter Beta)
        return anp.log(betainc(a, b, t))

    def logR(t,a,b): #Log SF (2 parameter Beta)
        return anp.log(1-betainc(a, b, t))

    def LL(params,T_f,T_rc,T_lc): #log likelihood function (2 parameter beta)
        LL_f = 0
        LL_rc = 0
        LL_lc = 0
        LL_f += Fit_Beta_2P.logf(T_f, params[0], params[1]).sum() #failure times
        LL_rc += Fit_Beta_2P.logR(T_rc, params[0], params[1]).sum() #right censored times
        LL_lc += Fit_Beta_2P.logF(T_lc, params[0], params[1]).sum() #left censored times
        return -(LL_f+LL_rc+LL_lc)

