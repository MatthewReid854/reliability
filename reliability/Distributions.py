'''
Probability Distributions Module

Available distributions are:
    Weibull_Distribution
    Normal_Distribution
    Lognormal_Distribution
    Exponential_Distribution
    Gamma_Distribution
    Beta_Distribution

Methods:
    name - the name of the distribution. eg. 'Weibull'
    name2 - the name of the distribution with the number of parameters. eg. 'Weibull_2P'
    param_title_long - Useful in plot titles, legends and in printing strings. Varies by distribution. eg. 'Weibull Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. Varies by distribution. eg. 'α=5,β=2'
    parameters - returns an array of parameters. eg. [alpha,beta,gamma]
    alpha, beta, gamma, Lambda, mu, sigma - these vary by distribution but will return the value of their respective parameter.
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.

Example usage:
dist = Weibull_Distribution(alpha = 8, beta = 1.2)
print(dist.mean)
    >> 7.525246866054174
print(dist.quantile(0.05))
    >> 0.6731943793488804
print(dist.mean_residual_life(15))
    >> 5.556500198354015
dist.plot()
    >> A figure of 5 plots and descriptive statistics will be displayed
dist.CHF()
    >> Cumulative Hazard Function plot will be displayed
values = dist.random_samples(number_of_samples=10000)
    >> random values will be generated from the distribution
'''

import scipy.stats as ss
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

_sigfig = 4  # number of significant figures to use when rounding descriptive statistics
np.seterr(divide='ignore', invalid='ignore')  # ignore the divide by zero warnings


class Weibull_Distribution:
    '''
    Weibull probability distribution

    Creates a Distribution object.

    inputs:
    alpha - scale parameter
    beta - shape parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name - 'Weibull'
    name2 = 'Weibull_2P' or 'Weibull_3P' depending on the value of the gamma parameter
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Weibull Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta,gamma]
    alpha
    beta
    gamma
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, alpha=None, beta=None, gamma=0):
        self.name = 'Weibull'
        self.alpha = alpha
        self.beta = beta
        if self.alpha is None or self.beta is None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Weibull_Distribution(alpha=5,beta=2)')
        self.gamma = gamma
        self.parameters = np.array([self.alpha, self.beta, self.gamma])
        mean, var, skew, kurt = ss.weibull_min.stats(self.beta, scale=self.alpha, loc=self.gamma, moments='mvsk')
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.weibull_min.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = self.alpha * ((self.beta - 1) / self.beta) ** (1 / self.beta) + self.gamma
        else:
            self.mode = r'No mode exists when $\beta$ < 1'
        if self.gamma != 0:
            self.param_title = str('α=' + str(self.alpha) + ',β=' + str(self.beta) + ',γ=' + str(self.gamma))
            self.param_title_long = str('Weibull Distribution (α=' + str(self.alpha) + ',β=' + str(self.beta) + ',γ=' + str(gamma) + ')')
            self.name2 = 'Weibull_3P'
        else:
            self.param_title = str('α=' + str(self.alpha) + ',β=' + str(self.beta))
            self.param_title_long = str('Weibull Distribution (α=' + str(self.alpha) + ',β=' + str(self.beta) + ')')
            self.name2 = 'Weibull_2P'
        self.b5 = ss.weibull_min.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.weibull_min.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed

        pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Weibull Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Weibull Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Weibull Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Weibull Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        hf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma) / ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Weibull Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Weibull Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.weibull_min.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.weibull_min.isf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.weibull_min.sf(x, self.beta, scale=self.alpha, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        if self.gamma == 0:
            print('Descriptive statistics for Weibull distribution with alpha =', self.alpha, 'and beta =', self.beta)
        else:
            print('Descriptive statistics for Weibull distribution with alpha =', self.alpha, ', beta =', self.beta, ', and gamma =', self.gamma)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.weibull_min.rvs(self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples)
        return RVS


class Normal_Distribution:
    '''
    Normal probability distribution

    Creates a Distribution object.

    inputs:
    mu - location parameter (mean)
    sigma - scale parameter (standard deviation)

    methods:
    name - 'Normal'
    name2 = 'Normal_2P'
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Normal Distribution (μ=5,σ=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'μ=5,σ=2'
    parameters - [mu,sigma]
    mu
    sigma
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, mu=None, sigma=None):
        self.name = 'Normal'
        self.name2 = 'Normal_2P'
        self.mu = mu
        self.sigma = sigma
        if self.mu is None or self.sigma is None:
            raise ValueError('Parameters mu and sigma must be specified. Eg. Normal_Distribution(mu=5,sigma=2)')
        self.parameters = np.array([self.mu, self.sigma])
        self.mean = mu
        self.variance = sigma ** 2
        self.standard_deviation = sigma
        self.skewness = 0
        self.kurtosis = 3
        self.excess_kurtosis = 0
        self.median = mu
        self.mode = mu
        self.param_title = str('μ=' + str(self.mu) + ',σ=' + str(self.sigma))
        self.param_title_long = str('Normal Distribution (μ=' + str(self.mu) + ',σ=' + str(self.sigma) + ')')
        self.b5 = ss.norm.ppf(0.05, loc=self.mu, scale=self.sigma)
        self.b95 = ss.norm.ppf(0.95, loc=self.mu, scale=self.sigma)

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed

        pdf = ss.norm.pdf(X, self.mu, self.sigma)
        cdf = ss.norm.cdf(X, self.mu, self.sigma)
        sf = ss.norm.sf(X, self.mu, self.sigma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Normal Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed
        pdf = ss.norm.pdf(X, self.mu, self.sigma)
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Normal Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed
        cdf = ss.norm.cdf(X, self.mu, self.sigma)
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Normal Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed
        sf = ss.norm.sf(X, self.mu, self.sigma)
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Normal Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed
        hf = ss.norm.pdf(X, self.mu, self.sigma) / ss.norm.sf(X, self.mu, self.sigma)
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Normal Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.norm.sf(X, self.mu, self.sigma))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Normal Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.norm.ppf(q, loc=self.mu, scale=self.sigma)

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.norm.isf(q, loc=self.mu, scale=self.sigma)

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.norm.sf(x, loc=self.mu, scale=self.sigma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        print('Descriptive statistics for Normal distribution with mu =', self.mu, 'and sigma =', self.sigma)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.norm.rvs(loc=self.mu, scale=self.sigma, size=number_of_samples)
        return RVS


class Lognormal_Distribution:
    '''
    Lognormal probability distribution

    Creates a Distribution object.

    inputs:
    mu - location parameter
    sigma - scale parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name - 'Lognormal'
    name2 = 'Lognormal_2P' or 'Lognormal_3P' depending on the value of the gamma parameter
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Lognormal Distribution (μ=5,σ=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'μ=5,σ=2'
    parameters - [mu,sigma,gamma]
    mu
    sigma
    gamma
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, mu=None, sigma=None, gamma=0):
        self.name = 'Lognormal'
        self.mu = mu
        self.sigma = sigma
        if self.mu is None or self.sigma is None:
            raise ValueError('Parameters mu and sigma must be specified. Eg. Lognormal_Distribution(mu=5,sigma=2)')
        self.gamma = gamma
        self.parameters = np.array([self.mu, self.sigma, self.gamma])
        mean, var, skew, kurt = ss.lognorm.stats(self.sigma, self.gamma, np.exp(self.mu), moments='mvsk')
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.lognorm.median(self.sigma, self.gamma, np.exp(self.mu))
        self.mode = np.exp(self.mu - self.sigma ** 2) + self.gamma
        if self.gamma != 0:
            self.param_title = str('μ=' + str(self.mu) + ',σ=' + str(self.sigma) + ',γ=' + str(self.gamma))
            self.param_title_long = str('Lognormal Distribution (μ=' + str(self.mu) + ',σ=' + str(self.sigma) + ',γ=' + str(gamma) + ')')
            self.name2 = 'Lognormal_3P'
        else:
            self.param_title = str('μ=' + str(self.mu) + ',σ=' + str(self.sigma))
            self.param_title_long = str('Lognormal Distribution (μ=' + str(self.mu) + ',σ=' + str(self.sigma) + ')')
            self.name2 = 'Lognormal_2P'
        self.b5 = ss.lognorm.ppf(0.05, self.sigma, self.gamma, np.exp(self.mu))  # note that scipy uses mu in a log way compared to most other software, so we must take the exp of the input
        self.b95 = ss.lognorm.ppf(0.95, self.sigma, self.gamma, np.exp(self.mu))

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed

        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))
        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))
        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Lognormal Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Lognormal Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Lognormal Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Lognormal Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        hf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu)) / ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Lognormal Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu)))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Lognormal Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.lognorm.ppf(q, self.sigma, self.gamma, np.exp(self.mu))

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.lognorm.isf(q, self.sigma, self.gamma, np.exp(self.mu))

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.lognorm.sf(x, self.sigma, self.gamma, np.exp(self.mu))
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        if self.gamma == 0:
            print('Descriptive statistics for Lognormal distribution with mu =', self.mu, 'and sigma =', self.sigma)
        else:
            print('Descriptive statistics for Lognormal distribution with mu =', self.mu, ', sigma =', self.sigma, ', and gamma =', self.gamma)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.lognorm.rvs(self.sigma, self.gamma, np.exp(self.mu), size=number_of_samples)
        return RVS


class Exponential_Distribution:
    '''
    Exponential probability distribution

    Creates a Distribution object.

    inputs:
    Lambda - scale (rate) parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name - 'Exponential'
    name2 = 'Exponential_1P' or 'Exponential_2P' depending on the value of the gamma parameter
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Exponential Distribution (λ=5)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'λ=5'
    parameters - [Lambda,gamma]
    Lambda
    gamma
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, Lambda=None, gamma=0):
        self.name = 'Exponential'
        self.Lambda = Lambda
        if self.Lambda is None:
            raise ValueError('Parameter Lambda must be specified. Eg. Exponential_Distribution(Lambda=3)')
        self.gamma = gamma
        self.parameters = np.array([self.Lambda, self.gamma])
        mean, var, skew, kurt = ss.expon.stats(scale=1 / self.Lambda, loc=self.gamma, moments='mvsk')
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.expon.median(scale=1 / self.Lambda, loc=self.gamma)
        self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str('λ=' + str(self.Lambda) + ',γ=' + str(self.gamma))
            self.param_title_long = str('Exponential Distribution (λ=' + str(self.Lambda) + ',γ=' + str(gamma) + ')')
            self.name2 = 'Exponential_2P'
        else:
            self.param_title = str('λ=' + str(self.Lambda))
            self.param_title_long = str('Exponential Distribution (λ=' + str(self.Lambda) + ')')
            self.name2 = 'Exponential_1P'
        self.b5 = ss.expon.ppf(0.05, scale=1 / self.Lambda, loc=self.gamma)
        self.b95 = ss.expon.ppf(0.95, scale=1 / self.Lambda, loc=self.gamma)

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed

        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)
        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)
        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Exponential Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Exponential Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Exponential Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Exponential Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        hf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma) / ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Exponential Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Exponential Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.expon.ppf(q, scale=1 / self.Lambda, loc=self.gamma)

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.expon.isf(q, scale=1 / self.Lambda, loc=self.gamma)

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.expon.sf(x, scale=1 / self.Lambda, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        if self.gamma == 0:
            print('Descriptive statistics for Exponential distribution with lambda =', self.Lambda)
        else:
            print('Descriptive statistics for Exponential distribution with lambda =', self.Lambda, ', and gamma =', self.gamma)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.expon.rvs(scale=1 / self.Lambda, loc=self.gamma, size=number_of_samples)
        return RVS


class Gamma_Distribution:
    '''
    Gamma probability distribution

    Creates a Distribution object.

    inputs:
    alpha - scale parameter
    beta - shape parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name - 'Gamma'
    name2 = 'Gamma_2P' or 'Gamma_3P' depending on the value of the gamma parameter
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Gamma Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta,gamma]
    alpha
    beta
    gamma
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, alpha=None, beta=None, gamma=0):
        self.name = 'Gamma'
        self.alpha = alpha
        self.beta = beta
        if self.alpha is None or self.beta is None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Gamma_Distribution(alpha=5,beta=2)')
        self.gamma = gamma
        self.parameters = np.array([self.alpha, self.beta, self.gamma])
        mean, var, skew, kurt = ss.gamma.stats(self.beta, scale=self.alpha, loc=self.gamma, moments='mvsk')
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.gamma.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = (self.beta - 1) * self.alpha + self.gamma
        else:
            self.mode = r'No mode exists when $\beta$ < 1'
        if self.gamma != 0:
            self.param_title = str('α=' + str(self.alpha) + ',β=' + str(self.beta) + ',γ=' + str(self.gamma))
            self.param_title_long = str('Gamma Distribution (α=' + str(self.alpha) + ',β=' + str(self.beta) + ',γ=' + str(gamma) + ')')
            self.name2 = 'Gamma_3P'
        else:
            self.param_title = str('α=' + str(self.alpha) + ',β=' + str(self.beta))
            self.param_title_long = str('Gamma Distribution (α=' + str(self.alpha) + ',β=' + str(self.beta) + ')')
            self.name2 = 'Gamma_2P'
        self.b5 = ss.gamma.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.gamma.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed

        pdf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.gamma.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Gamma Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        pdf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Gamma Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        cdf = ss.gamma.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Gamma Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        sf = ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Gamma Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        hf = ss.gamma.pdf(X, self.beta, scale=self.alpha, loc=self.gamma) / ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Gamma Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.gamma.sf(X, self.beta, scale=self.alpha, loc=self.gamma))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Gamma Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.gamma.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.gamma.isf(q, self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.gamma.sf(x, self.beta, scale=self.alpha, loc=self.gamma)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        if self.gamma == 0:
            print('Descriptive statistics for Gamma distribution with alpha =', self.alpha, 'and beta =', self.beta)
        else:
            print('Descriptive statistics for Gamma distribution with alpha =', self.alpha, ', beta =', self.beta, ', and gamma =', self.gamma)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.gamma.rvs(self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples)
        return RVS


class Beta_Distribution:
    '''
    Beta probability distribution

    Creates a Distribution object in the range 0-1.

    inputs:
    alpha - shape parameter 1
    beta - shape parameter 2

    methods:
    name - 'Beta'
    name2 = 'Beta_2P'
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Beta Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta]
    alpha
    beta
    mean
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    b5
    b95
    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plots the survival function (also known as reliability function)
    HF() - plots the hazard function
    CHF() - plots the cumulative hazard function
    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing.
                 Also known as b life where b5 is the time at which 5% have failed.
    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time.
                           Effectively the mean of the remaining amount (right side) of a distribution at a given time.
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    '''

    def __init__(self, alpha=None, beta=None):
        self.name = 'Beta'
        self.name2 = 'Beta_2P'
        self.alpha = alpha
        self.beta = beta
        if self.alpha is None or self.beta is None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Beta_Distribution(alpha=5,beta=2)')
        self.parameters = np.array([self.alpha, self.beta])
        mean, var, skew, kurt = ss.beta.stats(self.alpha, self.beta, 0, 1, moments='mvsk')
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var ** 0.5
        self.skewness = float(skew)
        self.kurtosis = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.beta.median(self.alpha, self.beta, 0, 1)
        if self.alpha > 1 and self.beta > 1:
            self.mode = (self.alpha - 1) / (self.beta + self.alpha - 2)
        else:
            self.mode = r'No mode exists unless $\alpha$ > 1 and $\beta$ > 1'
        self.param_title = str('α=' + str(self.alpha) + ',β=' + str(self.beta))
        self.param_title_long = str('Beta Distribution (α=' + str(self.alpha) + ',β=' + str(self.beta) + ')')
        self.b5 = ss.beta.ppf(0.05, self.alpha, self.beta, 0, 1)
        self.b95 = ss.beta.ppf(0.95, self.alpha, self.beta, 0, 1)

    def plot(self, xvals=None, xmin=None, xmax=None):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

        Inputs:
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *no plotting keywords are accepted

        Outputs:
        The plot will be shown. No need to use plt.show()
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed

        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)
        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)
        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str('Beta Distribution' + '\n' + self.param_title)
        plt.suptitle(text_title, fontsize=15)
        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(X, sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(X, hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(X, chf)
        plt.title('Cumulative Hazard\nFunction')

        # descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode))  # required when mode is str
        text_b5 = str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95 = str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.variance ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.variance), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skewness), _sigfig)))
        text_ex_kurt = str('Excess kurtosis = ' + str(round(float(self.excess_kurtosis), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed
        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)
        if show_plot == False:
            return pdf
        else:
            plt.plot(X, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Beta Distribution\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return pdf

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed
        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)
        if show_plot == False:
            return cdf
        else:
            plt.plot(X, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction failing')
            text_title = str('Beta Distribution\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the SF (survival function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed
        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        if show_plot == False:
            return sf
        else:
            plt.plot(X, sf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction surviving')
            text_title = str('Beta Distribution\n' + ' Survival Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the HF (hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed
        hf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1) / ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        if show_plot == False:
            return hf
        else:
            plt.plot(X, hf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str('Beta Distribution\n' + ' Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return hf

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        '''
        Plots the CHF (cumulative hazard function)

        Inputs:
        show_plot - True/False. Default is True
        xvals - x-values for plotting
        xmin - minimum x-value for plotting
        xmax - maximum x-value for plotting
        *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements
        will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters.
        *plotting keywords are also accepted (eg. color, linestyle)

        Outputs:
        yvals - this is the y-values of the plot
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is not None:
            X = xvals
        elif xmin is not None and xmax is not None:
            X = np.linspace(xmin, xmax, 1000)
        else:
            X = np.linspace(0, 1, 1000)  # if no limits are specified, they are assumed
        chf = -np.log(ss.beta.sf(X, self.alpha, self.beta, 0, 1))
        if show_plot == False:
            return chf
        else:
            plt.plot(X, chf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative hazard')
            text_title = str('Beta Distribution\n' + ' Cumulative Hazard Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return chf

    def quantile(self, q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.beta.ppf(q, self.alpha, self.beta, 0, 1)

    def inverse_SF(self, q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if type(q) == int or type(q) == float:
            if q < 0 or q > 1:
                raise ValueError('Quantile must be between 0 and 1')
        elif type(q) == np.ndarray or type(q) == list:
            if min(q) < 0 or max(q) > 1:
                raise ValueError('Quantile must be between 0 and 1')
        else:
            raise ValueError('Quantile must be of type int, float, list, array')
        return ss.beta.isf(q, self.alpha, self.beta, 0, 1)

    def mean_residual_life(self, t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.beta.sf(x, self.alpha, self.beta, 0, 1)
        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        print('Descriptive statistics for Beta distribution with alpha =', self.alpha, 'and beta =', self.beta)
        print('Mean = ', self.mean)
        print('Median =', self.median)
        print('Mode =', self.mode)
        print('5th quantile =', self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =', self.standard_deviation)
        print('Variance =', self.variance)
        print('Skewness =', self.skewness)
        print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples):
        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.beta.rvs(self.alpha, self.beta, 0, 1, size=number_of_samples)
        return RVS
