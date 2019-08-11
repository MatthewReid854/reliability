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
    parameter names - varies by distribution
    parameters - returns an array of parameters
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

from reliability import Plotting
import scipy.stats as ss
import numpy as np
from scipy import integrate

class Weibull_Distribution:
    '''
    Weibull probability distribution

    Creates a Distribution object.

    inputs:
    alpha - scale parameter
    beta - shape parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name
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
    '''
    def __init__(self,alpha=None,beta=None,gamma=0):
        self.name = 'Weibull'
        self.alpha = alpha
        self.beta = beta
        if self.alpha==None or self.beta==None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Weibull_Distribution(alpha=5,beta=2)')
        self.gamma = gamma
        self.parameters = np.array([self.alpha,self.beta,self.gamma])
        mean,var,skew,kurt = ss.weibull_min.stats(self.beta, scale=self.alpha, loc=self.gamma,moments='mvsk')
        self.mean = mean
        self.variance = var
        self.standard_deviation = var**0.5
        self.skewness = skew
        self.kurtosis = kurt+3
        self.excess_kurtosis = kurt
        self.median = ss.weibull_min.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta>=1:
            self.mode = self.alpha*((self.beta-1)/self.beta)**(1/self.beta)+self.gamma
        else:
            self.mode = 'No mode exists when beta < 1'
        self.b5 = ss.weibull_min.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.weibull_min.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)
    def plot(self=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('weibull', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.weibull_min.ppf(q,self.beta,scale=self.alpha,loc=self.gamma)
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.weibull_min.isf(q,self.beta, scale=self.alpha, loc=self.gamma)

    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.weibull_min.sf(x,self.beta,scale=self.alpha,loc=self.gamma)
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        if self.gamma==0:
            print('Descriptive statistics for Weibull distribution with alpha =',self.alpha,'and beta =',self.beta)
        else:
            print('Descriptive statistics for Weibull distribution with alpha =',self.alpha,', beta =',self.beta,', and gamma =',self.gamma)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)

    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.weibull_min.rvs(self.beta,scale=self.alpha,loc=self.gamma,size=number_of_samples)
        return RVS

class Normal_Distribution:
    '''
    Normal probability distribution

    Creates a Distribution object.

    inputs:
    mu - location parameter (mean)
    sigma - scale parameter (standard deviation)

    methods:
    name
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
    '''
    def __init__(self,mu=None,sigma=None):
        self.name = 'Normal'
        self.mu = mu
        self.sigma = sigma
        if self.mu==None or self.sigma==None:
            raise ValueError('Parameters mu and sigma must be specified. Eg. Normal_Distribution(mu=5,sigma=2)')
        self.parameters = np.array([self.mu,self.sigma])
        self.mean = mu
        self.variance = sigma**2
        self.standard_deviation = sigma
        self.skewness = 0
        self.kurtosis = 3
        self.excess_kurtosis = 0
        self.median = mu
        self.mode = mu
        self.b5 = ss.norm.ppf(0.05, loc=self.mu, scale=self.sigma)
        self.b95 = ss.norm.ppf(0.95, loc=self.mu, scale=self.sigma)
    def plot(self,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('normal', sigma=self.sigma, mu=self.mu, xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.norm.ppf(q,loc=self.mu, scale=self.sigma)
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.norm.isf(q,loc=self.mu, scale=self.sigma)
    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.norm.sf(x,loc=self.mu, scale=self.sigma)
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        print('Descriptive statistics for Normal distribution with mu =',self.mu,'and sigma =',self.sigma)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)
    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.norm.rvs(loc=self.mu, scale=self.sigma,size=number_of_samples)
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
    name
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
    '''
    def __init__(self,mu=None,sigma=None,gamma=0):
        self.name = 'Lognormal'
        self.mu = mu
        self.sigma = sigma
        if self.mu==None or self.sigma==None:
            raise ValueError('Parameters mu and sigma must be specified. Eg. Lognormal_Distribution(mu=5,sigma=2)')
        self.gamma = gamma
        self.parameters = np.array([self.mu,self.sigma,self.gamma])
        mean, var, skew, kurt = ss.lognorm.stats(self.sigma, self.gamma, np.exp(self.mu), moments='mvsk')
        self.mean = mean
        self.variance = var
        self.standard_deviation = var**0.5
        self.skewness = skew
        self.kurtosis = kurt+3
        self.excess_kurtosis = kurt
        self.median = ss.lognorm.median(self.sigma, self.gamma, np.exp(self.mu))
        self.mode = np.exp(self.mu - self.sigma ** 2) + self.gamma
        self.b5 = ss.lognorm.ppf(0.05, self.sigma, self.gamma, np.exp(self.mu))  # note that scipy uses mu in a log way compared to most other software, so we must take the exp of the input
        self.b95 = ss.lognorm.ppf(0.95, self.sigma, self.gamma, np.exp(self.mu))
    def plot(self=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('lognormal', mu=self.mu, sigma=self.sigma, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.lognorm.ppf(q, self.sigma, self.gamma, np.exp(self.mu))
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.lognorm.isf(q, self.sigma, self.gamma, np.exp(self.mu))
    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.lognorm.sf(x,self.sigma, self.gamma, np.exp(self.mu))
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        if self.gamma==0:
            print('Descriptive statistics for Lognormal distribution with mu =',self.mu,'and sigma =',self.sigma)
        else:
            print('Descriptive statistics for Lognormal distribution with mu =',self.mu,', sigma =',self.sigma,', and gamma =',self.gamma)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)
    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.lognorm.rvs(self.sigma, self.gamma, np.exp(self.mu),size=number_of_samples)
        return RVS

class Exponential_Distribution:
    '''
    Exponential probability distribution

    Creates a Distribution object.

    inputs:
    Lambda - scale (rate) parameter
    gamma - threshold (offset) parameter. Default = 0

    methods:
    name
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
    '''
    def __init__(self,Lambda=None,gamma=0):
        self.name = 'Exponential'
        self.Lambda = Lambda
        if self.Lambda==None:
            raise ValueError('Parameter Lambda must be specified. Eg. Exponential_Distribution(Lambda=3)')
        self.gamma = gamma
        self.parameters = np.array([self.Lambda,self.gamma])
        mean,var,skew,kurt = ss.expon.stats(scale=1/self.Lambda, loc=self.gamma,moments='mvsk')
        self.mean = mean
        self.variance = var
        self.standard_deviation = var**0.5
        self.skewness = skew
        self.kurtosis = kurt+3
        self.excess_kurtosis = kurt
        self.median = ss.expon.median(scale=1/self.Lambda, loc=self.gamma)
        self.mode = 0
        self.b5 = ss.expon.ppf(0.05, scale=1/self.Lambda, loc=self.gamma)
        self.b95 = ss.expon.ppf(0.95, scale=1/self.Lambda, loc=self.gamma)
    def plot(self=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('expon', Lambda=self.Lambda, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.expon.ppf(q,scale=1/self.Lambda, loc=self.gamma)
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.expon.isf(q,scale=1/self.Lambda, loc=self.gamma)
    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.expon.sf(x,scale=1/self.Lambda, loc=self.gamma)
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        if self.gamma==0:
            print('Descriptive statistics for Exponential distribution with lambda =',self.Lambda)
        else:
            print('Descriptive statistics for Exponential distribution with lambda =',self.Lambda,', and gamma =',self.gamma)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)
    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.expon.rvs(scale=1/self.Lambda, loc=self.gamma,size=number_of_samples)
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
    name
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
    '''
    def __init__(self,alpha=None,beta=None,gamma=0):
        self.name = 'Gamma'
        self.alpha = alpha
        self.beta = beta
        if self.alpha==None or self.beta==None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Gamma_Distribution(alpha=5,beta=2)')
        self.gamma = gamma
        self.parameters = np.array([self.alpha,self.beta,self.gamma])
        mean,var,skew,kurt = ss.gamma.stats(self.beta, scale=self.alpha, loc=self.gamma,moments='mvsk')
        self.mean = mean
        self.variance = var
        self.standard_deviation = var**0.5
        self.skewness = skew
        self.kurtosis = kurt+3
        self.excess_kurtosis = kurt
        self.median = ss.gamma.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta>=1:
            self.mode = (self.beta-1)*self.alpha
        else:
            self.mode = 'No mode exists when beta < 1'
        self.b5 = ss.gamma.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.gamma.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)
    def plot(self=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('gamma', alpha=self.alpha, beta=self.beta, gamma=self.gamma,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.gamma.ppf(q,self.beta,scale=self.alpha,loc=self.gamma)
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.gamma.isf(q,self.beta,scale=self.alpha,loc=self.gamma)
    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.gamma.sf(x,self.beta,scale=self.alpha,loc=self.gamma)
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        if self.gamma==0:
            print('Descriptive statistics for Gamma distribution with alpha =',self.alpha,'and beta =',self.beta)
        else:
            print('Descriptive statistics for Gamma distribution with alpha =',self.alpha,', beta =',self.beta,', and gamma =',self.gamma)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)
    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.gamma.rvs(self.beta,scale=self.alpha,loc=self.gamma,size=number_of_samples)
        return RVS

class Beta_Distribution:
    '''
    Beta probability distribution

    Creates a Distribution object in the range 0-1.

    inputs:
    alpha - shape parameter 1
    beta - shape parameter 2

    methods:
    name
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
    '''
    def __init__(self,alpha=None,beta=None):
        self.name = 'Beta'
        self.alpha = alpha
        self.beta = beta
        if self.alpha==None or self.beta==None:
            raise ValueError('Parameters alpha and beta must be specified. Eg. Beta_Distribution(alpha=5,beta=2)')
        self.parameters = np.array([self.alpha,self.beta])
        mean,var,skew,kurt = ss.beta.stats(self.alpha, self.beta, 0, 1, moments='mvsk')
        self.mean = mean
        self.variance = var
        self.standard_deviation = var**0.5
        self.skewness = skew
        self.kurtosis = kurt+3
        self.excess_kurtosis = kurt
        self.median = ss.beta.median(self.alpha, self.beta, 0, 1)
        if self.alpha >1 and self.beta>1:
            self.mode = (self.alpha-1) / (self.beta + self.alpha - 2)
        else:
            self.mode = 'No mode exists unless alpha > 1 and beta > 1'
        self.b5 = ss.beta.ppf(0.05, self.alpha, self.beta, 0, 1)
        self.b95 = ss.beta.ppf(0.95, self.alpha, self.beta, 0, 1)
    def plot(self=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        '''
        Plots the distribution
        Invokes the Plotting.plot.all_functions() which will provide all the plots of the distribution
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).all_functions()
        return yvals
    def PDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the PDF
        Invokes Plotting.plot.PDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).PDF(**kwargs)
        return yvals
    def CDF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CDF
        Invokes Plotting.plot.CDF() which will plot only the PDF of the function
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CDF(**kwargs)
        return yvals
    def SF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the SF
        Invokes Plotting.plot.SF() which will plot only the Survival Function
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).SF(**kwargs)
        return yvals
    def HF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the HF
        Invokes Plotting.plot.HF() which will plot only the Hazard Function
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).HF(**kwargs)
        return yvals
    def CHF(self,xvals=None,xmin=None,xmax=None,show_plot=True,**kwargs):
        '''
        Plots the CHF
        Invokes Plotting.plot.CHF which will plot only the Cumulative Hazard Function
        '''
        yvals = Plotting.plot('beta', alpha=self.alpha, beta=self.beta,xvals=xvals,xmin=xmin,xmax=xmax,show_plot=show_plot).CHF(**kwargs)
        return yvals
    def quantile(self,q):
        '''
        Quantile calculator

        :param q: quantile to be calculated
        :return: the probability (area under the curve) that a random variable from the distribution is < q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.beta.ppf(q, self.alpha, self.beta, 0, 1)
    def inverse_SF(self,q):
        '''
        Inverse Survival function calculator

        :param q: quantile to be calculated
        :return: the inverse of the survival function at q
        '''
        if min(q)<0 or max(q)> 1:
            raise ValueError('Quantile must be between 0 and 1')
        return ss.beta.isf(q, self.alpha, self.beta, 0, 1)
    def mean_residual_life(self,t):
        '''
        Mean Residual Life calculator

        :param t: time at which MRL is to be evaluated
        :return: MRL
        '''
        R = lambda x: ss.beta.sf(x,self.alpha, self.beta, 0, 1)
        integral_R,error = integrate.quad(R,t,np.inf)
        MRL = integral_R/R(t)
        return MRL
    def stats(self):
        print('Descriptive statistics for Beta distribution with alpha =',self.alpha,'and beta =',self.beta)
        print('Mean = ',self.mean)
        print('Median =',self.median)
        print('Mode =',self.mode)
        print('5th quantile =',self.b5)
        print('95th quantile =', self.b95)
        print('Standard deviation =',self.standard_deviation)
        print('Variance =',self.variance)
        print('Skewness =',self.skewness)
        print('Excess kurtosis =',self.excess_kurtosis)
    def random_samples(self,number_of_samples):
        if type(number_of_samples)!=int or number_of_samples<1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        RVS = ss.beta.rvs(self.alpha, self.beta, 0, 1,size=number_of_samples)
        return RVS
