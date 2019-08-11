import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution
_sigfig=4 #number of significant figures to use when rounding descriptive statistics
np.seterr(divide='ignore',invalid='ignore') #ignore the divide by zero warnings
class plot:
    '''
    Plotting Module

    This module is designed to provide a quick way to plot several functions from a specified reliability distribution.
    The functions plotted are:
        PDF - probability density function
        CDF - cumulative distribution function
        SF - Survival function (also known as reliability function)
        HF - hazard function
        CHF - cumulative hazard function
    These may be plotted all together [using all_functions()] or separately using their respective names.
    When plotted all together, several common descriptive statistics from the probability distribution are also provided.
    User input of parameters is fairly flexible, accepting either scale/shape/threshold or greek letter parameters.
    Available distributions are:
        Exponential
        Weibull
        Normal
        Lognormal
        Gamma
        Beta

    Example Usage:
        plot('weibull',scale=10,shape=2.2,threshold=3).all_functions()
        plot('exponential',Lambda=0.2).PDF() #note that for exponential, Lambda must be capitalized due to "lambda" being a python keyword
        plot('normal',mu=5.538,sigma=3).HF()
    '''
    def __init__(self,dist,Lambda=None,alpha=None,beta=None,gamma=None,mu=None,sigma=None,location=None,scale=None,shape=None,threshold=None,xvals=None,xmin=None,xmax=None,show_plot=True):
        self.show_plot = show_plot
        self.X = None
        if mu is not None:
            self.mu = mu
        if location is not None:
            self.mu = location
        if sigma is not None:
            self.sigma = sigma
        if scale is not None:
            self.sigma = scale
        if Lambda is not None:
            self.Lambda = 1 / Lambda
        if shape is not None:
            self.Lambda = 1 / shape
        if threshold is not None:
            self.gamma = threshold
        if gamma is not None:
            self.gamma = gamma
        if gamma is None and threshold is None:
            self.gamma = 0
        if alpha is not None:
            self.alpha = alpha
        if scale is not None:
            self.alpha = scale
        if beta is not None:
            self.beta = beta
        if shape is not None:
            self.beta = shape

        if xvals is not None:
            self.X = xvals
        if xmin is not None and xmax is not None:
            self.X = np.linspace(xmin, xmax, 1000)

        if dist in ['Norm','norm','normal','Normal']:
            self.name = 'Normal'
            self.b5 = ss.norm.ppf(0.05, self.mu, self.sigma)
            self.b95 = ss.norm.ppf(0.95, self.mu, self.sigma)
            if self.X is None:
                self.X = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma,1000)  # if no limits are specified, they are assumed
            self.pdf = ss.norm.pdf(self.X, self.mu, self.sigma)
            self.cdf = ss.norm.cdf(self.X, self.mu, self.sigma)
            self.sf = ss.norm.sf(self.X, self.mu, self.sigma)
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.norm.stats(loc=self.mu, scale=self.sigma, moments='mvsk')
            self.median = float(self.mu)
            self.mode = float(self.mu)
            self.param_title = str('$\mu$ = ' + str(self.mu) + ' , $\sigma$ = ' + str(self.sigma))
        elif dist in ['Weibull','weibull','weib','Weib']:
            self.name = 'Weibull'
            self.b5 = ss.weibull_min.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
            self.b95 = ss.weibull_min.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)
            if self.X is None:
                self.X = np.linspace(0, self.b95*1.5, 1000)  # if no limits are specified, they are assumed
            self.pdf = ss.weibull_min.pdf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.cdf = ss.weibull_min.cdf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.sf = ss.weibull_min.sf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.weibull_min.stats(self.beta, scale=self.alpha, loc=self.gamma, moments='mvsk')
            self.median = ss.weibull_min.median(self.beta, scale=self.alpha, loc=self.gamma)
            if self.beta >= 1:
                self.mode = round(self.alpha * ((self.beta - 1) / self.beta) ** (1 / self.beta) + self.gamma, _sigfig)
            else:
                self.mode = r'No mode exists when $\beta$ < 1'
            if self.gamma != 0:
                self.param_title = str(
                    r'$\alpha$ = ' + str(self.alpha) + r' , $\beta$ = ' + str(self.beta) + ' , $\gamma$ = ' + str(self.gamma))
            else:
                self.param_title = str(r'$\alpha$ = ' + str(self.alpha) + r' , $\beta$ = ' + str(self.beta))
        elif dist in ['expon', 'Expon', 'exponential', 'Exponential']:
            self.name = 'Exponential'
            self.b5 = ss.expon.ppf(0.05, scale=self.Lambda, loc=self.gamma)
            self.b95 = ss.expon.ppf(0.95, scale=self.Lambda, loc=self.gamma)
            if self.X is None:
                self.X = np.linspace(0, self.b95*1.5, 1000)  # if no limits are specified, they are assumed
            self.pdf = ss.expon.pdf(self.X, scale=self.Lambda, loc=self.gamma)
            self.cdf = ss.expon.cdf(self.X, scale=self.Lambda, loc=self.gamma)
            self.sf = ss.expon.sf(self.X, scale=self.Lambda, loc=self.gamma)
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.expon.stats(scale=self.Lambda, loc=self.gamma, moments='mvsk')
            self.median = np.log(2) / (1/self.Lambda) + self.gamma
            self.mode = 0
            if self.gamma != 0:
                self.param_title = str(r'$\lambda$ = ' + str(1 / self.Lambda) + ' , $\gamma$ = ' + str(self.gamma))
            else:
                self.param_title = str(r'$\lambda$ = ' + str(1 / self.Lambda))
        elif dist in ['lognorm','Lognorm','Lognormal','lognormal']:
            self.name = 'Lognormal'
            self.b5 = ss.lognorm.ppf(0.05, self.sigma, self.gamma, np.exp(self.mu)) #note that scipy uses mu in a log way compared to most other software, so we must take the exp of the input
            self.b95 = ss.lognorm.ppf(0.95, self.sigma, self.gamma, np.exp(self.mu))
            if self.X is None:
                self.X = np.linspace(0, self.b95*1.5,1000)  # if no limits are specified, they are assumed
            self.pdf = ss.lognorm.pdf(self.X, self.sigma, self.gamma, np.exp(self.mu))
            self.cdf = ss.lognorm.cdf(self.X, self.sigma, self.gamma, np.exp(self.mu))
            self.sf = ss.lognorm.sf(self.X, self.sigma, self.gamma, np.exp(self.mu))
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.lognorm.stats(self.sigma, self.gamma, np.exp(self.mu), moments='mvsk')
            self.median = ss.lognorm.median(self.sigma, self.gamma, np.exp(self.mu))
            self.mode = np.exp(self.mu-self.sigma**2)+self.gamma
            if self.gamma != 0:
                self.param_title = str('$\mu$ = ' + str(self.mu) + ' , $\sigma$ = ' + str(self.sigma)+ ' , $\gamma$ = ' + str(self.gamma))
            else:
                self.param_title = str('$\mu$ = ' + str(self.mu) + ' , $\sigma$ = ' + str(self.sigma))
        elif dist in ['Gamma','gamma','gam','Gam']:
            self.name = 'Gamma'
            self.b5 = ss.gamma.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
            self.b95 = ss.gamma.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)
            if self.X is None:
                self.X = np.linspace(0, self.b95 * 1.5, 1000)  # if no limits are specified, they are assumed
            self.pdf = ss.gamma.pdf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.cdf = ss.gamma.cdf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.sf = ss.gamma.sf(self.X, self.beta, scale=self.alpha, loc=self.gamma)
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.gamma.stats(self.beta, scale=self.alpha,loc=self.gamma, moments='mvsk')
            self.median = ss.gamma.median(self.beta, scale=self.alpha, loc=self.gamma)
            if self.beta >= 1:
                self.mode = round((self.beta-1)*self.alpha, _sigfig)
            else:
                self.mode = r'No mode exists when $\beta$ < 1'
            if self.gamma != 0:
                self.param_title = str(
                    r'$\alpha$ = ' + str(self.alpha) + r' , $\beta$ = ' + str(self.beta) + ' , $\gamma$ = ' + str(
                        self.gamma))
            else:
                self.param_title = str(r'$\alpha$ = ' + str(self.alpha) + r' , $\beta$ = ' + str(self.beta))
        elif dist in ['Beta','beta']:
            self.name = 'Beta'
            self.b5 = ss.beta.ppf(0.05, self.alpha, self.beta, 0, 1)
            self.b95 = ss.beta.ppf(0.95, self.alpha, self.beta, 0, 1)
            if self.X is None:
                self.X = np.linspace(0,1, 1000)  # if no limits are specified, they are assumed
            self.pdf = ss.beta.pdf(self.X, self.alpha, self.beta, 0, 1)
            self.cdf = ss.beta.cdf(self.X, self.alpha, self.beta, 0, 1)
            self.sf = ss.beta.sf(self.X, self.alpha, self.beta, 0, 1)
            self.hf = self.pdf / self.sf
            self.chf = -np.log(self.sf)
            self.mean, self.var, self.skew, self.kurt = ss.beta.stats(self.alpha, self.beta, 0, 1, moments='mvsk')
            self.median = ss.beta.median(self.alpha, self.beta, 0, 1)
            if self.alpha > 1 and self.beta > 1:
                self.mode = round((self.alpha-1) / (self.beta + self.alpha - 2), _sigfig)
            else:
                self.mode = r'No mode exists unless $\alpha>1$ and $\beta>1$'
            self.param_title = str(r'$\alpha$ = ' + str(self.alpha) + r' , $\beta$ = ' + str(self.beta))
        else:
            raise ValueError('Unknown distribution specified. Available distributions are Exponential, Weibull, Normal, Lognormal, Beta, Gamma')

    def all_functions(self):
        '''
        Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure
        '''
        #plotting section
        plt.figure(figsize=(9,7))
        text_title=str(self.name+' Distribution'+'\n'+self.param_title)
        plt.suptitle(text_title,fontsize=15)
        plt.subplot(231)
        plt.plot(self.X,self.pdf)
        plt.title('Probability Density\nFunction')
        plt.subplot(232)
        plt.plot(self.X,self.cdf)
        plt.title('Cumulative Distribution\nFunction')
        plt.subplot(233)
        plt.plot(self.X,self.sf)
        plt.title('Survival Function')
        plt.subplot(234)
        plt.plot(self.X,self.hf)
        plt.title('Hazard Function')
        plt.subplot(235)
        plt.plot(self.X,self.chf)
        plt.title('Cumulative Hazard\nFunction')

        #descriptive statistics section
        plt.subplot(236)
        plt.axis('off')
        plt.ylim([0,10])
        plt.xlim([0,10])
        text_mean=str('Mean = ' + str(round(float(self.mean), _sigfig)))
        text_median = str('Median = ' + str(round(self.median, _sigfig)))
        try:
            text_mode = str('Mode = ' + str(round(self.mode, _sigfig)))
        except:
            text_mode = str('Mode = ' + str(self.mode)) #required when mode is str
        text_b5=str('$5^{th}$ quantile = ' + str(round(self.b5, _sigfig)))
        text_b95=str('$95^{th}$ quantile = ' + str(round(self.b95, _sigfig)))
        text_std = str('Standard deviation = ' + str(round(self.var ** 0.5, _sigfig)))
        text_var = str('Variance = ' + str(round(float(self.var), _sigfig)))
        text_skew = str('Skewness = ' + str(round(float(self.skew), _sigfig)))
        text_kurt = str('Excess kurtosis = ' + str(round(float(self.kurt), _sigfig)))
        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self,**kwargs):
        '''
        Plots the Probability Density Function (PDF)
        '''
        if self.show_plot == False:
            return self.pdf
        else:
            plt.plot(self.X,self.pdf,**kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str(self.name+' Distribution\n'+' Probability Density Function '+ '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return self.pdf
    def CDF(self,**kwargs):
        '''
        Plots the Cumulative Density Function (CDF)
        '''
        if self.show_plot == False:
            return self.cdf
        else:
            plt.plot(self.X,self.cdf,**kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction Failing')
            text_title = str(self.name+' Distribution\n'+' Cumulative Distribution Function '+ '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return self.cdf
    def SF(self,**kwargs):
        '''
        Plots the Survival Function (SF) - also known as the reliability function
        '''
        if self.show_plot == False:
            return self.sf
        else:
            plt.plot(self.X,self.sf,**kwargs)
            plt.xlabel('x values')
            plt.ylabel('Fraction Surviving')
            text_title = str(self.name+' Distribution\n'+' Survival Function '+ '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return self.sf
    def HF(self,**kwargs):
        '''
        Plots the Hazard Function (HF)
        '''
        if self.show_plot == False:
            return self.hf
        else:
            plt.plot(self.X,self.hf,**kwargs)
            plt.xlabel('x values')
            plt.ylabel('Hazard')
            text_title = str(self.name+' Distribution\n'+' Hazard Function '+ '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return self.hf

    def CHF(self,**kwargs):
        '''
        Plots the Cumulative Hazard Function (CHF)
        '''
        if self.show_plot == False:
            return self.chf
        else:
            plt.plot(self.X,self.chf,**kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative Hazard')
            text_title = str(self.name+' Distribution\n'+' Cumulative Hazard Function '+ '\n' + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)
            return self.chf