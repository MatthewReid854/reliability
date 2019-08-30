'''
Probability plotting

This module contains the functions required to generate linearized probability plots of the six distributions included in reliability.
The most common use of probability plots is to assess goodness of fit.
The functions in this module are:
plotting_positions - using the median rank method, this function generates an empirical estimate of the CDF
Weibull_probability_plot - used for Weibull_2P and Weibull_3P plotting.
Normal_probability_plot - used for Normal_2P plotting.
Lognormal_probability_plot - used for Lognormal_2P plotting.
Exponential_probability_plot - used for Exponential_1P and Exponential_2P plotting.
Beta_probability_plot - used for Beta_2P plotting.
Gamma_probability_plot - used for Gamma_2P and Gamma_3P plotting.

This function also the axes scaling functions, though these are used internally by matplotlib and have no inputs or outputs.
If you would like to use these scaling functions, simply import this module and then in your plotting code use plt.gca().set_yscale('weibull').
The scaling options are 'weibull','exponential','normal','gamma',beta'.
Gamma and Beta are more complicated to use as they require the parameters of the distribution to be declared as global variables. Look through the code if you need to see how this is done.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import scale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator
import scipy.stats as ss
np.seterr('ignore')

class _WeibullScale(scale.ScaleBase):
    name = 'weibull'
    def __init__(self, axis, **kwargs):
        scale.ScaleBase.__init__(self)
        self.thresh = None
    def get_transform(self):
        return self.CustomTransform(self.thresh)
    def set_default_locators_and_formatters(self, axis):
        pass
    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, F):
            return np.log(-np.log(1-F))
        def inverted(self):
            return _WeibullScale.InvertedCustomTransform(self.thresh)
    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, R):
            return 1-np.exp(-np.exp(R))
        def inverted(self):
            return _WeibullScale.CustomTransform(self.thresh)

class _ExponentialScale(scale.ScaleBase):
    name = 'exponential'
    def __init__(self, axis, **kwargs):
        scale.ScaleBase.__init__(self)
        self.thresh = None
    def get_transform(self):
        return self.CustomTransform(self.thresh)
    def set_default_locators_and_formatters(self, axis):
        pass
    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, F):
            return ss.expon.ppf(F)
        def inverted(self):
            return _ExponentialScale.InvertedCustomTransform(self.thresh)
    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, R):
            return ss.expon.cdf(R)
        def inverted(self):
            return _ExponentialScale.CustomTransform(self.thresh)

class _NormalScale(scale.ScaleBase):
    name = 'normal'
    def __init__(self, axis, **kwargs):
        scale.ScaleBase.__init__(self)
        self.thresh = None
    def get_transform(self):
        return self.CustomTransform(self.thresh)
    def set_default_locators_and_formatters(self, axis):
        pass
    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, F):
            return ss.norm.ppf(F)
        def inverted(self):
            return _NormalScale.InvertedCustomTransform(self.thresh)
    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, R):
            return ss.norm.cdf(R)
        def inverted(self):
            return _NormalScale.CustomTransform(self.thresh)

class _GammaScale(scale.ScaleBase):
    name = 'gamma'
    def __init__(self, axis, **kwargs):
        scale.ScaleBase.__init__(self)
        self.thresh = None
    def get_transform(self):
        return self.CustomTransform(self.thresh)
    def set_default_locators_and_formatters(self, axis):
        pass
    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, F):
            return ss.gamma.ppf(F,a=gamma_beta)
        def inverted(self):
            return _GammaScale.InvertedCustomTransform(self.thresh)
    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, R):
            return ss.gamma.cdf(R,a=gamma_beta)
        def inverted(self):
            return _GammaScale.CustomTransform(self.thresh)

class _BetaScale(scale.ScaleBase):
    name = 'beta'
    def __init__(self, axis, **kwargs):
        scale.ScaleBase.__init__(self)
        self.thresh = None
    def get_transform(self):
        return self.CustomTransform(self.thresh)
    def set_default_locators_and_formatters(self, axis):
        pass
    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, F):
            return ss.beta.ppf(F,a=beta_alpha,b=beta_beta)
        def inverted(self):
            return _BetaScale.InvertedCustomTransform(self.thresh)
    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
        def transform_non_affine(self, R):
            return ss.beta.cdf(R,a=beta_alpha,b=beta_beta)
        def inverted(self):
            return _BetaScale.CustomTransform(self.thresh)

#register all the custom scales so that matplotlib can find them by name
scale.register_scale(_WeibullScale)
scale.register_scale(_ExponentialScale)
scale.register_scale(_NormalScale)
scale.register_scale(_GammaScale)
scale.register_scale(_BetaScale)

def plotting_positions(failures=None,right_censored=None,left_censored=None):
    '''
    Uses the median rank method to calculate the plotting positions for plotting on probability paper
    You may specify left or right censored data but not both.
    This function is primarily used by the probability plotting functions such as Weibull_probability_plot and the other 5.

    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times

    Outputs:
    x,y - the x and y plotting positions as arrays
    '''
    if left_censored is not None and right_censored is not None:
        raise ValueError('You can not specify both left and right censored data at the same time')
    if failures is None:
        raise ValueError('failures must be specified as an array or list')
    elif type(failures)==np.ndarray:
        f=np.sort(failures)
    elif type(failures)==list:
        f = np.sort(np.array(failures))
    else:
        raise ValueError('failures must be specified as an array or list')
    if right_censored is None:
        rc=np.array([])
    elif type(right_censored) == np.ndarray:
        rc = np.sort(right_censored)
    elif type(right_censored) == list:
        rc = np.sort(np.array(right_censored))
    else:
        raise ValueError('if specified, right_censored must be an array or list')
    if left_censored is None:
        lc=np.array([])
    elif type(left_censored) == np.ndarray:
        lc = np.sort(left_censored)
    elif type(left_censored) == list:
        lc = np.sort(np.array(left_censored))
    else:
        raise ValueError('if specified, left_censored must be an array or list')
    if left_censored is None: #used for uncensored and right censored data
        f_codes = np.ones_like(f)
        rc_codes = np.zeros_like(rc)
        cens_codes = np.hstack([f_codes,rc_codes])
        all_data = np.hstack([f,rc])
        data = {'times': all_data,'cens_codes': cens_codes}
        df = pd.DataFrame(data,columns=['times','cens_codes'])
        df_sorted = df.sort_values(by='times')
        df_sorted['counter']=np.arange(1,len(all_data)+1)
        i = df_sorted['counter'].values
        n = len(i)
        y0 = 1 - (i - 0.3) / (n + 0.4) #median rank method
        df_sorted['y']=y0[::-1]
        failure_rows = df_sorted.loc[df_sorted['cens_codes'] == 1.0]
        xvalues = failure_rows['times'].values
        x = xvalues[::-1]
        yvalues = failure_rows['y'].values
        y = yvalues[::-1]
    else: #used for left censored data
        f_codes = np.ones_like(f)
        lc_codes = np.zeros_like(lc)
        cens_codes = np.hstack([f_codes,lc_codes])
        all_data = np.hstack([f,lc])
        data = {'times': all_data,'cens_codes': cens_codes}
        df = pd.DataFrame(data,columns=['times','cens_codes'])
        df_sorted = df.sort_values(by='times')
        df_sorted['counter']=np.arange(1,len(all_data)+1)
        i = df_sorted['counter'].values
        n = len(i)
        y0 = 1 - (i - 0.3) / (n + 0.4) #median rank method
        df_sorted['y']=y0[::-1]
        failure_rows = df_sorted.loc[df_sorted['cens_codes'] == 1.0]
        xvalues = failure_rows['times'].values
        x = xvalues[::-1]
        yvalues = failure_rows['y'].values
        y = yvalues[::-1]
    return x,y

def Weibull_probability_plot(failures=None,right_censored=None,left_censored=None,fit_gamma=False):
    '''
    Weibull probability plot

    Generates a probability plot on Weibull scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times
    fit_gamma - True/False. Default is False. Specify This as true in order to fit the Weibull_3P distribution and scale the x-axis to time - gamma.

    Outputs:
    The plot is the only output. Use plt.show() to show it.
    '''
    from reliability.Fitters import Fit_Weibull_2P,Fit_Weibull_3P
    from reliability.Distributions import Weibull_Distribution
    #ensure the input data is arrays
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    if type(failures)==np.ndarray:
        pass
    elif type(failures)==list:
        failures = np.array(failures)
    else:
        raise ValueError('failures must be a list or an array')
    if right_censored is not None:
        if type(right_censored)==np.ndarray:
            pass
        elif type(right_censored)==list:
            right_censored = np.array(right_censored)
        else:
            raise ValueError('right_censored must be a list or an array')
    if left_censored is not None:
        if type(left_censored)==np.ndarray:
            pass
        elif type(left_censored)==list:
            left_censored = np.array(left_censored)
        else:
            raise ValueError('left_censored must be a list or an array')
    if fit_gamma==True and left_censored is not None:
        raise ValueError('cannot fit gamma if left censored data is specified')
    #generate the figure and fit the distribution
    xvals = np.logspace(-2, np.log10(max(failures))*10, 1000)
    if fit_gamma == False:
        fit = Fit_Weibull_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        wbf = Weibull_Distribution(alpha = fit.alpha,beta=fit.beta).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,wbf,color='red',label=str('Fitted Weibull_2P (α='+str(round(fit.alpha,2))+', β='+str(round(fit.beta,2))+')'))
        plt.xlabel('Time')
    elif fit_gamma == True:
        fit = Fit_Weibull_3P(failures=failures, right_censored=right_censored)
        wbf = Weibull_Distribution(alpha = fit.alpha,beta=fit.beta).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,wbf,color='red',label=str('Fitted Weibull_3P\n(α='+str(round(fit.alpha,2))+', β='+str(round(fit.beta,2))+', γ='+str(round(fit.gamma,2))+')'))
        plt.xlabel('Time - gamma')
        failures=failures-fit.gamma
        if right_censored is not None:
            right_censored=right_censored-fit.gamma
    #plot the failure points and format the scale and axes
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.gca().set_yscale('weibull')
    plt.xscale('log')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    plt.ylim([0.0001,0.9999])
    pts_min_log = 10**(int(np.floor(np.log10(x[-2])))) #second smallest point is rounded down to nearest power of 10
    pts_max_log = 10**(int(np.ceil(np.log10(x[0])))) #largest point is rounded up to nearest power of 10
    plt.xlim([pts_min_log,pts_max_log])
    plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0,1,51)))
    ytickvals = [0.0001,0.0003,0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999,0.9999]
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    plt.title('Probability plot\nWeibull CDF')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7) #adjust the figsize. This is done post figure creation so that layering is easier

def Normal_probability_plot(failures=None,right_censored=None,left_censored=None):
    '''
    Normal probability plot

    Generates a probability plot on Normal scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times

    Outputs:
    The plot is the only output. Use plt.show() to show it.
    '''
    from reliability.Fitters import Fit_Normal_2P
    from reliability.Distributions import Normal_Distribution
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.ylim([0.0001,0.9999])
    plt.xlim([min(x)-max(x)*0.2,max(x)*1.2])
    plt.gca().set_yscale('normal')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0,1,51)))
    ytickvals = [0.0001,0.001,0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.999,0.9999]
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    xvals = np.linspace(min(x)-max(x),max(x)*10,1000)
    fit = Fit_Normal_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
    nf = Normal_Distribution(mu= fit.mu,sigma=fit.sigma).CDF(show_plot=False,xvals=xvals)
    plt.plot(xvals,nf,color='red',label=str('Fitted Normal_2P (μ='+str(round(fit.mu,2))+', σ='+str(round(fit.sigma,2))+')'))
    plt.title('Probability plot\nNormal CDF')
    plt.xlabel('Time')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7) #adjust the figsize. This is done post figure creation so that layering is easier

def Lognormal_probability_plot(failures=None,right_censored=None,left_censored=None):
    '''
    Lognormal probability plot

    Generates a probability plot on Lognormal scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times

    Outputs:
    The plot is the only output. Use plt.show() to show it.

    Note that fit_gamma is not an option as the Fit_Lognormal_3P is not yet implemented.
    '''
    from reliability.Fitters import Fit_Lognormal_2P
    from reliability.Distributions import Lognormal_Distribution
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.ylim([0.0001,0.9999])
    xmin_log = 10**(int(np.floor(np.log10(min(x)))))
    xmax_log = 10**(int(np.ceil(np.log10(max(x)))))
    plt.xlim([xmin_log,xmax_log])
    plt.gca().set_yscale('normal')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    plt.xscale('log')
    plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0,1,51)))
    ytickvals = [0.0001,0.001,0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.999,0.9999]
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    xvals = np.logspace(np.log10(xmin_log)-1,np.log10(xmax_log)+1,1000)
    fit = Fit_Lognormal_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
    lnf = Lognormal_Distribution(mu= fit.mu,sigma=fit.sigma).CDF(show_plot=False,xvals=xvals)
    plt.plot(xvals,lnf,color='red',label=str('Fitted Lognormal_2P (μ='+str(round(fit.mu,2))+', σ='+str(round(fit.sigma,2))+')'))
    plt.title('Probability plot\nLognormal CDF')
    plt.xlabel('Time')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7) #adjust the figsize. This is done post figure creation so that layering is easier

def Beta_probability_plot(failures=None,right_censored=None,left_censored=None):
    '''
    Beta probability plot

    Generates a probability plot on Beta scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times

    Outputs:
    The plot is the only output. Use plt.show() to show it.
    '''
    from reliability.Fitters import Fit_Beta_2P
    from reliability.Distributions import Beta_Distribution
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.ylim([0.0001,0.9999])
    plt.xlim([-0.1,1.1])
    plt.gca().set_yscale('beta')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0,1,51)))
    ytickvals = [0.001,0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,0.999]
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    xvals = np.linspace(0,1,1000)
    fit = Fit_Beta_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
    bf = Beta_Distribution(alpha = fit.alpha,beta=fit.beta).CDF(show_plot=False,xvals=xvals)
    plt.plot(xvals,bf,color='red',label=str('Fitted Beta_2P (α='+str(round(fit.alpha,2))+', β='+str(round(fit.beta,2))+')'))
    global beta_alpha, beta_beta
    beta_beta = fit.beta #this is used in the axes scaling as the ppf and cdf need the shape parameter to scale the axes correctly
    beta_alpha = fit.alpha
    plt.title('Probability plot\nBeta CDF')
    plt.xlabel('Time')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7)  # adjust the figsize. This is done post figure creation so that layering is easier

def Gamma_probability_plot(failures=None,right_censored=None,left_censored=None,fit_gamma=False):
    '''
    Gamma probability plot

    Generates a probability plot on Gamma scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times
    fit_gamma - True/False. Default is False. Specify This as true in order to fit the Gamma_3P distribution and scale the x-axis to time - gamma.

    Outputs:
    The plot is the only output. Use plt.show() to show it.
    '''
    from reliability.Fitters import Fit_Gamma_2P,Fit_Gamma_3P
    from reliability.Distributions import Gamma_Distribution
    #ensure the input data is arrays
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    if type(failures)==np.ndarray:
        pass
    elif type(failures)==list:
        failures = np.array(failures)
    else:
        raise ValueError('failures must be a list or an array')
    if right_censored is not None:
        if type(right_censored)==np.ndarray:
            pass
        elif type(right_censored)==list:
            right_censored = np.array(right_censored)
        else:
            raise ValueError('right_censored must be a list or an array')
    if left_censored is not None:
        if type(left_censored)==np.ndarray:
            pass
        elif type(left_censored)==list:
            left_censored = np.array(left_censored)
        else:
            raise ValueError('left_censored must be a list or an array')
    if fit_gamma==True and left_censored is not None:
        raise ValueError('cannot fit gamma if left censored data is specified')
    if fit_gamma==True and left_censored is not None:
        raise ValueError('cannot fit gamma if left censored data is specified')
    #generate the figure and fit the distribution
    xvals = np.logspace(-2, np.log10(max(failures))*10, 1000)
    if fit_gamma == False:
        fit = Fit_Gamma_2P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        gf = Gamma_Distribution(alpha = fit.alpha,beta=fit.beta).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,gf,color='red',label=str('Fitted Gamma_2P (α='+str(round(fit.alpha,2))+', β='+str(round(fit.beta,2))+')'))
        plt.xlabel('Time')
    elif fit_gamma == True:
        fit = Fit_Gamma_3P(failures=failures, right_censored=right_censored)
        gf = Gamma_Distribution(alpha = fit.alpha,beta=fit.beta).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,gf,color='red',label=str('Fitted Gamma_3P\n(α='+str(round(fit.alpha,2))+', β='+str(round(fit.beta,2))+', γ='+str(round(fit.gamma,2))+')'))
        plt.xlabel('Time - gamma')
        failures=failures-fit.gamma
        if right_censored is not None:
            right_censored=right_censored-fit.gamma
    global gamma_beta
    gamma_beta = fit.beta #this is used in the axes scaling as the ppf and cdf need the shape parameter to scale the axes correctly
    #plot the failure points and format the scale and axes
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.gca().set_yscale('gamma')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    plt.xlim([0,max(x)*1.2])
    if max(y)<0.9:
        ytickvals = [0.05,0.2,0.4,0.6,0.7,0.8,0.9]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.9, 90)))
        plt.ylim([0, 0.9])
    elif max(y)<0.95:
        ytickvals = [0.05,0.2,0.4,0.6,0.7,0.8,0.9,0.95]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.95, 95)))
        plt.ylim([0, 0.95])
    elif max(y)<0.97:
        ytickvals = [0.05,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.97]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.97, 97)))
        plt.ylim([0, 0.97])
    elif max(y)<0.99:
        ytickvals = [0.05,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.97,0.99]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0, 0.99])
    elif max(y)<0.999:
        ytickvals = [0.05,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.999]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0, 0.999])
    else:
        ytickvals = [0.05,0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 0.9999]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0, 0.9999])
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    plt.title('Probability plot\nGamma CDF')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7) #adjust the figsize. This is done post figure creation so that layering is easier

def Exponential_probability_plot(failures=None,right_censored=None,left_censored=None,fit_gamma=False):
    '''
    Exponential probability plot

    Generates a probability plot on Exponential scaled probability paper so that the distribution appears linear.
    Inputs:
    failures - the array or list of failure times
    right_censored - the array or list of right censored failure times
    left_censored - the array or list of left censored failure times
    fit_gamma - True/False. Default is False. Specify This as true in order to fit the Exponential_2P distribution and scale the x-axis to time - gamma.

    Outputs:
    The plot is the only output. Use plt.show() to show it.
    '''
    from reliability.Fitters import Fit_Expon_1P,Fit_Expon_2P
    from reliability.Distributions import Exponential_Distribution
    if len(failures)<2:
        raise ValueError('Insufficient data to fit a distribution. Minimum number of points is 2')
    if fit_gamma==True and left_censored is not None:
        raise ValueError('cannot fit gamma if left censored data is specified')
    xvals = np.logspace(-2, np.log10(max(failures)) * 10, 1000)
    if fit_gamma == False:
        fit = Fit_Expon_1P(failures=failures,right_censored=right_censored,left_censored=left_censored)
        ef = Exponential_Distribution(Lambda = fit.Lambda).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,ef,color='red',label=str('Fitted Exponential_1P (λ='+str(round(fit.Lambda,2))+')'))
        plt.xlabel('Time')
    elif fit_gamma == True:
        fit = Fit_Expon_2P(failures=failures, right_censored=right_censored)
        ef = Exponential_Distribution(Lambda = fit.Lambda).CDF(show_plot=False,xvals=xvals)
        plt.plot(xvals,ef,color='red',label=str('Fitted Exponential_2P\n(λ='+str(round(fit.Lambda,2))+', γ='+str(round(fit.gamma,2))+')'))
        plt.xlabel('Time - gamma')
        failures=failures-fit.gamma
        if right_censored is not None:
            right_censored=right_censored-fit.gamma
    x,y = plotting_positions(failures=failures,right_censored=right_censored,left_censored=left_censored)
    plt.scatter(x, y, marker='.', linewidth=2, c='k',label='Failure data')
    plt.xlim([0,max(x)*1.2])
    plt.gca().set_yscale('exponential')
    plt.grid(b=True, which='major' ,color='k',alpha=0.3, linestyle='-')
    plt.grid(b=True, which='minor', color='k',alpha=0.08, linestyle='-')
    if max(y)<0.9:
        ytickvals = [0.1,0.2,0.3,0.5,0.7,0.8,0.9]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.9, 90)))
        plt.ylim([0.01, 0.9])
    elif max(y)<0.95:
        ytickvals = [0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.95, 95)))
        plt.ylim([0.01, 0.95])
    elif max(y)<0.97:
        ytickvals = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.97, 97)))
        plt.ylim([0.01, 0.97])
    elif max(y)<0.99:
        ytickvals = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0.01, 0.99])
    elif max(y)<0.999:
        ytickvals = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.999]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0.01, 0.999])
    else:
        ytickvals = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.9999]
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0.01, 0.99, 99)))
        plt.ylim([0.01, 0.9999])
    plt.yticks(ytickvals)
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in ytickvals]) #formats y ticks as percentage
    plt.title('Probability plot\nExponential CDF')
    plt.ylabel('Fraction failing')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(9, 7)  # adjust the figsize. This is done post figure creation so that layering is easier
