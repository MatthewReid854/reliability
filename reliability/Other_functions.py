'''
Other Functions

This is a collection of several other functions and statistical tests that did not otherwise fit within their own module.
Included functions are:
one_sample_proportion - Calculates the upper and lower bounds of reliability for a given number of trials and successes.
two_proportion_test - Calculates whether the difference in test results between two samples is statistically significant.
sample_size_no_failures - used to determine the sample size required for a test in which no failures are expected, and the desired
    outcome is the lower bound on the reliability based on the sample size and desired confidence interval.
sequential_sampling_chart - plots the accept/reject boundaries for a given set of quality and risk levels. If supplied, the test results
    are also plotted on the chart.
convert_dataframe_to_grouped_lists - groups values in a 2-column dataframe based on the values in the left column and returns those groups in a list of lists
QQ_plot_parametric - quantile-quantile plot. Compares two parametric distributions using shared quantiles. Useful for Field-to-Test conversions in ALT.
QQ_plot_semiparametric - quantile-quantile plot. Compares failure data with a hypothesised parametric distribution. Useful to assess goodness of fit.
PP_plot_parametric - probability-probability plot. Compares two parametric distributions using their CDFs. Useful to understand the differences between the quantiles of the distributions.
PP_plot_semiparametric - probability-probability plot. Compares failure data with a hypothesised parametric distribution. Useful to assess goodness of fit.
'''

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Exponential_Distribution
from reliability.Nonparametric import KaplanMeier, NelsonAalen

def one_sample_proportion(trials=None,successes=None,CI=0.95):
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
    if successes>trials:
        raise ValueError('successes cannot be greater than trials')
    if successes==0 or successes==trials: #calculate 1 sided CI in these cases
        n = 1
    else:
        n = 2
    V1_lower = 2*successes
    V2_lower = 2*(trials-successes+1)
    alpha_lower = (1-CI)/n
    F_lower = ss.f.ppf(alpha_lower, V1_lower, V2_lower)
    LOWER_LIM = (V1_lower*F_lower)/(V2_lower+V1_lower*F_lower)

    V1_upper = 2*(successes+1)
    V2_upper = 2*(trials-successes)
    alpha_upper = 1 - alpha_lower
    F_upper = ss.f.ppf(alpha_upper, V1_upper, V2_upper)
    UPPER_LIM = (V1_upper*F_upper)/(V2_upper+V1_upper*F_upper)

    return LOWER_LIM,UPPER_LIM #will return nan for lower or upper if only one sided CI is calculated (ie. when successes=0 or successes=trials).

def two_proportion_test(sample_1_trials=None,sample_1_successes=None,sample_2_trials=None,sample_2_successes=None,CI=0.95):
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
    if CI<0.5 or CI>=1:
        raise ValueError('CI must be between 0.5 and 1. Default is 0.95')
    if sample_1_trials is None or sample_1_successes is None or sample_2_trials is None or sample_2_successes is None:
        raise ValueError('You must specify the number of trials and successes for both samples.')
    if sample_1_successes>sample_1_trials or sample_2_successes>sample_2_trials:
        raise ValueError('successes cannot be greater than trials')
    p1 = sample_1_successes/sample_1_trials
    p2 = sample_2_successes/sample_2_trials
    diff = p1-p2
    Z = ss.norm.ppf(1-((1-CI)/2))
    k = Z*((p1*(1-p1)/sample_1_trials)+(p2*(1-p2)/sample_2_trials))**0.5
    lower = diff-k
    upper = diff+k
    if lower<0 and upper>0:
        result='non-significant'
    else:
        result='significant'
    return lower,upper,result

def sample_size_no_failures(reliability,CI=0.95,lifetimes=1,weibull_shape=1):
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
    if CI<0.5 or CI>=1:
        raise ValueError('CI must be between 0.5 and 1')
    if reliability<=0 or reliability>=1:
        raise ValueError('Reliability must be between 0 and 1')
    if weibull_shape<0:
        raise ValueError('Weibull shape must be greater than 0. Default (exponential distribution) is 1. If unknown then use 1.')
    if lifetimes>5:
        print('Testing for greater than 5 lifetimes is highly unlikely to result in zero failures.')
    if lifetimes<=0:
        raise ValueError('lifetimes must be >0. Default is 1. No more than 5 is recommended due to test feasibility.')
    n = int(np.ceil((np.log(1-CI))/(lifetimes**weibull_shape*np.log(reliability)))) #rounds up to nearest integer
    return n

def sequential_samling_chart(p1,p2,alpha,beta,show_plot = True, print_results=True, test_results=None,max_samples=100):
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
    if type(test_results)==list:
        F = np.array(test_results)
    elif type(test_results)==np.ndarray:
        F = test_results
    elif test_results is None:
        F = None
    else:
        raise ValueError('test_results must be a binary array or list with 1 as failures and 0 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')

    a = 1-alpha
    b = 1-beta
    d = np.log(p2/p1)+np.log((1-p1)/(1-p2))
    h1 = np.log((1-a)/b)/d
    h2 = np.log((1 - b) / a) / d
    s = np.log((1-p1)/(1-p2))/d

    xvals = np.arange(max_samples+1)
    rejection_line = s*xvals-h1
    acceptance_line = s*xvals+h2
    acceptance_line[acceptance_line<0]=0

    upper_line = np.ones_like(xvals)*(s * max_samples - h1)
    lower_line_range = np.linspace(-h2/s,max_samples,max_samples+1)
    acceptance_line2 = s*lower_line_range+h2 #this is the visible part of the line that starts beyond x=0

    acceptance_array = np.asarray(np.floor(s*xvals+h2), dtype=int)
    rejection_array = np.asarray(np.ceil(s*xvals-h1), dtype=int)
    for i,x in enumerate(xvals): # this replaces cases where the criteria exceeds the number of samples
        if rejection_array[i]>x:
            rejection_array[i]=-1

    data = {'Samples': xvals,'Failures to accept': acceptance_array,'Failures to reject': rejection_array}
    df = pd.DataFrame(data, columns=['Samples', 'Failures to accept','Failures to reject'])
    df.set_index('Samples', inplace=True)
    df.loc[df['Failures to accept'] < 0, 'Failures to accept'] = 'x'
    df.loc[df['Failures to reject'] < 0, 'Failures to reject'] = 'x'

    if print_results==True:
        print(df)

    if show_plot==True:
        #plots the results of tests if they are specified
        if type(F)==np.ndarray:
            if all(F) not in [0,1]:
                raise ValueError('test_results must be a binary array or list with 0 as failures and 1 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')
            nx = []
            ny = []
            failure_count = 0
            sample_count = 0
            for f in F:
                if f==0:
                    sample_count+=1
                    nx.append(sample_count)
                    ny.append(failure_count)
                elif f==1:
                    sample_count+=1
                    nx.append(sample_count)
                    ny.append(failure_count)
                    failure_count+=1
                    nx.append(sample_count)
                    ny.append(failure_count)
                else:
                    raise ValueError('test_results must be a binary array or list with 0 as failures and 1 as successes. eg. [0 0 0 1] for 3 successes and 1 failure.')
            plt.plot(nx, ny,label='test results')

        #plots the decision boundaries and shades the areas red and green
        plt.plot(lower_line_range,acceptance_line2,linestyle='--',color='green')
        plt.plot(xvals,rejection_line,linestyle='--',color='red')
        plt.fill_between(xvals,rejection_line,upper_line,color='red',alpha=0.3,label='Reject sample')
        plt.fill_between(xvals, acceptance_line, rejection_line, color='gray', alpha=0.1, label='Keep Testing')
        plt.fill_between(lower_line_range, 0, acceptance_line2, color='green', alpha=0.3, label='Accept Sample')
        plt.ylim([0,max(rejection_line)])
        plt.xlim([0,max(xvals)])
        plt.xlabel('Number of samples tested')
        plt.ylabel('Number of failures from samples tested')
        plt.title('Sequential sampling decision boundaries')
        plt.legend()
        plt.show()
    return df

def convert_dataframe_to_grouped_lists(input_dataframe):
    '''
    Accepts a dataframe containing 2 columns
    This function assumes the identifying column is the left column
    returns:
    lists , names - lists is a list of the grouped lists
                  - names is the identifying values used to group the lists from the first column

    Example usage:
    #create sample data
    import pandas as pd
    data = {'outcome': ['Failed', 'Censored', 'Failed', 'Failed', 'Censored'],
        'cycles': [1253,1500,1342,1489,1500]}
    df = pd.DataFrame(data, columns = ['outcome', 'cycles'])
    #usage of the function
    lists,names = convert_dataframe_to_grouped_lists(df)
    print(names[1]) >>> Failed
    print(lists[1]) >>> [1253, 1342, 1489]
    '''
    df = input_dataframe
    column_names = df.columns.values
    if len(column_names)>2:
        raise ValueError('Dataframe contains more than 2 columns. There should only be 2 columns with the first column containing the labels to group by and the second containing the values to be returned in groups.')
    grouped_lists = []
    group_list_names = []
    for key, items in df.groupby(column_names[0]):
        values = list(items.iloc[:,1].values)
        grouped_lists.append(values)
        group_list_names.append(key)
    return grouped_lists,group_list_names

def PP_plot_parametric(X_dist=None, Y_dist=None, y_quantile_lines=None, x_quantile_lines=None, show_diagonal_line=False,**kwargs):
    '''
    A PP_Plot is a probability-probability plot that consists of plotting the CDF of one distribution against the CDF of another distribution. If the distributions are similar, the PP_Plot will lie on the diagonal.
    This version of a PP_Plot is the fully parametric form in which we plot one distribution against another distribution. There is also a semi-parametric form offered in PP_plot_semiparametric.

    Inputs:
    X_dist - a probability distribution. The CDF of this distribution will be plotted along the X-axis.
    Y_dist - a probability distribution. The CDF of this distribution will be plotted along the Y-axis.
    y_quantile_lines - starting points for the trace lines to find the X equivalent of the Y-quantile. Optional input. Must be list or array.
    x_quantile_lines - starting points for the trace lines to find the Y equivalent of the X-quantile. Optional input. Must be list or array.
    show_diagonal_line - True/False. Default is False. If True the diagonal line will be shown on the plot.

    Outputs:
    The PP_plot is the only output. Use plt.show() to show it.
    '''

    if X_dist is None or Y_dist is None:
        raise ValueError('X_dist and Y_dist must both be specified as probability distributions generated using the Distributions module')
    if type(X_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution] or type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution]:
        raise ValueError('Invalid probability distribution. X_dist and Y_dist must both be specified as probability distributions generated using the Distributions module')

    # extract certain keyword arguments or specify them if they are not set
    if 'color' in kwargs:
        color = kwargs.pop('color')
    else:
        color = 'k'
    if 'marker' in kwargs:
        marker = kwargs.pop('marker')
    else:
        marker = '.'

    #generate plotting limits and create the PP_plot line
    dist_X_b01 = X_dist.quantile(0.01)
    dist_Y_b01 = Y_dist.quantile(0.01)
    dist_X_b99 = X_dist.quantile(0.99)
    dist_Y_b99 = Y_dist.quantile(0.99)
    xvals = np.linspace(min(dist_X_b01,dist_Y_b01),max(dist_X_b99,dist_Y_b99),100)
    dist_X_CDF = X_dist.CDF(xvals=xvals, show_plot=False)
    dist_Y_CDF = Y_dist.CDF(xvals=xvals, show_plot=False)
    plt.scatter(dist_X_CDF, dist_Y_CDF,marker=marker,color=color,**kwargs)

    #this creates the labels for the axes using the parameters of the distributions
    sigfig=2
    if X_dist.name== 'Weibull':
        X_label_str = str('Weibull CDF (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    elif X_dist.name== 'Gamma':
        X_label_str = str('Gamma CDF (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    elif X_dist.name== 'Exponential':
        X_label_str = str('Exponential CDF (λ=' + str(round(X_dist.Lambda,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    elif X_dist.name== 'Normal':
        X_label_str = str('Normal CDF (μ=' + str(round(X_dist.mu,sigfig)) + ', σ=' + str(round(X_dist.sigma,sigfig)) + ')')
    elif X_dist.name== 'Lognormal':
        X_label_str = str('Lognormal CDF (μ=' + str(round(X_dist.mu,sigfig)) + ', σ=' + str(round(X_dist.sigma,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    elif X_dist.name== 'Beta':
        X_label_str = str('Beta CDF (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ')')

    if Y_dist.name== 'Weibull':
        Y_label_str = str('Weibull CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    elif Y_dist.name== 'Gamma':
        Y_label_str = str('Gamma CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    elif Y_dist.name== 'Exponential':
        Y_label_str = str('Exponential CDF (λ=' + str(round(Y_dist.Lambda,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    elif Y_dist.name== 'Normal':
        Y_label_str = str('Normal CDF (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ')')
    elif Y_dist.name== 'Lognormal':
        Y_label_str = str('Lognormal CDF (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    elif Y_dist.name== 'Beta':
        Y_label_str = str('Beta CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ')')
    plt.xlabel(X_label_str)
    plt.ylabel(Y_label_str)

    #this draws on the quantile lines
    if y_quantile_lines is not None:
        for q in y_quantile_lines:
            quantile = X_dist.CDF(xvals=Y_dist.quantile(q), show_plot=False)
            plt.plot([0,quantile,quantile],[q,q,0],color='blue',linewidth=0.5)
            plt.text(0,q,str(q))
            plt.text(quantile,0,str(round(quantile,2)))
    if x_quantile_lines is not None:
        for q in x_quantile_lines:
            quantile = Y_dist.CDF(xvals=X_dist.quantile(q), show_plot=False)
            plt.plot([q,q,0],[0,quantile,quantile],color='red',linewidth=0.5)
            plt.text(q,0,str(q))
            plt.text(0,quantile,str(round(quantile,2)))
    if show_diagonal_line == True:
        plt.plot([0,1],[0,1],color='blue',alpha=0.5,label='Y = X')
    plt.title('Probability-Probability plot\nParametric')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

def QQ_plot_parametric(X_dist=None, Y_dist=None, show_fitted_lines=True, show_diagonal_line=False, **kwargs):
    '''
    A QQ plot is a quantile-quantile plot which consists of plotting failure units vs failure units for shared quantiles. A quantile is simply the fraction failing (ranging from 0 to 1).
    To generate this plot we calculate the failure units (these may be units of time, strength, cycles, landings, etc.) at which a certain fraction has failed (0.01,0.02,0.03...0.99) for each distribution and plot them together.
    The time (or any other failure unit) at which a given fraction has failed is found using the inverse survival function. If the distributions are similar in shape, then the QQ_plot should be a reasonably straight line.
    By plotting the failure times at equal quantiles for each distribution we can obtain a conversion between the two distributions which is useful for Field-to-Test conversions that are necessary during accelerated life testing (ALT).

    Inputs:
    X_dist - a probability distribution. The failure times at given quantiles from this distribution will be plotted along the X-axis.
    Y_dist - a probability distribution. The failure times at given quantiles from this distribution will be plotted along the Y-axis.
    show_fitted_lines - True/False. Default is True. These are the Y=mX and Y=mX+c lines of best fit.
    show_diagonal_line - True/False. Default is False. If True the diagonal line will be shown on the plot.

    Outputs:
    The QQ_plot will always be output. Use plt.show() to show it.
    [m,m1,c1] - these are the values for the lines of best fit. m is used in Y=mX, and m1 and c1 are used in Y=m1X+c1
    '''

    if X_dist is None or Y_dist is None:
        raise ValueError('dist_X and dist_Y must both be specified as probability distributions generated using the Distributions module')
    if type(X_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution] or type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution]:
        raise ValueError('dist_X and dist_Y must both be specified as probability distributions generated using the Distributions module')
    xvals = np.linspace(0.01,0.99,100)

    # extract certain keyword arguments or specify them if they are not set
    if 'color' in kwargs:
        color = kwargs.pop('color')
    else:
        color = 'k'
    if 'marker' in kwargs:
        marker = kwargs.pop('marker')
    else:
        marker = '.'
    #calculate the failure times at the given quantiles
    dist_X_ISF = []
    dist_Y_ISF = []
    for x in xvals:
        dist_X_ISF.append(X_dist.inverse_SF(float(x)))
        dist_Y_ISF.append(Y_dist.inverse_SF(float(x)))
    dist_X_ISF = np.array(dist_X_ISF)
    dist_Y_ISF = np.array(dist_Y_ISF)
    plt.scatter(dist_X_ISF,dist_Y_ISF,color=color,marker=marker,**kwargs)

    #fit lines and generate text for equations to go in legend
    x = dist_X_ISF[:, np.newaxis]
    y = dist_Y_ISF
    deg1 = np.polyfit(dist_X_ISF,dist_Y_ISF,deg=1) #fit y=mx+c
    m = np.linalg.lstsq(x, y, rcond=-1)[0][0] #fit y=mx
    x_fit = np.linspace(0,max(dist_X_ISF)*1.1,100)
    y_fit = m*x_fit
    text_str = str('Y = ' + str(round(m, 3)) + ' X')
    y1_fit = deg1[0]*x_fit+deg1[1]
    if deg1[1]<0:
        text_str1 = str('Y = ' + str(round(deg1[0], 3)) + ' X' + ' - ' + str(round(-1*deg1[1], 3)))
    else:
        text_str1 = str('Y = ' + str(round(deg1[0], 3)) + ' X'+' + '+str(round(deg1[1], 3)))
    xmax = max(dist_X_ISF)*1.1
    ymax = max(dist_Y_ISF)*1.1
    overall_max = max(xmax,ymax)
    if show_diagonal_line==True:
        plt.plot([0,overall_max],[0,overall_max],color='blue',alpha=0.5,label='Y = X')
    if show_fitted_lines==True:
        plt.plot(x_fit, y_fit, color='red',alpha=0.5,label=text_str)
        plt.plot(x_fit, y1_fit, color='green', alpha=0.5, label=text_str1)
        plt.legend(title='Fitted lines:')

    #this creates the labels for the axes using the parameters of the distributions
    sigfig=2
    if X_dist.name== 'Weibull':
        X_label_str = str('Weibull Quantiles (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    if X_dist.name== 'Gamma':
        X_label_str = str('Gamma Quantiles (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    if X_dist.name== 'Exponential':
        X_label_str = str('Exponential Quantiles (λ=' + str(round(X_dist.Lambda,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    if X_dist.name== 'Normal':
        X_label_str = str('Normal Quantiles (μ=' + str(round(X_dist.mu,sigfig)) + ', σ=' + str(round(X_dist.sigma,sigfig)) + ')')
    if X_dist.name== 'Lognormal':
        X_label_str = str('Lognormal Quantiles (μ=' + str(round(X_dist.mu,sigfig)) + ', σ=' + str(round(X_dist.sigma,sigfig)) + ', γ=' + str(round(X_dist.gamma,sigfig)) + ')')
    if X_dist.name== 'Beta':
        X_label_str = str('Beta Quantiles (α=' + str(round(X_dist.alpha,sigfig)) + ', β=' + str(round(X_dist.beta,sigfig)) + ')')

    if Y_dist.name== 'Weibull':
        Y_label_str = str('Weibull Quantiles (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Gamma':
        Y_label_str = str('Gamma Quantiles (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Exponential':
        Y_label_str = str('Exponential Quantiles (λ=' + str(round(Y_dist.Lambda,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Normal':
        Y_label_str = str('Normal Quantiles (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ')')
    if Y_dist.name== 'Lognormal':
        Y_label_str = str('Lognormal Quantiles (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Beta':
        Y_label_str = str('Beta Quantiles (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ')')
    plt.xlabel(X_label_str)
    plt.ylabel(Y_label_str)
    plt.title('Quantile-Quantile plot\nParametric')
    # plt.xlim([0,xmax])
    # plt.ylim([0,ymax])
    plt.axis('square')
    plt.xlim([0,overall_max])
    plt.ylim([0,overall_max])
    return [m,deg1[0],deg1[1]]

def PP_plot_semiparametric(X_data_failures=None, X_data_right_censored=None, Y_dist=None, show_diagonal_line=True, method='KM',**kwargs):
    '''
    A PP_Plot is a probability-probability plot that consists of plotting the CDF of one distribution against the CDF of another distribution. If we have both distributions we can use PP_plot_parametric.
    This function is for when we want to compare a fitted distribution to an empirical distribution for a given set of data.
    If the fitted distribution is a good fit the PP_Plot will lie on the diagonal line. Assessing goodness of fit in a graphical way is the main purpose of this type of plot.
    To create a semi-parametric PP_plot, we must provide the failure data and the method ('KM' or 'NA' for Kaplan-Meier or Nelson-Aalen) to estimate the empirical CDF, and we must also provide the parametric distribution for the parametric CDF.
    The failure times are the limiting values here so the parametric CDF is only calculated at the failure times since that is the result from the empirical CDF.
    Note that the empirical CDF also accepts X_data_right_censored just as Kaplan-Meier and Nelson-Aalen will also accept right censored data.

    Inputs:
    X_data_failures - the failure times in an array or list
    X_data_right_censored - the right censored failure times in an array or list. Optional input.
    Y_dist - a probability distribution. The CDF of this distribution will be plotted along the Y-axis.
    method - 'KM' or 'NA' for Kaplan-Meier and Nelson-Aalen. Default is 'KM'
    show_diagonal_line - True/False. Default is True. If True the diagonal line will be shown on the plot.

    Outputs:
    The PP_plot is the only output. Use plt.show() to show it.
    '''

    if X_data_failures is None or Y_dist is None:
        raise ValueError('X_data_failures and Y_dist must both be specified. X_data_failures can be an array or list of failure times. Y_dist must be a probability distribution generated using the Distributions module')
    if type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution] or type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution]:
        raise ValueError('Y_dist must be specified as a probability distribution generated using the Distributions module')
    if type(X_data_failures)==list:
        X_data_failures=np.sort(np.array(X_data_failures))
    elif type(X_data_failures)==np.ndarray:
        X_data_failures = np.sort(X_data_failures)
    else:
        raise ValueError('X_data_failures must be an array or list')
    if type(X_data_right_censored)==list:
        X_data_right_censored=np.sort(np.array(X_data_right_censored))
    elif type(X_data_right_censored)==np.ndarray:
        X_data_right_censored = np.sort(X_data_right_censored)
    elif X_data_right_censored is None:
        pass
    else:
        raise ValueError('X_data_right_censored must be an array or list')
    # extract certain keyword arguments or specify them if they are not set
    if 'color' in kwargs:
        color = kwargs.pop('color')
    else:
        color = 'k'
    if 'marker' in kwargs:
        marker = kwargs.pop('marker')
    else:
        marker = '.'
    if method=='KM':
        KM = KaplanMeier(failures=X_data_failures,right_censored=X_data_right_censored,show_plot=False,print_results=False)
        df = KM.results
        failure_rows = df.loc[df['Censoring code (censored=0)']==1.0]
        ecdf = 1-np.array(failure_rows['Kaplan-Meier Estimate'].values)
        xlabel = 'Empirical CDF (Kaplan-Meier estimate)'
    elif method=='NA':
        NA = NelsonAalen(failures=X_data_failures,right_censored=X_data_right_censored,show_plot=False,print_results=False)
        df = NA.results
        failure_rows = df.loc[df['Censoring code (censored=0)'] == 1.0]
        ecdf = 1 - np.array(failure_rows['Nelson-Aalen Estimate'].values)
        xlabel = 'Empirical CDF (Nelson-Aalen estimate)'
    else:
        raise ValueError('method must be "KM" for Kaplan-meier or "NA" for Nelson-Aalen. Default is KM')
    CDF = Y_dist.CDF(X_data_failures,show_plot=False)
    plt.scatter(ecdf,CDF,color=color,marker=marker,**kwargs)

    #this creates the labels for the axes using the parameters of the distributions
    sigfig = 2
    if Y_dist.name== 'Weibull':
        Y_label_str = str('Weibull CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Gamma':
        Y_label_str = str('Gamma CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Exponential':
        Y_label_str = str('Exponential CDF (λ=' + str(round(Y_dist.Lambda,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Normal':
        Y_label_str = str('Normal CDF (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ')')
    if Y_dist.name== 'Lognormal':
        Y_label_str = str('Lognormal CDF (μ=' + str(round(Y_dist.mu,sigfig)) + ', σ=' + str(round(Y_dist.sigma,sigfig)) + ', γ=' + str(round(Y_dist.gamma,sigfig)) + ')')
    if Y_dist.name== 'Beta':
        Y_label_str = str('Beta CDF (α=' + str(round(Y_dist.alpha,sigfig)) + ', β=' + str(round(Y_dist.beta,sigfig)) + ')')
    plt.ylabel(Y_label_str)
    plt.xlabel(xlabel)
    plt.axis('square')
    plt.xlim([0,1])
    plt.ylim([0,1])
    if show_diagonal_line==True:
        plt.plot([0,1],[0,1],color='blue',alpha=0.5)
    plt.title('Probability-Probability Plot\nSemi-parametric')

def QQ_plot_semiparametric(X_data_failures=None,X_data_right_censored=None,Y_dist=None,show_fitted_lines=True, show_diagonal_line=False, method='KM',**kwargs):
    '''
    A QQ plot is a quantile-quantile plot which consists of plotting failure units vs failure units for shared quantiles. A quantile is simply the fraction failing (ranging from 0 to 1).
    When we have two parametric distributions we can plot the failure times for common quanitles against one another using QQ_plot_parametric. QQ_plot_semiparametric is a semiparametric form of a QQ_plot in which we obtain theoretical quantiles using a non-parametric estimate and a specified distribution.
    To generate this plot we begin with the failure units (these may be units of time, strength, cycles, landings, etc.). We then obtain an emprical CDF using either Kaplan-Meier or Nelson-Aalen. The empirical CDF gives us the quantiles we will use to equate the actual and theoretical failure times.
    Once we have the empirical CDF, we use the inverse survival function of the specified distribution to obtain the theoretical failure times and then plot the actual and theoretical failure times together.
    If the specified distribution is a good fit, then the QQ_plot should be a reasonably straight line along the diagonal.
    The primary purpose of this plot is as a graphical goodness of fit test.

    Inputs:
    X_data_failures - the failure times in an array or list. These will be plotted along the X-axis.
    X_data_right_censored - the right censored failure times in an array or list. Optional input.
    Y_dist - a probability distribution. The quantiles of this distribution will be plotted along the Y-axis.
    method - 'KM' or 'NA' for Kaplan-Meier and Nelson-Aalen. Default is 'KM'
    show_fitted_lines - True/False. Default is True. These are the Y=mX and Y=mX+c lines of best fit.
    show_diagonal_line - True/False. Default is False. If True the diagonal line will be shown on the plot.

    Outputs:
    The QQ_plot will always be output. Use plt.show() to show it.
    [m,m1,c1] - these are the values for the lines of best fit. m is used in Y=mX, and m1 and c1 are used in Y=m1X+c1
    '''

    if X_data_failures is None or Y_dist is None:
        raise ValueError('X_data_failures and Y_dist must both be specified. X_data_failures can be an array or list of failure times. Y_dist must be a probability distribution generated using the Distributions module')
    if type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution] or type(Y_dist) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution]:
        raise ValueError('Y_dist must be specified as a probability distribution generated using the Distributions module')
    if type(X_data_failures)==list:
        X_data_failures=np.sort(np.array(X_data_failures))
    elif type(X_data_failures)==np.ndarray:
        X_data_failures = np.sort(X_data_failures)
    else:
        raise ValueError('X_data_failures must be an array or list')
    if type(X_data_right_censored)==list:
        X_data_right_censored=np.sort(np.array(X_data_right_censored))
    elif type(X_data_right_censored)==np.ndarray:
        X_data_right_censored = np.sort(X_data_right_censored)
    elif X_data_right_censored is None:
        pass
    else:
        raise ValueError('X_data_right_censored must be an array or list')
    # extract certain keyword arguments or specify them if they are not set
    if 'color' in kwargs:
        color = kwargs.pop('color')
    else:
        color = 'k'
    if 'marker' in kwargs:
        marker = kwargs.pop('marker')
    else:
        marker = '.'
    if method=='KM':
        KM = KaplanMeier(failures=X_data_failures,right_censored=X_data_right_censored,show_plot=False,print_results=False)
        df = KM.results
        failure_rows = df.loc[df['Censoring code (censored=0)']==1.0]
        ecdf = 1-np.array(failure_rows['Kaplan-Meier Estimate'].values)
        method_str = 'Kaplan-Meier'

    elif method=='NA':
        NA = NelsonAalen(failures=X_data_failures,right_censored=X_data_right_censored,show_plot=False,print_results=False)
        df = NA.results
        failure_rows = df.loc[df['Censoring code (censored=0)'] == 1.0]
        ecdf = 1 - np.array(failure_rows['Nelson-Aalen Estimate'].values)
        method_str = 'Nelson-Aalen'
    else:
        raise ValueError('method must be "KM" for Kaplan-meier or "NA" for Nelson-Aalen. Default is KM')

    # calculate the failure times at the given quantiles
    dist_Y_ISF = []
    for q in ecdf:
        dist_Y_ISF.append(Y_dist.inverse_SF(float(q)))
    dist_Y_ISF = np.array(dist_Y_ISF[::-1])

    dist_Y_ISF[dist_Y_ISF == -np.inf] = 0
    plt.scatter(X_data_failures,dist_Y_ISF,marker=marker,color=color)
    plt.ylabel(str('Theoretical Quantiles based on\n'+method_str+' estimate and '+Y_dist.name+' distribution'))
    plt.xlabel('Actual Quantiles')
    plt.axis('square')
    endval = max(max(dist_Y_ISF), max(X_data_failures)) * 1.1
    if show_diagonal_line==True:
        plt.plot([0,endval],[0,endval],color='blue',alpha=0.5,label='Y = X')

    #fit lines and generate text for equations to go in legend
    y = dist_Y_ISF[:, np.newaxis]
    x = X_data_failures[:, np.newaxis]
    deg1 = np.polyfit(X_data_failures,dist_Y_ISF,deg=1) #fit y=mx+c
    m = np.linalg.lstsq(x, y, rcond=-1)[0][0][0] #fit y=mx
    x_fit = np.linspace(0,endval,100)
    y_fit = m*x_fit
    text_str = str('Y = ' + str(round(m, 3)) + ' X')
    y1_fit = deg1[0]*x_fit+deg1[1]
    if deg1[1]<0:
        text_str1 = str('Y = ' + str(round(deg1[0], 3)) + ' X' + ' - ' + str(round(-1*deg1[1], 3)))
    else:
        text_str1 = str('Y = ' + str(round(deg1[0], 3)) + ' X'+' + '+str(round(deg1[1], 3)))
    if show_fitted_lines==True:
        plt.plot(x_fit, y_fit, color='red',alpha=0.5,label=text_str)
        plt.plot(x_fit, y1_fit, color='green', alpha=0.5, label=text_str1)
        plt.legend(title='Fitted lines:')
    plt.xlim([0,endval])
    plt.ylim([0,endval])
    plt.title('Quantile-Quantile Plot\nSemi-parametric')
    return [m,deg1[0],deg1[1]]
