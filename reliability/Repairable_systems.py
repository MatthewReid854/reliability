'''
Repairable systems
This is a collection of functions used for repairable systems.

Currently included functions are:
reliability_growth - uses the Duane method to find the instantaneous MTBF and produce a reliability growth plot.
optimal_replacement_time - Calculates the cost model to determine how cost varies with replacement time. The cost model may be NHPP (as good as old) or HPP (as good as new). Default is HPP.

Planned future functions are:
Parametric MCF
Nonparametric MCF
Laplace Test
ROCOF
'''
def reliability_growth(times,xmax=None,target_MTBF=None,show_plot=True,print_results=True,**kwargs):
    '''
    Uses the Duane method to find the instantaneous MTBF and produce a reliability growth plot.

    inputs:
    times - array of list of failure times
    xmax - xlim to plot up to. Default is 1.5*max(times)
    target_MTBF - specify the target MTBF to obtain the total time on test required to reach it.
    show_plot - True/False. Defaults to true. Other keyword arguments are passed to the plot for style
    print_results - True/False. Defaults to True.

    returns:
    If print_results is True it will print a summary of the results
    [Lambda, beta, time_to_target] - Array of results. Time to target is only returned if target_MTBF is specified.
    If show_plot is True, it will plot the reliability growth. Use plt.show() to show the plot.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    if type(times)==list:
        times = np.sort(np.array(times))
    elif type(times)==np.ndarray:
        times = np.sort(times)
    else:
        raise ValueError('times must be an array or list of failure times')
    if min(times)<0:
        raise ValueError('failure times cannot be negative. times must be an array or list of failure times')
    if xmax==None:
        xmax=int(max(times)*1.5)
    if 'color' in kwargs:
        c = kwargs.pop('color')
    else:
        c = 'steelblue'

    N = np.arange(1,len(times)+1)
    theta_c = times/N
    ln_t = np.log(times)
    ln_theta_c = np.log(theta_c)
    z = np.polyfit(ln_t,ln_theta_c,1) #fit a straight line to the data to get the parameters lambda and beta
    beta = 1-z[0]
    Lambda = np.exp(-z[1])
    xvals = np.linspace(0,xmax,1000)
    theta_i = (xvals**(1-beta))/(Lambda*beta) #the smooth line
    theta_i_points = (times**(1-beta))/(Lambda*beta) #the failure times highlighted along the line
    output = [Lambda, beta] #returns lambda and beta

    if print_results==True:
        print('Reliability growth model parameters:\nlambda:',Lambda,'\nbeta:',beta)

    if target_MTBF is not None:
        t_target = (target_MTBF * Lambda * beta) ** (1 / (1 - beta))
        output.append(t_target)  # also returns time to target MTBF if a target is specified
        print('Time to reach target MTBF:',t_target)

    if show_plot==True:
        plt.plot(xvals,theta_i,color=c,**kwargs)
        plt.plot(times,theta_i_points,'o',color=c,alpha=0.5)
        if target_MTBF is not None:
            plt.plot([0, t_target, t_target], [target_MTBF, target_MTBF, 0], 'red', label='Reliability target', linewidth=1)
        plt.title('Reliability Growth')
        plt.xlabel('Total time on test')
        plt.ylabel('Instantaneous MTBF')
        plt.xlim([0,max(xvals)])
        plt.ylim([0,max(theta_i)*1.2])
    return output

def optimal_replacement_time(cost_PM, cost_CM, weibull_alpha, weibull_beta, show_plot=True, print_results=True, q=0, **kwargs):
    '''
    Calculates the cost model to determine how cost varies with replacement time.
    The cost model may be HPP (good as new replacement) or NHPP (as good as old replacement). Default is HPP.

    inputs:
    Cost_PM - cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM - cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha - scale parameter of the underlying Weibull distribution
    weibull_beta - shape parameter of the underlying Weibull distribution. Should be greater than 1 otherwise conducting PM is not economical.
    q - restoration factor. q=1 is Power Law NHPP (as good as old), q=0 is HPP (as good as new). Default is q=0 (as good as new).
    show_plot - True/False. Defaults to True. Other plotting keywords are also accepted and used.
    print_results - True/False. Defaults to True

    outputs:
    [ORT, min_cost] - the optimal replacement time and minimum cost per unit time in an array
    Plot of cost model if show_plot is set to True. Use plt.show() to display it.
    Printed results if print_results is set to True.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    from scipy import integrate
    if 'color' in kwargs:
        c = kwargs.pop('color')
    else:
        c = 'steelblue'
    if cost_PM>cost_CM:
        raise ValueError('Cost_PM must be less than Cost_CM otherwise preventative maintenance should not be conducted.')
    if weibull_beta<1:
        warnings.warn('weibull_beta is < 1 so the hazard rate is decreasing, therefore preventative maintenance should not be conducted.')

    if q == 1: #as good as old
        alpha_multiple = 4 #just used for plot limits
        t = np.linspace(1, weibull_alpha * alpha_multiple, 100000)
        CPUT = ((cost_PM*(t/weibull_alpha)**weibull_beta)+cost_CM)/t
        ORT = weibull_alpha*((cost_CM/(cost_PM*(weibull_beta-1)))**(1/weibull_beta))
        min_cost = ((cost_PM * (ORT / weibull_alpha) ** weibull_beta) + cost_CM) / ORT
    elif q == 0: #as good as new
        alpha_multiple = 3
        t = np.linspace(1, weibull_alpha * alpha_multiple, 10000)
        CPUT = []  # cost per unit time
        R = lambda x: np.exp(-((x/weibull_alpha)**weibull_beta))
        for T in t:
            SF = np.exp(-((T/weibull_alpha)**weibull_beta))
            integral_R, error = integrate.quad(R, 0, T)
            CPUT.append((cost_PM * SF + cost_CM * (1 - SF)) / integral_R)
            idx = np.argmin(CPUT)
            min_cost = CPUT[idx]  # minimum cost per unit time
            ORT = t[idx]  # optimal replacement time
    else:
        raise ValueError('q must be 0 or 1. Default is 0. Use 0 for "as good as new" and use 1 for "as good as old".')

    if min_cost<1:
        min_cost_rounded = round(min_cost, -int(np.floor(np.log10(abs(min_cost)))) + 1)  # this rounds to exactly 2 sigfigs no matter the number of preceding zeros
    else:
        min_cost_rounded = round(min_cost,2)
    ORT_rounded = round(ORT, 2)

    if print_results==True:
        if q==0:
            print('Cost model assuming as good as new replacement (q=0):')
        else:
            print('Cost model assuming as good as old replacement (q=1):')
        print('The minimum cost per unit time is',min_cost_rounded,'\nThe optimal replacement time is',ORT_rounded)

    if show_plot==True:
        plt.plot(t,CPUT,color=c,**kwargs)
        plt.plot(ORT,min_cost,'o',color=c)
        text_str = str('\nMinimum cost per unit time is '+str(min_cost_rounded)+'\nOptimal replacement time is '+str(ORT_rounded))
        plt.text(ORT,min_cost,text_str,verticalalignment='top')
        plt.xlabel('Replacement time')
        plt.ylabel('Cost per unit time')
        plt.title('Optimal replacement time estimation')
        plt.ylim([0,min_cost*2])
        plt.xlim([0,weibull_alpha*alpha_multiple])
    return [ORT,min_cost]

