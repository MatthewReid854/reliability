'''
Non-parametric reliability analysis of failure data

Uses the Kaplan-Meier estimation method to calculate the reliability from failure data.
Right censoring is supported and confidence bounds are provided. Left censoring is not supported.
The confidence bounds are calculated using the Greenwood formula with Normal approximation, which is the same as
featured in Minitab.

inputs:
    failure - an array or list of failure times. Sorting is automatic so times do not need to be provided in any order.
     right_censored - an array or list of right censored failure times. Defaults to None.
    show_plot - True/False. Default is True
    print_results - True/False. Default is True. Will display a pandas dataframe in the console.
    plot_CI - plots the upper and lower confidence interval
    CI - confidence interval between 0 and 1. Default is 0.95 for 95% CI.
    shade_CI - True/False. True (default) will plot a solid fill, False will leave the CI areas unshaded.

outputs:
    results - dataframe of results
    KM - list of Kaplan Meier column from results dataframe. This column is the non parametric estimate of the Survival Function (reliability function).

Example Usage:
f = [5248,7454,16890,17200,38700,45000,49390,69040,72280,131900]
rc = [3961,4007,4734,6054,7298,10190,23060,27160,28690,37100,40060,45670,53000,67000,69630,77350,78470,91680,105700,106300,150400]
KaplanMeier(failures = f, right_censored = rc)

'''

#produce a kaplan meier plot of the data
class KaplanMeier:
    def __init__(self,failures=None,right_censored = None,show_plot=True,print_results=True,plot_CI=True,CI=0.95,shade_CI=True,**kwargs):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as ss
        np.seterr(divide='ignore') #divide by zero occurs if last detapoint is a failure so risk set is zero

        if failures is None:
            raise ValueError('failures must be provided to calculate non-parametric estimates.')
        if right_censored is None:
            right_censored = [] #create empty array so it can be added in hstack

        if shade_CI not in [True, False]:
            raise ValueError('shade_CI must be either True or False. Default is True.')

        #turn the failures and right censored times into a two lists of times and censoring codes
        times = np.hstack([failures, right_censored])
        F = np.ones_like(failures)
        RC = np.zeros_like(right_censored) #censored values are given the code of 0
        cens_code = np.hstack([F, RC])
        Data = {'times': times, 'cens_code': cens_code}
        df = pd.DataFrame(Data, columns=['times', 'cens_code'])
        df2 = df.sort_values(by='times')
        d = df2['times'].values
        c = df2['cens_code'].values

        if CI<0 or CI>1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals.')

        n = len(d) #number of items
        failures_array = np.arange(1, n + 1) #array of number of items (1 to n)
        remaining_array = failures_array[::-1] #items remaining (n to 1)
        KM = []
        KM_upper = [] #upper CI
        KM_lower = [] #lower CI
        z = ss.norm.ppf(1-(1-CI)/2)
        frac = []
        delta=0
        for i in failures_array:
            if i==1:
                KM.append((remaining_array[i-1]-c[i-1])/remaining_array[i-1])
            else:
                KM.append(((remaining_array[i-1] - c[i-1]) / remaining_array[i-1])*KM[i-2])
            #greenwood confidence interval calculations. Uses Normal approximation (same method as in Minitab)
            if c[i-1]==1:
                risk_set=n-i+1
                frac.append(1/((risk_set)*(risk_set-1)))
                sumfrac = sum(frac)
                R2 = KM[i-1]**2
                if R2>0: #required if the last piece of data is a failure
                    delta = ((sumfrac * R2) ** 0.5) * z
                else:
                    delta=0
            KM_upper.append(KM[i-1]+delta)
            KM_lower.append(KM[i-1]-delta)
        KM_lower = np.array(KM_lower)
        KM_upper = np.array(KM_upper)
        KM_upper[KM_upper>1]=1
        KM_lower[KM_lower<0]=0

        #assemble the pandas dataframe for the output
        DATA = {'Failure times': d,
                'Censoring code (censored=0)': c,
                'Items remaining': remaining_array,
                'Kaplan Meier Estimate': KM,
                'Lower CI bound': KM_lower,
                'Upper CI bound': KM_upper}
        dfx = pd.DataFrame(DATA,columns=['Failure times','Censoring code (censored=0)','Items remaining','Kaplan Meier Estimate','Lower CI bound','Upper CI bound'])
        dfy = dfx.set_index('Failure times')
        pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
        pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
        self.results = dfy
        self.KM = KM

        if print_results==True:
            print(dfy) #this will print the pandas dataframe
        #plotting section
        if show_plot==True:
            KM_x = [0]
            KM_y = [1] #adds a start point for 100% reliability at 0 time
            KM_y_upper = []
            KM_y_lower = []

            for i in failures_array:
                if i==1:
                    if c[i-1]==0: #if the first item is censored
                        KM_x.append(d[i-1])
                        KM_y.append(1)
                        KM_y_lower.append(1)
                        KM_y_upper.append(1)
                    else: #if the first item is a failure
                        KM_x.append(d[i-1])
                        KM_x.append(d[i-1])
                        KM_y.append(1)
                        KM_y.append(KM[i - 1])
                        KM_y_lower.append(1)
                        KM_y_upper.append(1)
                        KM_y_lower.append(1)
                        KM_y_upper.append(1)
                else:
                    if KM[i-2]==KM[i-1]: #if the next item is censored
                        KM_x.append(d[i-1])
                        KM_y.append(KM[i-1])
                        KM_y_lower.append(KM_lower[i-2])
                        KM_y_upper.append(KM_upper[i-2])
                    else: #if the next item is a failure
                        KM_x.append(d[i-1])
                        KM_y.append(KM[i-2])
                        KM_y_lower.append(KM_lower[i-2])
                        KM_y_upper.append(KM_upper[i-2])
                        KM_x.append(d[i-1])
                        KM_y.append(KM[i-1])
                        KM_y_lower.append(KM_lower[i-2])
                        KM_y_upper.append(KM_upper[i-2])
            plt.plot(KM_x,KM_y,**kwargs) #plot the main KM estimate

            #extract certain keyword arguments or specify them if they are not set. We cannot pass all kwargs to CI plots as some are not appropriate (eg. label)
            if 'alpha' in kwargs:
                CI_alpha=kwargs.get('alpha')*0.5
            else:
                CI_alpha=0.5
            if 'color' in kwargs:
                CI_color=kwargs.get('color')
            else:
                CI_color='steelblue'
            if 'linestyle' in kwargs:
                CI_linestyle=kwargs.get('linestyle')
            else:
                CI_linestyle='-'

            if plot_CI==True: #calculates and plots the confidence bounds
                KM_y_lower.append(KM_y_lower[-1])
                KM_y_upper.append(KM_y_upper[-1])
                plt.plot(KM_x,KM_y_lower,alpha=CI_alpha,color=CI_color,linestyle=CI_linestyle)
                plt.plot(KM_x, KM_y_upper,alpha=CI_alpha,color=CI_color,linestyle=CI_linestyle)
                title_text = str('Kaplan-Meier reliability estimate\n with ' + str(int(CI * 100)) + '% confidence bounds')
                if shade_CI is True:
                    plt.fill_between(KM_x,KM_y_lower,KM_y_upper,alpha=CI_alpha*0.6,color=CI_color)
            else:
                title_text = 'Kaplan-Meier reliability estimate'
            plt.xlabel('Failure units')
            plt.ylabel('Reliability')
            plt.title(title_text)
            plt.xlim([0,max(KM_x)])
            plt.ylim([0,1.1])
