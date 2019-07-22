'''
Non-parametric reliability analysis of failure data

Uses the Kaplan-Meier estimation method to calculate the reliability from failure data.
Censoring is supported and confidence bounds are provided.
The confidence bounds are calculated using the Greenwood formula with Normal approxmation, which is the same as
featured in Minitab.

inputs:
    data - list or numpy array of failure data. Automatic sorting is implemented only if censoring is not specified.
        If censoring is specified, the data must be presorted in ascending order so as to avoid any complications that
        may arise from pairing censoring codes with data.
    censoring - list or numpy array of censoring codes (0=censored, 1=failure). Only right censoring is supported.
        This is an optional input, if unspecified, all data will be treated as failures.
    show_plot - True/False. Default is True
    print_results - True/False. Default is True. Will display a pandas dataframe in the console.
    plot_CI - plots the upper and lower confidence interval
    CI - confidence interval between 0 and 1. Default is 0.95 for 95% CI.
    shade_CI - 'gradient' (default) will plot a gradient fill, 'solid' will plot a solid fill, None = CI unshaded.
    color_gradient - colormap to use if gradient shading the CI. Specify as string. Default is 'Blues'. Full list of
            supported colormaps is available at https://matplotlib.org/users/colormaps.html

outputs:
    results - dataframe of results
    KM - list of Kaplan Meier column from results dataframe. This column is the non parametric estimate of the Survival Function (reliability function).

Example Usage:
d = [3961, 4007, 4734, 5248, 6054, 7298, 7454, 10190, 16890, 17200, 23060, 27160, 28690, 37100, 38700, 40060, 45000, 45670, 49390, 53000, 67000, 69040, 69630, 72280, 77350, 78470, 91680, 105700, 106300, 131900, 150400]
c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]  # censoring codes, where censored = 0
KaplanMeier(data=d,censoring=c)

'''

#produce a kaplan meier plot of the data
class KaplanMeier:
    def __init__(self,data,censoring=None,show_plot=True,print_results=True,plot_CI=True,CI=0.95,shade_CI='gradient',color_gradient='Blues',**kwargs):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as ss
        np.seterr(divide='ignore') #divide by zero occurs if last detapoint is a failure so risk set is zero

        if type(data)==np.ndarray:
            d = list(data)
        elif type(data)==list:
            d = data
        else:
            raise TypeError('Incorrect type for data. Data must be numpy array or list.')

        if d != sorted(d):
            if censoring==None: #automatic sorting is done only if there is no censoring
                d = sorted(d)
            else:
                raise ValueError('Data input must be sorted in ascending order. Due to complications that may arise from pairing censoring codes with data, automatic sorting is not implemented if censoring is specified.')

        if censoring==None:
            censoring = list(np.ones(len(d)))

        if type(censoring)== np.ndarray:
            c = list(censoring)
        elif type(censoring)==list:
            c = censoring
        else:
            raise TypeError('Incorrect type for censoring. Censoring must be numpy array or list of same length as data.')

        if len(c)!= len(d):
            raise ValueError('Length of censoring array does not match length of data array.')
        if min(c)<0 or max(c)>1:
            raise ValueError('Censoring array must contain only the numbers 0 and 1. Use 0 for right censored failures and 1 for actual failures. If all data is actual failures then do not specify the censoring array.')

        if CI<0 or CI>1:
            raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals.')

        colormap= plt.get_cmap(color_gradient)

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

        #assemble the pandas dataframe
        DATA = {'Failure times': data,
                'Censoring code (censored=0)': c,
                'Items remaining': remaining_array,
                'Kaplan Meier Estimate': KM,
                'Lower CI bound': KM_lower,
                'Upper CI bound': KM_upper}
        df = pd.DataFrame(DATA,columns=['Failure times','Censoring code (censored=0)','Items remaining','Kaplan Meier Estimate','Lower CI bound','Upper CI bound'])
        df2 = df.set_index('Failure times')
        pd.set_option('display.width', 200)  # prevents wrapping after default 80 characters
        pd.set_option('display.max_columns', 9)  # shows the dataframe without ... truncation
        self.results = df2
        self.KM = KM

        if print_results==True:
            print(df2) #this will print the pandas dataframe
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

                if shade_CI in ['gradient',True]: # provides the gradient fill
                    idx = []
                    for i in range(len(KM_x)):
                        if i<len(KM_x)-1:
                            if KM_x[i]==KM_x[i+1]:
                                idx.append(i) # list of indexes for failures. Used in the gradient fill
                    for i,x in enumerate(idx):
                        try:
                            box_upper = [KM_x[x],KM_x[idx[i+1]],KM_y[x+1],KM_y_upper[x+1]]
                            box_lower = [KM_x[x],KM_x[idx[i+1]],KM_y[x+1],KM_y_lower[x+1]]
                            plt.imshow([[0, 0], [1, 1]], extent=box_upper, interpolation='bicubic', cmap=colormap,alpha=CI_alpha, vmin=-0.3, aspect='auto')
                            plt.imshow([[0, 0], [1, 1]], extent=box_lower, interpolation='bicubic', cmap=colormap,alpha=CI_alpha, vmin=-0.3, aspect='auto')
                        except: #the last box is treated differently due to index out of bounds
                            box_upper = [KM_x[x], KM_x[-1], KM_y[x+1], KM_y_upper[x+1]] #used for last shade rectangle
                            box_lower = [KM_x[x], KM_x[-1], KM_y[x+1], KM_y_lower[x+1]]
                            if KM_x[x] != KM_x[-1]: #prevents a warning. If the last datapoint is a failure then the box has no height and matplotlib gives a warning
                                plt.imshow([[0, 0], [1, 1]], extent=box_upper, interpolation='bicubic', cmap=colormap, alpha=CI_alpha,vmin=-0.3, aspect='auto')
                                plt.imshow([[0, 0], [1, 1]], extent=box_lower, interpolation='bicubic', cmap=colormap,alpha=CI_alpha,vmin=-0.3,aspect='auto')
                elif shade_CI=='solid':
                    plt.fill_between(KM_x,KM_y_lower,KM_y_upper,alpha=CI_alpha*0.6,color=CI_color)
                elif shade_CI in ['none','None',None,False]:
                    pass
                else:
                    raise ValueError('Invalid value for shade_CI. Must be either gradient, solid, or None.')

            plt.xlabel('Failure units')
            plt.ylabel('Reliability')
            if plot_CI==False:
                title_text = 'Kaplan-Meier reliability estimate'
            else:
                title_text = str('Kaplan-Meier reliability estimate\n with '+str(int(CI*100))+'% confidence bounds')
            plt.title(title_text)
            plt.xlim([0,max(KM_x)])
            plt.ylim([0,1.1])
