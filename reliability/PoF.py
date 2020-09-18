'''
Physics of Failure

Within the module PoF, are the following functions:
- SN_diagram - plots an S-N diagram from stress and life data from cyclic fatigue tests.
- stress_strain_life_parameters_from_data - finds the parameters K, n, sigma_F, epsilon_f, b, c, given data for stress, strain, and cycles
- stress_strain_diagram - plots the stress-strain diagram and hysteresis loop using model parameters
- strain_life_diagram - plots the strain-life diagram and finds fatigue life using model parameters and specified stress or strain
- creep_rupture_curves - plots the creep rupture curves for a given set of creep data (times, temps, and stresses)
- creep_failure_time - finds the time to failure at a higher temperature for a given lower temperature and lower failure time while keeping stress constant
- fracture_mechanics_crack_initiation - finds the cycles to initiate a crack
- fracture_mechanics_crack_growth - finds the cycles to grow a crack to failure (or a specified length) from an initial length
- palmgren_miner_linear_damage - uses the Palmgren-Miner linear damage model to find various damage and life outputs.
- acceleration_factor - Given T_use and two out of the three values for AF, T_acc, Ea, it will find the third value.
'''

import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.stats as ss
from scipy.optimize import fsolve


def SN_diagram(stress, cycles, stress_runout=None, cycles_runout=None, xscale='log', stress_trace=None, cycles_trace=None, show_endurance_limit=None, method_for_bounds='statistical', CI=0.95, **kwargs):
    '''
    This function will plot the stress vs number of cycles (S-N) diagram when supplied with data from a series of fatigue tests.

    Inputs:
    stress - an array or list of stress values at failure
    cycles - an array or list of cycles values at failure
    stress_runout - an array or list of stress values that did not result in failure Optional
    cycles_runout - an array or list of cycles values that did not result in failure. Optional
    xscale - 'log' or 'linear'. Default is 'log'.
    stress_trace - an array or list of stress values to be traced across to cycles values.
    cycles_trace - an array or list of cycles values to be traced across to stress values.
    show_endurance_limit - This will adjust all lines of best fit to be greater than or equal to the average stress_runout. Defaults to False if stress_runout is not specified. Defaults to True if stress_runout is specified.
    method_for_bounds - 'statistical', 'residual', or None. Defaults to 'statistical'. If set to 'statistical' the CI value is used, otherwise it is not used for the 'residual' method. Residual uses the maximum residual datapoint for symmetric bounds. Setting the method for bounds to None will turn off the confidence bounds.
    CI - Must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if method_for_bounds = 'statistical'
    Other plotting keywords (eg. color, linestyle, etc) are accepted for the line of best fit.

    Outputs:
    The plot is the only output. All calculated values are shown on the plot.

    Example usage:
    stress = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    stress_runout = [210, 210, 205, 205, 205]
    cycles_runout = [10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7]
    SN_diagram(stress=stress, cycles=cycles, stress_runout=stress_runout, cycles_runout=cycles_runout,method_for_bounds='residual',cycles_trace=[5 * 10 ** 5], stress_trace=[260])
    plt.show()
    '''

    # error checking of input and changing inputs to arrays
    if type(stress) == np.ndarray:
        pass
    elif type(stress) == list:
        stress = np.array(stress)
    else:
        raise ValueError('stress must be an array or list')
    if type(cycles) == np.ndarray:
        pass
    elif type(cycles) == list:
        cycles = np.array(cycles)
    else:
        raise ValueError('cycles must be an array or list')
    if len(cycles) != len(stress):
        raise ValueError('the number of datapoints for stress and cycles must be equal')

    if stress_runout is not None and cycles_runout is not None:
        if len(cycles_runout) != len(stress_runout):
            raise ValueError('the number of datapoints for stress_runout and cycles_runout must be equal')
        if type(stress_runout) == np.ndarray:
            pass
        elif type(stress_runout) == list:
            stress_runout = np.array(stress_runout)
        else:
            raise ValueError('stress_runout must be an array or list')
        if type(cycles_runout) == np.ndarray:
            pass
        elif type(cycles_runout) == list:
            cycles_runout = np.array(cycles_runout)
        else:
            raise ValueError('cycles_runout must be an array or list')

    if method_for_bounds not in ['statistical', 'residual', None]:
        raise ValueError('method_for_bounds must be either ''statistical'',''residual'',or None (for no bounds).')

    if CI <= 0 or CI >= 1:
        raise ValueError('CI must be between 0 and 1. Default is 0.95 for 95% Confidence intervals on statistical bounds')

    if stress_runout is None and show_endurance_limit is None:
        show_endurance_limit = False
    elif stress_runout is None and show_endurance_limit is True:
        warnings.warn('Unable to show endurance limit without entries for stress_runout and cycles_runout. show_endurance_limit has been changed to False.')
        show_endurance_limit = False
    elif stress_runout is not None and show_endurance_limit is None:
        show_endurance_limit = True

    if xscale not in ['log', 'linear']:
        raise ValueError('xscale must be ''log'' or ''linear''. Default is ''log''')

    if stress_trace is not None:
        if type(stress_trace) not in [np.ndarray, list]:
            raise ValueError('stress_trace must be an array or list. Default is None')
    if cycles_trace is not None:
        if type(cycles_trace) not in [np.ndarray, list]:
            raise ValueError('cycles_trace must be an array or list. Default is None')

    # fit the log-linear model
    log10_cycles = np.log10(cycles)
    linear_fit = np.polyfit(log10_cycles, stress, deg=1)
    m = linear_fit[0]
    c = linear_fit[1]
    xvals = np.logspace(0, max(log10_cycles) + 2, 1000)
    y = m * np.log10(xvals) + c
    y_pred = m * np.log10(cycles) + c
    residual = max(abs(y_pred - stress))  # largest difference between line of best fit and observed data.
    # this is for the plotting limits
    cycles_min_log = 10 ** (int(np.floor(np.log10(min(cycles)))))
    cycles_max_log = 10 ** (int(np.ceil(np.log10(max(cycles)))))
    ymax = max(stress) * 1.2
    ymin = min(stress) - max(stress) * 0.2

    # extract keyword arguments
    if 'label' in kwargs:
        label = kwargs.pop('label')
    else:
        label = str('$σ_a = ' + str(round(c, 3)) + ' - ' + str(round(m * -1, 3)) + r'\times log_{10}(N_f)$')
    if 'color' in kwargs:
        color = kwargs.pop('color')
    else:
        color = 'steelblue'

    if show_endurance_limit is True:
        endurance_limit = np.average(stress_runout)  # endurance limit is predicted as the average of the runout values
        y[y < endurance_limit] = endurance_limit
        y_upper = m * np.log10(xvals) + c + residual
        y_lower = m * np.log10(xvals) + c - residual
        y_upper[y_upper < endurance_limit + residual] = endurance_limit + residual
        y_lower[y_lower < endurance_limit - residual] = endurance_limit - residual

    # plot the data and lines of best fit
    plt.scatter(cycles, stress, marker='.', color='k', label='Failure data')
    plt.plot(xvals, y, label=label, color=color)

    if show_endurance_limit is True:  # this is separated from the other endurance limit calculations due to the order of entries in the legend
        plt.plot([0, max(np.hstack([cycles, cycles_runout])) * 10], [endurance_limit, endurance_limit], linestyle='--', color='orange', linewidth=1, label=str('Endurance limit = ' + str(round(endurance_limit, 2))))

    # trace the fatigue life at the specified stresses
    if stress_trace is not None:
        for stress_value in stress_trace:
            fatigue_life = 10 ** ((stress_value - c) / m)
            plt.plot([0, fatigue_life, fatigue_life], [stress_value, stress_value, 0], 'r', linewidth=0.5)
            plt.text(cycles_min_log - 1, stress_value, str(stress_value))
            plt.text(fatigue_life, ymin, str(int(fatigue_life)))

    # trace the fatigue life at the specified cycles
    if cycles_trace is not None:
        for cycles_value in cycles_trace:
            fatigue_strength = m * np.log10(cycles_value) + c
            plt.plot([cycles_value, cycles_value, 0], [0, fatigue_strength, fatigue_strength], 'b', linewidth=0.5)
            plt.text(cycles_min_log - 1, fatigue_strength, str(round(fatigue_strength, 2)))
            plt.text(cycles_value, ymin, str(int(cycles_value)))

    # set the plot limits and plot the runout data (the position for the runout data depends on the plotting limits)
    plt.gca().set_xscale(xscale)
    plt.xlabel('Number of cycles $(N_f)$')
    plt.ylabel('Stress $(σ_a)$')
    plt.title('S-N diagram')
    if xscale == 'log':
        plt.xlim([cycles_min_log, cycles_max_log])
        if stress_runout is not None:
            plt.scatter(np.ones_like(cycles_runout) * cycles_max_log, stress_runout, marker=5, color='k', label='Runout data')
    else:
        plt.xlim([0, max(cycles) * 1.2])
        if stress_runout is not None:
            plt.scatter(np.ones_like(cycles_runout) * max(cycles) * 1.2, stress_runout, marker=5, color='k', label='Runout data')
    plt.ylim([ymin, ymax])

    # upper and lower bounds
    if method_for_bounds == 'residual':
        plt.plot(xvals, y_upper, color=color, alpha=0.7, linestyle='--', label=str('Max residual bounds (±' + str(round(residual, 2)) + ')'))
        plt.plot(xvals, y_lower, color=color, alpha=0.7, linestyle='--')
    elif method_for_bounds == 'statistical':
        x_av = np.average(cycles)
        n = len(cycles)
        y_pred = m * log10_cycles + c
        STEYX = (((stress - y_pred) ** 2).sum() / (n - 2)) ** 0.5
        tinv = ss.t.ppf((CI + 1) / 2, n - 2)
        DEVSQ = ((cycles - x_av) ** 2).sum()
        CL = tinv * STEYX * (1 / n + (xvals - x_av) ** 2 / DEVSQ) ** 0.5
        y_upper_CI = y + CL
        y_lower_CI = y - CL
        plt.plot(xvals, y_lower_CI, color=color, linestyle='--', alpha=0.7)
        plt.plot(xvals, y_upper_CI, color=color, linestyle='--', alpha=0.7, label='Statistical bounds (95% CI)')
    plt.legend(loc='upper right')
    plt.subplots_adjust(top=0.9, bottom=0.135, left=0.12, right=0.93, hspace=0.2, wspace=0.2)


class stress_strain_life_parameters_from_data:
    '''
    This function will use stress and strain data to calculate the stress-strain parameters: K, n.
    If cycles is provided it will also produce the strain-life parameters: sigma_f, epsilon_f, b, c.
    You cannot find the strain-life parameters without stress as we use stress to find elastic strain.

    Note: If you already have the parameters K, n, sigma_f, epsilon_f, b, c, then you can use the function 'stress_strain_diagram'

    Inputs:
    strain - an array or list of strain
    stress - an array or list of stress
    E - The modulus of elasticity. Ensure this is in the same units as stress (typically MPa)
    cycles - the number of cycles to failure. Optional input. Required if you want to obtain the parameters sigma_f, epsilon_f, b, c
    print_results - True/False. Default is True.
    show_plot - True/False. Default is True.

    Outputs:
    The stress-strain plot will a be generated if show_plot is True. Use plt.show() to show it.
    The results will be printed in the console if print_results is True.
    K - the cyclic strength coefficient
    n - the cyclic strain hardening exponent
    sigma_f - the fatigue strength coefficient. Not generated if cycles is not provided.
    epsilon_f - the fatigue strain coefficient. Not generated if cycles is not provided.
    b - the elastic strain exponent. Not generated if cycles is not provided.
    c - the plastic strain exponent. Not generated if cycles is not provided.

    '''

    def __init__(self, strain, stress, E, cycles=None, print_results=True, show_plot=True):

        # note that strain is the total strain (elastic strain + plastic strain)
        if type(stress) == np.ndarray:
            pass
        elif type(stress) == list:
            stress = np.array(stress)
        else:
            raise ValueError('stress must be an array or list')
        if type(strain) == np.ndarray:
            pass
        elif type(strain) == list:
            strain = np.array(strain)
        else:
            raise ValueError('strain must be an array or list')
        if cycles is not None:
            if type(cycles) == np.ndarray:
                cycles_2Nf = 2 * cycles
            elif type(cycles) == list:
                cycles_2Nf = 2 * np.array(cycles)
            else:
                raise ValueError('cycles must be an array or list')

        # fit the ramberg osgood relationship to the data
        elastic_strain = stress / E
        plastic_strain = strain - elastic_strain
        fit = np.polyfit(np.log10(plastic_strain), np.log10(stress), deg=1)
        self.K = 10 ** fit[1]
        self.n = fit[0]

        # plot the data and fitted curve
        if show_plot is True:
            plt.figure()
            stress_array = np.linspace(0, max(stress), 10000)
            strain_array = stress_array / E + (stress_array / self.K) ** (1 / self.n)
            plt.plot(strain_array, stress_array, color='red', label=str(r'$\epsilon_{tot} = \frac{\sigma}{' + str(round(E, 4)) + r'} + (\frac{\sigma}{' + str(round(self.K, 4)) + r'})^{\frac{1}{' + str(round(self.n, 4)) + '}}$'))
            plt.scatter(strain, stress, marker='.', color='k', label='Stress-Strain data')
            plt.xlabel('Strain $(\epsilon_{tot})$')
            plt.ylabel('Stress $(\sigma)$')
            plt.title('Stress-Strain diagram')
            xmax = max(strain) * 1.2
            ymax = max(stress) * 1.2
            plt.xlim([0, xmax])
            plt.ylim([0, ymax])
            plt.grid(True)
            leg = plt.legend()
            # this is to make the first legend entry (the equation) bigger
            legend_texts = leg.get_texts()
            legend_texts[0]._fontproperties = legend_texts[1]._fontproperties.copy()
            legend_texts[0].set_size(15)

        # STRAIN-LIFE
        if cycles is not None:
            fit2 = np.polyfit(np.log10(cycles_2Nf), np.log10(elastic_strain), deg=1)
            self.sigma_f = E * 10 ** fit2[1]
            self.b = fit2[0]

            fit3 = np.polyfit(np.log10(cycles_2Nf), np.log10(plastic_strain), deg=1)
            self.epsilon_f = 10 ** fit3[1]
            self.c = fit3[0]

            # STRAIN-LIFE plot
            if show_plot is True:
                plt.figure()
                cycles_2Nt = (self.epsilon_f * E / self.sigma_f) ** (1 / (self.b - self.c))
                plt.scatter(cycles_2Nf, strain, marker='.', color='k', label='Stress-Life data')
                cycles_2Nf_array = np.logspace(1, 8, 1000)
                epsilon_total = (self.sigma_f / E) * cycles_2Nf_array ** self.b + self.epsilon_f * cycles_2Nf_array ** self.c
                epsilon_total_at_cycles_2Nt = (self.sigma_f / E) * cycles_2Nt ** self.b + self.epsilon_f * cycles_2Nt ** self.c
                plt.loglog(cycles_2Nf_array, epsilon_total, color='red', alpha=0.8, label=str(r'$\epsilon_{tot} = \frac{' + str(round(self.sigma_f, 4)) + '}{' + str(round(E, 4)) + '}(2N_f)^{' + str(round(self.b, 4)) + '} + ' + str(round(self.epsilon_f, 4)) + '(2N_f)^{' + str(round(self.c, 4)) + '}$'))
                plt.plot([cycles_2Nt, cycles_2Nt], [10 ** -6, epsilon_total_at_cycles_2Nt], 'red', linestyle='--', alpha=0.5)
                plastic_strain_line = self.epsilon_f * cycles_2Nf_array ** self.c
                elastic_strain_line = self.sigma_f / E * cycles_2Nf_array ** self.b
                plt.plot(cycles_2Nf_array, plastic_strain_line, 'orange', alpha=0.7, label='plastic strain')
                plt.plot(cycles_2Nf_array, elastic_strain_line, 'steelblue', alpha=0.8, label='elastic strain')
                plt.xlabel('Reversals to failure $(2N_f)$')
                plt.ylabel('Strain amplitude $(\epsilon_a)$')
                plt.title('Strain-Life diagram')
                cycles_min_log = 10 ** (int(np.floor(np.log10(min(cycles_2Nf)))) - 1)
                cycles_max_log = 10 ** (int(np.ceil(np.log10(max(cycles_2Nf)))) + 1)
                strain_min_log = 10 ** (int(np.floor(np.log10(min(strain)))) - 1)
                strain_max_log = 10 ** (int(np.ceil(np.log10(max(strain)))) + 1)
                plt.text(cycles_2Nt, strain_min_log, str('$2N_t = $' + str(int(cycles_2Nt))), verticalalignment='bottom')
                plt.xlim(cycles_min_log, cycles_max_log)
                plt.ylim(strain_min_log, strain_max_log)
                plt.grid(True)
                leg2 = plt.legend()
                # this is to make the first legend entry (the equation) bigger
                legend_texts2 = leg2.get_texts()
                legend_texts2[0]._fontproperties = legend_texts2[1]._fontproperties.copy()
                legend_texts2[0].set_size(13)

        if print_results is True:
            print('K (cyclic strength coefficient):', self.K)
            print('n (strain hardening exponent):', self.n)
            if cycles is not None:
                print('sigma_f (fatigue strength coefficient):', self.sigma_f)
                print('epsilon_f (fatigue strain coefficient):', self.epsilon_f)
                print('b (elastic strain exponent):', self.b)
                print('c (plastic strain exponent):', self.c)


class stress_strain_diagram:
    '''
    This function plots the stress-strain diagram.

    Note: If you do not have the parameters K, n, but you do have stress and strain data then you can use the function 'stress_strain_life_parameters_from_data'

    Inputs:
    K - cyclic strength coefficient
    n - strain hardening exponent
    E - The modulus of elasticity. Ensure this is in the same units for which K and n were obtained (typically MPa)
    max_strain - the maximum strain to use for cyclic loading when plotting the hysteresis loop.
    max_stress - the maximum stress to use for cyclic loading when plotting the hysteresis loop.
    min_strain - if this is not -max_strain then specify it here. Optional input.
    min_stress - if this is not -max_stress then specify it here. Optional input.
     *When specifying min and max stress or strain, Do not specify both stress and strain as the corresponding value will be automatically calculated. Only specify the min if it is not -max
    initial_load_direction - 'tension' or 'compression'. Default is tension.

    Outputs:
    The stress-strain plot will always be generated. Use plt.show() to show it.

    '''

    def __init__(self, K, n, E, max_strain=None, max_stress=None, min_stress=None, min_strain=None, print_results=True, initial_load_direction='tension'):

        if max_stress is not None and max_strain is not None:
            raise ValueError('Do not specify both max_stress and max_strain as the corresponding value will be automatically calculated')
        if min_stress is not None and min_strain is not None:
            raise ValueError('Do not specify both min_stress and min_strain as the corresponding value will be automatically calculated')
        if max_stress is None and max_strain is None:
            raise ValueError('You must specify either max_stress OR max_strain for the cyclic loading')
        if initial_load_direction not in ['tension', 'compression']:
            raise ValueError('initial_load_direction must be either tension or compression. Default is tension.')

        self.K = K
        self.n = n

        warnings.filterwarnings('ignore')  # sometimes fsolve has issues when delta_sigma crosses zero. It almost always resolves itself so the warning is just an annoyance

        # these functions are used for solving the equation for sigma_max as it can not be rearranged
        def ramberg_osgood(epsilon, sigma, E, K, n):
            return (sigma / E) + ((sigma / K) ** (1 / n)) - epsilon

        def ramberg_osgood_delta(delta_epsilon, delta_sigma, E, K, n):
            return (delta_sigma / E) + 2 * (delta_sigma / (2 * K)) ** (1 / n) - delta_epsilon

        if max_strain is None:
            self.max_strain = (max_stress / E) + ((max_stress / K) ** (1 / n))
        else:
            self.max_strain = max_strain

        if max_stress is not None:  # we have stress. Need to find strain
            self.max_stress = max_stress
            self.max_strain = max_stress / E + (max_stress / K) ** (1 / n)
            if min_stress is None:
                self.min_stress = -self.max_stress
                self.min_strain = -self.max_strain
            else:
                self.min_stress = min_stress
                self.min_strain = -(-min_stress / E + (-min_stress / K) ** (1 / n))
        else:  # we have strain. Don't need stress as it is found later
            self.max_strain = max_strain
            if min_strain is None:
                self.min_strain = -self.max_strain
            else:
                self.min_strain = min_strain
        strain_range = self.max_strain - self.min_strain

        # initial loading
        if initial_load_direction == 'tension':
            strain_array1 = np.linspace(0, self.max_strain, 1000)
        else:
            strain_array1 = np.linspace(0, self.min_strain, 1000)
        stress_array1 = []
        sigma = 10  # initial guess for fsolve which get updated once the first value is found
        for epsilon_1 in strain_array1:
            fun = lambda x: ramberg_osgood(epsilon=epsilon_1, sigma=x, E=E, K=self.K, n=self.n)  # lgtm [py/loop-variable-capture]
            result = fsolve(fun, np.array(sigma))
            sigma = result[0]
            stress_array1.append(sigma)

        # first reversal
        strain_delta_array2 = np.linspace(0, strain_range, 1000)
        if initial_load_direction == 'tension':
            strain_array2 = np.linspace(self.max_strain, self.min_strain, 1000)
        else:
            strain_array2 = np.linspace(self.min_strain, self.max_strain, 1000)
        stress_array2 = []
        initial_stress = stress_array1[-1]
        delta_sigma = 10  # initial guess for fsolve which get updated once the first value is found
        for delta_epsilon_2 in strain_delta_array2:
            fun_delta = lambda x: ramberg_osgood_delta(delta_epsilon=delta_epsilon_2, delta_sigma=x, E=E, K=self.K, n=self.n)  # lgtm [py/loop-variable-capture]
            result2 = fsolve(fun_delta, np.array(delta_sigma))
            delta_sigma = result2[0]
            if initial_load_direction == 'tension':
                current_stress = initial_stress - delta_sigma
            else:
                current_stress = initial_stress + delta_sigma
            stress_array2.append(current_stress)

        # second reversal
        strain_delta_array3 = np.linspace(0, strain_range, 1000)
        if initial_load_direction == 'tension':
            strain_array3 = np.linspace(self.min_strain, self.max_strain, 1000)
        else:
            strain_array3 = np.linspace(self.max_strain, self.min_strain, 1000)
        stress_array3 = []
        initial_stress = stress_array2[-1]
        for delta_epsilon_3 in strain_delta_array3:
            fun_delta = lambda x: ramberg_osgood_delta(delta_epsilon=delta_epsilon_3, delta_sigma=x, E=E, K=self.K, n=self.n)  # lgtm [py/loop-variable-capture]
            result3 = fsolve(fun_delta, np.array(delta_sigma))
            delta_sigma = result3[0]
            if initial_load_direction == 'tension':
                current_stress = initial_stress + delta_sigma
            else:
                current_stress = initial_stress - delta_sigma
            stress_array3.append(current_stress)

        # plot the initial loading and hysteresis loops
        plt.plot(strain_array1, stress_array1, color='red', label='Initial loading')
        plt.plot(strain_array2, stress_array2, color='steelblue', label='Hysteresis loop from subsequent load cycles')
        plt.plot(strain_array3, stress_array3, color='steelblue')

        # add text to provide the coordinates of the tips of the hysteresis loop
        if initial_load_direction == 'tension':
            left_x = strain_array2[-1]
            left_y = stress_array2[-1]
            right_x = strain_array3[-1]
            right_y = stress_array3[-1]
        else:
            left_x = strain_array3[-1]
            left_y = stress_array3[-1]
            right_x = strain_array2[-1]
            right_y = stress_array2[-1]
        plt.text(left_x, left_y, str('(' + str(round(left_x, 4)) + ',' + str(round(left_y, 3)) + ')'), verticalalignment='top', horizontalalignment='right')
        plt.text(right_x, right_y, str('(' + str(round(right_x, 4)) + ',' + str(round(right_y, 3)) + ')'), verticalalignment='top', horizontalalignment='left')

        if initial_load_direction == 'tension':
            self.max_stress = max(stress_array1)
            self.min_stress = min(stress_array2)
        else:
            self.max_stress = max(stress_array2)
            self.min_stress = min(stress_array1)

        plt.xlabel('Strain $(\epsilon_{tot})$')
        plt.ylabel('Stress $(\sigma)$')
        plt.title('Stress-Strain diagram')
        xmax = self.max_strain * 2
        ymax = self.max_stress * 1.4
        xmin = self.min_strain * 2
        ymin = self.min_stress * 1.4
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.grid(True)
        plt.plot([xmin, xmax], [0, 0], 'k', linewidth=1)
        plt.plot([0, 0], [ymin, ymax], 'k', linewidth=1)
        plt.legend(loc='upper left')
        plt.gcf().set_size_inches(10, 7)

        if print_results is True:
            print('Max stress:', self.max_stress)
            print('Min stress:', self.min_stress)
            print('Max strain:', self.max_strain)
            print('Min strain:', self.min_strain)


class strain_life_diagram:
    '''
    This function plots the strain-life diagram.

    Note: If you do not have the parameters sigma_f, epsilon_f, b, c, but you do have stress, strain, and cycles data then you can use the function 'stress_strain_life_parameters_from_data'

    Inputs:
    E - The modulus of elasticity. Ensure this is in the same units for which K and n were obtained (typically MPa)
    sigma_f - fatigue strength coefficient
    epsilon_f - fatigue strain coefficient
    b - elastic strain exponent
    c - plastic strain exponent
    K - cyclic strength coefficient. Optional input. Only required if you specify max_stress or max_strain.
    n - strain hardening exponent. Optional input. Only required if you specify max_stress or max_strain.
    mean_stress_correction_method - must be either 'morrow','modified_morrow', or 'SWT'. Default is 'SWT'. Only used if mean_stress is found to be non-zero.
    max_stress - specify the max_stress if you want cycles to failure. If specified, you will need to also specify K and n.
    max_strain - specify the max_strain if you want cycles to failure.
    min_stress - if this is not -max_stress then specify it here. Optional input.
    min_strain - if this is not -max_strain then specify it here. Optional input.
     *When specifying min and max stress or strain, Do not specify both stress and strain as the corresponding value will be automatically calculated. Only specify the min if it is not -max
    print_results - True/False. Defaults to True. If use_level_stress or use_level_strain is specified then the printed results will be the cycles_to_failure
    show_plot - True/False. Default is True

    Outputs:
    The strain-life plot will be generated if show_plot = True. Use plt.show() to show it.
    cycles_to_failure - only calculated if max_stress OR max_strain is specified. This will be printed if print_results = True.

    '''

    def __init__(self, E, sigma_f, epsilon_f, b, c, K=None, n=None, mean_stress_correction_method='SWT', max_strain=None, min_strain=None, max_stress=None, min_stress=None, print_results=True, show_plot=True):
        if max_stress is not None and max_strain is not None:
            raise ValueError('Do not specify both max_stress and max_strain as the corresponding value will be automatically calculated')
        if min_stress is not None and min_strain is not None:
            raise ValueError('Do not specify both min_stress and min_strain as the corresponding value will be automatically calculated')
        if (max_stress is not None or max_strain is not None) and (K is None or n is None):
            raise ValueError('K and n must be specified if you specify max_stress or max_strain. These values are required to calculate the corresponding stress or strain')
        if mean_stress_correction_method not in ['morrow', 'modified_morrow', 'SWT']:
            raise ValueError('mean_stress_correction_method must be either ''morrow'',''modified_morrow'',or ''SWT''.')

        self.sigma_f = sigma_f
        self.b = b
        self.epsilon_f = epsilon_f
        self.c = c

        if max_strain is not None or max_stress is not None:
            if max_stress is not None:  # we have stress. Need to find strain
                self.max_stress = max_stress
                self.max_strain = max_stress / E + (max_stress / K) ** (1 / n)
                if min_stress is None:
                    self.min_stress = -self.max_stress
                    self.min_strain = -self.max_strain
                else:
                    self.min_stress = min_stress
                    self.min_strain = min_stress / E + (min_stress / K) ** (1 / n)
            else:  # we have strain. Need to find stress
                self.max_strain = max_strain
                if min_strain is None:
                    self.min_strain = -self.max_strain
                else:
                    self.min_strain = min_strain

                # need to solve for stress iteratively
                def ramberg_osgood(epsilon, sigma, E, K, n):
                    return (sigma / E) + ((sigma / K) ** (1 / n)) - epsilon

                fun_max_strain = lambda x: ramberg_osgood(epsilon=self.max_strain, sigma=x, E=E, K=K, n=n)
                fun_min_strain = lambda x: ramberg_osgood(epsilon=self.min_strain, sigma=x, E=E, K=K, n=n)
                self.max_stress = fsolve(fun_max_strain, np.array(100))
                self.min_stress = fsolve(fun_min_strain, np.array(-100))

            mean_stress = (self.min_stress + (self.max_stress - self.min_stress) / 2)[0]
            delta_epsilon_half = (self.max_strain - self.min_strain) / 2

            # solve for number of cycles and the plot parameters
            cycles_2Nf_array = np.logspace(1, 8, 1000)
            if mean_stress == 0:
                print('here')

                def coffin_manson(eps_tot, sigma_f, E, cycles_2Nf, epsilon_f, b, c):
                    return sigma_f / E * cycles_2Nf ** b + epsilon_f * cycles_2Nf ** c - eps_tot

                fun_cm = lambda x: coffin_manson(eps_tot=self.max_strain, sigma_f=self.sigma_f, E=E, cycles_2Nf=x, epsilon_f=self.epsilon_f, b=self.b, c=self.c)
                use_cycles_2Nf = fsolve(fun_cm, np.array(10))
                cycles_2Nt = (self.epsilon_f * E / self.sigma_f) ** (1 / (self.b - self.c))
                epsilon_total = (self.sigma_f / E) * cycles_2Nf_array ** self.b + self.epsilon_f * cycles_2Nf_array ** self.c
                epsilon_total_at_cycles_2Nt = (self.sigma_f / E) * cycles_2Nt ** self.b + self.epsilon_f * cycles_2Nt ** self.c
                equation_str = str(r'$\epsilon_{tot} = \frac{' + str(round(self.sigma_f, 4)) + '}{' + str(round(E, 4)) + '}(2N_f)^{' + str(round(self.b, 4)) + '} + ' + str(round(self.epsilon_f, 4)) + '(2N_f)^{' + str(round(self.c, 4)) + '}$')
                plastic_strain_line = self.epsilon_f * cycles_2Nf_array ** self.c
                elastic_strain_line = self.sigma_f / E * cycles_2Nf_array ** self.b
            else:
                if mean_stress_correction_method == 'morrow':
                    def morrow(eps_tot, sigma_f, sigma_mean, E, cycles_2Nf, epsilon_f, b, c):
                        return (sigma_f - sigma_mean) / E * cycles_2Nf ** b + epsilon_f * cycles_2Nf ** c - eps_tot

                    fun_m = lambda x: morrow(eps_tot=delta_epsilon_half, sigma_f=self.sigma_f, sigma_mean=mean_stress, E=E, cycles_2Nf=x, epsilon_f=self.epsilon_f, b=self.b, c=self.c)
                    use_cycles_2Nf = fsolve(fun_m, np.array(10))
                    cycles_2Nt = (self.epsilon_f * E / (self.sigma_f - mean_stress)) ** (1 / (self.b - self.c))
                    epsilon_total = ((self.sigma_f - mean_stress) / E) * cycles_2Nf_array ** self.b + self.epsilon_f * cycles_2Nf_array ** self.c
                    epsilon_total_at_cycles_2Nt = ((self.sigma_f - mean_stress) / E) * cycles_2Nt ** self.b + self.epsilon_f * cycles_2Nt ** self.c
                    equation_str = str(r'$\epsilon_{tot} = \frac{' + str(round(self.sigma_f, 4)) + '-' + str(round(mean_stress, 4)) + '}{' + str(round(E, 4)) + '}(2N_f)^{' + str(round(self.b, 4)) + '} + ' + str(round(self.epsilon_f, 4)) + '(2N_f)^{' + str(round(self.c, 4)) + '}$')
                    plastic_strain_line = self.epsilon_f * cycles_2Nf_array ** self.c
                    elastic_strain_line = ((self.sigma_f - mean_stress) / E) * cycles_2Nf_array ** self.b

                elif mean_stress_correction_method == 'modified_morrow':
                    def modified_morrow(eps_tot, sigma_f, sigma_mean, E, cycles_2Nf, epsilon_f, b, c):
                        return (sigma_f - sigma_mean) / E * cycles_2Nf ** b + epsilon_f * ((sigma_f - sigma_mean) / sigma_f) ** (c / b) * cycles_2Nf ** c - eps_tot

                    fun_mm = lambda x: modified_morrow(eps_tot=delta_epsilon_half, sigma_f=self.sigma_f, sigma_mean=mean_stress, E=E, cycles_2Nf=x, epsilon_f=self.epsilon_f, b=self.b, c=self.c)
                    use_cycles_2Nf = fsolve(fun_mm, np.array(10))
                    cycles_2Nt = (self.epsilon_f * E * ((self.sigma_f - mean_stress) / self.sigma_f) ** (self.c / self.b) / (self.sigma_f - mean_stress)) ** (1 / (self.b - self.c))
                    epsilon_total = ((self.sigma_f - mean_stress) / E) * cycles_2Nf_array ** self.b + self.epsilon_f * ((self.sigma_f - mean_stress) / self.sigma_f) ** (self.c / self.b) * cycles_2Nf_array ** self.c
                    epsilon_total_at_cycles_2Nt = ((self.sigma_f - mean_stress) / E) * cycles_2Nt ** self.b + self.epsilon_f * ((self.sigma_f - mean_stress) / self.sigma_f) ** (self.c / self.b) * cycles_2Nt ** self.c
                    equation_str = str(r'$\epsilon_{tot} = \frac{' + str(round(self.sigma_f, 4)) + '-' + str(round(mean_stress, 4)) + '}{' + str(round(E, 4)) + '}(2N_f)^{' + str(round(self.b, 4)) + '} + ' + str(round(self.epsilon_f, 4)) + r'(\frac{' + str(round(self.sigma_f, 4)) + '-' + str(round(mean_stress, 4)) + '}{' + str(round(self.sigma_f, 4)) + '})^' + r'\frac{' + str(self.c) + '}{' + str(self.b) + '}' + '(2N_f)^{' + str(round(self.c, 4)) + '}$')
                    plastic_strain_line = self.epsilon_f * ((self.sigma_f - mean_stress) / self.sigma_f) ** (self.c / self.b) * cycles_2Nf_array ** self.c
                    elastic_strain_line = ((self.sigma_f - mean_stress) / E) * cycles_2Nf_array ** self.b
                elif mean_stress_correction_method == 'SWT':
                    def SWT(eps_tot, sigma_f, E, cycles_2Nf, epsilon_f, b, c, sigma_max):
                        return ((sigma_f ** 2) / E * cycles_2Nf ** (2 * b) + sigma_f * epsilon_f * (cycles_2Nf) ** (b + c)) / sigma_max - eps_tot

                    fun_swt = lambda x: SWT(eps_tot=delta_epsilon_half, sigma_f=self.sigma_f, E=E, cycles_2Nf=x, epsilon_f=self.epsilon_f, b=self.b, c=self.c, sigma_max=self.max_stress)
                    use_cycles_2Nf = fsolve(fun_swt, np.array(10))
                    cycles_2Nt = (self.epsilon_f * E / self.sigma_f) ** (1 / (self.b - self.c))
                    epsilon_total = ((self.sigma_f ** 2 / E) * cycles_2Nf_array ** (2 * self.b) + self.sigma_f * self.epsilon_f * cycles_2Nf_array ** (self.b + self.c)) / self.max_stress
                    epsilon_total_at_cycles_2Nt = ((self.sigma_f ** 2 / E) * cycles_2Nt ** (2 * self.b) + self.sigma_f * self.epsilon_f * cycles_2Nt ** (self.b + self.c)) / self.max_stress
                    equation_str = str(r'$\epsilon_{tot} = \frac{1}{' + str(round(self.max_stress[0], 4)) + '}' + r'(\frac{' + str(round(self.sigma_f, 4)) + '^2}{' + str(round(E, 4)) + '}(2N_f)^{(' + str(round(self.b, 4)) + '×2)} + ' + str(round(self.sigma_f, 4)) + '×' + str(round(self.epsilon_f, 4)) + '(2N_f)^{(' + str(round(self.b, 4)) + '+' + str(round(self.c, 4)) + ')})$')
                    plastic_strain_line = (self.sigma_f * self.epsilon_f * cycles_2Nf_array ** (self.b + self.c)) / self.max_stress
                    elastic_strain_line = ((self.sigma_f ** 2 / E) * cycles_2Nf_array ** (2 * self.b)) / self.max_stress

            self.cycles_to_failure = use_cycles_2Nf[0] / 2

            if print_results is True:
                if max_strain is not None:
                    print('Failure will occur in', round(self.cycles_to_failure, 2), 'cycles', str('(' + str(round(self.cycles_to_failure * 2, 2)) + ' reversals).'))
                else:
                    print('Failure will occur in', round(self.cycles_to_failure, 2), 'cycles', str('(' + str(round(self.cycles_to_failure * 2, 2)) + ' reversals).'))

            if show_plot is True:
                strain_amplitude = (self.max_strain - self.min_strain) / 2
                plt.loglog(cycles_2Nf_array, epsilon_total, color='red', alpha=0.8, label=equation_str)
                plt.plot([cycles_2Nt, cycles_2Nt], [10 ** -6, epsilon_total_at_cycles_2Nt], 'red', linestyle='--', alpha=0.5)
                plt.plot(cycles_2Nf_array, plastic_strain_line, 'orange', alpha=0.7, label='plastic strain')
                plt.plot(cycles_2Nf_array, elastic_strain_line, 'steelblue', alpha=0.8, label='elastic strain')
                plt.xlabel('Reversals to failure $(2N_f)$')
                plt.ylabel('Strain amplitude $(\epsilon_a)$')
                plt.title('Strain-Life diagram')
                plt.xlim(min(cycles_2Nf_array), max(cycles_2Nf_array))
                strain_min_log = 10 ** (int(np.floor(np.log10(min(plastic_strain_line)))))
                strain_max_log = 10 ** (int(np.ceil(np.log10(max(epsilon_total)))))
                plt.ylim(strain_min_log, strain_max_log)
                plt.scatter([self.cycles_to_failure * 2], [strain_amplitude], marker='o', color='k')
                plt.plot([1, self.cycles_to_failure * 2, self.cycles_to_failure * 2], [strain_amplitude, strain_amplitude, 0], linewidth=1, linestyle='--', color='k')
                if self.cycles_to_failure * 2 < cycles_2Nt:
                    plt.text(cycles_2Nt, strain_min_log, str(' $2N_t = $' + str(int(cycles_2Nt)) + ' reversals'), verticalalignment='bottom', ha='left')
                    plt.text(self.cycles_to_failure * 2, strain_min_log, str('Life = ' + str(int(np.floor(self.cycles_to_failure * 2))) + ' reversals '), ha='right', va='bottom')
                else:
                    plt.text(cycles_2Nt, strain_min_log, str('$2N_t = $' + str(int(cycles_2Nt)) + ' reversals'), verticalalignment='bottom', ha='right')
                    plt.text(self.cycles_to_failure * 2, strain_min_log, str(' Life = ' + str(int(np.floor(self.cycles_to_failure * 2))) + ' reversals '), ha='left', va='bottom')
                plt.text(10, strain_amplitude, str(' $\epsilon_a$ = ' + str(strain_amplitude)), va='bottom', ha='left')
                plt.grid(True)
                leg2 = plt.legend(loc='upper right')
                legend_texts2 = leg2.get_texts()  # this is to make the first legend entry (the equation) bigger
                legend_texts2[0]._fontproperties = legend_texts2[1]._fontproperties.copy()
                legend_texts2[0].set_size(13)
                plt.gcf().set_size_inches(10, 7)
        else:  # this is in the case that max stress or strain was not supplied
            if show_plot is True:
                cycles_2Nt = (self.epsilon_f * E / self.sigma_f) ** (1 / (self.b - self.c))
                cycles_2Nf_array = np.logspace(1, 8, 1000)
                epsilon_total = (self.sigma_f / E) * cycles_2Nf_array ** self.b + self.epsilon_f * cycles_2Nf_array ** self.c
                epsilon_total_at_cycles_2Nt = (self.sigma_f / E) * cycles_2Nt ** self.b + self.epsilon_f * cycles_2Nt ** self.c
                plt.loglog(cycles_2Nf_array, epsilon_total, color='red', alpha=0.8, label=str(r'$\epsilon_{tot} = \frac{' + str(round(self.sigma_f, 4)) + '}{' + str(round(E, 4)) + '}(2N_f)^{' + str(round(self.b, 4)) + '} + ' + str(round(self.epsilon_f, 4)) + '(2N_f)^{' + str(round(self.c, 4)) + '}$'))
                plt.plot([cycles_2Nt, cycles_2Nt], [10 ** -6, epsilon_total_at_cycles_2Nt], 'red', linestyle='--', alpha=0.5)
                plastic_strain_line = self.epsilon_f * cycles_2Nf_array ** self.c
                elastic_strain_line = self.sigma_f / E * cycles_2Nf_array ** self.b
                plt.plot(cycles_2Nf_array, plastic_strain_line, 'orange', alpha=0.7, label='plastic strain')
                plt.plot(cycles_2Nf_array, elastic_strain_line, 'steelblue', alpha=0.8, label='elastic strain')
                plt.xlabel('Reversals to failure $(2N_f)$')
                plt.ylabel('Strain amplitude $(\epsilon_a)$')
                plt.title('Strain-Life diagram')
                strain_min_log = 10 ** (int(np.floor(np.log10(min(plastic_strain_line)))))
                strain_max_log = 10 ** (int(np.ceil(np.log10(max(epsilon_total)))))
                plt.ylim(strain_min_log, strain_max_log)
                plt.xlim(min(cycles_2Nf_array), max(cycles_2Nf_array))
                plt.text(cycles_2Nt, strain_min_log, str('$2N_t = $' + str(int(cycles_2Nt))), verticalalignment='bottom')
                plt.grid(True)
                leg2 = plt.legend()
                # this is to make the first legend entry (the equation) bigger
                legend_texts2 = leg2.get_texts()
                legend_texts2[0]._fontproperties = legend_texts2[1]._fontproperties.copy()
                legend_texts2[0].set_size(13)
                self.cycles_to_failure = 'Not calculated. Specify max stress or strain to find cycles_to_failure'


def palmgren_miner_linear_damage(rated_life, time_at_stress, stress):
    '''
    Uses the Palmgren-Miner linear damage hypothesis to find the outputs:

    Inputs:
    - rated life - an array or list of how long the component will last at a given stress level
    - time_at_stress - an array or list of how long the component is subjected to the stress that gives the specified rated_life
    - stress - what stress the component is subjected to. Not used in the calculation but is required for printing the output.
    Ensure that the time_at_stress and rated life are in the same units as the answer will also be in those units

    Outputs:
    - Fraction of life consumed per load cycle
    - service life of the component
    - Fraction of damage caused at each stress level


    Example usage:
    Ball bearings are fail after 50000 hrs, 6500 hrs, and 1000 hrs, after being subjected to a stress of 1kN, 2kN, and 4kN respectively.
    If each load cycle involves 40 mins at 1kN, 15 mins at 2kN, and 5 mins at 4kN, how long will the ball bearings last?

    palmgren_miner_linear_damage(rated_life=[50000,6500,1000], time_at_stress=[40/60, 15/60, 5/60], stress=[1, 2, 4])
    '''
    if len(rated_life) != len(time_at_stress) or len(rated_life) != len(stress):
        raise ValueError('All inputs must be of equal length.')

    life_frac = []
    for i, x in enumerate(time_at_stress):
        life_frac.append(x / rated_life[i])
    life_consumed_per_load_cycle = sum(life_frac)
    service_life = 1 / life_consumed_per_load_cycle
    damage_frac = service_life * np.array(life_frac)
    print('Palmgren-Miner Linear Damage Model results:')
    print('Each load cycle uses', round(life_consumed_per_load_cycle * 100, 5), '% of the components life.')
    print('The service life of the component is', round(service_life, 5), 'load cycles.')
    print('The amount of damage caused at each stress level is:')
    for i, x in enumerate(stress):
        print('Stress = ', x, ', Damage fraction =', round(damage_frac[i] * 100, 5), '%.')


class fracture_mechanics_crack_initiation:
    '''
    This function uses the material properties, the local cross sectional area, and force applied to the component to determine how many cycles until crack initiation (of a 1mm crack).
    Units should always be in MPa (and mm^2 for area). This function may be used for an un-notched or notched component.
    If the component is un-notched, the parameters q and Kt may be left as their default values of 1.

    While there are formulas to find the parameters q and Kt, these formulas have not been included here so that the function is reasonably generic to different materials and geometries.
    Resources for finding some of these parameters if they are not given to you:
    q = 1/(1+a/r) Where r is the notch radius of curvature (in mm), and a is 0.025*(2070/Su). Su is the ultimate strength in MPa. This only applies to high strength steels where Su>550MPa.
    Kt ==> https://www.efatigue.com/constantamplitude/stressconcentration/  This website will provide you with the appropriate Kt for your notched geometry.

    Inputs:
    P - Force applied on the component [units of MPa]
    A - Cross sectional area of the component (at the point of crack initiation) [units of mm^2]
    Sy - Yield strength of the material [units of MPa]
    E - Elastic modulus (Young's modulus) [units of MPa]
    K - Strength coefficient of the material
    n - Strain hardening exponent of the material
    b - Elastic strain exponent of the material
    c - Plastic strain exponent of the material
    sigma_f - Fatigue strength coefficient of the material
    epsilon_f - Fatigue strain coefficient of the material
    q - Notch sensitivity factor (default is 1 for no notch)
    Kt - stress concentration factor (default is 1 for no notch)
    mean_stress_correction_method - must be either ‘morrow’, ’modified_morrow’, or ‘SWT'. Default is 'modified_morrow' as this is the same as the uncorrected Coffin-Manson relationship when mean stress is zero.

    Outputs:
    The results will be printed to the console if print_results is True
    The following results are also stored in the calculated object.
    sigma_max
    sigma_min
    sigma_mean
    epsilon_max
    epsilon_min
    epsilon_mean
    cycles_to_failure

    Example usage:
    fracture_mechanics_crack_initiation(P=0.15, A=5*80, Kt=2.41, q=0.9857, Sy=690, E=210000, K=1060, n=0.14, b=-0.081, c=-0.65, sigma_f=1160, epsilon_f=1.1,mean_stress_correction_method='SWT')

    '''

    def __init__(self, P, A, Sy, E, K, n, b, c, sigma_f, epsilon_f, Kt=1.0, q=1.0, mean_stress_correction_method='modified_morrow', print_results=True):
        if mean_stress_correction_method not in ['morrow', 'modified_morrow', 'SWT']:
            raise ValueError('mean_stress_correction_method must be either morrow,modified_morrow, or SWT. Default is modified_morrow.')
        S_net = 10 ** 6 * P / A
        Kf = 1 + q * (Kt - 1)
        sigma_epsilon = ((Kf * S_net) ** 2) / E
        delta_epsilon_delta_sigma = ((Kf * S_net * 2) ** 2) / E
        if Kt * S_net > Sy:  # elastic and plastic (monotonic load model)
            def ramberg_osgood(sigma_epsilon, sigma, E, K, n):
                return sigma / E + (sigma / K) ** (1 / n) - sigma_epsilon / sigma

            def massing(delta_sigma_delta_epsilon, delta_sigma, E, K, n):
                return delta_sigma / (2 * E) + (delta_sigma / (2 * K)) ** (1 / n) - delta_sigma_delta_epsilon / (2 * delta_sigma)

            ramberg_osgood_sigma = lambda x: ramberg_osgood(sigma_epsilon, x, E, K, n)
            sigma = fsolve(ramberg_osgood_sigma, x0=100)[0]
            epsilon = sigma_epsilon / sigma

            massing_delta_sigma = lambda x: massing(delta_epsilon_delta_sigma, x, E, K, n)
            delta_sigma = fsolve(massing_delta_sigma, x0=100)[0]
            delta_epsilon = delta_epsilon_delta_sigma / delta_sigma

            if delta_epsilon > 1:  # this checks that the delta_epsilon that was found is realistic
                raise ValueError('As a results of the inputs, delta_epsilon has been calculated to be greater than 1. This will result in immediate failure of the component. You should check your inputs to ensure they are in the correct units, especially for P (units of MPa) and A (units of mm^2).')

            self.sigma_min = sigma - delta_sigma
            self.epsilon_min = epsilon - delta_epsilon
            self.sigma_max = sigma
            self.epsilon_max = epsilon
            self.sigma_mean = self.sigma_min + delta_sigma * 0.5
            self.epsilon_mean = self.epsilon_min + delta_epsilon * 0.5

            if mean_stress_correction_method == 'morrow':
                def morrow(delta_epsilon, sigma_f, sigma_mean, E, Nf2, b, epsilon_f, c):
                    return ((sigma_f - sigma_mean) / E) * Nf2 ** b + epsilon_f * Nf2 ** c - delta_epsilon * 0.5

                morrow_2Nf = lambda x: morrow(delta_epsilon, sigma_f, self.sigma_mean, E, x, b, epsilon_f, c)
                Nf2 = fsolve(morrow_2Nf, x0=np.array([100]))[0]

            elif mean_stress_correction_method == 'modified_morrow':
                def modified_morrow(delta_epsilon, sigma_f, sigma_mean, E, Nf2, b, epsilon_f, c):
                    return ((sigma_f - sigma_mean) / E) * Nf2 ** b + epsilon_f * ((sigma_f - sigma_mean) / sigma_f) ** (c / b) * Nf2 ** c - delta_epsilon * 0.5

                modified_morrow_2Nf = lambda x: modified_morrow(delta_epsilon, sigma_f, self.sigma_mean, E, x, b, epsilon_f, c)
                Nf2 = fsolve(modified_morrow_2Nf, x0=np.array([100]))[0]

            elif mean_stress_correction_method == 'SWT':
                def SWT(sigma, delta_epsilon, sigma_f, E, Nf2, b, epsilon_f, c):
                    return ((sigma_f ** 2) / E) * Nf2 ** (2 * b) + sigma_f * epsilon_f * Nf2 ** (b + c) - sigma * delta_epsilon * 0.5

                SWT_2Nf = lambda x: SWT(sigma, delta_epsilon, sigma_f, E, x, b, epsilon_f, c)
                Nf2 = fsolve(SWT_2Nf, x0=np.array([100]))[0]
        else:  # fully elastic model
            def ramberg_osgood(sigma_epsilon, sigma, E):
                return sigma / E - sigma_epsilon / sigma

            def massing(delta_sigma_delta_epsilon, delta_sigma, E):
                return delta_sigma / (2 * E) - delta_sigma_delta_epsilon / (2 * delta_sigma)

            ramberg_osgood_sigma = lambda x: ramberg_osgood(sigma_epsilon, x, E)
            sigma = fsolve(ramberg_osgood_sigma, x0=100)[0]
            epsilon = sigma_epsilon / sigma

            massing_delta_sigma = lambda x: massing(delta_epsilon_delta_sigma, x, E)
            delta_sigma = fsolve(massing_delta_sigma, x0=100)[0]
            delta_epsilon = delta_epsilon_delta_sigma / delta_sigma

            self.sigma_min = sigma - delta_sigma
            self.epsilon_min = epsilon - delta_epsilon
            self.sigma_max = sigma
            self.epsilon_max = epsilon
            self.sigma_mean = self.sigma_min + delta_sigma * 0.5
            self.epsilon_mean = self.epsilon_min + delta_epsilon * 0.5

            if mean_stress_correction_method == 'morrow':
                def morrow(delta_epsilon, sigma_f, sigma_mean, E, Nf2, b):
                    return ((sigma_f - sigma_mean) / E) * Nf2 ** b - delta_epsilon * 0.5

                morrow_2Nf = lambda x: morrow(delta_epsilon, sigma_f, self.sigma_mean, E, x, b)
                Nf2 = fsolve(morrow_2Nf, x0=1000)[0]

            elif mean_stress_correction_method == 'modified_morrow':
                def modified_morrow(delta_epsilon, sigma_f, sigma_mean, E, Nf2, b):
                    return ((sigma_f - sigma_mean) / E) * Nf2 ** b - delta_epsilon * 0.5

                modified_morrow_2Nf = lambda x: modified_morrow(delta_epsilon, sigma_f, self.sigma_mean, E, x, b)
                Nf2 = fsolve(modified_morrow_2Nf, x0=1000)[0]

            elif mean_stress_correction_method == 'SWT':
                def SWT(sigma, delta_epsilon, sigma_f, E, Nf2, b):
                    return ((sigma_f ** 2) / E) * Nf2 ** (2 * b) - sigma * delta_epsilon * 0.5

                SWT_2Nf = lambda x: SWT(sigma, delta_epsilon, sigma_f, E, x, b)
                Nf2 = fsolve(SWT_2Nf, x0=1000)[0]
        Nf = Nf2 / 2
        self.cycles_to_failure = Nf
        if print_results is True:
            print('A crack of 1 mm will be formed after:', round(self.cycles_to_failure, 2), 'cycles (', round(self.cycles_to_failure * 2, 2), 'reversals )')
            print('Stresses in the component: Min =', round(self.sigma_min, 4), 'MPa , Max =', round(self.sigma_max, 4), 'MPa , Mean =', self.sigma_mean, 'MPa.')
            print('Strains in the component: Min =', round(self.epsilon_min, 4), ', Max =', round(self.epsilon_max, 4), ', Mean =', self.epsilon_mean)
            print('Mean stress correction method used:', mean_stress_correction_method)


class fracture_mechanics_crack_growth:
    '''
    This function uses the principles of fracture mechanics to find the number of cycles required to grow a crack from an initial length until a final length.
    The final length (a_final) may be specified, but if not specified then a_final will be set as the critical crack length (a_crit) which causes failure due to rapid fracture.
    This functions performs the same calculation using two methods: similified and iterative.
    The simplified method assumes that the geometry factor (f(g)), the stress (S_net), and the critical crack length (a_crit) are constant. THis method is the way most textbooks show these problems solved as they can be done in a few steps.
    The iterative method does not make the assumptions that the simplified method does and as a result, the parameters f(g), S_net and a_crit must be recalculated based on the current crack length at every cycle.

    This function is applicable only to thin plates with an edge crack or a centre crack (which is to be specified using the parameter crack_type).
    You may also use this function for notched components by specifying the parameters Kt and D which are based on the geometry of the notch.
    For any notched components, this method assumes the notched component has a "shallow notch" where the notch depth (D) is much less than the plate width (W).
    The value of Kt for notched components may be found at https://www.efatigue.com/constantamplitude/stressconcentration/
    In the case of notched components, the local stress concentration from the notch will often cause slower crack growth.
    In these cases, the crack length is calculated in two parts (stage 1 and stage 2) which can clearly be seen on the plot using the iterative method.
    The only geometry this function is designed for is unnotched and notched thin flat plates. No centre holes are allowed.

    Inputs:
    Kc - fracture toughness
    Kt - stress concentration factor (default is 1 for no notch).
    D - depth of the notch (default is None for no notch). A notched specimen is assumed to be doubly-notched (equal notches on both sides)
    C - material constant (sometimes referred to as A)
    m - material constant (sometimes referred to as n). This value must not be 2.
    P - external load on the material (MPa)
    t - plate thickness (mm)
    W - plate width (mm)
    a_initial - initial crack length (mm) - default is 1 mm
    a_final - final crack length (mm) - default is None in which case a_final is assumed to be a_crit (length at failure). It is useful to be able to enter a_final in cases where there are multiple loading regimes over time.
    crack_type - must be either 'edge' or 'center'. Default is 'edge'. The geometry factor used for each of these in the simplified method is 1.12 for edge and 1.0 for center. The iterative method calculates these values exactly using a_initial and W (plate width).
    print_results - True/False. Default is True
    show_plot - True/False. Default is True. If True the Iterative method's crack growth will be plotted.

    Outputs:
    If print_results is True, all outputs will be printed with some description of the process.
    If show_plot is True, the crack growth plot will be shown for the iterative method.
    You may also access the following parameters from the calculated object:
    - Nf_stage_1_simplified
    - Nf_stage_2_simplified
    - Nf_total_simplified
    - final_crack_length_simplified
    - transition_length_simplified
    - Nf_stage_1_iterative
    - Nf_stage_2_iterative
    - Nf_total_iterative
    - final_crack_length_iterative
    - transition_length_iterative

    Example usage:
    fracture_mechanics_crack_growth(Kc=66,C=6.91*10**-12,m=3,P=0.15,W=100,t=5,Kt=2.41,a_initial=1,D=10,crack_type='edge')
    fracture_mechanics_crack_growth(Kc=66,C=3.81*10**-12,m=3,P=0.103,W=100,t=5,crack_type='center')
    '''

    def __init__(self, Kc, C, m, P, W, t, Kt=1.0, a_initial=1.0, D=None, a_final=None, crack_type='edge', print_results=True, show_plot=True):
        if m == 2:
            raise ValueError('m can not be 2')
        if crack_type not in ['center', 'edge', 'centre']:
            raise ValueError('crack_type must be either edge or center. default is center')
        if D is None:
            d = 0
        else:
            d = D
        if W - 2 * d < 0:
            error_str = str('The specified geometry is invalid. A doubly notched specimen with specified values of the d = ' + str(d) + 'mm will have notches deeper than the width of the plate W = ' + str(W) + 'mm. This would result in a negative cross sectional area.')
            raise ValueError(error_str)
        # Simplified method (assuming fg, S_max, af to be constant)
        S_max = P / (t * (W - 2 * d)) * 10 ** 6
        if crack_type == 'edge':
            f_g_fixed = 1.12
        elif crack_type in ['center', 'centre']:
            f_g_fixed = 1.0
        m_exp = -0.5 * m + 1
        a_crit = 1 / np.pi * (Kc / (S_max * f_g_fixed)) ** 2 + d / 1000  # critical crack length to cause failure
        if a_final is None:
            a_f = a_crit
        elif a_final < a_crit * 1000 - d:  # this is approved early stopping
            a_f = (a_final + d) / 1000
        else:
            print('WARNING: In the simplified method, the specified a_final (', a_final, 'mm ) is greater than the critical crack length to cause failure (', round(a_crit * 1000 - d, 5), 'mm ).')
            print('         a_final has been set to equal a_crit since cracks cannot grow beyond the critical length.')
            a_f = a_crit
        lt = d / ((1.12 * Kt / f_g_fixed) ** 2 - 1) / 1000  # find the transition length due to the notch
        if lt > a_initial / 1000:  # two step process due to local stress concentration
            Nf_1 = (lt ** m_exp - (a_initial / 1000) ** m_exp) / (m_exp * C * S_max ** m * np.pi ** (0.5 * m) * f_g_fixed ** m)
            a_i = lt + d / 1000  # new initial length for stage 2
        else:
            a_i = a_initial / 1000
            Nf_1 = 0
        Nf_2 = (a_f ** m_exp - a_i ** m_exp) / (m_exp * C * S_max ** m * np.pi ** (0.5 * m) * f_g_fixed ** m)
        Nf_tot = Nf_1 + Nf_2
        self.Nf_stage_1_simplified = Nf_1
        print(Nf_1)
        self.Nf_stage_2_simplified = Nf_2
        self.Nf_total_simplified = Nf_tot
        self.final_crack_length_simplified = a_f * 1000 - d
        self.transition_length_simplified = lt * 1000
        if print_results is True:
            print('SIMPLIFIED METHOD (keeping f(g), S_max, and a_crit as constant):')
            if Nf_1 == 0:
                print('Crack growth was found in a single stage since the transition length (', round(self.transition_length_simplified, 2), 'mm ) was less than the initial crack length', round(a_initial, 2), 'mm.')
            else:
                print('Crack growth was found in two stages since the transition length (', round(self.transition_length_simplified, 2), 'mm ) due to the notch, was greater than the initial crack length (', round(a_initial, 2), 'mm ).')
                print('Stage 1 (a_initial to transition length):', int(np.floor(self.Nf_stage_1_simplified)), 'cycles')
                print('Stage 2 (transition length to a_final):', int(np.floor(self.Nf_stage_2_simplified)), 'cycles')
            if a_final is None or a_final >= a_crit * 1000 - d:
                print('Total cycles to failure:', int(np.floor(self.Nf_total_simplified)), 'cycles.')
                print('Critical crack length to cause failure was found to be:', round(self.final_crack_length_simplified, 2), 'mm.')
            else:
                print('Total cycles to reach a_final:', int(np.floor(self.Nf_total_simplified)), 'cycles.')
                print('Note that a_final will not result in failure. To find cycles to failure, leave a_final as None.')
            print('')

        # Iterative method (recalculating fg, S_max, af at each iteration)
        a = a_initial
        a_effective = a_initial + d
        if crack_type in ['center', 'centre']:
            f_g = (1 / np.cos(np.pi * a_effective / W)) ** 0.5
        elif crack_type == 'edge':
            f_g = 1.12 - 0.231 * (a_effective / W) + 10.55 * (a_effective / W) ** 2 - 21.72 * (a_effective / W) ** 3 + 30.39 * (a_effective / W) ** 4
        lt2 = d / ((1.12 * Kt / f_g) ** 2 - 1)
        self.transition_length_iterative = lt2
        self.Nf_stage_1_iterative = 0
        N = 0
        growth_finished = False
        a_array = []
        a_crit_array = []
        N_array = []
        while growth_finished is False:
            area = (t * (W - 2 * d - a))
            S = (P / area) * 10 ** 6  # local stress due to reducing cross-sectional area
            if a < lt2:  # crack growth slowed by transition length
                if crack_type in ['center', 'centre']:
                    f_g = (1 / np.cos(np.pi * a / W)) ** 0.5  # Ref: p92 of Bannantine, et al. (1997).
                elif crack_type == 'edge':
                    f_g = 1.12 - 0.231 * (a / W) + 10.55 * (a / W) ** 2 - 21.72 * (a / W) ** 3 + 30.39 * (a / W) ** 4  # Ref: p92 of Bannantine, et al. (1997).
                delta_K = f_g * S * (np.pi * a / 1000) ** 0.5
            else:
                if crack_type in ['center', 'centre']:
                    f_g = (1 / np.cos(np.pi * a / W)) ** 0.5
                elif crack_type == 'edge':
                    f_g = 1.12 - 0.231 * (a / W) + 10.55 * (a / W) ** 2 - 21.72 * (a / W) ** 3 + 30.39 * (a / W) ** 4
                delta_K = f_g * S * (np.pi * a_effective / 1000) ** 0.5
            da = (C * delta_K ** m) * 1000
            a_crit = 1 / np.pi * (Kc / (f_g * S)) ** 2 + d / 1000
            a_crit_array.append(a_crit * 1000 - d)
            a_effective += da  # grow the crack by da
            a += da  # grow the crack by da
            N += 1
            a_array.append(a)
            N_array.append(N)
            if a_array[N - 2] < lt2 and a_array[N - 1] > lt2:
                self.Nf_stage_1_iterative = N - 1
            if a_effective > a_crit * 1000:
                growth_finished = True
            if a_final is not None:
                if a_effective > a_final + d:
                    growth_finished = True
        self.Nf_total_iterative = N
        self.final_crack_length_iterative = a_crit * 1000 - d
        self.Nf_stage_2_iterative = N - self.Nf_stage_1_iterative
        if a_final is not None:
            if a_final > a_crit * 1000 - d:
                print('WARNING: During the iterative method, the specified a_final (', a_final, 'mm ) was found to be greater than the critical crack length to cause failure (', round(self.final_crack_length_iterative, 2), 'mm ).')
        if print_results is True:
            print('ITERATIVE METHOD (recalculating f(g), S_max, and a_crit for each cycle):')
            if a_initial > lt2:
                print('Crack growth was found in a single stage since the transition length (', round(self.transition_length_iterative, 2), 'mm ) was less than the initial crack length', round(a_initial, 2), 'mm.')
            else:
                print('Crack growth was found in two stages since the transition length (', round(self.transition_length_iterative, 2), 'mm ) due to the notch, was greater than the initial crack length (', round(a_initial, 2), 'mm ).')
                print('Stage 1 (a_initial to transition length):', round(self.Nf_stage_1_iterative, 2), 'cycles')
                print('Stage 2 (transition length to a_final):', round(self.Nf_stage_2_iterative, 2), 'cycles')
            if a_final is None or a_final >= a_crit * 1000 - d:
                print('Total cycles to failure:', round(self.Nf_total_iterative, 2), 'cycles.')
                print('Critical crack length to cause failure was found to be:', round(self.final_crack_length_iterative, 2), 'mm.')
            else:
                print('Total cycles to reach a_final:', round(self.Nf_total_iterative, 2), 'cycles.')
                print('Note that a_final will not result in failure. To find cycles to failure, leave a_final as None.')

        if show_plot is True:
            plt.plot(N_array, a_crit_array, label='Critical crack length', color='darkorange')
            plt.plot(N_array, a_array, label='Crack length', color='steelblue')
            plt.xlabel('Cycles')
            plt.ylabel('Crack length (mm)')
            plt.ylim(0, max(a_crit_array) * 1.2)
            plt.xlim(0, N * 1.1)
            plt.plot([0, N, N], [max(a_array), max(a_array), 0], linestyle='--', linewidth=1, color='k')
            plt.text(0, max(a_array), str(' ' + str(round(max(a_array), 2)) + ' mm'), va='bottom')
            plt.text(N, 0, str(str(N) + ' cycles '), ha='right', va='bottom')
            plt.legend(loc='upper right')
            plt.title('Crack growth using iterative method')


def creep_rupture_curves(temp_array, stress_array, TTF_array, stress_trace=None, temp_trace=None):
    '''
    Plots the creep rupture curves for a given set of creep data. Also fits the lines of best fit to each temperature.
    The time to failure for a given temperature can be found by specifying stress_trace and temp_trace.

    Inputs:
    temp_array: an array or list of temperatures
    stress_array: an array or list of stresses
    TTF_array: an array or list of times to failure at the given temperatures and stresses
    stress_trace: *only 1 value is accepted
    temp_trace: *only 1 value is accepted

    Outputs:
    The plot is the only output. Use plt.show() to show it.

    Example Usage:
    TEMP = [900,900,900,900,1000,1000,1000,1000,1000,1000,1000,1000,1100,1100,1100,1100,1100,1200,1200,1200,1200,1350,1350,1350]
    STRESS = [90,82,78,70,80,75,68,60,56,49,43,38,60.5,50,40,29,22,40,30,25,20,20,15,10]
    TTF = [37,975,3581,9878,7,17,213,1493,2491,5108,7390,10447,18,167,615,2220,6637,19,102,125,331,3.7,8.9,31.8]
    creep_rupture_curves(temp_array=TEMP, stress_array=STRESS, TTF_array=TTF, stress_trace=70, temp_trace=1100)
    plt.show()
    '''

    if (stress_trace is not None and temp_trace is None) or (stress_trace is None and temp_trace is not None):
        raise ValueError('You must enter both stress_trace and temp_trace to obtain the time to failure at a given stress and temperature.')
    if len(temp_array) < 2 or len(stress_array) < 2 or len(TTF_array) < 2:
        raise ValueError('temp_array, stress_array, and TTF_array must each have at least 2 data points for a line to be fitted.')
    if len(temp_array) != len(stress_array) or len(temp_array) != len(TTF_array):
        raise ValueError('The length of temp_array, stress_array, and TTF_array must all be equal')

    xmin = 10 ** (int(np.floor(np.log10(min(TTF_array)))) - 1)
    xmax = 10 ** (int(np.ceil(np.log10(max(TTF_array)))) + 1)

    delta = (max(stress_array) - min(stress_array)) * 0.2
    ymin = min(stress_array) - delta
    ymax = max(stress_array) + delta

    xvals = np.logspace(np.log10(xmin), np.log10(xmax), 100)

    unique_temps = []
    stress_indices = []
    for i, item in enumerate(temp_array):
        if item in unique_temps:
            pass
        else:
            unique_temps.append(item)
            stress_indices.append(i)
    stress_indices.append(len(stress_array))

    if temp_trace not in unique_temps:
        raise ValueError('temp_trace must be one of the temperatures provided in temp_array')

    for i, T in enumerate(unique_temps):
        xvalues = TTF_array[stress_indices[i]:stress_indices[i + 1]]
        yvalues = stress_array[stress_indices[i]:stress_indices[i + 1]]
        plt.scatter(xvalues, yvalues, label=T, alpha=0.8)
        fit = np.polyfit(np.log10(xvalues), yvalues, deg=1)
        m = fit[0]
        c = fit[1]
        plt.plot(xvals, m * np.log10(xvals) + c, alpha=0.8)
        if stress_trace is not None and temp_trace is not None:
            if T == temp_trace:
                y = stress_trace
                x = 10 ** ((y - c) / m)
                plt.plot([xmin, x, x], [y, y, ymin], linestyle='--', color='k', linewidth=1)
                plt.text(xmin, y, str(' Stress = ' + str(y)), va='bottom')
                plt.text(x, ymin, str(' Time to failure = ' + str(round(x, 3))), va='bottom')
    plt.xscale('log')
    plt.xlabel('Time to failure')
    plt.ylabel('Stress')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper right', title='Temperature')
    plt.title('Creep Rupture Curves')


def creep_failure_time(temp_low, temp_high, time_low, C=20, print_results=True):
    '''
    This function uses the Larson-Miller relation to find the time to failure due to creep.
    The method uses a known failure time (time_low) at a lower failure temperature (temp_low) to find the unknown failure time at the higher temperature (temp_high).
    This relation requires the input temperatures in Fahrenheit. To convert Celsius to Fahrenheit use F = C*(9/5)+32
    Note that the conversion between Fahrenheit and Rankine used in this calculation is R = F+459.67
    For more information see Wikipedia: https://en.wikipedia.org/wiki/Larson%E2%80%93Miller_relation

    Inputs:
    temp_low - temperature (in degrees Fahrenheit) where the time_low is known
    temp_high - temperature (in degrees Fahrenheit) which time_high is unknown and will be found by this function
    time_low - time to failure at temp_low
    C - creep constant (default is 20). Typically 20-22 for metals
    print_results - True/False

    Outputs:
    The time to failure at the higher temperature.
    If print_results is True, the output will also be printed to the console.
    '''
    LMP = (temp_low + 459.67) * (C + np.log10(time_low))  # larson-miller parameter. 459.67 converts Fahrenheit to Rankine
    time_high = 10 ** (LMP / (temp_high + 459.67) - C)
    if print_results is True:
        print('The time to failure at a temperature of', temp_high, '°F is', time_high)
        print('The Larson-Miller parameter was found to be', LMP)
    return time_high


class acceleration_factor:
    '''
    The Arrhenius model for Acceleration factor due to higher temperature is:
    AF = exp(Ea/K(1/T_use-1/T_acc))
    This function accepts T_use as a mandatory input and the user may specify any two of the three other variables, and the third variable will be found.

    Inputs:
    T_use - Temp of usage in Celsius
    T_acc - Temp of acceleration in Celsius (optional input)
    Ea - Activation energy in eV (optional input)
    AF - Acceleration factor (optional input)
    Two of the three optional inputs must be specified and the third one will be found.
    print_results - True/False. Default is True

    Outputs:
    Outputs will be printed to console if print_results is True
    AF - Acceleration Factor
    T_acc - Accelerated temperature
    T_use - Use temperature
    Ea - Activation energy (in eV)
    '''

    def __init__(self, AF=None, T_use=None, T_acc=None, Ea=None, print_results=True):
        if T_use is None:
            raise ValueError('T_use must be specified')
        args = [AF, T_acc, Ea]
        nonecounter = 0
        for item in args:
            if item is None:
                nonecounter += 1
        if nonecounter > 1:
            raise ValueError('You must specify two out of three of the optional inputs (T_acc, AF, Ea) and the third one will be found.')

        if AF is None:
            a = Ea / (8.617333262145 * 10 ** -5)
            AF = np.exp(a / (T_use + 273.15) - a / (T_acc + 273.15))
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if Ea is None:
            Ea = np.log(AF) * (8.617333262145 * 10 ** -5) / (1 / (T_use + 273.15) - 1 / (T_acc + 273.15))
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if T_acc is None:
            T_acc = (1 / (1 / (T_use + 273.15) - np.log(AF) * (8.617333262145 * 10 ** -5) / Ea)) - 273.15
            self.AF = AF
            self.Ea = Ea
            self.T_acc = T_acc
            self.T_use = T_use

        if print_results is True:
            print('Acceleration Factor:', self.AF)
            print('Use Temperature:', self.T_use, '°C')
            print('Accelerated Temperature:', self.T_acc, '°C')
            print('Activation Energy (eV):', self.Ea, 'eV')
