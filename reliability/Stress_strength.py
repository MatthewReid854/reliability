'''
Stress - Strength Interference
Given the probability distributions for stress and strength, this module can calculate the probability of failure.
Two functions are available:
Probability_of_failure - works with any distributions and uses numerical integration
Probability_of_failure_normdist - works only when both the stress and strength distributions are Normal Distributions.
    Uses an exact method (formula) rather than calculating the integral. Use this function if you have two Normal Distributions.
'''
from reliability.Distributions import Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution,\
    Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution
from reliability.Utils import round_to_decimals
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def Probability_of_failure(stress, strength, show_distribution_plot=True, print_results=True, warn=True):
    '''
    Stress - Strength Interference
    Given the probability distributions for stress and strength, this module will find the probability of failure due to stress-strength interference.
    Failure is defined as when stress>strength.
    The calculation is achieved using numerical integration.

    Inputs:
    stress - a probability distribution from the Distributions module
    strength - a probability distribution from the Distributions module
    show_distribution_plot - True/False (default is True)
    print_results - True/False (default is True)
    warn - a warning will be issued if both stress and strength are Normal as you should use Probability_of_failure_normdist. You can supress this using warn=False
         - a warning will be issued if the stress.mean > strength.mean as the user may have assigned the distributions to the wrong variables. You can supress this using warn=False

    Returns:
    probability of failure

    Example use:
    from reliability.Distributions import Weibull_Distribution, Gamma_Distribution
    stress = Weibull_Distribution(alpha=2,beta=3,gamma=1)
    strength = Gamma_Distribution(alpha=2,beta=3,gamma=3)
    Probability_of_failure(stress=stress, strength=strength)
    '''

    if type(stress) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution] \
            or type(strength) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution]:
        raise ValueError('Stress and Strength must both be probability distributions. First define the distribution using reliability.Distributions.___')
    if type(stress) == Normal_Distribution and type(strength) == Normal_Distribution and warn is True:  # supress the warning by setting warn=False
        print('If strength and stress are both Normal distributions, it is more accurate to use the exact formula. The exact formula is supported in the function Probability_of_failure_normdist. To supress this warning set warn=False')
    if stress.mean > strength.mean and warn == True:
        print('WARNING: The mean of the stress distribution is above the mean of the strength distribution. Please check you have assigned stress and strength to the correct variables. To supress this warning set warn=False')

    x = np.linspace(stress.quantile(1e-8), strength.quantile(1-1e-8), 1000)
    prob_of_failure = np.trapz(stress.PDF(x, show_plot=False) * strength.CDF(x, show_plot=False), x)

    if show_distribution_plot is True:
        xlims = plt.xlim(auto=None)
        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        if xmin < (xmax - xmin)/4:
            xmin = 0 #if the lower bound on xmin is near zero (relative to the entire range) then just make it zero
        if type(stress) == Beta_Distribution:
            xmin = 0
        if type(strength) == Beta_Distribution:
            xmax = 1
        xvals = np.linspace(xmin, xmax, 10000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
        Y = [(min(strength_PDF[i],stress_PDF[i])) for i in range(len(xvals))] #finds the lower of the two lines which is used as the upper boundary for fill_between
        plt.plot(xvals, stress_PDF, label='Stress')
        plt.plot(xvals, strength_PDF, label='Strength')
        intercept_idx = Y.index(max(Y))
        plt.fill_between(xvals,np.zeros_like(xvals),Y,color='salmon',alpha=1,linewidth=0, linestyle='--')
        plt.fill_between(xvals[0:intercept_idx],strength_PDF[0:intercept_idx],stress_PDF[0:intercept_idx],color='steelblue',alpha=0.3,linewidth=0, linestyle='--')
        plt.fill_between(xvals[intercept_idx::],stress_PDF[intercept_idx::],strength_PDF[intercept_idx::],color='darkorange',alpha=0.3,linewidth=0, linestyle='--')
        failure_text = str('Probability of\nfailure = ' + str(round_to_decimals(prob_of_failure, 4)))
        plt.legend(title=failure_text)
        plt.title('Stress - Strength Interference Plot')
        plt.ylabel('Probability Density')
        plt.xlabel('Stress and Strength Units')
        plt.subplots_adjust(left=0.16)
        if xlims != (0,1):
            plt.xlim(min(stress.b5,xlims[0]),max(strength.b95,xlims[1]),auto=None)
        else:
            plt.xlim(stress.b5, strength.b95,auto=None)
        plt.ylim(bottom=0,auto=None)

    if print_results is True:
        print('Probability of failure:', prob_of_failure)

    return prob_of_failure


def Probability_of_failure_normdist(stress=None, strength=None, show_distribution_plot=True, print_results=True):
    '''
    Stress - Strength Interference for two Normal Distributions
    Given the probability distributions for stress and strength, this module will find the probability of failure due to stress-strength interference.
    Failure is defined as when stress>strength.
    Uses the exact formula method which is only valid for two Normal Distributions.

    Inputs:
    stress - a probability distribution from the Distributions module
    strength - a probability distribution from the Distributions module
    show_distribution_plot - True/False (default is True)
    print_results - True/False (default is True)

    Returns:
    the probability of failure
    '''
    if type(stress) is not Normal_Distribution:
        raise ValueError('Both stress and strength must be a Normal_Distribution. If you need another distribution then use Probability_of_failure rather than Probability_of_failure_normdist')
    if type(strength) is not Normal_Distribution:
        raise ValueError('Both stress and strength must be a Normal_Distribution. If you need another distribution then use Probability_of_failure rather than Probability_of_failure_normdist')

    sigma_strength = strength.sigma
    mu_strength = strength.mu
    sigma_stress = stress.sigma
    mu_stress = stress.mu
    prob_of_failure = ss.norm.cdf(-(mu_strength - mu_stress) / ((sigma_strength ** 2 + sigma_stress ** 2) ** 0.5))

    if print_results is True:
        print('Probability of failure:', prob_of_failure)

    if show_distribution_plot is True:
        xlims = plt.xlim(auto=None)
        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        xvals = np.linspace(xmin, xmax, 1000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
        plt.plot(xvals, stress_PDF, label='Stress')
        plt.plot(xvals, strength_PDF, label='Strength')
        Y = [(min(strength_PDF[i], stress_PDF[i])) for i in range(len(xvals))]  # finds the lower of the two lines which is used as the upper boundary for fill_between
        intercept_idx = Y.index(max(Y))
        plt.fill_between(xvals,np.zeros_like(xvals),Y,color='salmon',alpha=1,linewidth=0, linestyle='--')
        plt.fill_between(xvals[0:intercept_idx],strength_PDF[0:intercept_idx],stress_PDF[0:intercept_idx],color='steelblue',alpha=0.3,linewidth=0, linestyle='--')
        plt.fill_between(xvals[intercept_idx::],stress_PDF[intercept_idx::],strength_PDF[intercept_idx::],color='darkorange',alpha=0.3,linewidth=0, linestyle='--')
        failure_text = str('Probability of\nfailure = ' + str(round_to_decimals(prob_of_failure, 4)))
        plt.legend(title=failure_text)
        plt.title('Stress - Strength Interference Plot')
        plt.ylabel('Probability Density')
        plt.xlabel('Stress and Strength Units')
        plt.subplots_adjust(left=0.15, right=0.93)
        if xlims != (0,1):
            plt.xlim(min(stress.b5,xlims[0]),max(strength.b95,xlims[1]),auto=None)
        else:
            plt.xlim(stress.b5, strength.b95,auto=None)
        plt.ylim(bottom=0,auto=None)

    return prob_of_failure
