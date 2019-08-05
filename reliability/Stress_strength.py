'''
Stress - Strength Interference
Given the probability distributions for stress and strength, this module can calculate the probability of failure.
Two functions are available:
Probability_of_failure - works with any distributions and uses a monte carlo simulation
Probability_of_failure_normdist - works only when both the stress and strength distributions are Normal Distributions.
    Uses an exact method (formula) rather than monte carlo simulation. Use this function if you have two Normal Distributions.
'''
from reliability.Distributions import Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

class Probability_of_failure:
    '''
    Stress - Strength Interference
    Given the probability distributions for stress and strength, this module will perform a monte carlo simulation to
    determine the probability of failure. Failure is defined as when stress>strength
    To ensure convergence is reached, a convergence plot is also provided.
    It is recommended to use a minimum of 10^4 monte carlo trials (default is 10^5), with more trials being more accurate.
    Using more than 10^7 monte carlo trials will take a long time to calculate.

    Inputs:
    stress - a probability distribution from the Distributions module
    strength - a probability distribution from the Distributions module
    monte_carlo_trials - number of MC trials. Default is 100000.
    show_distribution_plot - True/False (default is True)
    show_convergence_plot - True/False (default is True)
    print_results - True/False (default is True)

    returns:
    prob_of_failure - the probability of failure

    Example use:
    import Distributions
    stress = Distributions.Weibull_Distribution(alpha=2,beta=3,gamma=1)
    strength = Distributions.Gamma_Distribution(alpha=2,beta=3,gamma=3)
    Probability_of_failure(stress=stress, strength=strength, monte_carlo_trials=1000000)

    '''
    def __init__(self,stress,strength,monte_carlo_trials = 100000, show_distribution_plot=True, show_convergence_plot=True, print_results= True, warn=True):
        if type(stress) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution] or type(strength) not in [Weibull_Distribution, Normal_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution]:
            raise ValueError('Stress and Strength must both be probability distributions. First define the distribution using Reliability.Distributions.___')
        if monte_carlo_trials<100000:
            print('Warning: Using less than 100000 monte carlo trials will lead to very inaccurate results.')
        if type(stress)==Normal_Distribution and type(strength)==Normal_Distribution and warn==True: #supress the warning by setting warn=False
            print('If strength and stress are both Normal distributions, it is more accurate to use the exact formula rather than monte carlo estimation. The exact formula is supported in the function Probability_of_failure_normdist')


        #draw random samples from stress and strength distributions
        self.stress_RVS = stress.random_samples(number_of_samples=monte_carlo_trials)
        self.strength_RVS = strength.random_samples(number_of_samples=monte_carlo_trials)

        #calculate the probability of failure array and the final value
        pof_array = []
        failures = 0
        for i in range(monte_carlo_trials):
            if self.stress_RVS[i] > self.strength_RVS[i]:
                failures += 1
            pof_array.append(failures / (i + 1))
        self.prob_of_failure = failures / monte_carlo_trials

        if show_distribution_plot == True:
            xmin = stress.b5
            xmax = strength.b95
            if type(stress) == Beta_Distribution:
                xmin = 0
            if type(strength) == Beta_Distribution:
                xmax = 1
            xvals = np.linspace(xmin,xmax,100000)
            stress_PDF = stress.PDF(xvals=xvals,show_plot=False)
            strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
            plt.figure()
            plt.plot(xvals,stress_PDF,label='Stress')
            plt.plot(xvals,strength_PDF,label='Strength')
            intercept = np.argmin(abs(stress_PDF-strength_PDF)[1::]) #for use in fill_between. The slice ignores any instances where both distributions are 0 at x=0
            plt.fill_between(xvals[intercept::],0,stress_PDF[intercept::],color='peachpuff')
            plt.fill_between(xvals[0:intercept],0,strength_PDF[0:intercept],color='peachpuff')
            failure_text = str('Probability of\nfailure = ' + str(round(self.prob_of_failure * 100, int(np.log10(monte_carlo_trials)))) + '%')
            plt.legend(title=failure_text)
            plt.title('Stress - Strength Interference Plot')
            plt.xlabel('Probability Density')
            plt.ylabel('Stress and Strength Units')

        if show_convergence_plot == True:
            plt.figure()
            plt.plot(range(monte_carlo_trials),pof_array)
            plt.title('Monte Carlo convergence plot\nof probability of failure results')
            plt.xlabel('Monte Carlo trial')
            plt.ylabel('Probability of failure')
            plt.subplots_adjust(left=0.15,right=0.93)

        if print_results == True:
            print('Probability of failure:', self.prob_of_failure)

        if show_convergence_plot == True or show_distribution_plot == True:
            plt.show()

class Probability_of_failure_normdist:
    '''
    Stress-strength interference for two Normal distributions
    Not valid for any other distribution types.
    Uses the exact formula method (no use of monte carlo).

    Inputs:
    stress - a probability distribution from the Distributions module
    strength - a probability distribution from the Distributions module
    show_distribution_plot - True/False (default is True)
    print_results - True/False (default is True)

    returns:
    prob_of_failure - the probability of failure
    '''
    def __init__(self,stress=None,strength=None,show_distribution_plot=True, print_results= True):
        if type(stress) is not Normal_Distribution:
            raise ValueError('Both stress and strength must be a Normal_Distribution. If you need another distribution then use Probability_of_failure rather than Probability_of_failure_normdist')
        if type(strength) is not Normal_Distribution:
            raise ValueError('Both stress and strength must be a Normal_Distribution. If you need another distribution then use Probability_of_failure rather than Probability_of_failure_normdist')

        sigma_strength = strength.sigma
        mu_strength = strength.mu
        sigma_stress = stress.sigma
        mu_stress = stress.mu
        self.prob_of_failure = ss.norm.cdf(-(mu_strength-mu_stress)/((sigma_strength**2+sigma_stress**2)**0.5))

        if print_results == True:
            print('Probability of failure:', self.prob_of_failure)

        if show_distribution_plot == True:
            xmin = stress.b5
            xmax = strength.b95
            xvals = np.linspace(xmin,xmax,100000)
            stress_PDF = stress.PDF(xvals=xvals,show_plot=False)
            strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
            plt.figure()
            plt.plot(xvals,stress_PDF,label='Stress')
            plt.plot(xvals,strength_PDF,label='Strength')
            intercept = np.argmin(abs(stress_PDF-strength_PDF)[1::]) #for use in fill_between. The slice ignores any instances where both distributions are 0 at x=0
            plt.fill_between(xvals[intercept::],0,stress_PDF[intercept::],color='peachpuff')
            plt.fill_between(xvals[0:intercept],0,strength_PDF[0:intercept],color='peachpuff')
            failure_text = str('Probability of\nfailure = ' + str(round(self.prob_of_failure * 100, 4)) + '%')
            plt.legend(title=failure_text)
            plt.title('Stress - Strength Interference Plot')
            plt.xlabel('Probability Density')
            plt.ylabel('Stress and Strength Units')
            plt.subplots_adjust(left=0.15,right=0.93)
            plt.show()
