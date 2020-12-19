'''
This module has been deprecated and the functions for stress-strength interference have been moved into reliability.Other_functions.
Probability_of_failure has been renamed to stress_strength
Probability_of_failure_normdist has been renamed to stress_strength_normal
'''

from reliability.Utils import colorprint
from reliability.Other_functions import stress_strength, stress_strength_normal

def Probability_of_failure(stress, strength, show_distribution_plot=True, print_results=True, warn=True):
    '''
    This function is deprecated.
    Please use reliability.Other_functions.stress_strength
    It is the same function just in a different location with a different name.
    '''
    warning_str = 'DeprecationWarning: reliability.Stress_strength.Probability_of_failure was moved and renamed to reliability.Other_functions.stress_strength in version 0.5.5. Your function has still been run, however, this module will be fully deprecated in March 2021.'
    colorprint(warning_str,text_color='red')
    stress_strength(stress=stress,strength=strength,show_distribution_plot=show_distribution_plot,print_results=print_results,warn=warn)


def Probability_of_failure_normdist(stress=None, strength=None, show_distribution_plot=True, print_results=True, warn=True):
    '''
    This function is deprecated.
    Please use reliability.Other_functions.stress_strength_normal
    It is the same function just in a different location with a different name.
    '''
    warning_str = 'DeprecationWarning: reliability.Stress_strength.Probability_of_failure_normdist was moved and renamed to reliability.Other_functions.stress_strength_normal in version 0.5.5. Your function has still been run, however, this module will be fully deprecated in March 2021.'
    colorprint(warning_str,text_color='red')
    stress_strength_normal(stress=stress,strength=strength,show_distribution_plot=show_distribution_plot,print_results=print_results,warn=warn)
