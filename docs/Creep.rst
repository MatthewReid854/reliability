.. image:: images/logo.png

-------------------------------------

Creep
'''''

.. note:: This module is currently in development. The following documentation is correct, however, the most recent version of ``reliability`` on PyPI will not contain this module until Dec 2019.

Creep is the progressive accumulation of plastic strain in a component under stress at an elevated temperatureover a period of time. All creep modelling requires data that is unique to the material undergoing creep since all materials behave differently. This data may be stress, temperature, and time to failure data, or it may be material constants which are derived from the former. This secion of reliability contains two functions to determine time to failure due to creep. These functions are ``creep_rupture_curves`` and ``creep_failure_time``. Creep is generally modelled using the Larson-Miller theory or the Manson-Haferd theory. Further discussion on these models is available in 


creep_rupture_curves

    Plots the creep rupture curves for a given set of creep data. Also fits the lines of best fit to each temperature.
    The time to failure for a given temperature can be found by specifying stress_trace and temp_trace.

    Inputs:
    temp_array: an array or list of temperatures
    stress_array: an array or list of stresses
    TTF_array: an array or list of times to failure at the given temperatures and stresses
    stress_trace: *only 1 value is accepted
    temp_trace: *only 1 value is accepted

    Outputs:
    The plot if the only output. Use plt.show() to show it.

    Example Usage:
    TEMP = [900,900,900,900,1000,1000,1000,1000,1000,1000,1000,1000,1100,1100,1100,1100,1100,1200,1200,1200,1200,1350,1350,1350]
    STRESS = [90,82,78,70,80,75,68,60,56,49,43,38,60.5,50,40,29,22,40,30,25,20,20,15,10]
    TTF = [37,975,3581,9878,7,17,213,1493,2491,5108,7390,10447,18,167,615,2220,6637,19,102,125,331,3.7,8.9,31.8]
    creep_rupture_curves(temp_array=TEMP, stress_array=STRESS, TTF_array=TTF, stress_trace=70, temp_trace=1100)
    plt.show()






.. image:: images/creep_rupture_curves.png




creep_failure_time
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



