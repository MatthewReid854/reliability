.. image:: images/logo.png

-------------------------------------

Reliability test planner
''''''''''''''''''''''''

A solver to determine the parameters of a reliability test when given 3 out of the 4 unknowns (lower confidence bound on MTBF, test duration, number of failures, confidence interval).

The underlying assumption is that the failures follow an exponential distribution (ie. failures occur randomly and the hazard rate does not change with age). Using this assumption, the The Chi-squared distribution is used to find the lower confidence bound on MTBF for a given test duration, number of failures, and specified confidence interval.:

:math:`MTBF = \frac{2T}{\chi^2(1-CI,2F+2)}`

Where:

- MTBF = Mean time between failures (same as mean time to failure (MTTF) when the hazard rate is constant as it is here). Note that this is the lower confidence interval on MTBF. If you want the point estimate then see the example below.
- T = Test duration (this is the total time on test across all units being tested)
- CI = Confidence interval (the confidence interval to be used for the lower bound on the MTBF)
- F = number of failures during the test

The above formula can be rearranged, or solved iteratively to determine any of these parameters when given the other 3. The user must specify any 3 out of the 4 variables (not including two_sided or print_results) and the remaining variable will be calculated. This implementation is for a time-truncated test. If you want a failure-truncated test, the formula is different and this is not yet implemented within the python reliability library. Also note the difference between the one-sided and two-sided confidence intervals which are specified using the input two_sided=True/False described below. A description of the difference between one-sided and two-sided confidence intervals is provided at the end of this page.

A similar calculator is available in the `reliability analytics toolkit <https://reliabilityanalyticstoolkit.appspot.com/confidence_limits_exponential_distribution>`_.

Inputs:

-   MTBF - mean time between failures. This is the lower confidence bound on the MTBF. Units given in same units as the test_duration.
-   number_of_failures - the number of failures recorded (or allowed) to achieve the MTBF. Must be an integer.
-   test_duration - the amount of time on test required (or performed) to achieve the MTBF. May also be distance, rounds fires, cycles, etc. Units given in same units as MTBF.
-   CI - the confidence interval at which the lower confidence bound on the MTBF is given. Must be between 0.5 and 1. For example, specify 0.95 for 95% confidence interval.
-   print_results - True/False. Default is True.
-   two_sided - True/False. Default is True. If set to False, the 1 sided confidence interval will be returned.

Outputs:

-   If print_results is True, all the variables will be printed.
-   An output object is also returned with the same values as the inputs and the remaining value also calculated. This allows for any of the outputs to be called by name.

In the example below, we have a component that needs to perform with a MTBF of 500 hours (units are not important here as it may be days, cycles, rounds, etc.). We have been allocated 10000 hours of test time, and we want to know the number of failures permitted during the test to ensure we meet the MTBF to within an 80% confidence (two-sided).

.. code:: python

    from reliability.Other_functions import reliability_test_planner
    result = reliability_test_planner(MTBF=500,test_duration=10000,CI=0.8)

    #to access individual variables from the output, call them by name
    print('\n,result.number_of_failures)

    '''
    Reliability Test Planner
    Solving for number_of_failures
    Test duration: 10000
    MTBF (lower confidence bound): 500
    Number of failures: 16
    Confidence interval: 0.8

    16 #this was printed after we called it by name
    '''

One-sided vs two-sided confidence interval
==========================================

The below image illustrates the difference between one-sided and two-sided confidence interval. You can use either the one-sided or two-sided interval when you are seeking only the lower bound, but it is essential to understand that they will give very different results for the same CI. They will give equivalent results if the CI is set appropriately (eg. 90% one-sided is the same as 80% two-sided). If you are unsure which to use, the more conservative approach is to use the two-sided interval. If you want the point estimate, use the one-sided interval with a CI=0.5.

.. image:: images/CI_diagram.png
