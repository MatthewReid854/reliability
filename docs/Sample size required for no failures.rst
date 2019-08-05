.. image:: images/logo.png

-------------------------------------

Sample size required for no failures
''''''''''''''''''''''''''''''''''''
The function ``sample_size_no_failures`` is used to determine the minimum sample size required for a test in which no failures are expected, and the desired outcome is the lower bound on the reliability based on the sample size and desired confidence interval.
    
Inputs:

-   reliability - lower bound on product reliability (between 0 and 1)
-   CI - confidence interval of result (between 0.5 and 1). Defaults to 0.95 for 95% CI.
-   lifetimes - if testing the product for multiple lifetimes then more failures are expected so a smaller sample size will be required to demonstrate the desired reliability (assuming no failures). Conversely, if testing for less than one full lifetime then a larger sample size will be required. Default is 1.
-   weibull_shape - if the weibull shape (beta) of the failure mode is known, specify it here. Otherwise leave the default of 1 for the exponential distribution.
    
Outputs:

-   number of items required in the test. This will always be an integer (rounded up).

As an example, consider a scenario in which we want to be sure that a batch of LEDs meets the reliability target for on/off cycles. Testing is for the planned lifetime (1 million cycles) and tested items will have most or all of their lifetime used up during testing so we can't test everything. How many items from the batch do we need to test to ensure we achieve 99.9% reliability with a 95% confidence interval?

.. code:: python

    from reliability.Other_functions import sample_size_no_failures
    result = sample_size_no_failures(reliability=0.999)
    print(result)
    
    '''
    2995
    '''

Based on this result, we need to test 2995 items from the batch and not have a single failure in order to be 95% confident that the reliability of the batch meets or exceeds 99.9%. If we tested for more on/off cycles (lets say 3 million which is 3 lifetimes), then the number of successful results would only need to be 999. In this way, we can design our qualification test based on the desired reliability, CI, and lifetimes that are tested to.

In the event that we suffer a single failure during this test, then we will need to adjust the testing method, either by finishing the testing and calculating the lower bound on reliability using the `one_sample_proportion <https://reliability.readthedocs.io/en/latest/One%20sample%20proportion.html>`_ test, or by using a `sequential_sampling_chart <https://itl.nist.gov/div898/handbook/pmc/section2/pmc26.htm>`_.
