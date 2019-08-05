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

As an example, consider a scenario in which we have 

.. code:: python

    from reliability.Distributions import Weibull_Distribution

