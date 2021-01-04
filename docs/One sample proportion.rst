.. image:: images/logo.png

-------------------------------------

One sample proportion
'''''''''''''''''''''

This function calculates the upper and lower bounds of reliability for a given number of trials and successes. It is most applicable to analysis of test results in which there are only success/failure results and the analyst wants to know the reliability of the batch given those sample results.

Inputs:

-   trials - the number of trials which were conducted
-   successes - the number of trials which were successful
-   CI - the desired confidence interval. Defaults to 0.95 for 95% CI.
-   print_results - True/False. This will print the results if True. Defaults to True.

Outputs:

-   (lower, upper) - Tuple of the confidence interval limits. Note that this will return 0 for lower or 1 for upper if the one sided CI is calculated (ie. when successes=0 or successes=trials)

In this example, consider a scenario in which we have a large batch of items that we need to test for their reliability. The batch is large and testing is expensive so we will conduct the test on 30 samples. From those 30 samples, 29 passed the test. If the batch needs at least 85% reliability with a 95% confidence, then should we accept or reject the batch?

.. code:: python

    from reliability.Reliability_testing import one_sample_proportion
    one_sample_proportion(trials=30,successes=29)
    
    '''
    Results from one_sample_proportion:
    For a test with 30 trials of which there were 29 successes and 1 failures, the bounds on reliability are:
    Lower 95% confidence bound: 0.8278305443665873
    Upper 95% confidence bound: 0.9991564290733695
    '''

The lower bound (with 95% confidence interval) on the reliability was 82.78%. Since this is below our requirement of 85%, then we should reject the batch.
