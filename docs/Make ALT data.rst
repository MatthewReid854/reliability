.. image:: images/logo.png

-------------------------------------

Make ALT data
'''''''''''''

This will be written soon.

The following text is a template from make_right_censored_data so pleaase ignore it.








This function is a tool to convert complete data to complete and right censored data. Two methods are available which enable the production of either singly-censored or multiply-censored data. This function is often used in testing of the Fitters or Nonparametric functions when some right censored data is needed.

Inputs:

-   data - list or array of data
-   threshold - number or None. Default is None. If number this is the point to right censor (right censoring is done if data > threshold). This is known as "singly censored data" as everything is censored at a single point.
-   fraction_censored - number between 0 and 1. Deafult is 0.5. Censoring is done randomly. This is known as "multiply censored data" as there are multiple times at which censoring occurs. If both threshold and fraction_censored are None, fraction_censored will default to 0.5 to produce multiply censored data. If both threshold and fraction_censored are specified, an error will be raised since these methods conflict.
-   seed - sets the random seed. This is used for multiply censored data (i.e. when threshold is None). The data is shuffled to remove censoring bias that may be caused by any pre-sorting. Specifying the seed ensures a repeatable random shuffle.

Outputs:

-   failures - array of failure data
-   right_censored - array of right_censored data

Example 1
---------

In this first example we will look at the production of **singly censored data**. That is data which is all censored at the same value (defined by threshold).

.. code:: python

    from reliability.Other_functions import make_right_censored_data
    output = make_right_censored_data(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], threshold=6)
    print('Failures:',output.failures)
    print('Right Censored:',output.right_censored)
    
    '''
    Failures: [1 2 3 4 5 6]
    Right Censored: [6 6 6 6] #the numbers 7 to 10 have been set equal to the threshold
    '''

