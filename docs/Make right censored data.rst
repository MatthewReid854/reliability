.. image:: images/logo.png

-------------------------------------

Make right censored data
''''''''''''''''''''''''

This function is a simple tool to generate right censored data using complete data and a threshold from which to right censor the data. It is primarily used in testing of the Fitters functions when some right censored data is needed.

Inputs:

-   data - list or array of data
-   threshold - point to right censor (right censoring is done if value is > threshold)

Outputs:

-   failures - array of failures (data <= threshold)
-   right_censored - array of right_censored values (data > threshold). These will be set at the value of the threshold.

.. code:: python

    from reliability.Other_functions import make_right_censored_data
    output = make_right_censored_data(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], threshold=6)
    print('Failures:',output.failures)
    print('Right Censored:',output.right_censored)
    
    '''
    Failures: [1 2 3 4 5 6]
    Right Censored: [6 6 6 6] #the numbers 7 to 10 have been set equal to the threshold
    '''
    
