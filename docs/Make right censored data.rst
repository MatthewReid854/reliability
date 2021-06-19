.. image:: images/logo.png

-------------------------------------

Make right censored data
''''''''''''''''''''''''

This function is a tool to convert complete data to complete and right censored data. Two methods are available which enable the production of either singly-censored or multiply-censored data. This function is often used in testing of the Fitters or Nonparametric functions when some right censored data is needed.

.. admonition:: API Reference

   For inputs and outputs see the `API reference <https://reliability.readthedocs.io/en/latest/API/Other_functions/make_right_censored_data.html>`_.

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

Example 2
---------

In this second example we will look at the production of **multiply censored data**. That is data which is censored at different values. The amount of data to be censored is governed by fraction_censored. If unspecified it will default to 0.5 resulting in 50% of the data being right censored. Note that there is randomness to the censoring. For repeatability set the seed.

.. code:: python
    
    from reliability.Other_functions import make_right_censored_data
    output = make_right_censored_data(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fraction_censored=0.5, seed=1)
    print('Failures:', output.failures)
    print('Right Censored:', output.right_censored)
    
    '''
    Failures: [4 1 5 2 3] # half of the data has not been censored. It has been shuffled so its order will be different from the order of the input data.
    Right Censored: [5.89006504 8.71327034 4.27673283 3.11056676 2.728583] # half of the data has been censored at some value between 0 and the original value
    '''
