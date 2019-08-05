.. image:: images/logo.png

-------------------------------------

Two proportion test
'''''''''''''''''''

This function is a statistical test to determine if there is a statistically significant difference in the results from two different tests. Similar to the `One_sample_proportion <https://reliability.readthedocs.io/en/latest/One%20sample%20proportion.html>`_, we are interested in using results from a success/failure test, but we are now interested in whether the difference in results is significant when comparing results between two tests.

In this example, consider that sample 1 and sample 2 are batches of LEDs that two suppliers sent you as part of their contract bidding process. You test everything each supplier sent you and need to know whether the reliability difference between suppliers is significant. At first glance, the reliability for sample 1 is 490/500 = 98%, and for the second supplier is 


.. code:: python

    from reliability import Other_functions
    result = Other_functions.two_proportion_test(sample_1_trials=500,sample_1_successes=490,sample_2_trials=800,sample_2_successes=770)
    print(result)

