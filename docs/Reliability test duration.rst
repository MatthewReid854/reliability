.. image:: images/logo.png

-------------------------------------

Reliability test duration
'''''''''''''''''''''''''

To be written soon

Inputs:

-   MTBF_required - the required MTBF that the equipment must demonstrate during the test
-   MTBF_design - the design target for the MTBF that the producer aims to achieve
-   consumer_risk - the risk the consumer is accepting. This is the probability that a bad product will be accepted as a good product by the consumer.
-   producer_risk - the risk the producer is accepting. This is the probability that a good product will be rejected as a bad product by the consumer.
-   one_sided - default is True. The risk is analogous to the confidence interval, and the confidence interval can be one sided or two sided.
-   time_terminated - default is True. whether the test is time terminated or failure terminated. Typically it will be time terminated if the required test duration is sought.
-   show_plot - True/False. Default is True. This will create a plot of the risk vs test duration. Use plt.show() to show it.
-   print_results - True/False. Default is True. This will print the results to the console.

Outputs:

-   test duration
-   If print_results is True, all the variables will be printed.
-   If show_plot is True a plot of producer's and consumer's risk Vs test duration will be generated. Use plt.show() to display it.

In the example below...

.. code:: python

    #example 1
    from reliability.Reliability_testing import reliability_test_duration

**How does this work?**

The underlying method is as follows:
