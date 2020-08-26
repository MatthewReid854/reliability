.. image:: images/logo.png

-------------------------------------

Reliability test duration
'''''''''''''''''''''''''

This function is an extension of the reliability_test_planner which allows users to calculate the required duration for a reliability test to achieve the specified producers and consumers risks. This is done based on the specified MTBF (mean time between failure) required and MTBF design.

This type of determination must be made when organisations looking to test an item are uncertain of how much testing is required, but they know the amount of risk they are willing to accept as well as the MTBF required and the MTBF to which the item has been designed.

Inputs:

-   MTBF_required - the required MTBF that the equipment must demonstrate during the test.
-   MTBF_design - the design target for the MTBF that the producer aims to achieve.
-   consumer_risk - the risk the consumer is accepting. This is the probability that a bad product will be accepted as a good product by the consumer.
-   producer_risk - the risk the producer is accepting. This is the probability that a good product will be rejected as a bad product by the consumer.
-   one_sided - default is True. The risk is analogous to the confidence interval, and the confidence interval can be one sided or two sided.
-   time_terminated - default is True. whether the test is time terminated or failure terminated. Typically it will be time terminated if the required test duration is sought.
-   show_plot - True/False. Default is True. This will create a plot of the risk vs test duration. Use plt.show() to show it.
-   print_results - True/False. Default is True. This will print the results to the console.

Outputs:

-   test duration
-   If print_results is True, all the variables will be printed to the console.
-   If show_plot is True a plot of producer's and consumer's risk Vs test duration will be generated. Use plt.show() to display it.

In the example below the consumer requires a vehicle to achieve an MTBF of 2500km and is willing to accept 20% risk that they accept a bad item when they should have rejected it). The producer has designed the vehicle to have an MTBF of 3000km and they are willing to accept 20% risk that the consumer rejects a good item when they should have accepted it. How many kilometres should the reliability test be?

.. code:: python

    from reliability.Reliability_testing import reliability_test_duration
    import matplotlib.pyplot as plt
    
    reliability_test_duration(MTBF_required=2500, MTBF_design=3000, consumer_risk=0.2, producer_risk=0.2)
    plt.show()
    
    '''
    Reliability Test Duration Solver for time-terminated test
    Required test duration: 231615.79491309822 # Note that this duration is the total time on test and may be split across several vehicles.
    Specified consumer's risk: 0.2
    Specified producer's risk: 0.2
    Specified MTBF required by the consumer: 2500
    Specified MTBF designed to by the producer: 3000
    '''

.. image:: images/reliability_test_duration.png

**How does the algorithm work?**

The underlying method is as follows:

Step 1) Begin with failures = 1. This will be iterated later.

Step 2) Using the function `Repairable_systems.reliability_test_planner <https://reliability.readthedocs.io/en/latest/Reliability%20test%20planner.html>`_, we set CI = 1-consumer_risk, MTBF = MTBF_required to solve for the test_duration that is achieved by this test. This is the test duration required if there was 1 failure which would give the specified MTBF required and specified consumer's risk.

Step 3) We again use the function Repairable_systems.reliability_test_planner but this time we set MTBF = MTBF_design and use the test_duration as the output from step 2. Still keeping failures = 1 we are solving for the CI achieved. This is effectively the producer's risk for the given test_duration and number of failures.

Step 4) The answer is higher than the specified producer's risk, so we now repeat steps 1-3 by increasing the number of failures by 1 each iteration. This is continued until the producer's risk is below what was specified. We then go back 1 failure since is it standard that the producer's risk can't be below what was specified (or the consumer may think the producer is cheating by lowering their risk).

Step 5) We now have a value for test_duration that will give our required outputs in both equations. We also happen to arrive at the number of failures, though this is not particularly relevant since it is just part of the solution process and the actual number of failures will be determined based on the conduct of the reliability test.

The plot that is produced by Repairable_system.reliability_test_duration displays a scatter plot at each failure. Since the number of failures must be an integer, we get results for reliability test durations that go in steps. The result returned corresponds to the test_duration at the last failure before the producer's risk dropped below what was specified. Also note that if the consumer's risk is different from the producer's risk, the solution for test_duration will not occur near the point on the graph where producer's risk and consumer's risk are equal.
