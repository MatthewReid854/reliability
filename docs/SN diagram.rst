.. image:: images/logo.png

-------------------------------------

SN diagram
''''''''''

This function will plot the stress vs number of cycles (S-N) diagram when supplied with data from a series of fatigue tests. An S-N diagram is a common procedure used to model the fatigue life of metals which are subject to known cyclic loads. Typically, the plot is done using a semilog scale where the number of cycles is scaled logarithmically. This has the effect of linearizing the plot and making the accuracy of the model much easier to visualize. For steels, titanium alloys, and some other metals, there exists an endurance limit. This limit is the minimum stress required to propagate faigue cracks, and all stresses below this endurance limit do not contribute to fatigue growth. The plot can be adjusted to use an endurance limit using the optional inputs, however, there must be runout data (equivalent to right censored data) supplied in order for the program to determine where to set the endurance limit. 

Inputs:

-    stress - an array or list of stress values at failure
-    cycles - an array or list of cycles values at failure
-    stress_runout - an array or list of stress values that did not result in failure Optional
-    cycles_runout - an array or list of cycles values that did not result in failure. Optional
-    xscale - 'log' or 'linear'. Default is 'log'. Adjusts the x-scale.
-    stress_trace - an array or list of stress values to be traced across to cycles values.
-    cycles_trace - an array or list of cycles values to be traced across to stress values.
-    show_endurance_limit - This will adjust all lines of best fit to be greater than or equal to the average stress_runout. Defaults to False if stress_runout is not specified. Defaults to True if stress_runout is specified.
-    method_for_bounds - 'statistical', 'residual', or None. Defaults to 'statistical'. If set to 'statistical' the CI value is used, otherwise it is not used for the 'residual' method. Residual uses the maximum residual datapoint for symmetric bounds. Setting the method for bounds to None will turn off the confidence bounds.
-    CI - Must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if method_for_bounds = 'statistical'
-    Other plotting keywords (eg. color, linestyle, etc) are accepted for the line of best fit.

Outputs:

-    The plot is the only output. All calculated values are shown on the plot.

In this first example, we use the data for stress and cycles to produce an S-N diagram. We will not provide any runout data here so the endurance limit will not be calculated.

.. code:: python

    from reliability.PoF import SN_diagram
    import matplotlib.pyplot as plt
    stress = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    SN_diagram(stress=stress, cycles=cycles)
    plt.show()

.. image:: images/Fatigue_1.png

In this second example, we will use the same data as above, but also supply runout data so that the endurance limit will be calculated. We will also adjust the method_for_bounds to be 'residual'. We are also going to find the life (in cycles) at a stress of 260 by using stress_trace, and the stress required to achieve a life of 5x10^5 cycles using cycles_trace.

.. code:: python

    from reliability.PoF import SN_diagram
    import matplotlib.pyplot as plt
    stress = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    stress_runout = [210, 210, 205, 205, 205]
    cycles_runout = [10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7]
    SN_diagram(stress=stress, cycles=cycles, stress_runout=stress_runout, cycles_runout=cycles_runout,method_for_bounds='residual',cycles_trace=[5 * 10 ** 5], stress_trace=[260])
    plt.show()

.. image:: images/Fatigue_2.png

**References:**

Probabilistic Physics of Failure Approach to Reliability (2017), by M. Modarres, M. Amiri, and C. Jackson. pp 17-21.
