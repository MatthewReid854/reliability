.. image:: images/logo.png

-------------------------------------

Optimal replacement time
''''''''''''''''''''''''



Calculates the cost model to determine how cost varies with replacement time. The cost model assumes Power Law NHPP.

Inputs:

    Cost_PM - cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM - cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha - scale parameter of the underlying Weibull distribution
    weibull_beta - shape parameter of the underlying Weibull distribution. Should be greater than 1 otherwise conducting PM is not economical.
    show_plot - True/False. Defaults to True. Other plotting keywords are also accepted and used.
    print_results - True/False. Defaults to True

Outputs:

    [ORT, min_cost] - the optimal replacement time and minimum cost per unit time in an array
    Plot of cost model if show_plot is set to True. Use plt.show() to display it.
    Printed results if print_results is set to True.




.. code:: python

    from reliability.Other_functions import optimal_replacement_time
    import matplotlib.pyplot as plt
    optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5)
    plt.show()

    '''
    The minimum cost per unit time is 0.0035 
    The optimal replacement time is 493.05
    '''

.. image:: images/optimal_replacement_time.png