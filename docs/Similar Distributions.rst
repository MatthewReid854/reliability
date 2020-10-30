.. image:: images/logo.png

-------------------------------------

Similar Distributions
'''''''''''''''''''''

The function similar_distributions is a tool for finding the probability distributions that are most similar to an input distribution.
It samples the CDF of an input distribution and then fits all other distributions to those samples to determine the best fitting and therefore most similar distributions.

Inputs

-   distribution - a distribution object created using the reliability.Distributions module
-   include_location_shifted - True/False. Default is True. When set to True it will include Weibull_3P, Lognormal_3P, Gamma_3P, Expon_2P
-   show_plot - True/False. Default is True
-   print_results - True/False. Default is True
-   number_of_distributions_to_show - the number of similar distributions to show. Default is 3. If the number specified exceeds the number available (typically 8), then the number specified will automatically be reduced.

Outputs

-   If show_plot is True then the plot of PDF and CDF will automatically be shown.
-   If print_results is True then the parameters of the most similar distributions will be printed.
-   results - an array of distributions objects ranked in order of best fit.
-   most_similar_distribution - a distribution object. This is the first item from results.

In the example below, we create a Weibull Distribution object using the reliability.Distributions module. We then provide the Weibull Distribution as input to simiar_distributions and the output reveals the top 3 most similar distributions. The optional input of include_location_shifted has been set to False.

.. code:: python

    from reliability.Distributions import Weibull_Distribution
    from reliability.Other_functions import similar_distributions
    dist = Weibull_Distribution(alpha=50,beta=3.3)
    similar_distributions(distribution=dist,include_location_shifted=False)

    '''
    The input distribution was:
    Weibull Distribution (α=50,β=3.3)

    The top 3 most similar distributions are:
    Normal Distribution (μ=44.8471,σ=14.9226)
    Gamma Distribution (α=5.7607,β=7.7849)
    Loglogistic Distribution (α=43.465,β=4.7564)
    '''
    
.. image:: images/similar_distributions_V4.png
