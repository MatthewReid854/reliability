.. image:: images/logo.png

-------------------------------------

ALT probability plots
'''''''''''''''''''''

.. note:: This module is currently in development. The following documentation is correct, however, the most recent version of ``reliability`` on PyPI will not contain this module until Dec 2019.

Before reading this section, you should be familiar with what a probability plot is and how to use it. For a detailed explaination, please see the section on `probability plots <https://reliability.readthedocs.io/en/latest/Probability%20plots.html>`_.

The module ``reliability.ALT`` contains four ALT probability plotting functions. These functions are:

- ALT_probability_plot_Weibull
- ALT_probability_plot_Lognormal
- ALT_probability_plot_Normal
- ALT_probability_plot_Gamma

An ALT probability plot produces a multi-dataset probability plot which includes the probability plots for the data and the fitted distribution at each stress level, as well as a refitted distribution assuming a common shape parameter. All of these functions perform in a similar way, with the main difference being the distribution that is fitted. The Gamma ALT probability plot will not appear parallel because of the way the Gamma distribution works, but it is still useful to judge the goodness of fit to the entire dataset using a common shape parameter.

When producing the ALT probability plot, the function automates the following process; fit a distribution to the data for each unique stress level, find the common shape parameter (several methods are provided), refit the distribution to the data for each unique stress level whilst forcing the shape parameter to be equal to the common shape parameter, plot the data along with the original and new fitted distributions, calculate the change in the common shape parameter from the original shape parametter to see if the model is applicable to this dataset. Each of the ALT plotting functions listed above has the following inputs and outputs.

Inputs:

- failures - an array or list of all the failure times
- failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
- right_censored - an array or list of all the right censored failure times
- right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored datapoint was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
- print_results - True/False. Default is True
- show_plot - True/False. Default is True
- common_beta_method - 'BIC','weighted_average','average'. Default is 'BIC'. This is the method used to obtain the common_beta parameter. 'BIC' will find the common_beta that gives lowest total BIC (equivalent to the best overall fit), 'weighted_average' will perform a weighted average based on the amount of data (failures and right censored) for each stress, 'average' is simply the average. Note for the Lognormal and Normal plots, this variable is "common_sigma_method" as we are forcing sigma to be a common value.

Outputs:

- The plot will be produced if show_plot is True
- A dataframe of the fitted distributions parameters will be printed if print_results is True
- results - a dataframe of the fitted distributions parameters and change in the shape parameter
- common_beta - the common beta parameter. Note in the Lognormal and Normal plots, this variable is "common_sigma"

The time to run the function will be a few seconds if you have a large amount of data and the common_beta_method is set to 'BIC'. This is because the distributions need to be refitted for each iteration of the optimizer (which is usually around 20 to 30 iterations). With 100 datapoints this should take less than 5 seconds for the 'BIC' method, and less than 1 second for the 'average' and 'weighted_average' methods. The more data you have, the longer it will take, so please be patient as a lot of computation is required.

In the following example we will use a dataset from ``reliability.Datasets`` which contains failures and right_censored data for three stress levels. This is done using the Weibull and Lognormal ALT probability plots which are shown together for comparison.




.. code:: python

    from reliability.ALT import ALT_probability_plot_Weibull
    
.. image:: images/fix.png


Getting your input data in the right format
-------------------------------------------




What does an ALT probability plot show me?
------------------------------------------

An ALT probability plot shows
