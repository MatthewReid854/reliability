.. image:: images/logo.png

-------------------------------------

Competing risk models
'''''''''''''''''''''

.. note:: This documentation is valid for Version 0.5.2 which is currently unreleased.

What are competing risks models?
================================

Competing risks models are a combination of two or more distributions that represent failure modes which are "competing" to end the life of the system being modelled. This model is similar to a mixture model in the sense that it uses multiple distributions to create a new model that has a shape with more flexibility than a single distribution. However, unlike a mixture models, we are not adding proportions of the PDF or CDF, but are instead multiplying the survival functions. The formula for the competing risks model is typically written in terms of the survival function (SF). Since we may consider the system's reliability to depend on the reliability of all the parts of the system (each with its own failure modes), the equation is written as if the system was in series, using the product of the survival functions for each failure mode. For a competing risks model with 2 distributions, the equations are shown below:

:math:`{SF}_{Competing\,Risks} = {SF}_1 \times {SF}_2`

:math:`{CDF}_{Competing\,Risks} = 1-{SF}_{Competing\,Risks}`

Since :math:`{SF} = exp(-CHF)` we may equivalently write the competing risks model in terms of the hazard or cumulative hazard function as:

:math:`{HF}_{Competing\,Risks} = {HF}_1 + {HF}_2`

:math:`{CHF}_{Competing\,Risks} = {CHF}_1 + {CHF}_2`

:math:`{PDF}_{Competing\,Risks} = {HF}_{Competing\,Risks} \times {SF}_{Competing\,Risks}`

Another option to obtain the PDF, is to find the derivative of the CDF. This is easiest to do numerically since the formula for the SF of the competing risks model can become quite complex as more risks are added. Note that using the PDF = HF x SF method requires the conversion of nan to 0 in the PDF for high xvals. This is because the HF of the component distributions is obtained using PDF/SF and for the region where the SF and PDF of the component distributions is 0 the resulting HF will be nan.

The image below illustrates the difference between the competing risks model and the mixture model, each of which is made up of the same two component distributions. Note that the PDF of the competing risks model is always equal to or to the left of the component distributions, and the CDF is equal to or higher than the component distributions. This shows how a failure mode that occurs earlier in time can end the lives of units under observation before the second failure mode has the chance to. This behaviour is characteristic of real systems which experience multiple failure modes, each of which could cause system failure.

.. image:: images/CRvsMM.png

Competing risks models are useful when there is more than one failure mode that is generating the failure data. This can be recognised by the shape of the PDF and CDF being outside of what any single distribution can accurately model. On a probability plot, a combination of failure modes can be identified by bends in the data that you might otherwise expect to be linear. An example of this is shown in the image below. You should not use a competing risks model just because it fits your data better than a single distribution, but you should use a competing risks model if you suspect that there are multiple failure modes contributing to the failure data you are observing. To judge whether a competing risks model is justified, look at the goodness of fit criterion (AICc or BIC) which penalises the score based on the number of parameters in the model. The closer the goodness of fit criterion is to zero, the better the fit.

See also `mixture models <https://reliability.readthedocs.io/en/latest/Mixture%20models.html>`_ for another method of combining distributions using the sum of the CDF rather than the product of the SF.

.. image:: images/CRprobplot.png

Creating a competing risks model
================================

Within ``reliability.Distributions`` is the Competing_Risks_Model. This class accepts an array or list of distribution objects created using the reliability.Distributions module (available distributions are Exponential, Weibull, Normal, Lognormal, Gamma, Beta). There is no limit to the number of components you can add to the model, but is is generally preferable to use as few as are required to fit the data appropriately (typically 2 or 3). Unlike the mixture model, you do not need to specify any proportions.

As this process is multiplicative for the survival function (or additive for the hazard function), and may accept many distributions of different types, the mathematical formulation quickly gets complex. For this reason, the algorithm combines the models numerically rather than empirically so there are no simple formulas for many of the descriptive statistics (mean, median, etc.). Also, the accuracy of the model is dependent on xvals. If the xvals array is small (<100 values) then the answer will be “blocky” and inaccurate. The variable xvals is only accepted for PDF, CDF, SF, HF, and CHF. The other methods (like random samples) use the default xvals for maximum accuracy. The default number of values generated when xvals is not given is 1000. Consider this carefully when specifying xvals in order to avoid inaccuracies in the results.

The API is similar to the other probability distributions (Weibull, Normal, etc.) and has the following inputs and methods:

Inputs:

-   distributions - a list or array of probability distributions used to construct the model

Methods:

-   name - 'Competing risks'
-   name2 - 'Competing risks using 3 distributions'
-   mean
-   median
-   mode
-   variance
-   standard_deviation
-   skewness
-   kurtosis
-   excess_kurtosis
-   b5 - The time where 5% have failed. Same as quantile(0.05)
-   b95 - The time where 95% have failed. Same as quantile(0.95)
-   plot() - plots all functions (PDF,CDF,SF,HF,CHF)
-   PDF() - plots the probability density function
-   CDF() - plots the cumulative distribution function
-   SF() - plots the survival function (also known as reliability function)
-   HF() - plots the hazard function
-   CHF() - plots the cumulative hazard function
-   quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing. Also known as b life where b5 is the time at which 5% have failed.
-   inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
-   mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time. Effectively the mean of the remaining amount (right side) of a distribution at a given time.
-   stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
-   random_samples() - draws random samples from the distribution to which it is applied.

The following example shows how the Competing_Risks_Model object can be created, visualised and used.

.. code:: python

    from reliability.Distributions import Lognormal_Distribution, Gamma_Distribution, Weibull_Distribution, Competing_Risks_Model
    import matplotlib.pyplot as plt

    # create the competing risks model
    d1 = Lognormal_Distribution(mu=4, sigma=0.1)
    d2 = Weibull_Distribution(alpha=50, beta=2)
    d3 = Gamma_Distribution(alpha=30,beta=1.5)
    CR_model = Competing_Risks_Model(distributions=[d1, d2, d3])


    # plot the 5 functions using the plot() function
    CR_model.plot(xmin=0,xmax=100)

    # plot the PDF and CDF
    plot_components = True
    plt.figure(figsize=(9, 5))
    plt.subplot(121)
    CR_model.PDF(plot_components=plot_components, color='red', linestyle='--',xmin=0,xmax=130)
    plt.subplot(122)
    CR_model.CDF(plot_components=plot_components, color='red', linestyle='--',xmin=0,xmax=130)
    plt.subplots_adjust(left=0.1, right=0.95)
    plt.show()

    # extract the mean of the distribution
    print('The mean of the distribution is:', CR_model.mean)

    '''
    The mean of the distribution is: 27.04449126275214
    '''

.. image:: images/CR_model_plot.png

.. image:: images/CR_model_PDF_CDF.png

Fitting a competing risks model
===============================

This section will be written soon

.. note:: This documentation is valid for Version 0.5.2 which is currently unreleased.
