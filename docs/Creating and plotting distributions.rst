.. _code_directive:

.. image:: images/logo.png

-------------------------------------


Creating and plotting distributions
'''''''''''''''''''''''''''''''''''

Probability distributions within ``reliability`` are Python objects, which allows us to specify just the type of distribution and parameters, and from that we can access a large number of attributes and other methods. There are 6 different probability distributions available in ``reliability``. These are:
- Weibull Distribution
- Exponential Distribution
- Gamma Distribution
- Normal Distribution
- Lognormal Distribution
- Beta Distribution

Understanding how to create and plot distributions is easiest with an example. The following code will create a Lognormal Distribution with parameters mu=5 and sigma=1. From this distribution, we will use the plot() method which provides a quick way to visualise the five functions and also provides a summary of the descriptive statistics.

.. code:: python

    from reliability.Distributions import Lognormal_Distribution
    dist = Lognormal_Distribution(mu=5,sigma=1)
    dist.plot()

.. image:: images/Lognormal_plot1.png


Section title
-------------

