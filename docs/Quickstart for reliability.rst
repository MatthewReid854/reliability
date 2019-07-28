.. _code_directive:

.. image:: https://i.imgur.com/P7sjQ0C.png

-------------------------------------


Quickstart for reliability
''''''''''''''''''''''''''


Installation
------------

Install via ``pip``:

.. code-block:: console

    pip install reliability




A quick example:
----------------

In this example, we will create a Weibull Distribution, and from that distribuion we will draw 100 random samples. Using those samples we will obtain a non-parametric estimate of the survival function using the Kaplan-Meier method. Finally, we will plot the parametric and non-parametric distribuions together to see how they compare.

.. code-block:: console

    from reliability.Distributions import Weibull_Distribution
    from reliability.Nonparametric import KaplanMeier
    import numpy as np
    import matplotlib.pyplot as plt

    dist = Weibull_Distribution(alpha=30,beta=2)
    failures = dist.random_samples(100)
    xvals = np.linspace(0,max(failures),1000)
    KaplanMeier(failures=failures,label='Non-parametric')
    dist.SF(xvals=xvals,label='Parametric')
    plt.legend()
    plt.show()



.. image:: images/parametric_vs_nonparametric.png


A key feature of ``reliability`` is that probability distributions are created as objects, and these objects have many properties (such as the mean) that are set once the parameters of the distribution are defined. Using the dot operator allows us to access these properties as well as a large number of methods (such as drawing random samples as seen in the example above).

Each distribution may be visualised in five different plots. These are the Probability Density Function (PDF), Cumulative Distribution Function (CDF), Survival Function (SF) [also known as the reliabilty function], Hazard Function (HF), and the Cumulative Hazard Function (CHF). Accessing the plot of any of these is as easy as any of the other methods. Eg. ``dist.SF()`` in the above example is what plots the survival function.



