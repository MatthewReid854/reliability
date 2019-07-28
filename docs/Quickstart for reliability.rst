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

In this example, we will create a Weibull Distribution, and from that distribuion we will draw 100 random samples. To those samples we will obtain a non-parametric estimate of the survival function using the Kaplan-Meier method. Finally, we will plot the parametric and non-parametric distribuions together to see how they compare.

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



