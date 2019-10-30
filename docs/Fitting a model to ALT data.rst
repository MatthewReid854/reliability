.. image:: images/logo.png

-------------------------------------

Fitting a model to ALT data
'''''''''''''''''''''''''''

.. note:: This document and the associated functions are a work in progress. This notice will be removed when the models are available in the PyPI release of reliability.

Before reading this section, you should be familiar with `ALT probability plots <https://reliability.readthedocs.io/en/latest/ALT%20probability%20plots.html>`_, and `Fitting distributions <https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html>`_ to non-ALT datasets.

The module ``reliability.ALT`` contains fitting function for 15 different ALT life-stress models. Each model is a combination of the life model with the scale or location parameter replaced with life-stress model. For example, the Weibull-Exponential model is found by replacing the :math:`\alpha` parameter with the equation for the exponential life-stress model as follows:

:math:`\text{Weibull PDF:} \hspace{11mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} {\rm e}^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{Exponential Life-Stress Model:} \hspace{2mm} L(T) = A.exp\left(\frac{a}{T} \right)`

Replacing :math:`\alpha` with L(T) gives:

:math:`\text{Weibull PDF:} \hspace{11mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \left( A.exp\left(\frac{a}{T} \right) \right)^ \beta} {\rm e}^{-(\frac{t}{\left( A.exp\left(\frac{a}{T} \right) })^ \beta }`

The correct substitutions for each type of model are:

:math:`\text{Weibull:} \hspace{11mm} \alpha = L(T)`

:math:`\text{Lognormal:} \hspace{11mm} \mu = ln \left( L(T) \right)`

:math:`\text{Normal:} \hspace{11mm} \mu = L(T)`

The `life models <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ available in ``reliability.ALT`` are:

- Weibull_2P
- Lognormal_2P
- Normal_2P

The life-stress models avialble are:

- Exponential (also used for Arrhenius equation)
- Eyring
- Power (also known as inverse power)
- Dual_Exponential
- Dual_Power_Exponential

.. code:: python

    from reliability.ALT import ALT_probability_plot_Weibull
    
.. image:: images/example.png

