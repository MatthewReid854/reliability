.. image:: images/logo.png

-------------------------------------

Fitting a model to ALT data
'''''''''''''''''''''''''''

.. note:: This document and the associated functions are a work in progress. This notice will be removed when the models are available in the PyPI release of reliability.

Before reading this section, you should be familiar with `ALT probability plots <https://reliability.readthedocs.io/en/latest/ALT%20probability%20plots.html>`_, and `Fitting distributions <https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html>`_ to non-ALT datasets.

The module ``reliability.ALT`` contains fitting function for 15 different ALT life-stress models. Each model is a combination of the life model with the scale or location parameter replaced with life-stress model. For example, the Weibull-Exponential model is found by replacing the :math:`\alpha` parameter with the equation for the exponential life-stress model as follows:

:math:`\text{Weibull PDF:} \hspace{40mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} exp \left(-(\frac{t}{\alpha })^ \beta \right)`

:math:`\text{Exponential Life-Stress Model:} \hspace{5mm} L(T) = b.exp\left(\frac{a}{T} \right)`

Replacing :math:`\alpha` with :math:`L(T)` gives the PDF of the Weibull-Exponential model:

:math:`\text{Weibull-Exponential:} \hspace{25mm} f(t,T) = \frac{\beta t^{ \beta - 1}}{ \left(b.exp\left(\frac{a}{T} \right) \right)^ \beta} exp \left(-\left(\frac{t}{\left(b.exp\left(\frac{a}{T} \right) \right) }\right)^ \beta \right)` 

The correct substitutions for each type of model are:

:math:`\text{Weibull:} \hspace{12mm} \alpha = L(T)`

:math:`\text{Normal:} \hspace{12mm} \mu = L(T)`

:math:`\text{Lognormal:} \hspace{5mm} \mu = ln \left( L(T) \right)`

The `life models <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ available are:

- Weibull_2P
- Lognormal_2P
- Normal_2P

The life-stress models available are:

:math:`\text{Exponential (also used for Arrhenius equation):} \hspace{32mm} L(T)=b.exp \left(\frac{a}{T} \right)`

:math:`\text{Eyring:} \hspace{110mm} L(T)= \frac{1}{T} .exp \left( - \left( c - \frac{a}{T} \right) \right)`

:math:`\text{Power (also known as inverse power):} \hspace{51mm} L(S)=a .S^n`

:math:`\text{Dual-Exponential (also known as Temperature-Humidity):} \hspace{7mm} L(T,H)=c.exp \left(\frac{a}{T} + \frac{b}{H} \right)`

:math:`\text{Power-Exponential (also known as Thermal-Non-Thermal):} \hspace{5mm} L(T,S)=c.S^n.exp \left(\frac{a}{T} \right)`

.. code:: python

    from reliability.ALT import ALT_probability_plot_Weibull
    
.. image:: images/example.png

