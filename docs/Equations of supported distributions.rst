.. image:: images/logo.png

-------------------------------------

Equations of supported distributions
''''''''''''''''''''''''''''''''''''

The following expressions provide the equations for the Probability Density Function (PDF), Cumulative Distribution Function (CDF), Survival Function (SF), Hazard Function (HF), and Cumulative Hazard Function (CHF) of all supported distributions. Readers should note that there are many ways to write the equations for probability distributions and careful attention should be afforded to the parametrization to ensure you understand each parameter.

Weibull Distribution
====================

:math:`\alpha` = scale parameter (:math:`\alpha > 0`) 

:math:`\beta` = shape parameter (:math:`\beta > 0`)

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{SF:} \hspace{14mm} R(t) = e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\beta}{\alpha} (\frac{t}{\alpha})^{\beta -1}`

:math:`\text{CHF:} \hspace{10mm} H(t) = (\frac{t}{\alpha})^{\beta}`

Exponential Distribution
========================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = \lambda e^{-\lambda t}`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - e^{-\lambda t}`

:math:`\text{SF:} \hspace{14mm} R(t) = e^{-\lambda t}`

:math:`\text{HF:} \hspace{14mm} h(t) = \lambda`

:math:`\text{CHF:} \hspace{10mm} H(t) = \lambda t`

Normal Distribution
===================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{1}{\sigma \sqrt{2 \pi}}exp\left[-\frac{1}{2}(\frac{t - \mu}{\sigma})^2\right]`

:math:`\hspace{30mm} = \frac{1}{\sigma}\psi \left[ \frac{t - \mu}{\sigma} \right]`

where :math:`\psi` is the standard normal PDF with :math:`\mu = 0` and :math:`\sigma=1`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{\sigma \sqrt{2 \pi}} \int^t_{-\inf} exp\left[-\frac{1}{2}(\frac{\theta - \mu}{\sigma})^2\right] d \theta`

:math:`\hspace{30mm} =\frac{1}{2}+\frac{1}{2}erf\left(\frac{t - \mu}{\sigma \sqrt{2}}\right)`

:math:`\text{SF:} \hspace{14mm} R(t) = 1 - F(t)`

:math:`\text{HF:} \hspace{14mm} h(t) = 1`

:math:`\text{CHF:} \hspace{10mm} H(t) = 1`

Lognormal Distribution
======================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = 1`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1`

:math:`\text{HF:} \hspace{14mm} h(t) = 1`

:math:`\text{CHF:} \hspace{10mm} H(t) = 1`

Gamma Distribution
==================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = 1`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1`

:math:`\text{HF:} \hspace{14mm} h(t) = 1`

:math:`\text{CHF:} \hspace{10mm} H(t) = 1`

Beta Distribution
=================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = 1`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1`

:math:`\text{HF:} \hspace{14mm} h(t) = 1`

:math:`\text{CHF:} \hspace{10mm} H(t) = 1`
