.. image:: images/logo.png

-------------------------------------

Equations of supported distributions
''''''''''''''''''''''''''''''''''''

The following expressions provide the equations for the Probability Density Functions (PDF), Cumulative Distributions Function (CDF), Survival Function (SF), Hazard Function (HF), and Cumulative Hazard Function (CHF) of all supported distributions. Readers should note that there are many ways to write the equations for probability distributions and careful attention should be afforded to the parametrization to ensure you understand each parameter.

Weibull Distribution
====================

:math:`\alpha` = scale parameter (:math:`\alpha > 0`) 

:math:`\beta` = shape parameter (:math:`\beta > 0`)

:math:`\text{PDF:} \qquad f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{CDF:} \qquad F(t) = 1 - e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{SF:} \qquad R(t) = e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{HF:} \qquad h(t) = \beta t^{\beta -1}`

:math:`\text{CHF:} \qquad H(t) = t^{\beta -1}`

Exponential Distribution
========================

:math:`\text{PDF:} \qquad f(t) = 1`

:math:`\text{CDF:} \qquad F(t) = 1`

:math:`\text{SF:  } \qquad R(t) = 1`

:math:`\text{HF:  } \qquad h(t) = 1`

:math:`\text{CHF:} \qquad H(t) = 1`

