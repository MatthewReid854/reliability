.. image:: images/logo.png

-------------------------------------

Equations of supported distributions
''''''''''''''''''''''''''''''''''''

The following expressions provide the equations for the Probability Density Function (PDF), Cumulative Distribution Function (CDF), Survival Function (SF), Hazard Function (HF), and Cumulative Hazard Function (CHF) of all supported distributions. Readers should note that there are many ways to write the equations for probability distributions and careful attention should be afforded to the parametrization to ensure you understand each parameter. For more equations of these distributions, see the textbook "Probability Distributions Used in Reliability Engineering" listed in `recommended resources <https://reliability.readthedocs.io/en/latest/Recommended%20resources.html>`_. 

Weibull Distribution
====================

:math:`\alpha` = scale parameter :math:`( \alpha > 0 )`

:math:`\beta` = shape parameter :math:`( \beta > 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{SF:} \hspace{14mm} R(t) = e^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\beta}{\alpha} (\frac{t}{\alpha})^{\beta -1}`

:math:`\text{CHF:} \hspace{9mm} H(t) = (\frac{t}{\alpha})^{\beta}`

Exponential Distribution
========================

:math:`\lambda` = scale parameter :math:`( \lambda > 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \lambda {\rm e}^{-\lambda t}`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - {\rm e}^{-\lambda t}`

:math:`\text{SF:} \hspace{14mm} R(t) = {\rm e}^{-\lambda t}`

:math:`\text{HF:} \hspace{14mm} h(t) = \lambda`

:math:`\text{CHF:} \hspace{9mm} H(t) = \lambda t`

Normal Distribution
===================

:math:`\mu` = location parameter :math:`( -\infty < \mu < \infty )`

:math:`\sigma` = scale parameter :math:`( \sigma > 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{1}{\sigma \sqrt{2 \pi}}{\rm exp}\left[-\frac{1}{2}\left(\frac{t - \mu}{\sigma}\right)^2\right]`

:math:`\hspace{31mm} = \frac{1}{\sigma}\phi \left[ \frac{t - \mu}{\sigma} \right]`

where :math:`\phi` is the standard normal PDF with :math:`\mu = 0` and :math:`\sigma=1`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{\sigma \sqrt{2 \pi}} \int^t_{-\infty} {\rm exp}\left[-\frac{1}{2}\left(\frac{\theta - \mu}{\sigma}\right)^2\right] {\rm d} \theta`

:math:`\hspace{31mm} =\frac{1}{2}+\frac{1}{2}{\rm erf}\left(\frac{t - \mu}{\sigma \sqrt{2}}\right)`

:math:`\hspace{31mm} = \Phi \left( \frac{t - \mu}{\sigma} \right)`

where :math:`\Phi` is the standard normal CDF with :math:`\mu = 0` and :math:`\sigma=1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1 - \Phi \left( \frac{t - \mu}{\sigma} \right)`

:math:`\hspace{31mm} = \Phi \left( \frac{\mu - t}{\sigma} \right)`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\phi \left[\frac{t-\mu}{\sigma}\right]}{\sigma \left( \Phi \left[ \frac{\mu - t}{\sigma} \right] \right)}`

:math:`\text{CHF:} \hspace{9mm} H(t) = -{\rm ln}\left[\Phi \left(\frac{\mu - t}{\sigma}\right)\right]`

Lognormal Distribution
======================

:math:`\mu` = scale parameter :math:`( -\infty < \mu < \infty )`

:math:`\sigma` = shape parameter :math:`( \sigma > 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{1}{\sigma t \sqrt{2\pi}} {\rm exp} \left[-\frac{1}{2} \left(\frac{{\rm ln}(t)-\mu}{\sigma}\right)^2\right]`

:math:`\hspace{31mm} = \frac{1}{\sigma t}\phi \left[ \frac{{\rm ln}(t) - \mu}{\sigma} \right]`

where :math:`\phi` is the standard normal PDF with :math:`\mu = 0` and :math:`\sigma=1`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{\sigma \sqrt{2\pi}} \int^t_0 \frac{1}{\theta} {\rm exp} \left[-\frac{1}{2} \left(\frac{{\rm ln}(\theta)-\mu}{\sigma}\right)^2\right] {\rm d}\theta`

:math:`\hspace{31mm} =\frac{1}{2}+\frac{1}{2}{\rm erf}\left(\frac{{\rm ln}(t) - \mu}{\sigma \sqrt{2}}\right)`

:math:`\hspace{31mm} = \Phi \left( \frac{{\rm ln}(t) - \mu}{\sigma} \right)`

where :math:`\Phi` is the standard normal CDF with :math:`\mu = 0` and :math:`\sigma=1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1 - \Phi \left( \frac{{\rm ln}(t) - \mu}{\sigma} \right)`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\phi \left[ \frac{{\rm ln}(t) - \mu}{\sigma} \right]}{t \sigma \left(1 - \Phi \left( \frac{{\rm ln}(t) - \mu}{\sigma} \right)\right)}`

:math:`\text{CHF:} \hspace{9mm} H(t) = -{\rm ln}\left[1 - \Phi \left( \frac{{\rm ln}(t) - \mu}{\sigma} \right)\right]`

Gamma Distribution
==================

:math:`\alpha` = scale parameter :math:`( \alpha > 0 )`

:math:`\beta` = shape parameter :math:`( \beta > 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{t^{\beta-1}}{\Gamma(\beta)\alpha^\beta}{\rm e}^{-\frac{t}{\alpha}}`

where :math:`\Gamma(z)` is the Gamma function. :math:`\Gamma (z) = \int^\infty_0 x^{z-1}{\rm e}^{-x} {\rm d}x`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{\Gamma (\beta)} \gamma\left(\beta,\frac{t}{\alpha}\right)`

where :math:`\gamma(z,t)` is the lower incomplete gamma function. :math:`\gamma (z,t) = \frac{1}{\Gamma(z)} \int^t_0 x^{z-1}{\rm e}^{-x} {\rm d}x`

:math:`\text{SF:} \hspace{14mm} R(t) = \frac{1}{\Gamma (\beta)} \Gamma\left(\beta,\frac{t}{\alpha}\right)`

where :math:`\Gamma(z,t)` is the upper incomplete gamma function. :math:`\gamma (z,t) = \frac{1}{\Gamma(z)} \int^\infty_t x^{z-1}{\rm e}^{-x} {\rm d}x`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{t^{\beta-1}{\rm exp}\left(-\frac{t}{\alpha}\right)}{\alpha^\beta\Gamma\left(\beta,\frac{t}{\alpha}\right)}`

:math:`\text{CHF:} \hspace{9mm} H(t) = -{\rm ln}\left[\frac{1}{\Gamma (\beta)} \Gamma\left(\beta,\frac{t}{\alpha}\right)\right]`

Note that some parametrizations of the Gamma distribution use :math:`\frac{1}{\alpha}` in place of :math:`\alpha`. There is also an alternative parametrization which uses shape and rate instead of shape and scale. See `Wikipedia <https://en.wikipedia.org/wiki/Gamma_distribution>`_ for an example of this.

Beta Distribution
=================

This is a work in progress. Check back soon for more equations.

:math:`\text{PDF:} \hspace{11mm} f(t) = 1`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1`

:math:`\text{SF:} \hspace{14mm} R(t) = 1`

:math:`\text{HF:} \hspace{14mm} h(t) = 1`

:math:`\text{CHF:} \hspace{9mm} H(t) = 1`

Relationships between the five functions
========================================

The PDF, CDF, SF, HF, CHF of a probability distribution are inter-related and any of these functions can be obtained by applying the correct transformation to any of the others. The following list of transformations are some of the most useful:

:math:`{\rm PDF} = \frac{d}{dt} {\rm CDF}`

:math:`{\rm CDF} = \int_{-\infty}^t {\rm PDF}`

:math:`{\rm SF} = 1 - {\rm CDF}`

:math:`{\rm HF} = \frac{{\rm PDF}}{{\rm SF}}`

:math:`{\rm CHF} = -{\rm ln} \left({\rm SF} \right)`
