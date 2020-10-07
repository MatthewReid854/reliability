.. image:: images/logo.png

-------------------------------------

Equations of supported distributions
''''''''''''''''''''''''''''''''''''

The following expressions provide the equations for the Probability Density Function (PDF), Cumulative Distribution Function (CDF), Survival Function (SF) (this is the same as the reliability function R(t)), Hazard Function (HF), and Cumulative Hazard Function (CHF) of all supported distributions. Readers should note that there are many ways to write the equations for probability distributions and careful attention should be afforded to the parametrization to ensure you understand each parameter. For more equations of these distributions, see the textbook "Probability Distributions Used in Reliability Engineering" listed in `recommended resources <https://reliability.readthedocs.io/en/latest/Recommended%20resources.html>`_. 

Weibull Distribution
====================

:math:`\alpha` = scale parameter :math:`( \alpha > 0 )`

:math:`\beta` = shape parameter :math:`( \beta > 0 )`

Limits :math:`( t \geq 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} {\rm e}^{-(\frac{t}{\alpha })^ \beta }` 

:math:`\hspace{31mm} = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{(\beta-1)}{\rm e}^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - {\rm e}^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{SF:} \hspace{14mm} R(t) = {\rm e}^{-(\frac{t}{\alpha })^ \beta }`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\beta}{\alpha} (\frac{t}{\alpha})^{\beta -1}`

:math:`\text{CHF:} \hspace{9mm} H(t) = (\frac{t}{\alpha})^{\beta}`

Exponential Distribution
========================

:math:`\lambda` = scale parameter :math:`( \lambda > 0 )`

Limits :math:`( t \geq 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \lambda {\rm e}^{-\lambda t}`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1 - {\rm e}^{-\lambda t}`

:math:`\text{SF:} \hspace{14mm} R(t) = {\rm e}^{-\lambda t}`

:math:`\text{HF:} \hspace{14mm} h(t) = \lambda`

:math:`\text{CHF:} \hspace{9mm} H(t) = \lambda t`

Note that some parametrizations of the Exponential distribution (such as the one in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html>`_) use :math:`\frac{1}{\lambda}` in place of :math:`\lambda`. 

Normal Distribution
===================

:math:`\mu` = location parameter :math:`( -\infty < \mu < \infty )`

:math:`\sigma` = scale parameter :math:`( \sigma > 0 )`

Limits :math:`( -\infty < t < \infty )`

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

Limits :math:`( t \geq 0 )`

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

Limits :math:`( t \geq 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{t^{\beta-1}}{\Gamma(\beta)\alpha^\beta}{\rm e}^{-\frac{t}{\alpha}}`

where :math:`\Gamma(x)` is the complete gamma function. :math:`\Gamma (x) = \int^\infty_0 t^{x-1}{\rm e}^{-t} {\rm d}t`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{\Gamma (\beta)} \gamma\left(\beta,\frac{t}{\alpha}\right)`

where :math:`\gamma(x,y)` is the lower incomplete gamma function. :math:`\gamma (x,y) = \frac{1}{\Gamma(x)} \int^y_0 t^{x-1}{\rm e}^{-t} {\rm d}t`

:math:`\text{SF:} \hspace{14mm} R(t) = \frac{1}{\Gamma (\beta)} \Gamma\left(\beta,\frac{t}{\alpha}\right)`

where :math:`\Gamma(x,y)` is the upper incomplete gamma function. :math:`\Gamma (x,y) = \frac{1}{\Gamma(x)} \int^\infty_y t^{x-1}{\rm e}^{-t} {\rm d}t`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{t^{\beta-1}{\rm exp}\left(-\frac{t}{\alpha}\right)}{\alpha^\beta\Gamma\left(\beta,\frac{t}{\alpha}\right)}`

:math:`\text{CHF:} \hspace{9mm} H(t) = -{\rm ln}\left[\frac{1}{\Gamma (\beta)} \Gamma\left(\beta,\frac{t}{\alpha}\right)\right]`

Note that some parametrizations of the Gamma distribution use :math:`\frac{1}{\alpha}` in place of :math:`\alpha`. There is also an alternative parametrization which uses shape and rate instead of shape and scale. See `Wikipedia <https://en.wikipedia.org/wiki/Gamma_distribution>`_ for an example of this.

Beta Distribution
=================

:math:`\alpha` = shape parameter :math:`( \alpha > 0 )`

:math:`\beta` = shape parameter :math:`( \beta > 0 )`

Limits :math:`(0 \leq t \leq 1 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}.t^{\alpha-1}(1-t)^{\beta-1}`

:math:`\hspace{31mm} =\frac{1}{B(\alpha,\beta)}.t^{\alpha-1}(1-t)^{\beta-1}`

where :math:`\Gamma(x)` is the complete gamma function. :math:`\Gamma (x) = \int^\infty_0 t^{x-1}{\rm e}^{-t} {\rm d}t`

where :math:`B(x,y)` is the complete beta function. :math:`B(x,y) = \int^1_0 t^{x-1}(1-t)^{y-1} {\rm d}t`

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \int^t_0 \theta^{\alpha-1}(1-\theta)^{\beta-1} {\rm d}\theta`

:math:`\hspace{31mm} =\frac{B_t(t|\alpha,\beta)}{B(\alpha,\beta)}`

:math:`\hspace{31mm} =I_t(t|\alpha,\beta)`

where :math:`B_t(t|x,y)` is the incomplete beta function. :math:`B_t(t|x,y) = \int^t_0 \theta^{x-1}(1-\theta)^{y-1} {\rm d}\theta`

where :math:`I_t(t|x,y)` is the regularized incomplete beta function which is defined in terms of the incomplete beta function and the complete beta function. :math:`I_t(t|x,y)=\frac{B_t(t|x,y)}{B(x,y)}`

:math:`\text{SF:} \hspace{14mm} R(t) = 1 - I_t(t|\alpha,\beta)`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{t^{\alpha-1}(1-t)}{B(\alpha,\beta)-B_t(t|\alpha,\beta)}`

:math:`\text{CHF:} \hspace{9mm} H(t) = -{\rm ln}\left[1 - I_t(t|\alpha,\beta)\right]`

Note that there is a parameterization of the Beta distribution that changes the lower and upper limits beyond 0 and 1. For this parametrization, see the reference listed in the opening paragraph of this page.

Loglogistic Distribution
====================

:math:`\alpha` = scale parameter :math:`( \alpha > 0 )`

:math:`\beta` = shape parameter :math:`( \beta > 0 )`

Limits :math:`( t \geq 0 )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{\left( \frac{\beta}{\alpha}\right) {\left( \frac{t}{\alpha} \right)}^{\beta - 1}}{{\left(1+{\left(\frac{t}{\alpha}\right)}^{\beta} \right)}^{2}}` 

:math:`\text{CDF:} \hspace{10mm} F(t) = \frac{1}{1+{\left(\frac{t}{\alpha} \right)}^{-\beta}}`

:math:`\hspace{31mm} = \frac{{\left(\frac{t}{\alpha} \right)}^{\beta}}{1+{\left(\frac{t}{\alpha} \right)}^{\beta}}`

:math:`\hspace{31mm} = \frac{{t}^{\beta}}{{\alpha}^{\beta}+{t}^{\beta}}`

:math:`\text{SF:} \hspace{14mm} R(t) = \frac{1}{1+{\left(\frac{t}{\alpha} \right)}^{\beta}}`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{\left( \frac{\beta}{\alpha}\right) {\left( \frac{t}{\alpha} \right)}^{\beta - 1}}{1+{\left(\frac{t}{\alpha} \right)}^{\beta}}`

:math:`\text{CHF:} \hspace{9mm} H(t) = ln \left(1+{\left(\frac{t}{\alpha} \right)}^{\beta} \right)`

There is another parameterization of the loglogistic distribution using :math:`\mu` and :math:`\sigma` which is designed to look more like the parametrization of the `logistic distribution <https://en.wikipedia.org/wiki/Logistic_distribution>`_ and is related to the above parametrization by :math:`\mu = ln(\alpha)` and :math:`\sigma = \frac{1}{\beta}`. This parametrisation can be found `here <http://reliawiki.org/index.php/The_Loglogistic_Distribution>`_.

Gumbel Distribution
====================

:math:`\mu` = location parameter :math:`( -\infty < \mu < \infty )`

:math:`\sigma` = scale parameter :math:`( \sigma > 0 )`

Limits :math:`( -\infty < t < \infty )`

:math:`\text{PDF:} \hspace{11mm} f(t) = \frac{{\rm e}^{z-{\rm e}^{z}}}{\sigma}`

where :math:`z = \frac{t-\mu}{\sigma}`

:math:`\text{CDF:} \hspace{10mm} F(t) = 1-{\rm e}^{-{\rm e}^{z}}`

:math:`\text{SF:} \hspace{14mm} R(t) = {\rm e}^{-{\rm e}^{z}}`

:math:`\text{HF:} \hspace{14mm} h(t) = \frac{{\rm e}^{z}}{\sigma}`

:math:`\text{CHF:} \hspace{9mm} H(t) = {\rm e}^{z}`

The Gumbel Distribution is similar to the Normal Distribution in the sense that it has a consistent shape, with the location controlled by the location parameter and the spread (or scale) controlled by the scale parameter. The location parameter may be positive or negative, making the Gumbel and Normal Distributions the only distributions within the Python reliability library that can be defined below t=0. There is no threshold (gamma) parameter since the location parameter achieves this.

Location shifting the distributions
===================================

Within ``reliability`` the parametrization of the Exponential, Weibull, Gamma, Lognormal, and Loglogistic distributions allows for location shifting using the gamma parameter. This will simply shift the distribution's lower limit to the right from 0 to :math:`\gamma`. In the location shifted form of the distributions, the equations listed above are almost identical, except everywhere you see :math:`t` replace it with :math:`t - \gamma`. The reason for using the location shifted form of the distribution is because some phenomena that can be modelled well by a certain probability distribution do not begin to occur immediately, so it becomes necessary to shift the lower limit of the distribution so that the data can be accurately modelled by the distribution.

If implementing this yourself, ensure you set all y-values to 0 for :math:`t \leq \gamma` as the raw form of the location shifted distributions above will not automatically zeroise these values for you and may result in negative values. This zeroizing is done automatically within ``reliability``.

Relationships between the five functions
========================================

The PDF, CDF, SF, HF, CHF of a probability distribution are inter-related and any of these functions can be obtained by applying the correct transformation to any of the others. The following list of transformations are some of the most useful:

:math:`{\rm PDF} = \frac{d}{dt} {\rm CDF}`

:math:`{\rm CDF} = \int_{-\infty}^t {\rm PDF}`

:math:`{\rm SF} = 1 - {\rm CDF}`

:math:`{\rm HF} = \frac{{\rm PDF}}{{\rm SF}}`

:math:`{\rm CHF} = -{\rm ln} \left({\rm SF} \right)`
