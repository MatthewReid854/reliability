.. image:: images/logo.png

-------------------------------------

Equations of ALT models
'''''''''''''''''''''''

Constructing an ALT model
"""""""""""""""""""""""""

ALT models are probability distributions with a stress dependent model replacing their scale (or rate) parameter. For example, the Weibull-Exponential model is obtained by replacing the :math:`\alpha` parameter with the equation for the exponential life-stress model as follows:

:math:`\text{Weibull PDF:} \hspace{40mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} .exp \left(-(\frac{t}{\alpha })^ \beta \right)`

:math:`\text{Exponential Life-Stress model:} \hspace{5mm} L(S) = b.exp\left(\frac{a}{S} \right)`

Replacing :math:`\alpha` with :math:`L(S)` gives the PDF of the Weibull-Exponential model:

:math:`\text{Weibull_Exponential:} \hspace{25mm} f(t,S) = \frac{\beta t^{ \beta - 1}}{ \left(b.exp\left(\frac{a}{S} \right) \right)^ \beta} .exp \left(-\left(\frac{t}{\left(b.exp\left(\frac{a}{S} \right) \right) }\right)^ \beta \right)` 

By replacing the scale parameter with a stress dependent model, the scale parameter of the distribution can be varied as the stress varies. The shape parameter (:math:`\beta` in the above example) is kept constant. On a probability plot (which is scaled appropriately such that the distribution appears as a straight line), the process of changing the scale parameter has the effect of moving the line to the left or right.

ALT models can use any probability distribution which does not scale the axes based on the shape or scale parameters. The Gamma and Beta distributions do scale their axes based on their parameters (which is why you'll never find gamma or beta probability paper) so these probability distributions could not be used for ALT models. Within `reliability` the Weibull_2P, Exponential_1P, Lognormal_2P, and Normal_2P `distributions <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ are used.

In the above example we saw that :math:`\alpha` was replaced with the life model L(S). A direct substitution is not always the case. The correct substitutions for each of the four models used in `reliability` are as follows:

:math:`\text{Weibull:} \hspace{12mm} \alpha = L(S)`

:math:`\text{Normal:} \hspace{12mm} \mu = L(S)`

:math:`\text{Lognormal:} \hspace{5mm} \mu = ln \left( L(S) \right)`

:math:`\text{Exponential:} \hspace{3mm} \lambda = \frac{1}{L(S)}`

The life-stress models available are:

:math:`\text{Exponential (also used for Arrhenius equation):} \hspace{29mm} L(S) = b.exp \left(\frac{a}{S} \right)`

:math:`\text{Eyring:} \hspace{108mm} L(S) = \frac{1}{S} .exp \left( - \left( c - \frac{a}{S} \right) \right)`

:math:`\text{Power (also known as inverse power):} \hspace{48mm} L(S) = a.S^n`

:math:`\text{Dual_Exponential (also known as Temperature-Humidity):} \hspace{7mm} L(S_1,S_2) = c.exp \left(\frac{a}{S_1} + \frac{b}{S_2} \right)`

:math:`\text{Dual_Power (also known as Non-Thermal-Non-Thermal):} \hspace{4mm} L(S1,S2)=c.{S_1}^m.{S_2}^n`

:math:`\text{Power_Exponential (also known as Thermal-Non-Thermal):} \hspace{4mm} L(S1,S2) = c.exp \left(\frac{a}{S1} \right).S_2^n`

Note that while this model is named "Power_Exponential" (keeping in line with academic literature), it would be more appropriate to call it the Exponential_Power model since the stresses are modelled in the "Thermal-Non-Thermal" stress order. This means that the first stress (S1) is modelled by the exponential model (typically used for thermal stresses) and the second stress (S2) is modelled by the power model (typically used for non-thermal stresses). The model may perform differently if given S1 and S2 in the opposite order.
