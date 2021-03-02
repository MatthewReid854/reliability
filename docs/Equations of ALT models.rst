.. image:: images/logo.png

-------------------------------------

Equations of ALT models
'''''''''''''''''''''''

Constructing an ALT model
"""""""""""""""""""""""""

ALT models are probability distributions with a stress dependent model replacing their scale (or rate) parameter. For example, the Weibull-Exponential model is obtained by replacing the :math:`\alpha` parameter with the equation for the exponential life-stress model as follows:

:math:`\text{Weibull PDF:} \hspace{40mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} .{\rm exp} \left( -\left(\frac{t}{\alpha }\right)^ \beta \right)`

:math:`\text{Exponential Life-Stress model:} \hspace{5mm} L(S) = b.{\rm exp} \left( \frac{a}{S} \right)`

Replacing :math:`\alpha` with :math:`L(S)` gives the PDF of the Weibull-Exponential model:

:math:`\text{Weibull-Exponential:} \hspace{25mm} f(t,S) = \frac{\beta t^{ \beta - 1}}{ \left(b.{\rm exp}\left(\frac{a}{S} \right) \right)^ \beta} .{\rm exp} \left(-\left(\frac{t}{b.{\rm exp}\left(\frac{a}{S} \right)}\right)^ \beta \right)`

By replacing the scale parameter with a stress dependent model, the scale parameter of the distribution can be varied as the stress varies. The shape parameter (:math:`\beta` in the above example) is kept constant. On a probability plot (which is scaled appropriately such that the distribution appears as a straight line), the process of changing the scale parameter has the effect of moving the line to the left or right.

ALT models can use any probability distribution which does not scale the axes based on the shape or scale parameters. The Gamma and Beta distributions do scale their axes based on their parameters (which is why you'll never find Gamma or Beta probability paper) so these probability distributions could not be used for ALT models. Within `reliability` the Weibull_2P, Exponential_1P, Lognormal_2P, and Normal_2P `distributions <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ are used.

In the above example we saw that :math:`\alpha` was replaced with the life model L(S). A direct substitution is not always the case. The correct substitutions for each of the four models used in `reliability` are as follows:

:math:`\text{Weibull:} \hspace{12mm} \alpha = L(S)`

:math:`\text{Normal:} \hspace{12mm} \mu = L(S)`

:math:`\text{Lognormal:} \hspace{5mm} \mu = ln \left( L(S) \right)`

:math:`\text{Exponential:} \hspace{3mm} \lambda = \frac{1}{L(S)}`

The life-stress models available within `reliability` are:

:math:`\text{Exponential:} \hspace{56mm} L(S) = b.{\rm exp} \left(\frac{a}{S} \right)`

:math:`\text{(also known as Arrhenius)}\hspace{30mm} \text{limits:}\hspace{2mm}(-\infty < a < \infty)\hspace{1mm},\hspace{1mm} (b > 0)`

:math:`\text{Eyring:} \hspace{67mm} L(S) = \frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right)`

:math:`\hspace{82mm} \text{limits:}\hspace{2mm}(-\infty < a < \infty)\hspace{1mm},\hspace{1mm} (-\infty < c < \infty)`

:math:`\text{Power:} \hspace{68mm} L(S) = a.S^n`

:math:`\text{(also known as Inverse Power Law)}\hspace{12mm} \text{limits:}\hspace{2mm}(a > 0)\hspace{1mm},\hspace{1mm} (-\infty < n < \infty)`

:math:`\text{Dual-Exponential:} \hspace{45mm} L({S_1},{S_2}) = c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right)`

:math:`\text{(also known as Temperature-Humidity)}\hspace{4mm} \text{limits:}\hspace{2mm}(-\infty < a < \infty)\hspace{1mm},\hspace{1mm} (-\infty < b < \infty)\hspace{1mm},\hspace{1mm}(c > 0)`

:math:`\text{Dual-Power:} \hspace{57mm} L(S_1,S_2) = c.S_1^m.S_2^n`

:math:`\hspace{82mm} \text{limits:}\hspace{2mm}(c > 0)\hspace{1mm},\hspace{1mm} (-\infty < m < \infty)\hspace{1mm},\hspace{1mm}(-\infty < n < \infty)`

:math:`\text{Power-Exponential:} \hspace{42mm} L(S_1,S_2) = c.{\rm exp} \left(\frac{a}{S_1} \right).S_2^n`

:math:`\text{(also known as Thermal-Nonthermal)}\hspace{7mm} \text{limits:}\hspace{2mm}(-\infty < a < \infty)\hspace{1mm},\hspace{1mm} (c>0)\hspace{1mm},\hspace{1mm}(-\infty < n < \infty)`

Note that while this last model is named "Power-Exponential" (keeping in line with academic literature), it would be more appropriate to call it the Exponential-Power model since the stresses are modelled in the "Thermal-Nonthermal" stress order. This means that the first stress (:math:`S_1`) is modelled by the Exponential model (typically used for thermal stresses) and the second stress (:math:`S_2`) is modelled by the Power model (typically used for nonthermal stresses). The model may perform quite differently if given :math:`S_1` and :math:`S_2` in the opposite order.

Since each ALT model is a combination of a life model (Weibull, Exponential, Lognormal, Normal) and a life-stress model (Exponential, Eyring, Power, Dual-Exponential, Dual-Power, Power-Exponential), there are 24 possible models (12 for single stress and 12 for dual stress).

Weibull ALT models
""""""""""""""""""

:math:`\text{Weibull-Exponential:} \hspace{18mm} f(t,S) = \frac{\beta t^{ \beta - 1}}{ \left(b.{\rm exp}\left(\frac{a}{S} \right) \right)^ \beta} .{\rm exp} \left(-\left(\frac{t}{b.{\rm exp}\left(\frac{a}{S} \right) }\right)^ \beta \right)` 

:math:`\text{Weibull-Eyring:} \hspace{28mm} f(t,S) = \frac{\beta t^{ \beta - 1}}{ \left(\frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right) \right)^ \beta} .{\rm exp} \left(-\left(\frac{t}{\frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right) }\right)^ \beta \right)` 

:math:`\text{Weibull-Power:} \hspace{29mm} f(t,S) = \frac{\beta t^{ \beta - 1}}{ \left( a.S^n \right)^ \beta}. {\rm exp}\left(-\left(\frac{t}{ a.S^n }\right)^ \beta \right)` 

:math:`\text{Weibull-Dual-Exponential:} \hspace{5mm} f(t,S_1,S_2) = \frac{\beta t^{ \beta - 1}}{ \left( c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right) \right)^ \beta}. {\rm exp}\left(-\left(\frac{t}{ c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right) }\right)^ \beta \right)` 

:math:`\text{Weibull-Dual-Power:} \hspace{17mm} f(t,S_1,S_2) = \frac{\beta t^{ \beta - 1}}{ \left( c.S_1^m.S_2^n \right)^ \beta} .{\rm exp}\left(-\left(\frac{t}{c.S_1^m.S_2^n }\right)^ \beta \right)` 

:math:`\text{Weibull-Power-Exponential:} \hspace{4mm} f(t,S_1,S_2) = \frac{\beta t^{ \beta - 1}}{ \left( c.{\rm exp} \left(\frac{a}{S_2} \right).S_1^n \right)^ \beta} .{\rm exp}\left(-\left(\frac{t}{c.{\rm exp} \left(\frac{a}{S_2} \right).S_1^n}\right)^ \beta \right)` 
 
Lognormal ALT models
""""""""""""""""""""

:math:`\text{Lognormal-Exponential:} \hspace{18mm} f(t,S) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left(b.{\rm exp}\left(\frac{a}{S} \right) \right)}{\sigma}\right)^2\right)`

:math:`\text{Lognormal-Eyring:} \hspace{28mm} f(t,S) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left( \frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right) \right)}{\sigma}\right)^2\right)`

:math:`\text{Lognormal-Power:} \hspace{29mm} f(t,S) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left( a.S^n \right)}{\sigma}\right)^2\right)`

:math:`\text{Lognormal-Dual-Exponential:} \hspace{5mm} f(t,S_1,S_2) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left( c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right) \right)}{\sigma}\right)^2\right)`

:math:`\text{Lognormal-Dual-Power:} \hspace{17mm} f(t,S_1,S_2) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left( c.{S_1}^m.{S_2}^n \right)}{\sigma}\right)^2\right)`

:math:`\text{Lognormal-Power-Exponential:} \hspace{4mm} f(t,S_1,S_2) = \frac{1}{\sigma t \sqrt{2\pi}} . {\rm exp} \left(-\frac{1}{2} \left(\frac{{\rm ln}(t)-{\rm ln}\left( c.{S_1}^n.{\rm exp} \left(\frac{a}{S_2} \right) \right)}{\sigma}\right)^2\right)`


Normal ALT models
"""""""""""""""""

:math:`\text{Normal-Exponential:} \hspace{18mm} f(t,S) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - b.{\rm exp}\left(\frac{a}{S} \right)}{\sigma}\right)^2\right)`

:math:`\text{Normal-Eyring:} \hspace{28mm} f(t,S) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - \frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right)}{\sigma}\right)^2\right)`

:math:`\text{Normal-Power:} \hspace{29mm} f(t,S) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - a.S^n}{\sigma}\right)^2\right)`

:math:`\text{Normal-Dual-Exponential:} \hspace{5mm} f(t,S_1,S_2) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right)}{\sigma}\right)^2\right)`

:math:`\text{Normal-Dual-Power:} \hspace{17mm} f(t,S_1,S_2) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - c.{S_1}^m.{S_2}^n}{\sigma}\right)^2\right)`

:math:`\text{Normal-Power-Exponential:} \hspace{4mm} f(t,S_1,S_2) = \frac{1}{\sigma \sqrt{2 \pi}}. {\rm exp}\left(-\frac{1}{2}\left(\frac{t - c.{S_1}^n.{\rm exp} \left(\frac{a}{S_2} \right)}{\sigma}\right)^2\right)`

Exponential ALT models
""""""""""""""""""""""

:math:`\text{Exponential-Exponential:} \hspace{18mm} f(t,S) = b.{\rm exp}\left(\frac{a}{S} \right) . {\rm exp}\left(\frac{-t}{b.{\rm exp}\left(\frac{a}{S} \right)} \right)`

:math:`\text{Exponential-Eyring:} \hspace{28mm} f(t,S) = \frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right) . {\rm exp}\left(\frac{-t}{\frac{1}{S} .{\rm exp} \left( - \left( c - \frac{a}{S} \right) \right)} \right)`

:math:`\text{Exponential-Power:} \hspace{29mm} f(t,S) = a.S^n . {\rm exp}\left(\frac{-t}{a.S^n} \right)`

:math:`\text{Exponential-Dual-Exponential:} \hspace{5mm} f(t,S_1,S_2) = c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right) . {\rm exp}\left(\frac{-t}{c.{\rm exp} \left(\frac{a}{S_1} + \frac{b}{S_2} \right)} \right)`

:math:`\text{Exponential-Dual-Power:} \hspace{17mm} f(t,S_1,S_2) = c.{S_1}^m.{S_2}^n . {\rm exp}\left(\frac{-t}{c.{S_1}^m.{S_2}^n} \right)`

:math:`\text{Exponential-Power-Exponential:} \hspace{4mm} f(t,S_1,S_2) = c.{S_1}^n.{\rm exp} \left(\frac{a}{S_2} \right) . {\rm exp}\left(\frac{-t}{c.{S_1}^n.{\rm exp} \left(\frac{a}{S_2} \right)} \right)`

Acceleration factor
"""""""""""""""""""

The acceleration factor is a value used to show by how much the life is being accelerated. The acceleration factor is given by the equation:

:math:`AF = \frac{L_{USE}}{L_{ACCELERATED}}`

**References:**

- Probabilistic Physics of Failure Approach to Reliability (2017), by M. Modarres, M. Amiri, and C. Jackson. pp. 136-168
- Accelerated Life Testing Data Analysis Reference - ReliaWiki, Reliawiki.com, 2019. [`Online <http://reliawiki.com/index.php/Accelerated_Life_Testing_Data_Analysis_Reference>`_].
