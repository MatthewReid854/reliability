.. image:: images/logo.png

-------------------------------------

Competing risk models
'''''''''''''''''''''

This section will be written soon.






The formula for the competing risks model is written in terms of the survival function (SF). Since we may consider the system's reliability to depend on the reliability of each component (or the absence of a failure mode), the equation is written in the same way as a series system, using the product of the survival functions for each failure mode. 

:math:`{SF}_{Competing_Risks} = {SF}_1 \times {SF}_2`

TEST

:math:`{CDF}_{Competing\;Risks} = 1-{SF}_{Competing Risks}`

:math:`{CDF}_{Competing\,Risks} = 1-{SF}_{Competing Risks}`

:math:`{CDF}_{Competing\!Risks} = 1-{SF}_{Competing Risks}`

:math:`{CDF}_{Competing\:Risks} = 1-{SF}_{Competing Risks}`

To obtain the PDF, we must find the derivative of the CDF. This is easiest to do numerically since the formula for the SF of the competing risks model can get quite complex as more risks are added. Since :math:`{SF} = exp(-CHF)` we may equivalently write the competing risks model in terms of the hazard or cumulative hazard function as:

:math:`{HF}_{Competing_Risks} = {HF}_1 + {HF}_2`

:math:`{CHF}_{Competing_Risks} = {CHF}_1 + {CHF}_2`


