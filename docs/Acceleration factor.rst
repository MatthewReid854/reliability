.. image:: images/logo.png

-------------------------------------

Acceleration factor
'''''''''''''''''''

.. note:: This module is currently in development. The following documentation is correct, however, the most recent version of ``reliability`` on PyPI will not contain this module until Dec 2018.

This function calculates the acceleration factor at a higher temperature based on the Arrhenius model.
It solves the equation :math:`AF = exp\left[\frac{E_a}{K_B}\left(\frac{1}{T_{use}}-\frac{1}{T_{acc}}\right)\right]`

Inputs:

-   T_use - Temp of usage in Celsius
-   T_acc - Temp of acceleration in Celsius
-   Ea - Activation energy in eV

Outputs:

-   Acceleration Factor
 
In the example below, the acceleration factor is found for an accelerated test at 100:math:`^oC` for a component that is normally run at 60:math:`^oC` and has an activation energy of 1.2 eV.


.. code:: python

    from reliability.PoF import acceleration_factor
    AF = acceleration_factor(T_use=60,T_acc=100,Ea=1.2)
    print(AF)

    '''
    88.29574588463338
    '''
