.. image:: images/logo.png

-------------------------------------

Acceleration factor
'''''''''''''''''''

The Arrhenius model for Acceleration factor due to higher temperature is :math:`AF = exp\left[\frac{E_a}{K_B}\left(\frac{1}{T_{use}}-\frac{1}{T_{acc}}\right)\right]`
This function accepts T_use as a mandatory input and you may specify any two of the three other variables, and the third variable will be found.

Inputs:

-   T_use - Temp of usage in Celsius
-   T_acc - Temp of acceleration in Celsius (optional input)
-   Ea - Activation energy in eV (optional input)
-   AF - Acceleration factor (optional input)
-   print_results - True/False. Default is True

Outputs:

-   Results will be printed to console if print_results is True
-   AF - Acceleration Factor
-   T_acc - Accelerated temperature (°C)
-   T_use - Use temperature (°C)
-   Ea - Activation energy (eV)
 
In the example below, the acceleration factor is found for an accelerated test at 100°C for a component that is normally run at 60°C and has an activation energy of 1.2 eV.

.. code:: python

    from reliability.PoF import acceleration_factor
    acceleration_factor(T_use=60,T_acc=100,Ea=1.2)

    '''
    Results from acceleration_factor:
    Acceleration Factor: 88.29574588463338
    Use Temperature: 60 °C
    Accelerated Temperature: 100 °C
    Activation Energy: 1.2 eV
    '''
