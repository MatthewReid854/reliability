.. image:: images/logo.png

-------------------------------------

Palmgren-Miner linear damage model
''''''''''''''''''''''''''''''''''

The function ``palmgren_miner_linear_damage`` uses the Palmgren-Miner linear damage hypothesis to find the outputs listed below.

Inputs:

    - rated_life - an array or list of how long the component will last at a given stress level
    - time_at_stress - an array or list of how long the component is subjected to the stress that gives the rated_life
    - stress - what stress the component is subjected to. Not used in the calculation but is required for printing the output.

Ensure that the time_at_stress and rated_life are in the same units. The answer will also be in those units. The number of items in each input must be the same.

Outputs:

    - Fraction of life consumed per load cycle
    - Service life of the component
    - Fraction of damage caused at each stress level

In the following example, we consider a scenario in which ball bearings fail after 50000 hrs, 6500 hrs, and 1000 hrs, after being subjected to a stress of 1kN, 2kN, and 4kN respectively. If each load cycle involves 40 mins at 1kN, 15 mins at 2kN, and 5 mins at 4kN, how long will the ball bearings last?

.. code:: python
    
    from reliability.PoF import palmgren_miner_linear_damage
    palmgren_miner_linear_damage(rated_life=[50000,6500,1000], time_at_stress=[40/60, 15/60, 5/60], stress=[1, 2, 4])
    
    '''
    Palmgren-Miner Linear Damage Model results:
    Each load cycle uses 0.01351 % of the components life.
    The service life of the component is 7400.37951 load cycles.
    The amount of damage caused at each stress level is:
    Stress =  1 , Damage fraction = 9.86717 %.
    Stress =  2 , Damage fraction = 28.463 %.
    Stress =  4 , Damage fraction = 61.66983 %.
    '''
