.. image:: images/logo.png

-------------------------------------

Fracture mechanics
''''''''''''''''''

.. note:: This module is currently in development. The following documentation is correct, however, the most recent version of ``reliability`` on PyPI will not contain this module until Dec 2019.



Crack initiation
----------------

This function will be written soon


Crack growth
------------

The ``function fracture_mechanics_crack_growth`` uses the principles of fracture mechanics to find the number of cycles required to grow a crack from an initial length until a final length.
The final length (a_final) may be specified, but if not specified then a_final will be set as the critical crack length (a_crit) which causes failure due to rapid fracture.
This functions performs the same calculation using two methods: similified and iterative.
The simplified method assumes that the geometry factor (f(g)), the stress (S_net), and the critical crack length (a_crit) are constant. THis method is the way most textbooks show these problems solved as they can be done in a few steps.
The iterative method does not make the assumptions that the simplified method does and as a result, the parameters f(g), S_net and a_crit must be recalculated based on the current crack length at every cycle.

This function is applicable only to thin plates with an edge crack or a centre crack (which is to be specified using the parameter crack_type).
You may also use this function for notched components by specifying the parameters Kt and D which are based on the geometry of the notch.
For any notched components, this method assumes the notched component has a "shallow notch" where the notch depth (D) is much less than the plate width (W).
The value of Kt for notched components may be found on the `eFatigue <https://www.efatigue.com/constantamplitude/stressconcentration/>`_ website.
In the case of notched components, the local stress concentration from the notch will often cause slower crack growth.
In these cases, the crack length is calculated in two parts (stage 1 and stage 2) which can clearly be seen on the plot using the iterative method.

Inputs:

- Kc - fracture toughness
- Kt - stress concentration factor (default is 1 for no notch).
- D - depth of the notch (default is None for no notch). A nothed specimen is assumed to be doubly-notched (equal notches on both sides)
- C - material constant (sometimes referred to as A)
- m - material constant (sometimes referred to as n). This value must not be 2.
- P - external load on the material (MPa)
- t - plate thickness (mm)
- W - plate width (mm)
- a_initial - initial crack length (mm) - default is 1 mm
- a_final - final crack length (mm) - default is None in which case a_final is assumed to be a_crit (length at failure). It is useful to be able to enter a_final in cases where there are multiple loading regimes over time.
- crack_type - must be either 'edge' or 'center'. Default is 'edge'. The geometry factor used for each of these in the simplified method is 1.12 for edge and 1.0 for center. The iterative method calculates these values exactly using a_initial and W (plate width).
- print_results - True/False. Default is True
- show_plot - True/False. Default is True. If True the Iterative method's crack growth will be plotted.

Outputs:

- If print_results is True, all outputs will be printed with some description of the process.
- If show_plot is True, the crack growth plot will be shown for the iterative method.
- Nf_stage_1_simplified
- Nf_stage_2_simplified
- Nf_total_simplified
- final_crack_length_simplified
- transition_length_simplified
- Nf_stage_1_iterative
- Nf_stage_2_iterative
- Nf_total_iterative
- final_crack_length_iterative
- transition_length_iterative

.. code:: python

  from reliability.PoF import fracture_mechanics_crack_growth
  import matplotlib.pyplot as plt
  fracture_mechanics_crack_growth(Kc=66,C=6.91*10**-12,m=3,P=0.15,W=100,t=5,Kt=2.41,a_initial=1,D=10)
  plt.show()

  '''
  SIMPLIFIED METHOD (keeping f(g), S_max, and a_crit as constant):
  Crack growth was found in two stages since the transition length ( 2.08 mm ) due to the notch, was greater than the initial crack length ( 1 mm ).
  Stage 1 (a_initial to transition length): 6802 cycles
  Stage 2 (transition length to a_final): 1133 cycles
  Total cycles to failure: 7935 cycles.
  Critical crack length to cause failure was found to be: 7.86 mm.

  ITERATIVE METHOD (recalculating f(g), S_max, and a_crit for each cycle):
  Crack growth was found in two stages since the transition length ( 2.45 mm ) due to the notch, was greater than the initial crack length ( 1 mm ).
  Stage 1 (a_initial to transition length): 7576 cycles
  Stage 2 (transition length to a_final): 671 cycles
  Total cycles to failure: 8247 cycles.
  Critical crack length to cause failure was found to be: 6.39 mm.
  '''

.. image:: images/fracture_mechanics_growth.png


