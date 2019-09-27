.. image:: images/logo.png

-------------------------------------

Solving simultaneous equations with sympy
'''''''''''''''''''''''''''''''''''''''''

This document is a tutorial for how to use the Python module ``sympy`` to solve simultaneous equations. Since sympy does this so well, there is no need to implement it within ``reliability``, but users may find this tutorial helpful as problems involving physics of failure will often require the solution of simultaneous equations.
The library ``sympy`` is not installed by default when you install ``reliability`` so users following this tutorial will need to ensure sympy is installed on their machine.
The following three examples should be sufficient to illustrate how to use sympy for solving simultaneous equations.

Example 1:

:math:`\text{Eqn 1:} \hspace{11mm} x + y = 5` 

:math:`\text{Eqn 2:} \hspace{11mm} x^2 + y^2 = 17`

Solving with sympy:

.. code:: python

    import sympy as sym
    x,y = sym.symbols('x,y')
    eq1 = sym.Eq(x+y,5)
    eq2 = sym.Eq(x**2+y**2,17)
    result = sym.solve([eq1,eq2],(x,y))
    print(result)

    '''
    [(1, 4), (4, 1)] #these are the solutions for x,y. There are 2 solutions because the equations represent a line passing through a circle.
    '''

Example 2:

:math:`\text{Eqn 1:} \hspace{11mm} a1000000^b = 119.54907` 

:math:`\text{Eqn 2:} \hspace{11mm} a1000^b = 405`

Solving with sympy:

.. code:: python

    import sympy as sym
    a,b = sym.symbols('a,b')
    eq1 = sym.Eq(a*1000000**b,119.54907)
    eq2 = sym.Eq(a*1000**b,405)
    result = sym.solve([eq1,eq2],(a,b))
    print(result)

    '''
    [(1372.03074854535, -0.176636273742481)] #these are the solutions for a,b
    '''

Example 3:

:math:`\text{Eqn 1:} \hspace{11mm} 2x^2 +y + z = 1` 

:math:`\text{Eqn 2:} \hspace{11mm} x + 2y + z = c_1`

:math:`\text{Eqn 3:} \hspace{11mm} -2x + y = -z`

The actual solution to the above set of equations is:

:math:`\hspace{21mm} x = -\frac{1}{2}+\frac{\sqrt{3}}{2}` 

:math:`\hspace{21mm} y = c_1 - \frac{3\sqrt{3}}{2}+\frac{3}{2}` 

:math:`\hspace{21mm} z = -c_1 - \frac{5}{2}+\frac{5\sqrt{3}}{2}` 

and a second solution:

:math:`\hspace{21mm} x = -\frac{1}{2}-\frac{\sqrt{3}}{2}` 

:math:`\hspace{21mm} y = c_1 + \frac{3\sqrt{3}}{2}+\frac{3}{2}` 

:math:`\hspace{21mm} z = -c_1 - \frac{5}{2}-\frac{5\sqrt{3}}{2}` 

Solving with sympy:

.. code:: python

    import sympy as sym
    x,y,z = sym.symbols('x,y,z')
    c1 = sym.Symbol('c1')
    eq1 = sym.Eq(2*x**2+y+z,1)
    eq2 = sym.Eq(x+2*y+z,c1)
    eq3 = sym.Eq(-2*x+y,-z)
    result = sym.solve([eq1,eq2,eq3],(x,y,z))
    print(result)

    '''
    [(-1/2 + sqrt(3)/2, c1 - 3*sqrt(3)/2 + 3/2, -c1 - 5/2 + 5*sqrt(3)/2), (-sqrt(3)/2 - 1/2, c1 + 3/2 + 3*sqrt(3)/2, -c1 - 5*sqrt(3)/2 - 5/2)]
    '''

.. note:: If you are using an iPython notebook, the display abilities are much better than the command line interface, so you can simply add sym.init_printing() after the import line and your equations should be displayed nicely.

A special thanks to Brigham Young University for offering `this tutorial <https://apmonitor.com/che263/index.php/Main/PythonSolveEquations>`_.
