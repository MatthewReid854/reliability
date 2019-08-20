.. image:: images/logo.png

-------------------------------------

Quantile-Quantile plots
'''''''''''''''''''''''

This section contains two different styles of quantile-quantile plots. These are the fully parametric quantile-quantile plot (``reliability.Other_functions.QQ_plot_parametric``) and the semi-parametric quantile-quantile plot (``reliability.Other_functions.QQ_plot_semiparametric``). These will be described separately below. A quantile-quantile (QQ) plot is made by plotting failure units vs failure units for shared quantiles. A quantile is the fraction failing (ranging from 0 to 1).

Parametric Quantile-Quantile plot
---------------------------------

To generate this plot we calculate the failure units (these may be units of time, strength, cycles, landings, rounds fired, etc.) at which a certain fraction has failed (0.01,0.02,0.03...0.99). We do this for each distribution so we have an array of failure units and then we plot these failure units against eachother. The time (or any other failure unit) at which a given fraction has failed is found using the inverse survival function. If the distributions are similar in shape, then the QQ plot should be a reasonably straight line (but not necessarily a 45 degree line). By plotting the failure times at equal quantiles for each distribution we can obtain a conversion between the two distributions. Such conversions are useful for accelerated life testing (ALT) to easily convert field time to test time.

Inputs:

-   X_dist - a probability distribution. The failure times at given quantiles from this distribution will be plotted along the X-axis.
-   Y_dist - a probability distribution. The failure times at given quantiles from this distribution will be plotted along the Y-axis.
-   show_fitted_lines - True/False. Default is True. These are the Y=mX and Y=mX+c lines of best fit.
-   show_diagonal_line - True/False. Default is False. If True the diagonal line will be shown on the plot.

Outputs:

-   The QQ_plot will always be output. Use plt.show() to show it.
-   [m,m1,c1] - these are the values for the lines of best fit. m is used in Y=mX, and m1 and c1 are used in Y=m1X+c1

In the example below, we have determined that the field failures follow a Weibull distribution (alpha=350,beta=2.01) with time represented in months. By using an accelerated life test we have replicated the failure mode and Weibull shape parameter reasonably closely and the Lab failures follow a Weibull distribution (alpha=128,beta=2.11) with time measured in hours. We would like to obtain a simple Field-to-Lab conversion for time so we know how much lab time is required to simulate 10 years of field time. The QQ-plot will automatically provide the equations for the lines of best fit. If we use the Y=mX equation we see that Field(months)=2.757 * Lab(hours). Therefore, to simulate 10 years of field time (120 months) we need to run the accelerated life test for approximately 43.5 hours in the Lab.

.. code:: python

    from reliability.Other_functions import QQ_plot_parametric
    Field = Weibull_Distribution(alpha=350,beta=2.01)
    Lab = Weibull_Distribution(alpha=128,beta=2.11)
    QQ_plot_parametric(X_dist=Lab, Y_dist=Field)
    plt.show()
    
.. image:: images/QQparametric.png

Semiparametric Quantile-Quantile plot
-------------------------------------



In the example below,

.. code:: python

    from reliability.Other_functions import QQ_plot_semiparametric
    from reliability.Fitters import Fit_Weibull_2P
    DATA = Normal_Distribution(mu=50,sigma=12).random_samples(100)
    wbf = Fit_Weibull_2P(failures=DATA)
    dist = Weibull_Distribution(alpha=wbf.alpha,beta=wbf.beta,gamma=wbf.gamma)
    QQ_plot_semiparametric(X_data_failures=DATA,Y_dist=dist)
    plt.show()
    
.. image:: images/QQsemiparametric.png

In the example below,

.. code:: python

    from reliability.Other_functions import QQ_plot_parametric, PP_plot_parametric
    Field = Weibull_Distribution(alpha=350,beta=2.01)
    Lab = Weibull_Distribution(alpha=128,beta=2.11)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    QQ_plot_parametric(X_dist=Lab, Y_dist=Field,show_diagonal_line=True,show_fitted_lines=False)
    plt.subplot(122)
    PP_plot_parametric(X_dist=Lab, Y_dist=Field,show_diagonal_line=True)
    plt.show()

.. image:: images/PPvsQQparametric.png

In the example below,

.. code:: python

    from reliability.Other_functions import PP_plot_semiparametric, QQ_plot_semiparametric
    from reliability.Fitters import Fit_Normal_2P
    DATA = Weibull_Distribution(alpha=100,beta=3).random_samples(100)
    nf = Fit_Normal_2P(failures=DATA)
    dist = Normal_Distribution(mu=nf.mu,sigma=nf.sigma)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    QQ_plot_semiparametric(X_data_failures=DATA,Y_dist=dist,show_fitted_lines=False,show_diagonal_line=True)
    plt.subplot(122)
    PP_plot_semiparametric(X_data_failures=DATA,Y_dist=dist)
    plt.show()

.. image:: images/PPvsQQsemiparametric.png
