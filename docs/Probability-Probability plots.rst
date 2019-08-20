.. image:: images/logo.png

-------------------------------------

Probability-Probability plots
'''''''''''''''''''''''''''''

This section contains two different styles of probability-probability plots. These are the fully parametric probability-probability plot (``reliability.Other_functions.PP_plot_parametric``) and the semi-parametric probability-probability plot (``reliability.Other_functions.PP_plot_semiparametric``). These will be described separately below. A probability-probability (PP) plot is made by plotting the fraction failing (CDF) of one distribution vs the fraction failing (CDF) of another distribution. In the semiparametric form, when we only have the failure data and one hypothesised distribution, the CDF for the data can be obtained non-parametrically to generate an Empirical CDF.

Parametric Quantile-Quantile plot
---------------------------------

To generate this plot we simply plot the CDF of one distribution vs the CDF of another distribution. If the distributions are very similar, the points will lie on the 45 degree diagonal. Any deviation from this diagonal indicates that one distribution is leading or lagging the other. Fully parametric PP plots are rarely used as their utility is limited to providing a graphical comparison of the similarity between two CDFs. To aide this comparison, the PP_plot_parametric function accepts x and y quantile lines that will be traced across to the other distribution.

Inputs:

-   X_dist - a probability distribution. The CDF of this distribution will be plotted along the X-axis.
-   Y_dist - a probability distribution. The CDF of this distribution will be plotted along the Y-axis.
-   y_quantile_lines - starting points for the trace lines to find the X equivalent of the Y-quantile. Optional input. Must be list or array.
-   x_quantile_lines - starting points for the trace lines to find the Y equivalent of the X-quantile. Optional input. Must be list or array.
-   show_diagonal_line - True/False. Default is False. If True the diagonal line will be shown on the plot.

Outputs:

-   The PP_plot is the only output. Use plt.show() to show it.

In the example below, we generate two parametric distributions and compare them using a PP plot. We are interested in the differences at specific quantiles so these are specified and the plot traces them across to the opposing distribution.

.. code:: python

    from reliability.Other_functions import PP_plot_parametric
    Field = Normal_Distribution(mu=100,sigma=30)
    Lab = Weibull_Distribution(alpha=120,beta=3)
    PP_plot_parametric(X_dist=Field, Y_dist=Lab, x_quantile_lines=[0.3, 0.6], y_quantile_lines=[0.1, 0.6])
    plt.show()

.. image:: images/PPparametric.png

Semiparametric Quantile-Quantile plot
---------------------------------

In the example below, 

.. code:: python

    from reliability.Other_functions import PP_plot_semiparametric
    from reliability.Fitters import Fit_Normal_2P
    DATA = Weibull_Distribution(alpha=5,beta=3).random_samples(100)
    nf = Fit_Normal_2P(failures=DATA)
    dist = Normal_Distribution(mu=nf.mu,sigma=nf.sigma)
    PP_plot_semiparametric(X_data_failures=DATA,Y_dist=dist)
    plt.show()

.. image:: images/PPsemiparametric.png
