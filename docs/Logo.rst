.. image:: images/logo.png

-------------------------------------

Logo
''''

The logo for `reliability` can be created using the code below. The logo was generated using matplotlib version 3.3.3. The image produced requires subsequent cropping to remove surrounding white space.

.. code:: python

    from reliability.Distributions import Weibull_Distribution
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 4))
    
    # blue distribution
    x_blue_fill = np.linspace(0, 19, 100)
    bluedist = Weibull_Distribution(alpha=5.5, beta=2, gamma=0.63)
    y_blue_fill = bluedist.PDF(linewidth=3, xvals=x_blue_fill, show_plot=False)
    plt.fill_between(
        x=x_blue_fill,
        y1=np.zeros_like(y_blue_fill),
        y2=y_blue_fill,
        color="steelblue",
        alpha=0.2,
    )
    bluedist.PDF(linewidth=3, xvals=np.linspace(1.5, 19, 100))
    
    # orange distribution
    orange_dist = Weibull_Distribution(alpha=5, beta=3, gamma=9.5)
    x_orange = np.linspace(0, 19, 100)
    orange_dist.PDF(linewidth=3, xvals=x_orange)
    plt.plot([-4, orange_dist.gamma + 0.1], [0, 0], linewidth=5.5, color="darkorange")
    
    # orange histogram
    samples = orange_dist.random_samples(5000, seed=3)
    plt.hist(
        x=samples[samples < max(x_orange)],
        density=True,
        alpha=0.4,
        color="darkorange",
        bins=25,
        edgecolor="k",
    )
    
    # text objects
    plt.text(x=-4, y=0.005, s="RELIABILITY", size=70, fontname="Calibri")
    plt.text(
        x=-4,
        y=-0.005,
        va="top",
        s="A Python library for reliability engineering",
        size=34.85,
        fontname="Calibri",
    )
    
    plt.xlim(-5, 20)
    plt.title("")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

If you have any suggestions for future versions of this logo, please send them through by email to alpha.reliability@gmail.com
