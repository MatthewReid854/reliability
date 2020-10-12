.. image:: images/logo.png

-------------------------------------

Recommended resources
'''''''''''''''''''''

The following collection of resources are things I have found useful during my reliability engineering studies and also while writing the Python reliability library. There are many other resources available (especially textbooks and academic papers), so I encourage you to do your own research. If you find something you think is worth adding here, please send me an email (alpha.reliability@gmail.com).

**Textbooks**

-    Reliability Engineering and Risk Analysis: A practical Guide, Third Edition (2017), by M. Modarres, M. Kaminskiy, and V. Krivtsov.
-    Probabilistic Physics of Failure Approach to Reliability (2017), by M. Modarres, M. Amiri, and C. Jackson.
-    `Probability Distributions Used in Reliability Engineering (2011), by A. O'Conner, M. Modarres, and A. Mosleh. <https://crr.umd.edu/sites/crr.umd.edu/files/Free%20Ebook%20Probability%20Distributions%20Used%20in%20Reliability%20Engineering.pdf>`_
-    Practical Reliability Engineering, Fifth Edition (2012), by P. O'Conner and A. Kleyner.
-    Recurrent Events Data Analysis for Product Repairs, Disease Recurrences, and Other Applications (2003), by W. Nelson
-    Reliasoft has compiled a much more comprehensive `list of textbooks <https://www.weibull.com/knowledge/books.htm>`_.
-    The reliability analytics toolkit (linked below in free online tools and calculators) has also compiled a much more comprehensive `list of textbooks <https://reliabilityanalyticstoolkit.appspot.com/static/books.htm#0486438678>`_.

**Free software**

-    `Lifelines <https://lifelines.readthedocs.io/en/latest/index.html>`_ - a Python library for survival analysis. Very powerful collection of tools, only a few of which overlap with the Python reliability library.
-    `Parameter Solver v3.0 <https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/6>`_ - a biostatistics tool for quickly making some simple calculations with probability distributions.
-    `Orange <https://orange.biolab.si/>`_ - a standalone data mining and data visualization program that runs using Python. Beautifully interactive data analysis workflows with a large toolbox. Not much reliability related content but good for data preprocessing.
-    R (Programming Language) - R is one of the most popular programming languages for data science, and it has several libraries that are targeted towards reliability engineering and survival analysis. These include `WeibullR <https://github.com/openrelia/WeibullR.gallery>`_, `abrem <https://rdrr.io/rforge/abrem/>`_, and `survival <https://cran.r-project.org/web/packages/survival/index.html>`_. 
-    `CumFreq <https://www.waterlog.info/cumfreq.htm>`_ - a program for cumulative frequency analysis with probability distribution fitting for a wide range of distributions. Limited functionality beyond fitting distributions.

**Paid software**

The listing of a software package here does not imply my endorsement, and is only intended to give readers an understanding of the broad range of reliability engineering software packages that are available. It is difficult to find a comprehensive list of software resources since most developers of proprietary software rarely acknowledge the existence of any software other than their own. I have not used most of the paid software listed here due to the high cost, so most of my comments in this section are based purely on the content from their websites.

-    `Minitab <https://www.minitab.com/en-us/>`_ - a great collection of statistical tools. A few reliability focussed tools included.
-    `Reliasoft <https://www.reliasoft.com/products/reliability-analysis/weibull>`_ - the industry leader for reliability engineering software.
-    `SAS JMP <https://www.jmp.com/en_us/software/predictive-analytics-software.html>`_ - lots of statistical tools for data modelling and visualization. A few purpose built reliability tools. Its utility for reliability engineering will depend on your application. SAS has also released the `SAS University Edition <https://www.sas.com/en_us/software/university-edition.html>`_ which is a free software package that runs in VirtualBox and offers a reduced set of tools compared to the paid package.
-    `PTC Windchill <https://www.ptc.com/en/products/plm/capabilities/quality/>`_ - a powerful tool for risk and reliability. Similar to Reliasoft but it forms one part of the larger PTC suite of tools.
-    `Isograph Reliability Workbench <https://www.isograph.com/software/reliability-workbench/>`_ - A collection of tools designed specifically for reliability engineering.
-    `Item Software <https://www.itemsoft.com/reliability_prediction.html>`_ - A collection of tools for reliability engineering including FMECA, fault trees, reliability prediction, and many others.
-    `SuperSMITH <https://fultonfindings.com/>`_ - This software is designed specifically for reliability engineering and has many useful tools. The user interface looks like it is from the early 1990s but the methods used are no less relevant today. This software was developed alongside the New Weibull Handbook, an excellent resource for interpreting the results of reliability engineering software.
-   `RAM Commander <http://www.reliability-safety-software.com/products/ram-commander/>`_ - A software tool for Reliability and Maintainability Analysis and Prediction, Spares Optimisation, FMEA/FMECA, Testability, Fault Tree Analysis, Event Tree Analysis and Safety Assessment.
-   `RelCalc <http://t-cubed.com/features.htm>`_ - RelCalc for Windows automates the reliability prediction procedure of Telcordia SR-332, or MIL-HDBK-217, providing an alternative to tedious, time consuming, and error prone manual methods.
-   `Relyence <https://www.relyence.com/products/>`_ - Relyence offers a range of products, similar to Reliasoft, each with a focus on a different area including Life Data Analysis, Accelerated Life Testing, Reliability Block Diagrams, FMEA, and several more. 
-   `@RISK <https://www.palisade.com/risk/key-features.asp>`_ - A comprehensive Excel addon that allows for distribution fitting, reliability modelling, MC simulation and much more.
-    `Quanterion Automated Reliability Toolkit (QuART) <https://www.quanterion.com/projects/quart/>`_ - A collection of reliability tools including reliability prediction, FMECA, derating, stress-strength interference, and many other. Quanterion produces several software products so their tools are not all available in one place.
-    `TopEvent FTA <https://www.fault-tree-analysis.com/>`_ - Fault Tree Analysis software. Tailored specifically for fault tree analysis so it lacks other RAM tools but it is good at its intended function. A demo version is available with size and data export limitations.
-   `Maintenance Aware Design (MADe) <https://www.phmtechnology.com/>`_ - FMECA and RCM software that is extremely useful at the product design stage to inform the design and service plan which then improves the inherent reliability and maintainability. There is an academic license which allows non-profit users to run the software for free.

**Free online tools and calculators**

-    `Reliability Analytics Toolkit <https://reliabilityanalyticstoolkit.appspot.com/>`_ - a collection of tools which run using the Google App Engine. Includes a `tool <https://reliabilityanalyticstoolkit.appspot.com/weibull_analysis>`_ for fitting a Weibull_2P distribution.
-    `Weibull App <https://carlescg.shinyapps.io/abrem_app/>`_ - An online tool for fitting a Weibull_2P distibution. Download the example template to see what format the app is expecting your data to be in before you can upload your own data. The backend is powered by the abrem R package. This tool has limited applications beyond fitting a Weibull_2P distribution.
-    `Distributome <http://www.distributome.org/V3/calc/index.html>`_ - Provides PDF and CDF of a large number of probability distributions that can be easily changed using sliders for their parameters. It also includes a quantile / CDF calculator. Similar to the Distribution calculator below.
-    `Distribution Calculator <https://www.randomservices.org/random/apps/SpecialCalculator.html>`_ - Provides PDF and CDF of a large number of probability distributions that can be easily changed using sliders for their parameters. It also includes a quantile / CDF calculator. Similar to Distributome above.
-    `Kijima G-renewal process <http://www.soft4structures.com/WeibullGRP/JSPageGRP.jsp>`_ - an online calculator for simulating the G-renewal process.
-    `Prediction of future recurrent events <http://www.soft4structures.com/WeibullGRP/JSPageGRPinverse_1.jsp>`_ - an online calculator for predicting future recurrent events with different underlying probability functions.
-    `Maintenance optimization <http://www.soft4structures.com/WeibullGRP/JSPageMTN.jsp>`_ - an online calculator for optimal replacement policy (time) under Kijima imperfect repair model.
-    `e-Fatigue <https://www.efatigue.com/constantamplitude/stressconcentration/>`_ - This website provides stress concentration factors (Kt) for various notched geometries. You will need this if using the functions for fracture machanics in the Physics of Failure section.
-    `Fault Tree Analyser <https://www.fault-tree-analysis-software.com/fault-tree-analysis>`_ - A simple online tool where you can build a fault tree, give each branch a failure rate and run a variety of reports including reliability prediction at time, minimal cut sets, and several others.
-    `Wolfram Alpha <https://www.wolframalpha.com/>`_ - an amazing computational knowledge engine. Great for checking your calculations.
-    `Derivative calculator <https://www.derivative-calculator.net/>`_ - calculates derivatives. Slightly more user friendly input method than Wolfram alpha and doesn't time out as easily for big calculations.
-    `Integral calculator <https://www.integral-calculator.com/>`_ - calculates integrals. Slightly more user friendly input method than Wolfram alpha and doesn't time out as easily for big calculations.
-    `GeoGebra <https://www.geogebra.org/calculator>`_ - An interactive calculator that is extremely useful for plotting equations. Also includes many mathematical operations (such as integrals and derivatives) that allow you to keep your equations in symbolic form. You can download your current calculator to save it. The only downside is that there are not many probability distribution functions inbuilt so you will need to enter the `equations <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ manually.

**Online information resources**

-    `Reliawiki <http://reliawiki.org/index.php/Life_Data_Analysis_Reference_Book>`_ - an excellent reference written by Reliasoft that is intended largely as a guide to reliability engineering when using Reliasoft's software but is equally as good to understand concepts without using their software.
-    `Reliasoft's Accelerated Life Testing Data Analysis Reference <http://reliawiki.com/index.php/Accelerated_Life_Testing_Data_Analysis_Reference>`_
-    `Reliasoft's collection of Military Directives, Handbooks and Standards Related to Reliability <https://www.weibull.com/knowledge/milhdbk.htm>`_
-    `Univariate distributions relationships <http://www.math.wm.edu/~leemis/chart/UDR/UDR.html>`_ - a great interactive diagram for understanding more about probability distributions and how they are related. Some strange parametrisations are used in the documentation.
-    `Cross Validated <https://stats.stackexchange.com/>`_ - an forum for asking statistics and mathematics questions. Check for existing answers before posting your own question.
-    `Stack Overflow <https://stackoverflow.com/>`_ - a forum for programmers where you can post questions and answers related to programming. Check for existing answers before posting your own question.
-    `Wikipedia <https://en.wikipedia.org/wiki/Reliability_engineering>`_ - it's always worth checking if there's an article on there about the topic you're trying to understand.

**Getting free access to academic papers**

-    `arXiv <https://arXiv.org>`_ - a database run by Cornell university that provides open access to over 1.5 million academic papers that have been submitted. If you can't find it here then check on Sci-Hub.
-    `Sci-Hub <https://sci-hub.tw/>`_ - paste in a DOI to get a copy of the academic paper. Accessing academic knowledge should be free and this site makes it possible.
