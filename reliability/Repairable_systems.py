"""
Repairable systems

reliability_growth - uses the Duane method to find the instantaneous MTBF and
    produce a reliability growth plot.
optimal_replacement_time - Calculates the cost model to determine how cost
    varies with replacement time. The cost model may be NHPP (as good as old)
    or HPP (as good as new).
ROCOF - rate of occurrence of failures. Uses the Laplace test to determine if
    there is a trend in the failure times.
MCF_nonparametric - Mean CUmulative Function Non-parametric. Used to determine
    if a repairable system (or collection of identical systems) is improving,
    constant, or worsening based on the rate of failures over time.
MCF_parametric - Mean Cumulative Function Parametric. Fits a parametric model to
    the data obtained from MCF_nonparametric
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import scipy.stats as ss
from scipy.optimize import curve_fit
from reliability.Utils import colorprint, round_to_decimals
from matplotlib.axes import SubplotBase


class reliability_growth:
    """
    Uses the Duane method to find the instantaneous MTBF and produce a
    reliability growth plot.

    Parameters
    ----------
    times : list, array
        The failure times.
    xmax : int, float, optional
        The xlim to plot up to. Default is 1.5*max(times)
    target_MTBF : int, float, optional
        Specify the target MTBF to obtain the total time on test required to
        reach it. Default = None
    show_plot : bool, optional
        If True the plot will be produced. Default is True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    Lambda : float
        The lambda parameter from the Duane model
    Beta : float
        The beta parameter from the Duane model
    time_to_target : float
        The time to reach the target is only returned if target_MTBF is
        specified.

    Notes
    -----
    If show_plot is True, the reliability growth plot will be produced. Use
    plt.show() to show the plot.
    """

    def __init__(
        self,
        times=None,
        xmax=None,
        target_MTBF=None,
        show_plot=True,
        print_results=True,
        **kwargs
    ):
        if times is None:
            raise ValueError("times must be an array or list of failure times")
        if type(times) == list:
            times = np.sort(np.array(times))
        elif type(times) == np.ndarray:
            times = np.sort(times)
        else:
            raise ValueError("times must be an array or list of failure times")
        if min(times) < 0:
            raise ValueError(
                "failure times cannot be negative. times must be an array or list of failure times"
            )
        if xmax is None:
            xmax = int(max(times) * 1.5)
        if "color" in kwargs:
            c = kwargs.pop("color")
        else:
            c = "steelblue"

        N = np.arange(1, len(times) + 1)
        theta_c = times / N
        ln_t = np.log(times)
        ln_theta_c = np.log(theta_c)
        z = np.polyfit(
            ln_t, ln_theta_c, 1
        )  # fit a straight line to the data to get the parameters lambda and beta
        beta = 1 - z[0]
        Lambda = np.exp(-z[1])
        xvals = np.linspace(0, xmax, 1000)
        theta_i = (xvals ** (1 - beta)) / (Lambda * beta)  # the smooth line
        theta_i_points = (times ** (1 - beta)) / (
            Lambda * beta
        )  # the failure times highlighted along the line
        self.Lambda = Lambda
        self.Beta = beta

        if print_results is True:
            colorprint(
                "Reliability growth model parameters:", bold=True, underline=True
            )
            print("lambda:", Lambda)
            print("beta:", beta)

        if target_MTBF is not None:
            t_target = (target_MTBF * Lambda * beta) ** (1 / (1 - beta))
            self.time_to_target = t_target
            print("Time to reach target MTBF:", t_target)
        else:
            self.time_to_target = "specify a target to obtain the time_to_target"

        if show_plot is True:
            plt.plot(xvals, theta_i, color=c, **kwargs)
            plt.plot(times, theta_i_points, "o", color=c, alpha=0.5)
            if target_MTBF is not None:
                plt.plot(
                    [0, t_target, t_target],
                    [target_MTBF, target_MTBF, 0],
                    "red",
                    label="Reliability target",
                    linewidth=1,
                )
            plt.title("Reliability Growth")
            plt.xlabel("Total time on test")
            plt.ylabel("Instantaneous MTBF")
            plt.xlim([0, max(xvals)])
            plt.ylim([0, max(theta_i) * 1.2])


class optimal_replacement_time:
    """
    Calculates the cost model to determine how cost varies with replacement time.
    The cost model may be HPP (good as new replacement) or NHPP (as good as old
    replacement). Default is HPP.

    Parameters
    ----------
    Cost_PM : int, float
        The cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM : int, float
        The cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha : int, float
        The scale parameter of the underlying Weibull distribution.
    weibull_beta : int, float
        The shape parameter of the underlying Weibull distribution. Should be
        greater than 1 otherwise conducting PM is not economical.
    q : int, optional
        The restoration factor. Must be 0 or 1. Use q=1 for Power Law NHPP
        (as good as old) or q=0 for HPP (as good as new). Default is q=0 (as
        good as new).
    show_time_plot : bool, axes, optional
        If True the plot of replacment time vs cost per unit time will be
        produced in a new figure. If an axes subclass is passed then the plot
        be generated in that axes. If False then no plot will be generated.
        Default is True.
    show_ratio_plot : bool, axes, optional
        If True the plot of cost ratio vs replacement interval will be
        produced in a new figure. If an axes subclass is passed then the plot
        be generated in that axes. If False then no plot will be generated.
        Default is True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    ORT : float
        The optimal replacement time
    min_cost : float
        The minimum cost per unit time
    """

    def __init__(
        self,
        cost_PM,
        cost_CM,
        weibull_alpha,
        weibull_beta,
        show_time_plot=True,
        show_ratio_plot=True,
        print_results=True,
        q=0,
        **kwargs
    ):
        if "color" in kwargs:
            c = kwargs.pop("color")
        else:
            c = "steelblue"
        if cost_PM > cost_CM:
            raise ValueError(
                "Cost_PM must be less than Cost_CM otherwise preventative maintenance should not be conducted."
            )
        if weibull_beta < 1:
            colorprint(
                "WARNING: weibull_beta is < 1 so the hazard rate is decreasing, therefore preventative maintenance should not be conducted.",
                text_color="red",
            )

        if q == 1:  # as good as old
            alpha_multiple = 4
            t = np.linspace(1, weibull_alpha * alpha_multiple, 100000)
            CPUT = ((cost_PM * (t / weibull_alpha) ** weibull_beta) + cost_CM) / t
            ORT = weibull_alpha * (
                (cost_CM / (cost_PM * (weibull_beta - 1))) ** (1 / weibull_beta)
            )
            min_cost = (
                (cost_PM * (ORT / weibull_alpha) ** weibull_beta) + cost_CM
            ) / ORT
        elif q == 0:  # as good as new
            alpha_multiple = 3
            t = np.linspace(1, weibull_alpha * alpha_multiple, 10000)

            # survival function and its integral
            calc_SF = lambda x: np.exp(-((x / weibull_alpha) ** weibull_beta))
            integrate_SF = lambda x: integrate.quad(calc_SF, 0, x)[0]

            # vectorize them
            calc_SFv = np.vectorize(calc_SF)
            integrate_SFv = np.vectorize(integrate_SF)

            # calculate the SF and intergral at each time
            sf = calc_SFv(t)
            integral = integrate_SFv(t)

            CPUT = (cost_PM * sf + cost_CM * (1 - sf)) / integral
            idx = np.argmin(CPUT)
            min_cost = CPUT[idx]  # minimum cost per unit time
            ORT = t[idx]  # optimal replacement time
        else:
            raise ValueError(
                'q must be 0 or 1. Default is 0. Use 0 for "as good as new" and use 1 for "as good as old".'
            )
        self.ORT = ORT
        self.min_cost = min_cost

        if min_cost < 1:
            min_cost_rounded = round(
                min_cost, -int(np.floor(np.log10(abs(min_cost)))) + 1
            )  # this rounds to exactly 2 sigfigs no matter the number of preceding zeros
        else:
            min_cost_rounded = round(min_cost, 2)
        ORT_rounded = round(ORT, 2)

        if print_results is True:
            colorprint(
                "Results from optimal_replacement_time:", bold=True, underline=True
            )
            if q == 0:
                print("Cost model assuming as good as new replacement (q=0):")
            else:
                print("Cost model assuming as good as old replacement (q=1):")
            print(
                "The minimum cost per unit time is",
                min_cost_rounded,
                "\nThe optimal replacement time is",
                ORT_rounded,
            )

        if (
            show_time_plot is True
            or issubclass(type(show_time_plot), SubplotBase) is True
        ):
            if issubclass(type(show_time_plot), SubplotBase) is True:
                plt.sca(ax=show_time_plot)  # use the axes passed
            else:
                plt.figure()  # if no axes is passed, make a new figure
            plt.plot(t, CPUT, color=c, **kwargs)
            plt.plot(ORT, min_cost, "o", color=c)
            text_str = str(
                "\nMinimum cost per unit time is "
                + str(min_cost_rounded)
                + "\nOptimal replacement time is "
                + str(ORT_rounded)
            )
            plt.text(ORT, min_cost, text_str, verticalalignment="top")
            plt.xlabel("Replacement time")
            plt.ylabel("Cost per unit time")
            plt.title("Optimal replacement time estimation")
            plt.ylim([0, min_cost * 2])
            plt.xlim([0, weibull_alpha * alpha_multiple])

        if (
            show_ratio_plot is True
            or issubclass(type(show_ratio_plot), SubplotBase) is True
        ):
            if issubclass(type(show_ratio_plot), SubplotBase) is True:
                plt.sca(ax=show_ratio_plot)  # use the axes passed
            else:
                plt.figure()  # if no axes is passed, make a new figure
            xupper = np.round(cost_CM / cost_PM, 0) * 2
            CC_CP = np.linspace(1, xupper, 200)  # cost unscheduled / cost preventative
            CC = CC_CP * cost_PM
            ORT_array = []

            if q == 1:
                for _cost_CM in CC:
                    CPUT = (
                        (cost_PM * (t / weibull_alpha) ** weibull_beta) + _cost_CM
                    ) / t
                    ORT = weibull_alpha * (
                        (_cost_CM / (cost_PM * (weibull_beta - 1)))
                        ** (1 / weibull_beta)
                    )
                    ORT_array.append(ORT)
            else:  # q = 0
                for _cost_CM in CC:
                    CPUT = (cost_PM * sf + _cost_CM * (1 - sf)) / integral
                    idx = np.argmin(CPUT)
                    ORT = t[idx]  # optimal replacement time
                    ORT_array.append(ORT)

            plt.plot(CC_CP, ORT_array)
            plt.xlim(0, xupper)
            plt.ylim(0, self.ORT * 2)
            plt.scatter(cost_CM / cost_PM, self.ORT)
            # vertical alignment based on plot increasing or decreasing
            if ORT_array[50] > ORT_array[40]:
                va = "top"
                mult = 0.95
            else:
                va = "bottom"
                mult = 1.05
            plt.text(
                s=str(
                    "$cost_{CM} = $"
                    + str(cost_CM)
                    + "\n$cost_{PM} = $"
                    + str(cost_PM)
                    + "\nInterval = "
                    + str(round(self.ORT, 2))
                ),
                x=cost_CM / cost_PM * 1.05,
                y=self.ORT * mult,
                ha="left",
                va=va,
            )
            plt.xlabel(r"Cost ratio $\left(\frac{CM}{PM}\right)$")
            plt.ylabel("Replacement Interval")
            plt.title("Optimal replacement interval\nacross a range of CM costs")


class ROCOF:
    """
    Uses the failure times or failure interarrival times to determine if there
    is a trend in those times. The test for statistical significance is the
    Laplace test which compares the Laplace test statistic (U) with the z value
    (z_crit) from the standard normal distribution. If there is a statistically
    significant trend, the parameters of the model (Lambda_hat and Beta_hat) are
    calculated. By default the results are printed and a plot of the times and
    MTBF is plotted.

    Parameters
    ----------
    times_between_failures : array, list, optional
        The failure interarrival times. See the Notes below.
    failure_times : array, list, optional
        The actual failure times. See the Notes below.
    test_end : int, float, optional
        Use this to specify the end of the test if the test did not end at the
        time of the last failure. Default = None which will result in the last
        failure being treated as the end of the test.
    CI : float
        The confidence interval for the Laplace test. Must be between 0 and 1.
        Default is 0.95 for 95% CI.
    show_plot : bool, optional
        If True the plot will be produced. Default = True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    U : float
        The Laplace test statistic
    z_crit : tuple
        (lower,upper) bound on z value. This is based on the CI.
    trend : str
        'improving','worsening','constant'. This is based on the comparison of U
        with z_crit
    Beta_hat : float, str
        The Beta parameter for the NHPP Power Law model. Only calculated if the
        trend is not constant, else a string is returned.
    Lambda_hat : float, str
        The Lambda parameter for the NHPP Power Law model. Only calculated if
        the trend is not constant.
    ROCOF : float, str
        The Rate of OCcurrence Of Failures. Only calculated if the trend is
        constant. If trend is not constant then ROCOF changes over time in
        accordance with Beta_hat and Lambda_hat. In this case a string will be
        returned.

    Notes
    -----
    You can specify either times_between_failures OR failure_times but not both.
    Both options are provided for convenience so the conversion between the two
    is done internally. failure_times should be the same as
    np.cumsum(times_between_failures).

    The repair time is assumed to be negligible. If the repair times are not
    negligibly small then you will need to manually adjust your input to factor
    in the repair times.

    If show_plot is True, the ROCOF plot will be produced. Use plt.show() to
    show the plot.
    """

    def __init__(
        self,
        times_between_failures=None,
        failure_times=None,
        CI=0.95,
        test_end=None,
        show_plot=True,
        print_results=True,
        **kwargs
    ):
        if times_between_failures is not None and failure_times is not None:
            raise ValueError(
                "You have specified both times_between_failures and failure times. You can specify one but not both. Use times_between_failures for failure interarrival times, and failure_times for the actual failure times. failure_times should be the same as np.cumsum(times_between_failures)"
            )
        if times_between_failures is not None:
            if any(t <= 0 for t in times_between_failures):
                raise ValueError("times_between_failures cannot be less than zero")
            if type(times_between_failures) == list:
                ti = times_between_failures
            elif type(times_between_failures) == np.ndarray:
                ti = list(times_between_failures)
            else:
                raise ValueError("times_between_failures must be a list or array")
        if failure_times is not None:
            if any(t <= 0 for t in failure_times):
                raise ValueError("failure_times cannot be less than zero")
            if type(failure_times) == list:
                failure_times = np.sort(np.array(failure_times))
            elif type(failure_times) == np.ndarray:
                failure_times = np.sort(failure_times)
            else:
                raise ValueError("failure_times must be a list or array")
            failure_times[1:] -= failure_times[
                :-1
            ].copy()  # this is the opposite of np.cumsum
            ti = list(failure_times)
        if test_end is not None and type(test_end) not in [float, int]:
            raise ValueError(
                "test_end should be a float or int. Use test_end to specify the end time of a test which was not failure terminated."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
            )
        if test_end is None:
            tn = sum(ti)
            n = len(ti) - 1
        else:
            tn = test_end
            n = len(ti)
            if tn < sum(ti):
                raise ValueError("test_end cannot be less than the final test time")

        if "linestyle" in kwargs:
            ls = kwargs.pop("linestyle")
        else:
            ls = "--"
        if "label" in kwargs:
            label_1 = kwargs.pop("label")
        else:
            label_1 = "Failure interarrival times"

        tc = np.cumsum(ti[0:n])
        sum_tc = sum(tc)
        z_crit = ss.norm.ppf((1 - CI) / 2)  # z statistic based on CI
        U = (sum_tc / n - tn / 2) / (tn * (1 / (12 * n)) ** 0.5)
        self.U = U
        self.z_crit = (z_crit, -z_crit)
        results_str = str(
            "Laplace test results: U = "
            + str(round(U, 3))
            + ", z_crit = ("
            + str(round(z_crit, 2))
            + ",+"
            + str(round(-z_crit, 2))
            + ")"
        )

        x = np.arange(1, len(ti) + 1)
        if U < z_crit:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = "improving"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        elif U > -z_crit:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = "worsening"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        else:
            rocof = (n + 1) / sum(ti)
            self.trend = "constant"
            self.ROCOF = rocof
            self.Beta_hat = "not calculated when trend is constant"
            self.Lambda_hat = "not calculated when trend is constant"
            x_to_plot = x
            MTBF = np.ones_like(x_to_plot) / rocof

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)

        if print_results is True:
            colorprint("Results from ROCOF analysis:", bold=True, underline=True)
            print(results_str)
            if U < z_crit:
                print(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is IMPROVING. Assume NHPP."
                    )
                )
                print(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_to_decimals(B, 3),
                    ", Lambda_hat =",
                    round_to_decimals(L, 4),
                )
            elif U > -z_crit:
                print(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is WORSENING. Assume NHPP."
                    )
                )
                print(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_to_decimals(B, 3),
                    ", Lambda_hat =",
                    round_to_decimals(L, 4),
                )
            else:
                print(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is CONSTANT. Assume HPP."
                    )
                )
                print(
                    "ROCOF assuming HPP is",
                    round_to_decimals(rocof, 4),
                    "failures per unit time.",
                )

        if show_plot is True:
            plt.plot(x_to_plot, MTBF, linestyle=ls, label="MTBF")
            plt.scatter(x, ti, label=label_1, **kwargs)
            plt.ylabel("Times between failures")
            plt.xlabel("Failure number")
            title_str = str(
                "Failure interarrival times vs failure number\nAt "
                + str(CI_rounded)
                + "% confidence level the ROCOF is "
                + self.trend.upper()
            )
            plt.title(title_str)
            plt.legend()


class MCF_nonparametric:
    """
    The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical, but it does not assume
    that each system's MCF is identical (which is an assumption of the
    parametric MCF). The non-parametric estimate of the MCF provides both the
    estimate of the MCF and the confidence bounds at a particular time.

    The shape of the MCF is a key indicator that shows whether the systems are
    improving, worsening, or staying the same over time. If the MCF is concave
    down (appearing to level out) then the system is improving. A straight line
    (constant increase) indicates it is staying the same. Concave up (getting
    steeper) shows the system is worsening as repairs are required more
    frequently as time progresses.

    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    print_results : bool, optional
        Prints the table of MCF results (state, time, MCF_lower, MCF, MCF_upper,
        variance). Default = True.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).
    show_plot : bool, optional
        If True the plot will be shown. Default = True. Use plt.show() to show
        it.
    plot_CI : bool, optional
        If True, the plot will include the confidence intervals. Default = True.
        Set as False to remove the confidence intervals from the plot.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    results : dataframe
        This is a dataframe of the results that are printed. It includes the
        blank lines for censored values.
    time : array
        This is the time column from results. Blank lines for censored values
        are removed.
    MCF : array
        This is the MCF column from results. Blank lines for censored values are
        removed.
    variance : array
        This is the Variance column from results. Blank lines for censored
        values are removed.
    lower : array
        This is the MCF_lower column from results. Blank lines for censored
        values are removed.
    upper : array
        This is the MCF_upper column from results. Blank lines for censored
        values are removed

    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:

    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+

    .. code:: python

        from reliability.Repairable_systems import MCF_nonparametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_nonparametric(data=times)
    """

    def __init__(
        self, data, CI=0.95, print_results=True, show_plot=True, plot_CI=True, **kwargs
    ):

        # check input is a list
        if type(data) == list:
            pass
        elif type(data) == np.ndarray:
            data = list(data)
        else:
            raise ValueError("data must be a list or numpy array")

        # check each item is a list and fix up any ndarrays to be lists.
        test_for_single_system = []
        for i, item in enumerate(data):
            if type(item) == list:
                test_for_single_system.append(False)
            elif type(item) == np.ndarray:
                data[i] = list(item)
                test_for_single_system.append(False)
            elif type(item) == int or type(item) == float:
                test_for_single_system.append(True)
            else:
                raise ValueError(
                    "Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]]"
                )
        # Wraps the data in another list if all elements were numbers.
        if all(test_for_single_system):  # checks if all are True
            data = [data]
        elif not any(test_for_single_system):  # checks if all are False
            pass
        else:
            raise ValueError(
                "Mixed data types found in the data. Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]]."
            )

        end_times = []
        repair_times = []
        for system in data:
            system.sort()  # sorts the values in ascending order
            for i, t in enumerate(system):
                if i < len(system) - 1:
                    repair_times.append(t)
                else:
                    end_times.append(t)

        if CI < 0 or CI > 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals (two sided)."
            )

        if max(end_times) < max(repair_times):
            raise ValueError(
                "The final end time must not be less than the final repair time."
            )
        last_time = max(end_times)
        C_array = ["C"] * len(end_times)
        F_array = ["F"] * len(repair_times)

        Z = -ss.norm.ppf(1 - CI)  # confidence interval converted to Z-value

        # sort the inputs and extract the sorted values for later use
        times = np.hstack([repair_times, end_times])
        states = np.hstack([F_array, C_array])
        data = {"times": times, "states": states}
        df = pd.DataFrame(data, columns=["times", "states"])
        df_sorted = df.sort_values(
            by=["times", "states"], ascending=[True, False]
        )  # sorts the df by times and then by states, ensuring that states are F then C where the same time occurs. This ensures a failure is counted then the item is retired.
        times_sorted = df_sorted.times.values
        states_sorted = df_sorted.states.values

        # MCF calculations
        MCF_array = []
        Var_array = []
        MCF_lower_array = []
        MCF_upper_array = []
        r = len(end_times)
        r_inv = 1 / r
        C_seq = 0  # sequential number of censored values
        for i in range(len(times)):
            if i == 0:
                if states_sorted[i] == "F":  # first event is a failure
                    MCF_array.append(r_inv)
                    Var_array.append(
                        (r_inv ** 2) * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                    )
                    MCF_lower_array.append(
                        MCF_array[i] / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                    )
                    MCF_upper_array.append(
                        MCF_array[i] * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                    )
                else:  # first event is censored
                    MCF_array.append("")
                    Var_array.append("")
                    MCF_lower_array.append("")
                    MCF_upper_array.append("")
                    r -= 1
                    if (
                        times_sorted[i] not in end_times
                    ):  # check if this system only has one event. If not then increment the number censored count for this system
                        C_seq += 1
            else:  # everything after the first time
                if states_sorted[i] == "F":  # failure event
                    i_adj = i - C_seq
                    r_inv = 1 / r
                    if (
                        MCF_array[i_adj - 1] == ""
                    ):  # this is the case where the first system only has one event that was censored and there is no data on the first line
                        MCF_array.append(r_inv)
                        Var_array.append(
                            (r_inv ** 2)
                            * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                        )
                        MCF_lower_array.append(
                            MCF_array[i]
                            / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                        MCF_upper_array.append(
                            MCF_array[i]
                            * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                    else:  # this the normal case where there was previous data
                        MCF_array.append(r_inv + MCF_array[i_adj - 1])
                        Var_array.append(
                            (r_inv ** 2)
                            * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                            + Var_array[i_adj - 1]
                        )
                        MCF_lower_array.append(
                            MCF_array[i]
                            / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                        MCF_upper_array.append(
                            MCF_array[i]
                            * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                    C_seq = 0
                else:  # censored event
                    r -= 1
                    C_seq += 1
                    MCF_array.append("")
                    Var_array.append("")
                    MCF_lower_array.append("")
                    MCF_upper_array.append("")
                    if r > 0:
                        r_inv = 1 / r

        # format output as dataframe
        data = {
            "state": states_sorted,
            "time": times_sorted,
            "MCF_lower": MCF_lower_array,
            "MCF": MCF_array,
            "MCF_upper": MCF_upper_array,
            "variance": Var_array,
        }
        printable_results = pd.DataFrame(
            data, columns=["state", "time", "MCF_lower", "MCF", "MCF_upper", "variance"]
        )

        indices_to_drop = printable_results[printable_results["state"] == "C"].index
        plotting_results = printable_results.drop(indices_to_drop, inplace=False)
        RESULTS_time = plotting_results.time.values
        RESULTS_MCF = plotting_results.MCF.values
        RESULTS_variance = plotting_results.variance.values
        RESULTS_lower = plotting_results.MCF_lower.values
        RESULTS_upper = plotting_results.MCF_upper.values

        self.results = printable_results
        self.time = RESULTS_time
        self.MCF = RESULTS_MCF
        self.lower = RESULTS_lower
        self.upper = RESULTS_upper
        self.variance = RESULTS_variance

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)

        if print_results is True:
            pd.set_option(
                "display.width", 200
            )  # prevents wrapping after default 80 characters
            pd.set_option(
                "display.max_columns", 9
            )  # shows the dataframe without ... truncation
            colorprint(
                str("Mean Cumulative Function results (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")

        if show_plot is True:
            x_MCF = [0, RESULTS_time[0]]
            y_MCF = [0, 0]
            y_upper = [0, 0]
            y_lower = [0, 0]
            x_MCF.append(RESULTS_time[0])
            y_MCF.append(RESULTS_MCF[0])
            y_upper.append(RESULTS_upper[0])
            y_lower.append(RESULTS_lower[0])
            for i, t in enumerate(RESULTS_time):
                if i > 0:
                    x_MCF.append(RESULTS_time[i])
                    y_MCF.append(RESULTS_MCF[i - 1])
                    y_upper.append(RESULTS_upper[i - 1])
                    y_lower.append(RESULTS_lower[i - 1])
                    x_MCF.append(RESULTS_time[i])
                    y_MCF.append(RESULTS_MCF[i])
                    y_upper.append(RESULTS_upper[i])
                    y_lower.append(RESULTS_lower[i])
            x_MCF.append(last_time)  # add the last horizontal line
            y_MCF.append(RESULTS_MCF[-1])
            y_upper.append(RESULTS_upper[-1])
            y_lower.append(RESULTS_lower[-1])
            title_str = "Non-parametric estimate of the Mean Cumulative Function"

            if "color" in kwargs:
                col = kwargs.pop("color")
            else:
                col = "steelblue"
            if plot_CI is True:
                plt.fill_between(
                    x_MCF, y_lower, y_upper, color=col, alpha=0.3, linewidth=0
                )
                title_str = str(
                    title_str
                    + "\nwith "
                    + str(CI_rounded)
                    + "% one-sided confidence interval bounds"
                )
            plt.plot(x_MCF, y_MCF, color=col, **kwargs)
            plt.xlabel("Time")
            plt.ylabel("Mean cumulative number of failures")
            plt.title(title_str)
            plt.xlim(0, last_time)
            plt.ylim(0, max(RESULTS_upper) * 1.05)


class MCF_parametric:
    """
    The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical. In the case of the fitted
    paramertic MCF, it is assumed that each system's MCF is identical.

    The shape (beta parameter) of the MCF is a key indicator that shows whether
    the systems are improving (beta<1), worsening (beta>1), or staying the same
    (beta=1) over time. If the MCF is concave down (appearing to level out) then
    the system is improving. A straight line (constant increase) indicates it is
    staying the same. Concave up (getting steeper) shows the system is worsening
    as repairs are required more frequently as time progresses.

    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    print_results : bool, optional
        Prints the table of MCF results (state, time, MCF_lower, MCF, MCF_upper,
        variance). Default = True.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).
    show_plot : bool, optional
        If True the plot will be shown. Default = True. Use plt.show() to show
        it.
    plot_CI : bool, optional
        If True, the plot will include the confidence intervals. Default = True.
        Set as False to remove the confidence intervals from the plot.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).

    Returns
    -------
    times : array
        This is the times (x values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    MCF : array
        This is the MCF (y values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    alpha : float
        The calculated alpha parameter from MCF = (t/alpha)^beta
    beta : float
        The calculated beta parameter from MCF = (t/alpha)^beta
    alpha_SE : float
        The standard error in the alpha parameter
    beta_SE : float
        The standard error in the beta parameter
    cov_alpha_beta : float
        The covariance between the parameters
    alpha_upper : float
        The upper CI estimate of the parameter
    alpha_lower : float
        The lower CI estimate of the parameter
    beta_upper : float
        The upper CI estimate of the parameter
    beta_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)

    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:

    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+

    .. code:: python

        from reliability.Repairable_systems import MCF_parametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_parametric(data=times)
    """

    def __init__(
        self, data, CI=0.95, plot_CI=True, print_results=True, show_plot=True, **kwargs
    ):

        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        MCF_NP = MCF_nonparametric(
            data=data, print_results=False, show_plot=False
        )  # all the MCF calculations to get the plot points are done in MCF_nonparametric
        self.times = MCF_NP.time
        self.MCF = MCF_NP.MCF

        # initial guess using least squares regression of linearised function
        # we must convert this back to list due to an issue within numpy dealing with the log of floats
        ln_x = np.log(list(self.times))
        ln_y = np.log(list(self.MCF))
        guess_fit = np.polyfit(ln_x, ln_y, deg=1)
        beta_guess = guess_fit[0]
        alpha_guess = np.exp(-guess_fit[1] / beta_guess)
        guess = [
            alpha_guess,
            beta_guess,
        ]  # guess for curve_fit. This guess is good but curve fit makes it much better.

        # actual fitting using curve_fit with initial guess from least squares
        def __MCF_eqn(t, a, b):  # objective function for curve_fit
            return (t / a) ** b

        fit = curve_fit(__MCF_eqn, self.times, self.MCF, p0=guess)
        alpha = fit[0][0]
        beta = fit[0][1]
        var_alpha = fit[1][0][
            0
        ]  # curve_fit returns the variance and covariance from the optimizer
        var_beta = fit[1][1][1]
        cov_alpha_beta = fit[1][0][1]

        Z = -ss.norm.ppf((1 - CI) / 2)
        self.alpha = alpha
        self.alpha_SE = var_alpha ** 0.5
        self.beta = beta
        self.beta_SE = var_beta ** 0.5
        self.cov_alpha_beta = cov_alpha_beta
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            colorprint(
                str(
                    "Mean Cumulative Function Parametric Model ("
                    + str(CI_rounded)
                    + "% CI):"
                ),
                bold=True,
                underline=True,
            )
            print("MCF = (t/α)^β")
            print(self.results.to_string(index=False), "\n")
            if self.beta_upper <= 1:
                print(
                    "Since Beta is less than 1, the system repair rate is IMPROVING over time."
                )
            elif self.beta_lower < 1 and self.beta_upper > 1:
                print(
                    "Since Beta is approximately 1, the system repair rate is remaining CONSTANT over time."
                )
            else:
                print(
                    "Since Beta is greater than 1, the system repair rate is WORSENING over time."
                )

        if show_plot is True:
            if "color" in kwargs:
                color = kwargs.pop("color")
                marker_color = "k"
            else:
                color = "steelblue"
                marker_color = "k"

            if "marker" in kwargs:
                marker = kwargs.pop("marker")
            else:
                marker = "."

            if "label" in kwargs:
                label = kwargs.pop("label")
            else:
                label = r"$\hat{MCF} = (\frac{t}{\alpha})^\beta$"

            x_line = np.linspace(0.001, max(self.times) * 10, 1000)
            y_line = (x_line / alpha) ** beta
            plt.plot(x_line, y_line, color=color, label=label, **kwargs)

            if plot_CI is True:
                p1 = -(beta / alpha) * (x_line / alpha) ** beta
                p2 = ((x_line / alpha) ** beta) * np.log(x_line / alpha)
                var = (
                    var_alpha * p1 ** 2
                    + var_beta * p2 ** 2
                    + 2 * p1 * p2 * cov_alpha_beta
                )
                SD = var ** 0.5
                y_line_lower = y_line * np.exp((-Z * SD) / y_line)
                y_line_upper = y_line * np.exp((Z * SD) / y_line)
                plt.fill_between(
                    x_line,
                    y_line_lower,
                    y_line_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                )

            plt.scatter(
                self.times, self.MCF, marker=marker, color=marker_color, **kwargs
            )
            plt.ylabel("Mean cumulative number of failures")
            plt.xlabel("Time")
            title_str = str(
                "Parametric estimate of the Mean Cumulative Function\n"
                + r"$MCF = (\frac{t}{\alpha})^\beta$ with α="
                + str(round(alpha, 4))
                + ", β="
                + str(round(beta, 4))
            )
            plt.xlim(0, max(self.times) * 1.2)
            plt.ylim(0, max(self.MCF) * 1.4)
            plt.title(title_str)
