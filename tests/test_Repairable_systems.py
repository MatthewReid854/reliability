from reliability.Datasets import MCF_1
from reliability.Repairable_systems import reliability_growth, optimal_replacement_time, ROCOF, MCF_nonparametric, MCF_parametric
from numpy.testing import assert_allclose
atol = 1e-8
rtol = 1e-7

def test_reliability_growth():
    times = [10400, 26900, 43400, 66400, 89400, 130400, 163400, 232000, 242000, 340700]
    rg = reliability_growth(times=times, target_MTBF=50000, print_results=False,show_plot=False)
    assert_allclose(rg.Beta,0.6638280053477188,rtol=rtol,atol=atol)
    assert_allclose(rg.Lambda, 0.002355878294089656, rtol=rtol, atol=atol)
    assert_allclose(rg.time_to_target, 428131.18039448344, rtol=rtol, atol=atol)


def test_optimal_replacement_time():
    ort0 = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=0)
    assert_allclose(ort0.ORT,493.1851185118512,rtol=rtol,atol=atol)
    assert_allclose(ort0.min_cost, 0.0034620429189943167, rtol=rtol, atol=atol)
    ort1 = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=1)
    assert_allclose(ort1.ORT,1618.644582767346,rtol=rtol,atol=atol)
    assert_allclose(ort1.min_cost, 0.0051483404213951, rtol=rtol, atol=atol)


def test_ROCOF():
    times = [104, 131, 1597, 59, 4, 503, 157, 6, 118, 173, 114, 62, 101, 216, 106, 140, 1, 102, 3, 393, 96, 232, 89, 61, 37, 293, 7, 165, 87, 99]
    results = ROCOF(times_between_failures=times, show_plot=False, print_results=False)
    assert_allclose(results.U, 2.4094382960447107, rtol=rtol, atol=atol)
    assert_allclose(results.z_crit, (-1.959963984540054, 1.959963984540054), rtol=rtol, atol=atol)
    assert results.trend == 'worsening'
    assert_allclose(results.Beta_hat, 1.5880533880966818, rtol=rtol, atol=atol)
    assert_allclose(results.Lambda_hat, 3.702728848984535e-05, rtol=rtol, atol=atol)
    assert results.ROCOF == 'ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t.'

def test_MCF_nonparametric():
    times = MCF_1().times
    results = MCF_nonparametric(data=times, show_plot=False, print_results=False)
    assert_allclose(sum(results.MCF), 22.833333333333332, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 3.933518518518521, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 13.992740081348929, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 38.23687898478023, rtol=rtol, atol=atol)


def test_MCF_parametric():
    times = MCF_1().times
    results = MCF_parametric(data=times, show_plot=False, print_results=False)
    assert_allclose(sum(results.MCF), 22.833333333333332, rtol=rtol, atol=atol)
    assert_allclose(sum(results.times), 214, rtol=rtol, atol=atol)
    assert_allclose(results.alpha, 11.980589826209348, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.6736221860957468, rtol=rtol, atol=atol)
    assert_allclose(results.cov_alpha_beta, 0.034638880600157955, rtol=rtol, atol=atol)
    assert_allclose(results.alpha_lower, 11.219187030973842, rtol=rtol, atol=atol)
    assert_allclose(results.alpha_upper, 12.793666081829453, rtol=rtol, atol=atol)
    assert_allclose(results.beta_lower, 1.4980169559010625, rtol=rtol, atol=atol)
    assert_allclose(results.beta_upper, 1.8698127619704332, rtol=rtol, atol=atol)


