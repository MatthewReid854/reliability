from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Normal_Distribution
from reliability.Other_functions import stress_strength, stress_strength_normal, similar_distributions, make_right_censored_data, crosshairs, distribution_explorer, histogram
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import warnings

atol = 1e-8
rtol = 1e-7

def test_stress_strength():
    stress = Weibull_Distribution(alpha=40, beta=4)
    strength = Lognormal_Distribution(mu=1.8, sigma=0.25, gamma=50)
    result = stress_strength(stress=stress, strength=strength, print_results=False, show_plot=False)
    assert_allclose(result,0.021559141113795574,rtol=rtol,atol=atol)

def test_stress_strength_normal():
    stress = Normal_Distribution(mu=50, sigma=5)
    strength = Normal_Distribution(mu=80, sigma=7)
    result = stress_strength_normal(stress=stress, strength=strength, print_results=False, show_plot=False)
    assert_allclose(result,0.00024384404803800858,rtol=rtol,atol=atol)

def test_similar_distributions():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Weibull_Distribution(alpha=50, beta=3.3)
    results = similar_distributions(distribution=dist, include_location_shifted=True, show_plot=False, print_results=False)
    assert_allclose(results.results[0].alpha, 49.22622520639563, rtol=rtol, atol=atol)
    assert_allclose(results.results[0].beta, 3.2573074120881964, rtol=rtol, atol=atol)
    assert_allclose(results.results[0].gamma, 0.7236421159037678, rtol=rtol, atol=atol)
    assert_allclose(results.results[1].mu, 44.847138326837566, rtol=rtol, atol=atol)
    assert_allclose(results.results[1].sigma, 14.922616862230697, rtol=rtol, atol=atol)
    assert_allclose(results.results[2].alpha, 5.760746660148767, rtol=rtol, atol=atol)
    assert_allclose(results.results[2].beta, 7.784952297226461, rtol=rtol, atol=atol)
    assert_allclose(results.results[2].gamma, 0, rtol=rtol, atol=atol)

def test_make_right_censored_data():
    results = make_right_censored_data(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fraction_censored=0.5, seed=1)
    assert_allclose(results.failures, [3,5,1,4,2], rtol=rtol, atol=atol)
    assert_allclose(results.right_censored, [6.02771433, 2.06619492, 3.74089736, 2.5841914, 0.46400404], rtol=rtol, atol=atol)

def test_crosshairs():
    plt.ion()  # this is the key to enabling plt.close() to take control of a plot that is being blocked by plt.show() inside the plot generating function.
    Weibull_Distribution(alpha=50, beta=2).CDF()
    crosshairs(xlabel='t', ylabel='F')
    plt.close()
    plt.ioff()

def test_distribution_explorer():
    plt.ion()
    distribution_explorer()
    plt.close()
    plt.ioff()

def test_histogram():
    plt.ion()
    dist = Weibull_Distribution(alpha=30, beta=4)
    samples = dist.random_samples(500, seed=2)
    plt.subplot(121)
    histogram(samples, white_above=dist.mean)
    plt.subplot(122)
    histogram(samples, white_above=dist.mean, cumulative=True)
    plt.show()
    plt.close()
    plt.ioff()
