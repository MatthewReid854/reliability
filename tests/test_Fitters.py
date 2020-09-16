from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P, Fit_Gamma_2P, Fit_Gamma_3P, Fit_Lognormal_2P, Fit_Lognormal_3P, Fit_Loglogistic_2P, Fit_Loglogistic_3P, Fit_Normal_2P, Fit_Expon_1P, Fit_Expon_2P, Fit_Beta_2P, Fit_Everything
from reliability.Distributions import Weibull_Distribution, Gamma_Distribution, Lognormal_Distribution, Loglogistic_Distribution, Normal_Distribution, Exponential_Distribution, Beta_Distribution
from reliability.Other_functions import make_right_censored_data
from numpy.testing import assert_allclose
atol = 0
rtol = 1e-10

def test_Fit_Weibull_2P():
    dist = Weibull_Distribution(alpha=50, beta=2)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Weibull_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 45.099010886086354,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 2.7827531773597975,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 0,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 115.66971887883678,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 0.9178064889295382,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -55.4819182629478,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 2.96571536864614,rtol=rtol,atol=atol)


def test_Fit_Weibull_3P():
    dist = Weibull_Distribution(alpha=50, beta=2, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Weibull_3P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 41.38429989624438,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 0.6941872050001636,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 514.5074549826453,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 114.68964821342475,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 1.3076497743444904,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -53.59482410671237,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 1.2001459605994476,rtol=rtol,atol=atol)


def test_Fit_Gamma_2P():
    dist = Gamma_Distribution(alpha=50, beta=2)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Gamma_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 30.895318057143644,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 2.5300452644724367,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 0,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 154.33194705093553,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 11.610666775358292,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -74.81303234899717,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 4.878548406768285,rtol=rtol,atol=atol)


def test_Fit_Gamma_3P():
    dist = Gamma_Distribution(alpha=50, beta=2, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Gamma_3P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 161.86588917785375,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 0.5429156736189351,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 515.4451173341464,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 150.01356065455715,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 11.303022319282686,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -71.25678032727858,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 0.5957708701483985,rtol=rtol,atol=atol)


def test_Fit_Lognormal_2P():
    dist = Lognormal_Distribution(mu=1,sigma=0.5)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Lognormal_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.mu, 0.9494189618970151,rtol=rtol,atol=atol)
    assert_allclose(fit.sigma, 0.4267323807168996,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 0,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 49.69392320890684,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_mu_sigma, 0.0025054526707355687,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -22.494020427982832,rtol=rtol,atol=atol)


def test_Fit_Lognormal_3P():
    dist = Lognormal_Distribution(mu=1,sigma=0.5, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Lognormal_3P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.mu, 6.2074393111799395,rtol=rtol,atol=atol)
    assert_allclose(fit.sigma, 0.0018428336205591194,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 6.216971087828216,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 54.382774526710904,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_mu_sigma, 4.689716427080205e-08,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -23.441387263355452,rtol=rtol,atol=atol)


def test_Fit_Loglogistic_2P():
    dist = Loglogistic_Distribution(alpha=50, beta=8)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Loglogistic_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 50.25178196536296,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 7.869850445508078,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 0,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 941.946173470838,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 0.14731251998744946,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -468.9426298826271,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 7.304622930677989,rtol=rtol,atol=atol)


def test_Fit_Loglogistic_3P():
    dist = Loglogistic_Distribution(alpha=50, beta=8, gamma=500)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Loglogistic_3P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 64.54473158929677,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 10.513230464353654,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 485.6731344659153,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 943.8101901715909,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 0.18812547180218483,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -468.84387059599953,rtol=rtol,atol=atol)
    assert_allclose(fit.initial_guess[1], 4.981027237709373,rtol=rtol,atol=atol)


def test_Fit_Normal_2P():
    dist = Normal_Distribution(mu=50,sigma=8)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Normal_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.mu, 49.01641765388186,rtol=rtol,atol=atol)
    assert_allclose(fit.sigma, 6.653242153943476,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 91.15205546551915,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_mu_sigma, 1.0395713921235965,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -43.22308655628899,rtol=rtol,atol=atol)


def test_Fit_Expon_1P():
    dist = Exponential_Distribution(Lambda=5)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Expon_1P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.Lambda, 6.101199434421275,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 0,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, -22.032339191099254,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, 12.127280706660738,rtol=rtol,atol=atol)


def test_Fit_Expon_2P():
    dist = Exponential_Distribution(Lambda=5, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Expon_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.Lambda, 7.00351280734533,rtol=rtol,atol=atol)
    assert_allclose(fit.gamma, 500.015837532126,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, -23.686473231109936,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, 14.196177792025557,rtol=rtol,atol=atol)


def test_Fit_Beta_2P():
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Beta_2P(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False, print_results=False)
    assert_allclose(fit.alpha, 7.429034112498652,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 6.519320902041194,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 4.947836247294108,rtol=rtol,atol=atol)
    assert_allclose(fit.Cov_alpha_beta, 9.9955246167663,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -0.1209769471764659,rtol=rtol,atol=atol)


def test_Fit_Everything():
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fit = Fit_Everything(failures=data.failures, right_censored=data.right_censored, show_probability_plot=False,show_histogram_plot=False,show_PP_plot=False,print_results=False)
    assert_allclose(fit.best_distribution.alpha, 0.5796887217806559,rtol=rtol,atol=atol)
    assert_allclose(fit.best_distribution.beta, 4.205258772699503,rtol=rtol,atol=atol)
    assert_allclose(fit.Beta_2P_BIC, 30.739845510058352,rtol=rtol,atol=atol)
