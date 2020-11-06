from reliability.ALT_fitters import Fit_Weibull_Power, Fit_Weibull_Eyring, Fit_Weibull_Exponential, Fit_Weibull_Power_Exponential, Fit_Weibull_Dual_Exponential
from reliability.Datasets import ALT_load2, ALT_temperature, ALT_temperature_voltage
from numpy.testing import assert_allclose
atol = 1e-8
rtol = 1e-7

def test_Fit_Weibull_Power():
    data = ALT_load2()
    fit = Fit_Weibull_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=60, show_plot=False, print_results=False)
    assert_allclose(fit.a, 398816.3314852511,rtol=rtol,atol=atol)
    assert_allclose(fit.n, -1.4173056075704444,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 3.017297319975304,rtol=rtol,atol=atol)
    assert_allclose(fit.a_SE, 519397.96045041195,rtol=rtol,atol=atol)
    assert_allclose(fit.n_SE, 0.24394440446615118,rtol=rtol,atol=atol)
    assert_allclose(fit.beta_SE, 0.7164263380293138,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -76.85410524535483,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 161.42249620499538, rtol=rtol, atol=atol)
    assert_allclose(fit.mean_life, 1075.3284454453265, rtol=rtol, atol=atol)


def test_Fit_Weibull_Eyring():
    data = ALT_load2()
    fit = Fit_Weibull_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=60, show_plot=False, print_results=False)
    assert_allclose(fit.a, 62.44413131984349,rtol=rtol,atol=atol)
    assert_allclose(fit.c, -10.344697097286081,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 2.9844223968011,rtol=rtol,atol=atol)
    assert_allclose(fit.a_SE, 40.49190392457896,rtol=rtol,atol=atol)
    assert_allclose(fit.c_SE, 0.23684494557879077,rtol=rtol,atol=atol)
    assert_allclose(fit.beta_SE, 0.7113932688904033,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -77.12812653241326,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 161.97053877911225, rtol=rtol, atol=atol)
    assert_allclose(fit.mean_life, 1309.8505815393235, rtol=rtol, atol=atol)


def test_Fit_Weibull_Exponential():
    data = ALT_temperature()
    fit = Fit_Weibull_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=150, show_plot=False, print_results=False)
    assert_allclose(fit.a, 208.3339653554636,rtol=rtol,atol=atol)
    assert_allclose(fit.b, 157.57405443712548,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 1.3998311623096287,rtol=rtol,atol=atol)
    assert_allclose(fit.a_SE, 33.28758176621911,rtol=rtol,atol=atol)
    assert_allclose(fit.b_SE, 85.92939609115767,rtol=rtol,atol=atol)
    assert_allclose(fit.beta_SE, 0.19952700393786083,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -341.5911775717569,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 689.3628062713333, rtol=rtol, atol=atol)
    assert_allclose(fit.mean_life, 575.9724247466862, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power_Exponential():
    data = ALT_temperature_voltage()
    fit = Fit_Weibull_Power_Exponential(failures=data.failures, failure_stress_thermal=data.failure_stress_temp, failure_stress_nonthermal=data.failure_stress_voltage, use_level_stress=[325, 0.5],print_results=False,show_plot=False)
    assert_allclose(fit.a, 3404.4856914313013,rtol=rtol,atol=atol)
    assert_allclose(fit.c, 0.08761034746246833,rtol=rtol,atol=atol)
    assert_allclose(fit.n, -0.7134237397366352,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 4.997526737186063,rtol=rtol,atol=atol)
    assert_allclose(fit.a_SE, 627.6747158448553,rtol=rtol,atol=atol)
    assert_allclose(fit.c_SE, 0.14121717084704113,rtol=rtol,atol=atol)
    assert_allclose(fit.n_SE, 0.27756118916078854,rtol=rtol,atol=atol)
    assert_allclose(fit.beta_SE, 1.1739975765365167,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -72.66388632019523,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 158.32777264039046, rtol=rtol, atol=atol)
    assert_allclose(fit.mean_life, 4673.15346828878, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Exponential():
    data = ALT_temperature_voltage()
    fit = Fit_Weibull_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stress_temp, failure_stress_2=data.failure_stress_voltage, use_level_stress=[325, 0.5],print_results=False,show_plot=False)
    assert_allclose(fit.a, 3404.485920027689,rtol=rtol,atol=atol)
    assert_allclose(fit.c, 0.016087369599405824,rtol=rtol,atol=atol)
    assert_allclose(fit.b, 2.7332636043656793,rtol=rtol,atol=atol)
    assert_allclose(fit.beta, 4.997525569531898,rtol=rtol,atol=atol)
    assert_allclose(fit.a_SE, 627.6623048081617,rtol=rtol,atol=atol)
    assert_allclose(fit.c_SE, 0.03108914926305635,rtol=rtol,atol=atol)
    assert_allclose(fit.b_SE, 1.0633829555837704,rtol=rtol,atol=atol)
    assert_allclose(fit.beta_SE, 1.1739967754195961,rtol=rtol,atol=atol)
    assert_allclose(fit.loglik, -72.66388632019535,rtol=rtol,atol=atol)
    assert_allclose(fit.AICc, 158.3277726403907, rtol=rtol, atol=atol)
    assert_allclose(fit.mean_life, 123839.91259121677, rtol=rtol, atol=atol)
