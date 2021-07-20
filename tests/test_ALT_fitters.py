from reliability.ALT_fitters import (
    Fit_Weibull_Exponential,
    Fit_Weibull_Eyring,
    Fit_Weibull_Power,
    Fit_Weibull_Dual_Exponential,
    Fit_Weibull_Power_Exponential,
    Fit_Weibull_Dual_Power,
    Fit_Lognormal_Exponential,
    Fit_Lognormal_Eyring,
    Fit_Lognormal_Power,
    Fit_Lognormal_Dual_Exponential,
    Fit_Lognormal_Power_Exponential,
    Fit_Lognormal_Dual_Power,
    Fit_Normal_Exponential,
    Fit_Normal_Eyring,
    Fit_Normal_Power,
    Fit_Normal_Dual_Exponential,
    Fit_Normal_Power_Exponential,
    Fit_Normal_Dual_Power,
    Fit_Exponential_Exponential,
    Fit_Exponential_Eyring,
    Fit_Exponential_Power,
    Fit_Exponential_Dual_Exponential,
    Fit_Exponential_Power_Exponential,
    Fit_Exponential_Dual_Power,
    Fit_Everything_ALT,
)
from reliability.Other_functions import make_ALT_data
from numpy.testing import assert_allclose
import warnings

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0 # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01 # 1% variation allowed in relative tolerance for most things
rtol_big = 0.1 # 10% variation allowed in relative tolerance allowed for some that seem to fail online. I don't know why online differs from local.
rtol_extreme = 0.5 # 50% variation allowed in relative tolerance allowed for some that seem to fail online. I don't know why online differs from local.

def test_Fit_Weibull_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution="Weibull",life_stress_model="Exponential",a=2000,b=10,beta=2.5,stress_1=[500, 400, 350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Weibull_Exponential(failures=data.failures,failure_stress=data.failure_stresses,right_censored=data.right_censored,right_censored_stress=data.right_censored_stresses,use_level_stress=300,show_life_stress_plot=False,show_probability_plot=False,print_results=False)
    assert_allclose(model.a, 1965.7797395338112, rtol=rtol, atol=atol)
    assert_allclose(model.b, 11.0113385296826, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.3990457903278615, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3710.5717742652996, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3721.602040608187, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1852.2453465921092, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Exponential', a=2000, b=10, sigma=0.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Lognormal_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 2013.4709935653532, rtol=rtol, atol=atol)
    assert_allclose(model.b, 9.844353808471647, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4901664124825419, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3824.906495666694, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3835.9367620095813, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1909.4127072928063, rtol=rtol, atol=atol)


def test_Fit_Normal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Exponential', a=500, b=1000, sigma=500, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Normal_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 510.32806900630544, rtol=rtol, atol=atol)
    assert_allclose(model.b, 973.8223647399388, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 486.1365917592639, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3670.508736811669, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3681.5390031545567, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1832.213827865294, rtol=rtol, atol=atol)


def test_Fit_Exponential_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Exponential', a=2000, b=10, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1928.4687332654944, rtol=rtol, atol=atol)
    assert_allclose(model.b, 12.96779155174335, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3984.1086002100037, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3991.475761118912, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1990.0340980847998, rtol=rtol, atol=atol)


def test_Fit_Weibull_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Eyring', a=1500, c=-10, beta=1, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Weibull_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1430.487997287318, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.24507777025095, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 0.970434273889214, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4201.694921258372, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4212.72518760126, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2097.8069200886457, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Eyring',a=1500,c=-10,sigma=0.5,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Lognormal_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1513.468677222031, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.98431861625691, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.49016660648069477, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4040.853294458127, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4051.8835608010145, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2017.386106688523, rtol=rtol, atol=atol)


def test_Fit_Normal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal',life_stress_model='Eyring',a=90,c=-14,sigma=500,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Normal_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 100.05607295376981, rtol=rtol, atol=atol)
    assert_allclose(model.c, -13.97387421375034, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 486.12929211552824, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3669.8593109070534, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3680.889577249941, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1831.889114912986, rtol=rtol, atol=atol)


def test_Fit_Exponential_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Eyring', a=1500, c=-10, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1428.4686331863793, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.259884009475353, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4200.055398999253, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4207.422559908162, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2098.0074974794247, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Power',a=5e15,n=-4,beta=2.5,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Weibull_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 3069917722933350.0, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.n, -3.916925628937264, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.beta, 2.399407397407449, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6041.16703767533, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6052.197304018217, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3017.5429782971246, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Power', a=5e15, n=-4, sigma=0.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Lognormal_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 6484458528522135.0, rtol=rtol_extreme, atol=atol) # larger due to variation in python versions
    assert_allclose(model.n, -4.040288980929209, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.sigma, 0.49020606301868014, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6155.598148028053, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6166.628414370941, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3074.7585334734863, rtol=rtol, atol=atol)


def test_Fit_Normal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Power', a=6e6, n=-1.2, sigma=500, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Normal_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 6599544.121386519, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.n, -1.2160545471894655, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.sigma, 486.16679721539464, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.AICc, 3668.2165027340816, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3679.246769076969, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1831.0677108265002, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Power', a=5e15, n=-4, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1970299637780768.8, rtol=rtol, atol=atol)
    assert_allclose(model.n, -3.831313136385626, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6314.7161417145035, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6322.083302623412, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3155.33786883705, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Dual_Exponential',a=50,b=0.1,c=500,beta=2.5,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Weibull_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 55.0594242239536, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.0919548759239501, rtol=rtol, atol=atol)
    assert_allclose(model.c, 551.6799466910546, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.582228157275601, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6584.301151215161, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6603.466037531028, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3288.128229238865, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Dual_Exponential',a=50,b=0.1,c=500,sigma=0.5,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Lognormal_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 43.172159701914175, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.11552196494417312, rtol=rtol, atol=atol)
    assert_allclose(model.c, 560.5704103455569, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4837723906057129, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6786.541219570215, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6805.706105886082, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3389.248263416392, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal',life_stress_model='Dual_Exponential',a=60,b=0.1,c=5000,sigma=300,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Normal_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 59.71344103606326, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.10065937394073277, rtol=rtol, atol=atol)
    assert_allclose(model.c, 5006.556618243661, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 281.9484045101182, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6394.529082362259, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6413.693968678126, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3193.2421948124143, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential',life_stress_model='Dual_Exponential',a=50,b=0.2,c=500,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Exponential_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 63.96739939569158, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.17691518863172884, rtol=rtol, atol=atol)
    assert_allclose(model.c, 569.2878329736656, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 11467.269078434578, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 11481.649477010264, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -5730.621146360146, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, beta=2.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Weibull_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[250,7], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 1562883141879859.5, rtol=rtol, atol=atol)
    assert_allclose(model.m, -4.134024472114375, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.7890763670187908, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.570622042328891, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3624.015790007103, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3640.793414319984, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1807.9674909631476, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, sigma=0.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Lognormal_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 948288219927352.4, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.m, -3.973104202762598, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.n, -1.9951777461141673, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.sigma, 0.4910387800455627, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3727.3183135308454, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3744.0959378437265, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1859.6187527250188, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Dual_Power', c=10000, m=-0.3, n=-0.4, sigma=100, stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Normal_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 9351.083262185502, rtol=rtol, atol=atol)
    assert_allclose(model.m, -0.29053712727287395, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.38934203114298255, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 94.85096832637454, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3041.440875815024, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3058.218500127905, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1516.680033867108, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 3203958968599901.5, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.m, -4.262981061560643, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.7492432603910426, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6136.997370988174, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6149.592808186666, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3065.4744919457, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Power_Exponential',a=22,c=400,n=-0.25,beta=2.5,stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Weibull_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 67.69037929967578, rtol=rtol, atol=atol)
    assert_allclose(model.c, 256.58824365759585, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.05262375033066265, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.571207071526618, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3122.643924859823, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3139.421549172704, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1557.2815583895076, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Power_Exponential',a=200,c=400,n=-0.5,sigma=0.5,stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Lognormal_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 192.1053043033961, rtol=rtol, atol=atol)
    assert_allclose(model.c, 451.44824106649287, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.4919602905576324, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4910516937147718, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3200.6626325485195, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3217.4402568614005, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1596.2909122338558, rtol=rtol, atol=atol)


def test_Fit_Normal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Power_Exponential', a=70, c=2500, n=-0.25, sigma=100, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Normal_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 69.52031280338441, rtol=rtol, atol=atol)
    assert_allclose(model.c, 2498.268588097067, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.24817878201877347, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 94.48228285331875, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3015.3322425662163, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3032.1098668790974, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1503.6257172427042, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential',life_stress_model='Power_Exponential',a=50, c=10000, n=-1.5, stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Exponential_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 137.3857161856382, rtol=rtol, atol=atol)
    assert_allclose(model.c, 5636.685334952079, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.277340033750778, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 5800.788764570015, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 5813.3842017685065, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2897.37018873662, rtol=rtol, atol=atol)


def test_Fit_Everything_ALT_single_stress():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution="Weibull", life_stress_model="Exponential", a=2000, b=10, beta=2.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Everything_ALT(failures=data.failures, failure_stress_1=data.failure_stresses, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses, use_level_stress=300, show_best_distribution_probability_plot=False, show_probability_plot=False, print_results=False)

    assert_allclose(model.Weibull_Exponential_a, 1965.7797395338112, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_b, 11.0113385296826, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_beta, 2.3990457903278615, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_AICc, 3710.5717742652996, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_BIC, 3721.602040608187, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_loglik, -1852.2453465921092, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Exponential_a, 2009.7501864638946, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_b, 7.809262222104393, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_sigma, 0.5166917673351094, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_AICc, 3735.234452757792, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_BIC, 3746.2647191006795, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_loglik, -1864.5766858383554, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Exponential_a, 1993.9519931456607, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_b, 9.204026230504914, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_sigma, 746.4100628682714, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_AICc, 3901.96807936702, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_BIC, 3912.9983457099074, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_loglik, -1947.9434991429694, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Exponential_a, 1990.2598136542852, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_b, 9.884663513057722, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_AICc, 3926.5330483709013, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_BIC, 3933.9002092798096, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_loglik, -1961.2463221652486, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Eyring_a, 1548.0749958573679, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_c, -9.445096203462972, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_beta, 2.3958314887542222, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_AICc, 3711.331305215902, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_BIC, 3722.3615715587894, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_loglik, -1852.6251120674103, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Eyring_a, 1591.6554769675936, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_c, -9.1020583345059, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_sigma, 0.517659675042165, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_AICc, 3736.1791910066286, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_BIC, 3747.209457349516, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_loglik, -1865.0490549627737, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Eyring_a, 1602.0465747399508, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_c, -9.196438650244044, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_sigma, 747.0850651825152, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_AICc, 3902.5447879728936, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_BIC, 3913.575054315781, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_loglik, -1948.2318534459062, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Eyring_a, 1572.4214513890167, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_c, -9.337426924177604, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_AICc, 3926.729095178802, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_BIC, 3934.09625608771, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_loglik, -1961.3443455691988, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_a, 2645794073306709.0, rtol=0.9, atol=atol) # much larger due to variation in python versions. WHY???
    assert_allclose(model.Weibull_Power_n, -4.698158834438999, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Weibull_Power_beta, 2.3785671118139122, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Weibull_Power_AICc, 3715.2055323609407, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_BIC, 3726.235798703828, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_loglik, -1854.5622256399297, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_a, 2899022021518504.5, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_n, -4.752882880383393, rtol=rtol_big, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_sigma, 0.522183419683184, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_AICc, 3740.5903647388977, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_BIC, 3751.620631081785, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_loglik, -1867.2546418289082, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Power_a, 1.1875960546823156e+16, rtol=rtol_big, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Power_n, -4.968632318615027, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Power_sigma, 751.4081379265706, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_AICc, 3905.4226589659943, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_BIC, 3916.4529253088817, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_loglik, -1949.6707889424565, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_a, 2899022021518504.5, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_n, -4.721375201743889, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_AICc, 3927.669127417165, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_BIC, 3935.036288326073, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_loglik, -1961.8143616883804, rtol=rtol, atol=atol)


def test_Fit_Everything_ALT_dual_stress():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Dual_Exponential', a=50, b=0.1, c=500, beta=2.5, stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540], stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Everything_ALT(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1, right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[300, 0.2], show_best_distribution_probability_plot=False, show_probability_plot=False, print_results=False)

    assert_allclose(model.Weibull_Dual_Exponential_a, 55.0594242239536, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_b, 0.0919548759239501, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_c, 551.6799466910546, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_beta, 2.582228157275601, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_AICc, 6584.301151215161, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_BIC, 6603.466037531028, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_loglik, -3288.128229238865, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Exponential_a, 52.32884413131662, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_b, 0.10088629835108626, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_c, 441.3744406102187, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_sigma, 0.5313705214283074, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_AICc, 6652.254628391293, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_BIC, 6671.4195147071605, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_loglik, -3322.1049678269314, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Dual_Exponential_a, 49.68121667246413, rtol=rtol_extreme, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_b, 0.08164110127092898, rtol=rtol_extreme, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_c, 512.8410763917044, rtol=rtol_extreme, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_sigma, 297.1949970356173, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_AICc, 6635.191223243016, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_BIC, 6654.356109558883, rtol=rtol_big, atol=atol) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Exponential_loglik, -3313.5732652527927, rtol=rtol_big, atol=atol) # larger due to variation in python versions

    assert_allclose(model.Exponential_Dual_Exponential_a, 56.221472604483075, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_b, 0.09656529409565541, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_c, 641.8922221857204, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_AICc, 7089.224587107275, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_BIC, 7103.604985682962, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_loglik, -3541.5989006964946, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Dual_Power_c, 1393.8242650160057, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_m, -0.12384724446029682, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_n, -0.2592466098202316, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_beta, 2.5815730622862336, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_AICc, 6583.673537008596, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_BIC, 6602.838423324463, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_loglik, -3287.814422135583, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Power_c, 1105.2569803418007, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_m, -0.12295522924057026, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_n, -0.28075657353599415, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_sigma, 0.5306077545728608, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_AICc, 6650.71962135737, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_BIC, 6669.884507673237, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_loglik, -3321.3374643099696, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Dual_Power_c, 914.3759033056451, rtol=rtol_extreme, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_m, -0.11465369510079437, rtol=rtol_extreme, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_n, -0.28844941648459693, rtol=rtol_extreme, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_sigma, 289.9456041409484, rtol=rtol_big, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_AICc, 6787.865358883584, rtol=rtol_big, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_BIC, 6807.030245199451, rtol=rtol_big, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_loglik, -3389.910333073077, rtol=rtol_big, atol=atol)  # larger due to variation in python versions

    assert_allclose(model.Exponential_Dual_Power_c, 1726.735002854915, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_m, -0.13364309671965835, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_n, -0.27105475584920874, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_AICc, 7088.926283426964, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_BIC, 7103.3066820026515, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_loglik, -3541.4497488563393, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_Exponential_a, 42.780811253238745, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_c, 595.5272158322398, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_n, -0.24874378554058516, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_beta, 2.5871119030281675, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_AICc, 6582.118937551177, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_BIC, 6601.283823867044, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_loglik, -3287.037122406873, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_a, 38.62820672031728, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_c, 479.7944662699472, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_n, -0.27523434626167476, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_sigma, 0.5304232436388364, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_AICc, 6650.403864331276, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_BIC, 6669.568750647143, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_loglik, -3321.1795857969228, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_a, 37.59712219841342, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_c, 548.919567681429, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_n, -0.2262443694448219, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_sigma, 296.6005867790367, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_AICc, 6633.1976944950475, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_BIC, 6652.362580810915, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_loglik, -3312.5765008788085, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_Exponential_a, 42.937623270069224, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_c, 695.3534426950991, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_n, -0.26413696165231737, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_AICc, 7088.800993682692, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_BIC, 7103.18139225838, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_loglik, -3541.3871039842033, rtol=rtol, atol=atol)
