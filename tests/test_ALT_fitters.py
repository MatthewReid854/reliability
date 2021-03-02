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
atol = 1e-3
rtol = 1e-3


def test_Fit_Weibull_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution="Weibull",life_stress_model="Exponential",a=2000,b=10,beta=2.5,stress_1=[500, 400, 350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Weibull_Exponential(failures=data.failures,failure_stress=data.failure_stresses,right_censored=data.right_censored,right_censored_stress=data.right_censored_stresses,use_level_stress=300,show_life_stress_plot=False,show_probability_plot=False,print_results=False)
    assert_allclose(model.a, 1936.3688190658345, rtol=rtol, atol=atol)
    assert_allclose(model.b, 11.308521997679772, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.3298257418546253, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3706.429938184347, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3717.4602045272345, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1850.174428551633, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Exponential', a=2000, b=10, sigma=0.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Lognormal_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1988.0495905259006, rtol=rtol, atol=atol)
    assert_allclose(model.b, 9.801624494598041, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4904393881039806, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3799.2513441930246, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3810.281610535912, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1896.5851315559717, rtol=rtol, atol=atol)


def test_Fit_Normal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Exponential', a=500, b=1000, sigma=500, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Normal_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 501.7293251600865, rtol=rtol, atol=atol)
    assert_allclose(model.b, 985.8942609995661, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 487.32084165860937, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3672.8947423752716, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3683.925008718159, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1833.4068306470951, rtol=rtol, atol=atol)


def test_Fit_Exponential_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Exponential', a=2000, b=10, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Exponential(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 2027.055631091525, rtol=rtol, atol=atol)
    assert_allclose(model.b, 8.907483857459324, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3919.882527782735, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3927.2496886916433, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1957.9210618711654, rtol=rtol, atol=atol)


def test_Fit_Weibull_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Eyring', a=1500, c=-10, beta=1, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Weibull_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1531.2238610041022, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.864422561498607, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 0.9528000462635589, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4136.93400700384, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4147.964273346727, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2065.4264629613795, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Eyring',a=1500,c=-10,sigma=0.5,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Lognormal_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1488.0362814357086, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.97999064378494, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.49043921782156963, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4015.1981429466014, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4026.228409289489, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2004.55853093276, rtol=rtol, atol=atol)


def test_Fit_Normal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal',life_stress_model='Eyring',a=90,c=-14,sigma=500,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Normal_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 92.66498061993232, rtol=rtol, atol=atol)
    assert_allclose(model.c, -13.983700193304404, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 486.96957445638174, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3672.599393693993, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3683.6296600368805, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1833.259156306456, rtol=rtol, atol=atol)


def test_Fit_Exponential_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Eyring', a=1500, c=-10, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1527.0636662965046, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.884286999041207, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4135.829326569959, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4143.196487478868, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2065.894461264778, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Power',a=5e15,n=-4,beta=2.5,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Weibull_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 3925605570237214.5, rtol=rtol, atol=atol)
    assert_allclose(model.n, -3.9652234855813573, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.3255321818860417, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6037.469208963296, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6048.499475306184, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3015.6940639411077, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Power', a=5e15, n=-4, sigma=0.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Lognormal_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 3491334253992439.0, rtol=0.06, atol=2e14) # larger due to variation in python versions
    assert_allclose(model.n, -3.9485517292713515, rtol=0.01, atol=0.01) # larger due to variation in python versions
    assert_allclose(model.sigma, 0.49040270483915543, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6129.883772738487, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6140.9140390813745, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3061.901345828703, rtol=rtol, atol=atol)


def test_Fit_Normal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Power', a=6e6, n=-1.2, sigma=500, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Normal_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 5535409.190946918, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.1877975512832173, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 485.6163925856839, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3669.0822731568105, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3680.112539499698, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1831.5005960378646, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Power', a=5e15, n=-4, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1.265065761069241e+16, rtol=rtol, atol=atol)
    assert_allclose(model.n, -4.162257759592342, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6250.618290860572, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6257.985451769481, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3123.2889434100844, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Dual_Exponential',a=50,b=0.1,c=500,beta=2.5,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Weibull_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 72.70458292197362, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.07187386863016872, rtol=rtol, atol=atol)
    assert_allclose(model.c, 481.6168222048736, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.4249656667486277, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6596.195726006904, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6615.360612322771, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3294.0755166347367, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Dual_Exponential',a=50,b=0.1,c=500,sigma=0.5,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Lognormal_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 43.08752282053009, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.1189315432463663, rtol=rtol, atol=atol)
    assert_allclose(model.c, 489.158614782064, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.47004948565478233, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6690.00527426434, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6709.170160580207, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3340.980290763455, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal',life_stress_model='Dual_Exponential',a=60,b=0.1,c=5000,sigma=300,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Normal_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 60.35437729968465, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.09981855412212184, rtol=rtol, atol=atol)
    assert_allclose(model.c, 4996.585098542188, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 279.85638158333495, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6377.720218019938, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6396.885104335805, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3184.837762641254, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential',life_stress_model='Dual_Exponential',a=50,b=0.2,c=500,stress_1=[500,400,350,300,200,180,390,250,540],stress_2=[0.9,0.8,0.7,0.6,0.3,0.3,0.2,0.7,0.5],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Exponential_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 88.9138420005482, rtol=0.003, atol=0.3) # larger due to variation in python versions
    assert_allclose(model.b, 0.15559480591338323, rtol=rtol, atol=atol)
    assert_allclose(model.c, 484.42992854739765, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 11283.120647795477, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 11297.501046371162, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -5638.546931040595, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, beta=2.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Weibull_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[250,7], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 2147917445796472.5, rtol=rtol, atol=atol)
    assert_allclose(model.m, -4.19698891031934, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.8188837776084512, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.573265136861532, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3607.244015664953, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3624.021639977834, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1799.5816037920727, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, sigma=0.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Lognormal_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 812819384851496.5, rtol=rtol, atol=atol)
    assert_allclose(model.m, -3.9812182899327944, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.9654140737394001, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4668564083341826, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3659.6871770336484, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3676.4648013465294, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1825.8031844764203, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Dual_Power', c=10000, m=-0.3, n=-0.4, sigma=100, stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Normal_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 8002.021009698654, rtol=rtol, atol=atol)
    assert_allclose(model.m, -0.2717479839737406, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.3758558923943934, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 97.09886297315518, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3052.5373106357024, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3069.3149349485834, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1522.2282512774473, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Dual_Power(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[100,0.2], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.c, 8217056028827617.0, rtol=0.02, atol=1e14) # larger due to variation in python versions
    assert_allclose(model.m, -4.4290315992680895, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.7851821319822152, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6041.4828154995885, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6054.07825269808, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3017.717214201407, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull',life_stress_model='Power_Exponential',a=22,c=400,n=-0.25,beta=2.5,stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Weibull_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 87.27474468196765, rtol=rtol, atol=atol)
    assert_allclose(model.c, 233.69041245172616, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.08955095819729136, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.57340107289803, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3105.946142550132, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3122.723766863013, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1548.9326672346622, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Lognormal',life_stress_model='Power_Exponential',a=200,c=400,n=-0.5,sigma=0.5,stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.5,seed=1)
    model = Fit_Lognormal_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 192.65970442010217, rtol=rtol, atol=atol)
    assert_allclose(model.c, 369.5260544068452, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.46381125969363973, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4668444200819145, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3133.009344984278, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3149.786969297159, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1562.464268451735, rtol=rtol, atol=atol)


def test_Fit_Normal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Normal', life_stress_model='Power_Exponential', a=70, c=2500, n=-0.25, sigma=100, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Normal_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 68.00256140337429, rtol=rtol, atol=atol)
    assert_allclose(model.c, 2466.2001120276796, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.24142018246269986, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 94.81300438484078, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3012.995178718041, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3029.772803030922, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1502.4571853186167, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential',life_stress_model='Power_Exponential',a=50, c=10000, n=-1.5, stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.2,seed=1)
    model = Fit_Exponential_Power_Exponential(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1,right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[200,5], show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 192.088621998411, rtol=rtol, atol=atol)
    assert_allclose(model.c, 4830.831901703228, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.3309081355641201, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 5705.289499683757, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 5717.884936882248, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2849.620556293491, rtol=rtol, atol=atol)


def test_Fit_Everything_ALT_single_stress():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution="Weibull", life_stress_model="Exponential", a=2000, b=10, beta=2.5, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Everything_ALT(failures=data.failures, failure_stress_1=data.failure_stresses, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses, use_level_stress=300, show_best_distribution_probability_plot=False, show_probability_plot=False, print_results=False)

    assert_allclose(model.Weibull_Exponential_a, 1936.3688190658345, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_b, 11.308521997679772, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_beta, 2.3298257418546253, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_AICc, 3706.429938184347, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_BIC, 3717.4602045272345, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_loglik, -1850.174428551633, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Exponential_a, 2024.7445929016446, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_b, 7.1938303858071055, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_sigma, 0.5175936302119497, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_AICc, 3717.61714155573, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_BIC, 3728.6474078986175, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_loglik, -1855.7680302373244, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Exponential_a, 1944.6559973964154, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_b, 9.923244470035774, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_sigma, 701.2726420148408, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_AICc, 3881.5070977044597, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_BIC, 3892.537364047347, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_loglik, -1937.7130083116892, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Exponential_a, 1994.3280823815207, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_b, 9.447099334611114, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_AICc, 3909.5886314069853, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_BIC, 3916.9557923158936, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_loglik, -1952.7741136832906, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Eyring_a, 1518.6648984357357, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_c, -9.471675146379535, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_beta, 2.3276534089926026, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_AICc, 3707.0416877391667, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_BIC, 3718.071954082054, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_loglik, -1850.4803033290427, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Eyring_a, 1606.908147563152, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_c, -9.019438435183952, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_sigma, 0.5184229852931106, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_AICc, 3718.474200140114, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_BIC, 3729.5044664830016, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_loglik, -1856.1965595295164, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Eyring_a, 1551.836191728938, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_c, -9.27426477047246, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_sigma, 701.81395017954, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_AICc, 3882.024455385868, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_BIC, 3893.0547217287553, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_loglik, -1937.9716871523933, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Eyring_a, 1576.4074594032234, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_c, -9.292349877329539, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_AICc, 3909.784187121611, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_BIC, 3917.1513480305193, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_loglik, -1952.8718915406034, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_a, 1704571726306460.2, rtol=0.02, atol=3e13) # larger due to variation in python versions
    assert_allclose(model.Weibull_Power_n, -4.632734486704249, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_beta, 2.314945291433339, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_AICc, 3710.2611745093573, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_BIC, 3721.2914408522447, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_loglik, -1852.090046714138, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_a, 3736761709377962.0, rtol=0.05, atol=2e14) # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_n, -4.802472968284754, rtol=0.002, atol=0.008)  # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_sigma, 0.522471206271303, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_AICc, 3722.611042306387, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_BIC, 3733.6413086492744, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_loglik, -1858.2649806126528, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Power_a, 5382471637730783.0, rtol=0.06, atol=3e14)  # larger due to variation in python versions
    assert_allclose(model.Normal_Power_n, -4.844657942648189, rtol=0.002, atol=0.01) # larger due to variation in python versions
    assert_allclose(model.Normal_Power_sigma, 705.2782926967965, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_AICc, 3884.6006939979784, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_BIC, 3895.630960340866, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_loglik, -1939.2598064584486, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_a, 3210488760191281.5, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_n, -4.744177818025859, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_AICc, 3910.7238668159157, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_BIC, 3918.091027724824, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_loglik, -1953.3417313877558, rtol=rtol, atol=atol)


def test_Fit_Everything_ALT_dual_stress():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Weibull', life_stress_model='Dual_Exponential', a=50, b=0.1, c=500, beta=2.5, stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540], stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5], number_of_samples=100, fraction_censored=0.5, seed=1)
    model = Fit_Everything_ALT(failures=data.failures, failure_stress_1=data.failure_stresses_1, failure_stress_2=data.failure_stresses_2, right_censored=data.right_censored, right_censored_stress_1=data.right_censored_stresses_1, right_censored_stress_2=data.right_censored_stresses_2, use_level_stress=[300, 0.2], show_best_distribution_probability_plot=False, show_probability_plot=False, print_results=False)

    assert_allclose(model.Weibull_Dual_Exponential_a, 72.70458292197362, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_b, 0.07187386863016872, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_c, 481.6168222048736, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_beta, 2.4249656667486277, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_AICc, 6596.195726006904, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_BIC, 6615.360612322771, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_loglik, -3294.0755166347367, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Exponential_a, 75.41297843984876, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_b, 0.057051226262971266, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_c, 402.8817086778958, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_sigma, 0.5797526820747794, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_AICc, 6647.4829119679625, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_BIC, 6666.64779828383, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_loglik, -3319.719109615266, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Dual_Exponential_a, 63.71771099895282, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_b, 0.05481867058120912, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_c, 456.0884657689198, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_sigma, 273.0230433833476, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_AICc, 6661.742701385689, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_BIC, 6680.907587701556, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Dual_Exponential_loglik, -3326.849004324129, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Dual_Exponential_a, 77.68419617087345, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_b, 0.06141350780484315, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_c, 598.0886735973173, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_AICc, 7017.033369550939, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_BIC, 7031.413768126627, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_loglik, -3505.503291918327, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Dual_Power_c, 1862.6163327808374, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_m, -0.19022378193666223, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_n, -0.21591496265071694, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_beta, 2.4298355941698535, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_AICc, 6594.617340511946, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_BIC, 6613.782226827813, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_loglik, -3293.286323887258, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Power_c, 1869.900973241963, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_m, -0.22046498927399688, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_n, -0.16991933949885826, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_sigma, 0.5789359518369657, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_AICc, 6646.11611739848, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_BIC, 6665.281003714347, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_loglik, -3319.035712330525, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Dual_Power_c, 1660.148499564434, rtol=0.11, atol=180)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_m, -0.18518820513140954, rtol=0.21, atol=0.04)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_n, -0.1626895367508826, rtol=0.13, atol=0.03)  # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_sigma, 272.48450130744226, rtol=0.04, atol=9) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_AICc, 6659.922216946836, rtol=0.05, atol=270) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_BIC, 6679.087103262703, rtol=0.05, atol=270) # larger due to variation in python versions
    assert_allclose(model.Normal_Dual_Power_loglik, -3325.9387621047026, rtol=0.05, atol=140) # larger due to variation in python versions

    assert_allclose(model.Exponential_Dual_Power_c, 2774.674297444099, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_m, -0.2191235132567922, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_n, -0.18452556112438676, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_AICc, 7016.779352700353, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_BIC, 7031.159751276041, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_loglik, -3505.376283493034, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_Exponential_a, 61.8031788732226, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_c, 509.408154921238, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_n, -0.2046637316779507, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_beta, 2.4355835569545903, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_AICc, 6593.113810396184, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_BIC, 6612.278696712051, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_loglik, -3292.5345588293767, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_a, 66.32839044639019, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_c, 421.93975893816895, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_n, -0.16331254935382397, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_sigma, 0.5788828408130797, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_AICc, 6646.083837922607, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_BIC, 6665.248724238474, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_loglik, -3319.0195725925882, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Power_Exponential_a, 54.586699746347655, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_c, 477.08319152466174, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_n, -0.15742111853292584, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_sigma, 272.3597735570058, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_AICc, 6659.480475337503, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_BIC, 6678.64536165337, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_loglik, -3325.717891300036, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_Exponential_a, 68.01310079531774, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_c, 628.3662308657258, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_n, -0.17571975297412967, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_AICc, 7016.635002980417, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_BIC, 7031.015401556105, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_loglik, -3505.304108633066, rtol=rtol, atol=atol)





