from reliability.Distributions import Normal_Distribution, Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution, Competing_Risks_Model, Mixture_Model
from numpy.testing import assert_allclose

atol = 1e-8
rtol = 1e-7


def test_Weibull_Distribution():
    dist = Weibull_Distribution(alpha=5, beta=2, gamma=10)
    assert_allclose(dist.mean, 14.4311346272637895, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2.316256875880522, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 5.365045915063796, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0.6311106578189344, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3.2450893006876456, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Weibull Distribution (α=5,β=2,γ=10)'
    assert_allclose(dist.quantile(0.2), 12.361903635387193, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 12.9861134604144417, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 1.1316926249544481, rtol=rtol, atol=atol)
    xvals = [dist.gamma - 1, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.012639622357755485, 0.03969953988653618, 0.11685342455082046, 0.06069708517540586, 0.008583864105157392, 0.0010513043539513882], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.012652274632387873, 0.04010054533993554, 0.12983713838980052, 0.6069708517540585, 0.8583864105157389, 1.0513043539513862], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [0.0, 0.0010005003335835354, 0.010050335853501409, 0.10536051565782631, 2.3025850929940455, 4.605170185988091, 6.907755278982135], rtol=rtol, atol=atol)


def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert_allclose(dist.mean, 5, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 4, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Normal Distribution (μ=5,σ=2)'
    assert_allclose(dist.quantile(0.2), 3.3167575328541714, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 3.9511989745839187, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(10), 0.6454895953278145, rtol=rtol, atol=atol)
    xvals = [0, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.00876415024678427, 0.001683545038531998, 0.01332607110172904, 0.08774916596624342, 0.08774916596624342, 0.01332607110172904, 0.001683545038531998], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.006209665325776132, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [0.9937903346742238, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.00881891274345837, 0.0016852302688007987, 0.013460677880534384, 0.09749907329582604, 0.8774916596624335, 1.332607110172904, 1.6835450385319983], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [0.006229025485860027, 0.0010005003335835344, 0.01005033585350145, 0.1053605156578264, 2.302585092994045, 4.605170185988091, 6.907755278982137], rtol=rtol, atol=atol)


def test_Lognormal_Distribution():
    dist = Lognormal_Distribution(mu=2, sigma=0.8, gamma=10)
    assert_allclose(dist.mean, 20.175674306073336, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 9.634600550542682, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 92.82552776851736, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 3.689292296091298, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 34.36765343083244, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Lognormal Distribution (μ=2,σ=0.8,γ=10)'
    assert_allclose(dist.quantile(0.2), 13.7685978648453116, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 14.857284757111664, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 9.143537277214762, rtol=rtol, atol=atol)
    xvals = [dist.gamma - 1, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.006748891633682291, 0.028994071579561444, 0.08276575111567319, 0.01064970121764939, 0.0007011277158027589, 4.807498012690953e-05], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.006755647280963254, 0.029286940989456004, 0.09196194568408134, 0.10649701217649381, 0.07011277158027589, 0.04807498012690954], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [-0.0, 0.0010005003335835344, 0.01005033585350145, 0.1053605156578264, 2.302585092994045, 4.605170185988091, 6.907755278982137], rtol=rtol, atol=atol)


def test_Exponential_Distribution():
    dist = Exponential_Distribution(Lambda=0.2, gamma=10)
    assert_allclose(dist.mean, 15, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 5, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 25, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 2, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 9, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Exponential Distribution (λ=0.2,γ=10)'
    assert_allclose(dist.quantile(0.2), 11.11571775657105, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 11.783374719693661, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 5, rtol=rtol, atol=atol)
    xvals = [dist.gamma - 1, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.19980000000000003, 0.198, 0.18, 0.019999999999999997, 0.002000000000000001, 0.0002000000000000004], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [0.0, 0.0010005003335834318, 0.01005033585350148, 0.10536051565782643, 2.3025850929940463, 4.605170185988091, 6.907755278982136], rtol=rtol, atol=atol)


def test_Gamma_Distribution():
    dist = Gamma_Distribution(alpha=5, beta=2, gamma=10)
    assert_allclose(dist.mean, 20, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 7.0710678118654755, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 50, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 1.414213562373095, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 6, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Gamma Distribution (α=5,β=2,γ=10)'
    assert_allclose(dist.quantile(0.2), 14.121941545164923, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 15.486746053517457, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 6.666666666666647, rtol=rtol, atol=atol)
    xvals = [dist.gamma - 1, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.008677353779839614, 0.02560943552734864, 0.06249207734544239, 0.015909786387521992, 0.001738163417685293, 0.00018045617911753266], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.008686039819659272, 0.025868116694291555, 0.06943564149493599, 0.15909786387522004, 0.17381634176852898, 0.18045617911753245], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [-0.0, 0.0010005003335835344, 0.01005033585350145, 0.10536051565782628, 2.3025850929940463, 4.605170185988089, 6.907755278982136], rtol=rtol, atol=atol)


def test_Beta_Distribution():
    dist = Beta_Distribution(alpha=5, beta=2)
    assert_allclose(dist.mean, 0.7142857142857143, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 0.15971914124998499, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 0.025510204081632654, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -0.5962847939999439, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 2.88, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Beta Distribution (α=5,β=2)'
    assert_allclose(dist.quantile(0.2), 0.577552475153728, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 0.6396423096199797, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(0.5), 0.2518796992481146, rtol=rtol, atol=atol)
    xvals = [0, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.026583776746547504, 0.15884542294682907, 0.8802849346924463, 1.883276908534153, 0.7203329063913153, 0.23958712288762668], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.026610387133681187, 0.16044992216851423, 0.9780943718804959, 18.832769085341553, 72.03329063913147, 239.58712288762646], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [-0.0, 0.0010005003335835344, 0.01005033585350145, 0.10536051565782628, 2.3025850929940472, 4.605170185988091, 6.907755278982136], rtol=rtol, atol=atol)


def test_Loglogistic_Distribution():
    dist = Loglogistic_Distribution(alpha=50, beta=8, gamma=10)
    assert_allclose(dist.mean, 61.308607648851535, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 12.009521950735257, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 144.228617485192, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 1.2246481827926854, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 8.342064360132765, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Loglogistic Distribution (α=50,β=8,γ=10)'
    assert_allclose(dist.quantile(0.2), 52.044820762685724, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 54.975179587474166, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 41.308716243335226, rtol=rtol, atol=atol)
    xvals = [dist.gamma - 1, dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.0003789929723245846, 0.0028132580909498313, 0.01895146578651591, 0.010941633873382936, 0.0008918684027148376, 6.741239934687115e-05], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0, 0.00037975209676602, 0.002870378625599256, 0.02339687134137767, 1.0941633873382928, 8.918684027148261, 67.412399346859], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [0.0, 0.001000500333583622, 0.010050335853501506, 0.10536051565782635, 2.302585092994045, 4.605170185988085, 6.907755278982047], rtol=rtol, atol=atol)


def test_Gumbel_Distribution():
    dist = Gumbel_Distribution(mu=15, sigma=2)
    assert_allclose(dist.mean, 13.845568670196934, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2.565099660323728, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 6.579736267392906, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -1.1395470994046486, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 5.4, rtol=rtol, atol=atol)
    assert dist.param_title_long == 'Gumbel Distribution (μ=15,σ=2)'
    assert_allclose(dist.quantile(0.2), 12.00012002648097, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 12.938139133682554, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(10), 4.349172610672009, rtol=rtol, atol=atol)
    xvals = [dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0004997499166249747, 0.0049749162474832164, 0.04741223204602183, 0.11512925464970217, 0.0230258509299404, 0.003453877639491069], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0009999999999999998, 0.010000000000000002, 0.09999999999999999, 0.9000000000000001, 0.99, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [0.999, 0.99, 0.9, 0.09999999999999984, 0.009999999999999969, 0.0010000000000000002], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.0005002501667917664, 0.0050251679267507236, 0.05268025782891314, 1.1512925464970236, 2.3025850929940472, 3.4538776394910684], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [0.0010005003335835344, 0.01005033585350145, 0.10536051565782628, 2.3025850929940472, 4.6051701859880945, 6.907755278982137], rtol=rtol, atol=atol)


def test_Competing_Risks_Model():
    distributions = [Weibull_Distribution(alpha=30, beta=2), Normal_Distribution(mu=35, sigma=5)]
    dist = Competing_Risks_Model(distributions=distributions)
    assert_allclose(dist.mean, 23.707625152181073, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 9.832880925543204, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 96.68554729591138, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, -0.20597940178753704, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 2.1824677678598667, rtol=rtol, atol=atol)
    assert dist.name2 == 'Competing risks using 2 distributions'
    assert_allclose(dist.quantile(0.2), 14.170859470541174, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 17.908811127053173, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 9.862745898092886, rtol=rtol, atol=atol)
    xvals = [dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.00210671, 0.00661657, 0.01947571, 0.02655321, 0.00474024, 0.00062978], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.0010001,  0.00999995, 0.09999943, 0.90000184, 0.99000021, 0.99900003], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [0.9989999,  0.99000005, 0.90000057, 0.09999816, 0.00999979, 0.00099997], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.00210882, 0.00668341, 0.02163966, 0.265537, 0.47403341, 0.62980068], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [1.00059934e-03, 1.00502826e-02, 1.05359884e-01, 2.30260350e+00, 4.60519097e+00, 6.90778668e+00], rtol=rtol, atol=atol)


def test_Mixture_Model():
    distributions = [Weibull_Distribution(alpha=30, beta=2), Normal_Distribution(mu=35, sigma=5)]
    dist = Mixture_Model(distributions=distributions, proportions=[0.6,0.4])
    assert_allclose(dist.mean, 29.952084649328917, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 11.95293368817564, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 142.87262375392413, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0.015505959874527537, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3.4018343377801674, rtol=rtol, atol=atol)
    assert dist.name2 == 'Mixture using 2 distributions'
    assert_allclose(dist.quantile(0.2), 19.085648329240094, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 24.540270766923847, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 14.686456940211107, rtol=rtol, atol=atol)
    xvals = [dist.quantile(0.001), dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.9), dist.quantile(0.99), dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0016309, 0.00509925, 0.01423464, 0.01646686, 0.00134902, 0.00016862], rtol=rtol, atol=atol)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False), [0.00099994, 0.00999996, 0.10000006, 0.90000056, 0.99000001, 0.999], rtol=rtol, atol=atol)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False), [0.99900006, 0.99000004, 0.89999994, 0.09999944, 0.00999999, 0.001], rtol=rtol, atol=atol)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False), [0.00163253, 0.00515076, 0.01581627, 0.16466956, 0.13490177, 0.16861429], rtol=rtol, atol=atol)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False), [1.00043950e-03, 1.00502998e-02, 1.05360581e-01, 2.30259070e+00, 4.60517090e+00, 6.90775056e+00], rtol=rtol, atol=atol)

