from reliability.Distributions import Normal_Distribution, Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution
from numpy.testing import assert_allclose
tolerance = 1e-10

def test_Weibull_Distribution():
    dist = Weibull_Distribution(alpha=5, beta=2, gamma=10)
    assert dist.mean == 14.4311346272637895
    assert dist.standard_deviation == 2.316256875880522
    assert dist.variance == 5.365045915063796
    assert dist.skewness == 0.6311106578189344
    assert dist.kurtosis == 3.2450893006876456
    assert dist.param_title_long == 'Weibull Distribution (α=5,β=2,γ=10)'
    assert dist.quantile(0.2) == 12.361903635387193
    assert dist.inverse_SF(q=0.7) == 12.9861134604144417
    assert dist.mean_residual_life(20) == 1.1316926249544481
    xvals=[dist.gamma-1, dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.012639622357755485, 0.03969953988653618, 0.11685342455082046, 0.06069708517540586, 0.008583864105157392, 0.0010513043539513882], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.012652274632387873, 0.04010054533993554, 0.12983713838980052, 0.6069708517540585, 0.8583864105157389, 1.0513043539513862], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[0.0, 0.0010005003335835354, 0.010050335853501409, 0.10536051565782631, 2.3025850929940455, 4.605170185988091, 6.907755278982135], rtol=tolerance)

def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 5
    assert dist.standard_deviation == 2
    assert dist.variance == 4
    assert dist.skewness == 0
    assert dist.kurtosis == 3
    assert dist.param_title_long == 'Normal Distribution (μ=5,σ=2)'
    assert dist.quantile(0.2) == 3.3167575328541714
    assert dist.inverse_SF(q=0.7) == 3.9511989745839187
    assert dist.mean_residual_life(10) == 0.6454895953278145
    xvals=[0, dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False),[0.00876415024678427, 0.001683545038531998, 0.01332607110172904, 0.08774916596624342, 0.08774916596624342, 0.01332607110172904, 0.001683545038531998], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.006209665325776132, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[0.9937903346742238, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.00881891274345837, 0.0016852302688007987, 0.013460677880534384, 0.09749907329582604, 0.8774916596624335, 1.332607110172904, 1.6835450385319983], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[0.006229025485860027, 0.0010005003335835344, 0.01005033585350145, 0.1053605156578264, 2.302585092994045, 4.605170185988091, 6.907755278982137], rtol=tolerance)

def test_Lognormal_Distribution():
    dist = Lognormal_Distribution(mu=2, sigma=0.8, gamma=10)
    assert dist.mean == 20.175674306073336
    assert dist.standard_deviation == 9.634600550542682
    assert dist.variance == 92.82552776851736
    assert dist.skewness == 3.689292296091298
    assert dist.kurtosis == 34.36765343083244
    assert dist.param_title_long == 'Lognormal Distribution (μ=2,σ=0.8,γ=10)'
    assert dist.quantile(0.2) == 13.7685978648453116
    assert dist.inverse_SF(q=0.7) == 14.857284757111664
    assert dist.mean_residual_life(20) == 9.143537277214762
    xvals=[dist.gamma-1,dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False),[0.0, 0.006748891633682291, 0.028994071579561444, 0.08276575111567319, 0.01064970121764939, 0.0007011277158027589, 4.807498012690953e-05], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.006755647280963254, 0.029286940989456004, 0.09196194568408134, 0.10649701217649381, 0.07011277158027589, 0.04807498012690954], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[-0.0, 0.0010005003335835344, 0.01005033585350145, 0.1053605156578264, 2.302585092994045, 4.605170185988091, 6.907755278982137], rtol=tolerance)

def test_Exponential_Distribution():
    dist = Exponential_Distribution(Lambda=0.2, gamma=10)
    assert dist.mean == 15
    assert dist.standard_deviation == 5
    assert dist.variance == 25
    assert dist.skewness == 2
    assert dist.kurtosis == 9
    assert dist.param_title_long == 'Exponential Distribution (λ=0.2,γ=10)'
    assert dist.quantile(0.2) == 11.11571775657105
    assert dist.inverse_SF(q=0.7) == 11.783374719693661
    assert dist.mean_residual_life(20) == 5
    xvals=[dist.gamma-1, dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False),[0.0, 0.19980000000000003, 0.198, 0.18, 0.019999999999999997, 0.002000000000000001, 0.0002000000000000004], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[0.0, 0.0010005003335834318, 0.01005033585350148, 0.10536051565782643, 2.3025850929940463, 4.605170185988091, 6.907755278982136], rtol=tolerance)

def test_Gamma_Distribution():
    dist = Gamma_Distribution(alpha=5, beta=2, gamma=10)
    assert dist.mean == 20
    assert dist.standard_deviation == 7.0710678118654755
    assert dist.variance == 50
    assert dist.skewness == 1.414213562373095
    assert dist.kurtosis == 6
    assert dist.param_title_long == 'Gamma Distribution (α=5,β=2,γ=10)'
    assert dist.quantile(0.2) == 14.121941545164923
    assert dist.inverse_SF(q=0.7) == 15.486746053517457
    assert dist.mean_residual_life(20) == 6.666666666666647
    xvals=[dist.gamma-1, dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False),[0.0, 0.008677353779839614, 0.02560943552734864, 0.06249207734544239, 0.015909786387521992, 0.001738163417685293, 0.00018045617911753266], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.008686039819659272, 0.025868116694291555, 0.06943564149493599, 0.15909786387522004, 0.17381634176852898, 0.18045617911753245], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[-0.0, 0.0010005003335835344, 0.01005033585350145, 0.10536051565782628, 2.3025850929940463, 4.605170185988089, 6.907755278982136], rtol=tolerance)

def test_Beta_Distribution():
    dist = Beta_Distribution(alpha=5, beta=2)
    assert dist.mean == 0.7142857142857143
    assert dist.standard_deviation == 0.15971914124998499
    assert dist.variance == 0.025510204081632654
    assert dist.skewness == -0.5962847939999439
    assert dist.kurtosis == 2.88
    assert dist.param_title_long == 'Beta Distribution (α=5,β=2)'
    assert dist.quantile(0.2) == 0.577552475153728
    assert dist.inverse_SF(q=0.7) == 0.6396423096199797
    assert dist.mean_residual_life(0.5) == 0.2518796992481146
    xvals=[0,dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False),[0.0, 0.026583776746547504, 0.15884542294682907, 0.8802849346924463, 1.883276908534153, 0.7203329063913153, 0.23958712288762668], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.026610387133681187, 0.16044992216851423, 0.9780943718804959, 18.832769085341553, 72.03329063913147, 239.58712288762646], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[-0.0, 0.0010005003335835344, 0.01005033585350145, 0.10536051565782628, 2.3025850929940472, 4.605170185988091, 6.907755278982136], rtol=tolerance)

def test_Loglogistic_Distribution():
    dist = Loglogistic_Distribution(alpha=50, beta=8, gamma=10)
    assert dist.mean == 61.308607648851535
    assert dist.standard_deviation == 12.009521950735257
    assert dist.variance == 144.228617485192
    assert dist.skewness == 1.2246481827926854
    assert dist.kurtosis == 8.342064360132765
    assert dist.param_title_long == 'Loglogistic Distribution (α=50,β=8,γ=10)'
    assert dist.quantile(0.2) == 52.044820762685724
    assert dist.inverse_SF(q=0.7) == 54.975179587474166
    assert dist.mean_residual_life(20) == 41.308716243335226
    xvals=[dist.gamma-1, dist.quantile(0.001),dist.quantile(0.01),dist.quantile(0.1),dist.quantile(0.9),dist.quantile(0.99),dist.quantile(0.999)]
    assert_allclose(dist.PDF(xvals=xvals, show_plot=False), [0.0, 0.0003789929723245846, 0.0028132580909498313, 0.01895146578651591, 0.010941633873382936, 0.0008918684027148376, 6.741239934687115e-05], rtol=tolerance)
    assert_allclose(dist.CDF(xvals=xvals, show_plot=False),[0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999], rtol=tolerance)
    assert_allclose(dist.SF(xvals=xvals, show_plot=False),[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001], rtol=tolerance)
    assert_allclose(dist.HF(xvals=xvals, show_plot=False),[0.0, 0.00037975209676602, 0.002870378625599256, 0.02339687134137767, 1.0941633873382928, 8.918684027148261, 67.412399346859], rtol=tolerance)
    assert_allclose(dist.CHF(xvals=xvals, show_plot=False),[0.0, 0.001000500333583622, 0.010050335853501506, 0.10536051565782635, 2.302585092994045, 4.605170185988085, 6.907755278982047], rtol=tolerance)
