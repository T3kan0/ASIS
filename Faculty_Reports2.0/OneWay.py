from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression

def anova(x, y, z):
    f_stat, p = f_oneway(x, y, z)
    eta2 = f_stat**2 / (1 + f_stat**2)
    print('F-stat: %.3f' % f_stat, '\n',
     'p-value: %.1E' % p, '\n')
    print('eta2: %.3f' % eta2,'\n')

    if eta2 <= 0.1:
        print('An effect size was calculated, and a small effect is found: %.3f' % eta2, '\n' )
    elif 0.1 < eta2 <= 0.6:
        print('An effect size was calculated, and a medium effect is found: %.3f' % eta2, '\n' )
    elif 0.6 < eta2 <= 0.14:
        print('An effect size was calculated, and an intermedium effect is found: %.3f' % eta2, '\n' )
    elif eta2 > 0.14:
        print('An effect size was calculated, and a large effect is found: %.3f' % eta2, '\n' )     
    if p > 0.05:
        print('The Multi-sample F-statistic determined is F-stat: %.3f' % f_stat, '.The calculated p-value: %.1E' % p, ',is greater than 0.05 therefore we accept the Null Hypothesis.')
    elif p < 0.05:
        print('The Multi-sample F-statistic determined is F-stat: %.3f' % f_stat, '.The calculated p-value: %.1E' % p, ',is less than 0.05 therefore we reject the Null Hypothesis.')
    if f_stat == 1:
        print('The Multi-sample F-statistic determined is F-stat: %.3f' % f_stat, ',is equal to 1.0 therefore we accept the Null Hypothesis.') 
    return [f_stat, p, eta2]
# Determination Coefficient (linear Regression)   

def Regression(x, y):
    reg = LinearRegression().fit(x, y)
    r2 = reg.score(x, y) # Determination Coefficient
    r2_perc = r2*100
    print('The determination coefficient between student attendance and final mark is %.3f' % r2, ', i.e %.2f' %r2_perc, '% of the variation in final marks of students can be explained by the variation in tutorial attendance')
    return r2_perc 
