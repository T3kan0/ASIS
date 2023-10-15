import scipy.stats as stats

def T_test(x, y): # running the t-test analysis 
    t_analysis = stats.ttest_ind(x,y)
    test, p = t_analysis[0], t_analysis[1]
    print('T-test = %.3f' % test, 'p-value = %.3E'%p, '\n')
    
    if p > 0.05:
        print('Since the p-value: %.1E' % p,'> 0.05, we accept the Null Hypothesis, i.e, students who attend the A_Step Supplemental Instruction tutorials at least once perform similar to those who do not attend the tutorials.')
    elif p < 0.05:
        print('Since the p-value: %.2E' % p,'< 0.05, we reject the Null Hypothesis, i.e, students who attend the A_Step Supplemental Instruction tutorials DO perform significantly better than those that do not attend the tutorials at all.')

    return [test, p]

# Effect Size Calculations
def cohend(X, Y):
    grpA_avg, grpB_avg = X.mean(), Y.mean()
    n1, n2 = len(X), len(Y)
    s1, s2 = X.std(), Y.std()
    s = (((n1 - 1) * (s1)**2 + (n2 - 1) * (s2)**2) / (n1 + n2 - 2))**(0.5)
    d = (grpA_avg - grpB_avg) / s
    if 0 < d <= 0.2:
        print('The effect size determined is small, i.e, d: %.3f' % d)
    elif 0.2 < d <= 0.5:
        print('The effect size determined is medium, i.e, d: %.3f' % d)
    elif 0.5 < d < 0.8:
        print('The effect size determined is intermediate, i.e, d: %.3f' % d)    
    elif d >= 0.8:
        print('The effect size determined is large, i.e, d: %.3f' % d)
    elif d <= 0:
        print('The effect size determined is very small, i.e, d: %.3f' % d)        
        
    return d
