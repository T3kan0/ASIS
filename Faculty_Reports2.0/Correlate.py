import pingouin as pg
def Corr(d): 
    corr = pg.corr(d['freq'], d['FINAL.MARK'])        
    #p_corr = d.partial_corr(x='freq', y='FINAL.MARK', covar='GR_12_ADSCORE', method='pearson').round(3)        
    pg.print_table(corr, floatfmt='.3f')
    #pg.print_table(p_corr, floatfmt='.3f')
    print('Pearson p-val~: %.3E' %corr['p-val'])
    #print('Partial Pearson p-val~: %.3E' %p_corr['p-val'])
    return [corr]
    

