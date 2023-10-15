import pingouin as pg
def Post_hoc(a, b, c, d, f , p):
    if p < 0.05:
        pst_hoc = pg.pairwise_gameshowell(data=d, dv='FINAL.MARK', between='freq', effsize='cohen')
        pg.print_table(pst_hoc, floatfmt='.3f')
    else:
        print('The One-Way ANOVA has completed running......','\n','\n', 'The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')
    return pst_hoc
