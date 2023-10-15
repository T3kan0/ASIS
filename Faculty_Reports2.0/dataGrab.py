#!/usr/bin/env python
# coding: utf-8
import pyreadr
import scipy.stats as stats
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import numpy as np

def data_read(dat):
    r_dt = pyreadr.read_r(dat)
#  objects
   # print(r_dt.keys()) # let's check what objects we got
    df = r_dt['GroupedData']
    return df
def dataSelector(fac, term, camp):

    x = 'complete_data.RData'
    df = data_read(x)
    df['FACULTY'] = df['FACULTY'].replace(['MHSC'],'HEALTH SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MLAW'],'LAW')
    df['FACULTY'] = df['FACULTY'].replace(['MHUM'],'HUMANITIES')
    df['FACULTY'] = df['FACULTY'].replace(['MEMS'],'ECONOMIC AND MANAGEMENT SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MNAS'],'NATURAL AND AGRICULTURAL SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MTHL'],'THEOLOGY')
    df['FACULTY'] = df['FACULTY'].replace(['MEDU'],'EDUCATION')
    df['FINAL.MARK'] = df['FINAL.MARK'].str.replace(',','')
    df['FINAL.MARK'] = df['FINAL.MARK'].astype("int")
    faclty = df.loc[lambda df: (df['FACULTY'] == fac) & (df['Term'] == term)] # data based on the selected faculty & semester
    modCode = faclty.loc[lambda faclty: faclty['Campus'] == camp] # data based on module code
    
    Groupa, Groupb = modCode.loc[lambda modCode: modCode['freq'] > 0], modCode.loc[lambda modCode: modCode['freq'] == 0] #Selecting Ttest Analysis GroupA&B
    GroupA = Groupa.loc[lambda Groupa: Groupa['FINAL.MARK'] > 0]
    GroupB = Groupb.loc[lambda Groupb: Groupb['FINAL.MARK'] > 0]
    na, nb = len(GroupA['freq']), len(GroupB['freq'])
    ntot = na + nb
    avg1, avg2 = round(np.average(GroupA['FINAL.MARK']), 2), round(np.average(GroupB['FINAL.MARK']), 2)
    s_1 = 'Out of a total number of {} students enrolled in the faculty of '
    Ttest_in_string = (s_1 + fac + ' on the '+camp+' Campus, {} attended at least one A_Step SI tutorial and scored an average of {} % on their final mark,  while {} did not attend any of the tutorials and obtained an average of {} % on their final mark.').format(ntot, na, avg1, nb, avg2)         
    
    # Selecting ANOVA Analysis GroupA&B&C 
    grpa = modCode.loc[lambda modCode: modCode['freq'] >= 5]
    grpb = modCode.loc[lambda modCode: (modCode['freq'] > 0) & (modCode['freq'] <= 4)]
    grpA = grpa.loc[lambda grpa: grpa['FINAL.MARK'] > 0]
    grpB = grpb.loc[lambda grpb: grpb['FINAL.MARK'] > 0]
    grpC = GroupB
    nA, nB, nC = len(grpA['freq']), len(grpB['freq']), len(grpC['freq'])
    avg3, avg4 = round(np.average(grpA['FINAL.MARK']), 2), round(np.average(grpB['FINAL.MARK']), 2)
    Term = term.lower()
    ANOVA_in_string = ('Furthermore, {} students attended at least 5 of the offered tutorials (Group A) and reached an average of {} % on their final mark, while {} students attended between 1-4 tutorials (Group B) and scored an average {} %. Finally the {} students that did not attend any of the tutorials are labelled (Group C). Refer to plots above for visualization of the data distributions.').format(nA, avg3, nB, avg4, nC)
    Intro = ('This is a CTL report on the impact of the Academic Student and Excellence Tutorial Program (A_STEP) focusing on the faculty of _'+fac+'_ \
,  during the semester _'+Term+'_.  This report is based on the quantitative data analysis carried on the \
samples of modules in the faculty that formed part of the A_STEP. The CTL A_STEP program/initiative is based on the principles of Supplemental \
Instruction  (SI),  where it is the high risk modules that are identified,  as opposed to high risk students.  In essence,  when students learn \
collaboratively in high quality tutorials which are facilitated by well trained tutors they are more likely to master course material \
and be successful. The CTL & A_STEP encourages regular attendance of tutorials,  and believes that continued participation \
over time plays an important role for student success. Towards the goal of seeing continued improvement and critical self - reflection, \
 the A_STEP examines its own impact on student academic performances and evaluates tutors on a semester-by-semester basis.  Between the years 2019 - 2022, there were seven terms, four being term 1 and three being term 2.').format()    
    print('\n',Ttest_in_string)
    print('\n',ANOVA_in_string)

    return [GroupA, GroupB, grpA, grpB, grpC, Ttest_in_string, ANOVA_in_string, Intro, modCode]
              
def dataSelector1(fac, term, mod, camp):

    x = 'complete_data.RData'
    df = data_read(x)
    df['FACULTY'] = df['FACULTY'].replace(['MHSC'],'HEALTH SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MLAW'],'LAW')
    df['FACULTY'] = df['FACULTY'].replace(['MHUM'],'HUMANITIES')
    df['FACULTY'] = df['FACULTY'].replace(['MEMS'],'ECONOMIC AND MANAGEMENT SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MNAS'],'NATURAL AND AGRICULTURAL SCIENCES')
    df['FACULTY'] = df['FACULTY'].replace(['MTHL'],'THEOLOGY')
    df['FACULTY'] = df['FACULTY'].replace(['MEDU'],'EDUCATION')
    df['FINAL.MARK'] = df['FINAL.MARK'].str.replace(',','')
    df['FINAL.MARK'] = df['FINAL.MARK'].astype("int")
    df = df.loc[lambda df: df['FINAL.MARK'] > 0]
    
    faclty = df.loc[lambda df: (df['FACULTY'] == fac) & (df['Term'] == term)] # data based on the selected faculty & semester
    modCode = faclty.loc[lambda faclty: (faclty['Module.Code'] == mod) & (faclty['Campus'] == camp)] # data based on module code
    
    GroupA, GroupB = modCode.loc[lambda modCode: modCode['freq'] > 0], modCode.loc[lambda modCode: modCode['freq'] == 0] #Selecting Ttest Analysis GroupA&B
    na, nb = len(GroupA['freq']), len(GroupB['freq'])
    ntot = na + nb
    avg1, avg2 = round(np.average(GroupA['FINAL.MARK']), 2), round(np.average(GroupB['FINAL.MARK']), 2)
    s_1 = 'Out of a total number of {} students enrolled for the module '
    Ttest_in_string = (s_1 + mod + ' on the '+camp+' Campus, {} attended at least one A_Step SI tutorial and scored an average of {} % on their final mark, while {} did not attend any of the tutorials and obtained an average of {} % on their final mark.').format(ntot, na, avg1, nb, avg2)         
    thresh = 0.8*modCode['freq'].max()
    # Selecting ANOVA Analysis GroupA&B&C 
    grpA = modCode.loc[lambda modCode: modCode['freq'] >= 5]
    grpB = modCode.loc[lambda modCode: (modCode['freq'] > 0) & (modCode['freq'] <= 4)]
    grpC = GroupB
    nA, nB, nC = len(grpA['freq']), len(grpB['freq']), len(grpC['freq'])
    avg3, avg4 = round(np.average(grpA['FINAL.MARK']), 2), round(np.average(grpB['FINAL.MARK']), 2)

    ANOVA_in_string = ('Furthermore,  {} students attended at least 5 of the offered tutorials (Group A) and reached an average of {} % on their final mark,  while {} students attended between 1-4 tutorials (Group B) and scored an average {} %. Finally the {} students that did not attend any of the tutorials are labelled (Group C). Refer to plots above for visualization of the data distributions.').format(nA, avg3, nB, avg4, nC)
       
    print('\n',Ttest_in_string)
    print('\n',ANOVA_in_string)

    return [GroupA, GroupB, grpA, grpB, grpC, Ttest_in_string, ANOVA_in_string]
          
    
    
    
    
    
    
    
    
    
    
    
    




