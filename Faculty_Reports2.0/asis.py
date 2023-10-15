#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyreadr
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import base64
from datetime import datetime
import time
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

import scipy.stats as stats
from scipy.stats import pearsonr
from numpy.random import randn
from numpy.random import seed
from datetime import date
from fpdf import FPDF
import dataGrab
import Ttest
import OneWay
import Posthoc
import Correlate
import os
import glob
# data visualization packages
import seaborn as sbn
import matplotlib.pyplot as plt
import plotly.express as px



names = ['Tekano Mbonani', 'Evodia Mohoanyane']
usernames = ['tmbonani', 'emohoanyane']

file_path = Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
        }            
    }
}

with open('style1.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    authenticator = stauth.Authenticate(credentials,
                                    'tutorial_report', 'asteptuts', cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('A-STEP Impact on Student Success (ASIS) Reporter 📖', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
if authentication_status == None:
    st.warning('Please enter your username and password')
if authentication_status:
    with st.spinner('Wait for it...'):
        time.sleep(3)

    dat = 'complete_data.RData'
    @st.cache_data
    def data_read():
        r_dt = pyreadr.read_r(dat)
        df = r_dt['GroupedData']
        df['FINAL.MARK'] = df['FINAL.MARK'].str.replace(',','')
        df['FINAL.MARK'] = df['FINAL.MARK'].astype("float64")
        df['FACULTY'] = df['FACULTY'].replace(['MHSC'],'HEALTH SCIENCES')
        df['FACULTY'] = df['FACULTY'].replace(['MLAW'],'LAW')
        df['FACULTY'] = df['FACULTY'].replace(['MHUM'],'HUMANITIES')
        df['FACULTY'] = df['FACULTY'].replace(['MEMS'],'ECONOMIC AND MANAGEMENT SCIENCES')
        df['FACULTY'] = df['FACULTY'].replace(['MNAS'],'NATURAL AND AGRICULTURAL SCIENCES')
        df['FACULTY'] = df['FACULTY'].replace(['MTHL'],'THEOLOGY')
        df['FACULTY'] = df['FACULTY'].replace(['MEDU'],'EDUCATION')
        df = df.loc[lambda df: df['FINAL.MARK'] > 0]
        return df
    df = data_read()

    st.markdown("<h3 style='text-align: center; color: darkred;'>Academic Student Tutorial Excellence Programme</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.25, 0.5, 0.25], gap='small')

    with col1:
        st.write(' ')

    with col2:
        st.markdown("![Alt Text](https://i.postimg.cc/dtqz6njz/log.png)")

    with col3:
        st.write(' ')
    Time = {datetime.now().strftime('%Y-%m-%d')}
    st.markdown("<h2 style='text-align: center; color: darkred;'>A-STEP ASIS User</h2>", unsafe_allow_html=True)
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.columns(3)[1].title(f'**Welcome**'+'\n'+f'**{name}**')
    empt, ufslog = st.sidebar.columns([0.10, 0.90], gap='small')
    with empt:
        st.write(' ')
    with ufslog:
        st.markdown("![Alt Text](https://i.postimg.cc/gJzPdRYd/logio.png)")
    
    st.sidebar.markdown("<h2 style='text-align: center; color: darkred;'>About A-STEP</h2>", unsafe_allow_html=True)

    with open('style2.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    exp = st.sidebar.expander(':blue[Read More:]')
    exp.write('The Academic Student Tutorial and Excellence Programme (A-STEP) provides both face-to-face and blended tutorials for students.\
                 These tutorials are led by trained senior students across all seven faculties on the Bloemfontein campus as well as the four faculties on QwaQwa campus.\
                 The tutors themselves are either under- or postgraduate students, thereby making communication easier between the relevant parties. A-STEP sessions\
                 offer regular, peer-facilitated sessions that occur out of class and after lectures, thereby integrating content with learning skills and study strategies.\
                 The work covered and facilitated in the tutorial sessions is therefore embedded within the context of a particular discipline, which is dependent on the faculty.')

    st.sidebar.markdown("<h2 style='text-align: center; color: darkred;'>Web App Instructions</h2>", unsafe_allow_html=True)

    st.sidebar.markdown('- Select Faculty')
    st.sidebar.markdown("- Select Campus")
    st.sidebar.markdown("- Select Term")
    st.sidebar.markdown("- Generate Report")
    st.sidebar.markdown("- Download PDF Report")

    st.sidebar.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)


    st.sidebar.markdown("<h2 style='text-align: center; color: darkred;'>How the Web App Works</h2>", unsafe_allow_html=True)
    works = st.sidebar.expander(':blue[Read More:]')
    works.write('The Web Application makes use of the A-STEP tutorial attendance data, along with student final exam performances to study relationships\
                 between tutorial attendance frequency and student academic success, through hypothesis testing and Linear Regression studies.\
                 Students are divided and grouped according to their A-STEP attendance frequency, and average final marks of the different groups\
                 are compared to find if the differences are statitiscally significant, and therefore, the impact of A-STEP on its attendees.')
    
    st.sidebar.markdown("<h2 style='text-align: center; color: darkred;'>How To Contact Us</h2>", unsafe_allow_html=True)
    st.sidebar.write('A-STEP Call Centre')
    st.sidebar.write('T: +27 51 401 9111 (Option 4 and then Option 2)')
    st.sidebar.write('E: AStep@ufs.ac.za')
    st.sidebar.write('A-STEP Data Analytics')
    st.sidebar.write('E: Mbonanits@ufs.ac.za')
    st.write('Welcome to the A-STEP Impact Report Web Application. An app developed for the automation of data and statistical\
         analysis of the tutorial attendance and student performance data, for the purpose of investigating potential quantitative relations\
         between student tutorial attendance and performance. Additionally, the app automatically compiles a report (PDF) of the research,\
         highlighting the impact of A-STEP on its students\' academic performance. Here is a preview of the average final marks of A-STEP students from the data currently loaded:')
    st.sidebar.info('Web App Developer: Tekano Mbonani', icon="ℹ️")
    #st.write(df.head())
    chart_dat = df[['ACAD_YEAR', 'FINAL.MARK']]
     
    Fmark = []
    for i in df['ACAD_YEAR'].unique():
        avg = df.loc[lambda df: df['ACAD_YEAR'] == i]['FINAL.MARK'].mean()
        Fmark.append(avg)
        
    dt = {'Avg (%)': Fmark,
      'Year': [2022, 2019, 2020, 2021]}
    dt = pd.DataFrame.from_dict(dt)
    with st.expander(':blue[Data Preview:]'):
        st.bar_chart(dt,
             x= 'Year',
             y = 'Avg (%)')

    plt.bar(dt['Year'], dt['Avg (%)'], color ='maroon',
        width = 0.4)
    plt.savefig('png.png')
    with open('style1.css') as fl:
        st.markdown(f'<style>{fl.read()}</style>', unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns(3, gap='small')
    # Selecting the faculty     
    with col1:
        acad_grp = df['FACULTY'].unique()
        fac = st.selectbox(
                    ':red[**Select the academic faculty:**] 👇🏿',
                     acad_grp)

    # Select the campus
    with col2:
        acad_camp = df['Campus'].unique()
        camp = st.selectbox(
                    ':red[**Select the academic campus:**] 👇🏿',
                    acad_camp)

    # Select the semester
    with col3:
        acad_term = df['Term'].unique()
        term = st.selectbox(
                    ':red[**Select the academic term:**] 👇🏿',
                     acad_term)
        
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('')
    with col3:
        st.write('')
    with col2:
        st.write(":blue[**User selections**:] :mag_right:")
        st.markdown('- :rainbow[**Faculty** :] '+str(fac))
        st.markdown("- :rainbow[**Campus** : ] "+str(camp))
        st.markdown("- :rainbow[**Term** : ] "+str(term))
        st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
                }
                </style>
                ''', unsafe_allow_html=True)

                
    if st.button('Generate Report 🚀'):
        style = "<style>h2 {text-align: center;}</style>"
        st.markdown(style, unsafe_allow_html=True)

        # save FPDF() class into a
        # variable pdf
        pdf = FPDF()
        pdf.page_no()
        # Add a page
        pdf.add_page()

        # set style and size of font
        # that you want in the pdf
        pdf.set_font("Times", 'B', size = 13)
        pdf.image('ctl-10yr_rgb-01.jpeg', x = 70, y = 10, w = 80, h = 40, type = 'PNG')
        pdf.cell(0, 5, txt = '', ln =1)
        pdf.cell(0, 5, txt = '', ln =1)
        pdf.cell(0, 5, txt = '', ln =1)
        pdf.cell(0, 5, txt = '', ln =1)
        pdf.cell(0, 5, txt = '', ln =2)
        pdf.cell(0, 5, txt = '', ln =4)
        pdf.cell(0, 5, txt = '', ln =5)
        pdf.cell(0, 5, txt = '', ln =6)
        pdf.cell(0, 5, txt = '', ln =7)
        pdf.cell(0, 5, txt = 'ASIS Impact Report: 2019-2022', align = 'C', ln=8)
        pdf.cell(0, 5, txt = '', ln =9)
        pdf.cell(0, 5, txt = 'CTL ASIS User', align = 'C', ln=10)
        pdf.cell(0, 5, txt = '', ln =11)
        today = date.today()
        print('Today\'s date:', today)
        pdf.cell(0, 5, txt = str(today), ln =12, align = 'C')
        pdf.cell(0, 5, txt = ' ', ln =13, align = 'C') 
        pdf.ln(0.25)
        pdf.set_font('Arial','B',10.0)
        pdf.cell(0, 5, txt = '1. Executive Summary.', ln=14, align='L')
        pdf.cell(0, 5, txt = '', ln =15, align = 'C')
        pdf.ln(0.25)
        pdf.set_font('Arial','',10.0)
        ss = ('In the majority of the modules, students who attended tutorials performed significantly better than students \
who did not. There is some evidence to suggest that students who attend at least  5  of the tutorials perform even better \
than those students who attend between 1-4 tutorials, however, this is not consistent over all modules. In the majority of modules, \
increased tutorial attendance is associated with improved academic performance.').format()
        pdf.multi_cell(0, 5, txt = str(ss), align = 'L', fill = False)
        pdf.cell(0, 5, txt = '', ln =16, align = 'C')
        pdf.ln(0.25)
        pdf.set_font('Arial','B',10.0)
        pdf.cell(0, 5, txt = '2. Introduction.', ln =17, align = 'L')
        pdf.cell(0, 5, txt = '', ln =18, align = 'C')
        pdf.ln(0.25)
        pdf.set_font('Arial','',10.0)
        dt = dataGrab.dataSelector(fac, term, camp)

        GroupA, GroupB, grpA, grpB, grpC, s12, s13, s14, mod_dat = dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dt[6], dt[7], dt[8]
        #GroupA = GroupA.loc[lambda GroupA: GroupA['FINAL.MARK'] > 0]
        n1, n2 = len(GroupA['freq']), len(GroupB['freq'])
        nt = n1 + n2

        un = GroupA['Module.Code'].unique()
        #un
        fig1, ax1 = plt.subplots()
        np.random.seed(1234)
        GroupTtest_final_marks = {('Students who did not' '\n' 'attend any tutorial'): GroupB['FINAL.MARK'], 
                          ('Students who attended''\n''at least one tutorial'): GroupA['FINAL.MARK']}
        Aavg, Bavg = np.average(GroupA['FINAL.MARK']), np.average(GroupB['FINAL.MARK'])
        dF1 = pd.DataFrame.from_dict(data = GroupTtest_final_marks)
        a,b = [1,Bavg],[2, Aavg]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        ax1.plot(x_values, y_values, 'r-', linewidth=1.0)             
        dF1.boxplot(grid=True, showmeans=True)
        ax1.set_ylabel('Final Marks (%)')
        plt.text(x=2.1,y=Aavg,s = '%.2f'%Aavg+'%', fontsize=10)
        plt.text(x=0.65,y=Bavg,s = '%.2f'%Bavg+'%', fontsize=10)
        ax1.set_title('Student Final Marks Per Independent Group.')
        #plt.savefig('ttest_'+fac+'_'+term+'.png')

        fig2, ax2 = plt.subplots()

        GroupANOVA_final_marks = {'Group C:''\n''0 tutorials': grpC['FINAL.MARK'],
                          'Group B:''\n''1-4 tutorials': grpB['FINAL.MARK'], 
                          'Group A:''\n''5 tutorials': grpA['FINAL.MARK']}
        grpA_avg, grpB_avg, grpC_avg = np.average(grpA['FINAL.MARK']), np.average(grpB['FINAL.MARK']), np.average(grpC['FINAL.MARK'])
        dF = pd.DataFrame.from_dict(data = GroupANOVA_final_marks)
        A,B,C = [1,grpC_avg],[2, grpB_avg],[3, grpA_avg]
        ABx_values = [A[0], B[0]]
        ABy_values = [A[1], B[1]]
        BCx_values = [B[0], C[0]]
        BCy_values = [B[1], C[1]]
        ax2.plot(ABx_values, ABy_values, 'r-', linewidth=1.0)
        ax2.plot(BCx_values, BCy_values, 'r-', linewidth=1.0) 
        dF.boxplot(grid=True, showmeans=True)
        ax2.set_ylabel('Final Marks (%)')
        plt.text(x=3.2,y=grpA_avg,s = '%.2f'%grpA_avg+'%', fontsize=10)
        plt.text(x=2.2,y=grpB_avg,s = '%.2f'%grpB_avg+'%', fontsize=10)
        plt.text(x=1.2,y=grpC_avg,s = '%.2f'%grpC_avg+'%', fontsize=10)
        ax2.set_title('Student Final Marks Per Independent Group.')
        #plt.savefig('anova_'+fac+'_'+term+'.png')


        ## line & bar plots
        fig, ax = plt.subplots()
        sa = {('Students who did not''\n''attend any tutorial'): Bavg,
    ('Students who attended''\n''at least one tutorial'): Aavg}
        a,b = [0,Bavg],[1, Aavg]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        ax.plot(x_values, y_values, 'k-', linewidth=1.0)
        ax.scatter(x_values, y_values,  marker = 's', s = 100.0, c = 'k')
        ax.bar(sa.keys(), sa.values(), color = ['darkblue', 'firebrick'], alpha =1.0, width = 0.35)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Final Marks (%)')
        ax.grid(color='k', linestyle='--', linewidth=.1)
        plt.text(x=1.0,y=Aavg+3,s = '%.2f'%Aavg+'%', fontsize=10)
        plt.text(x=0.0,y=Bavg+3,s = '%.2f'%Bavg+'%', fontsize=10)
        ax.set_title('Student Final Marks Per Independent Group.')
        plt.savefig('ttest_'+fac+'_'+term+'.png')

        ## line & bar plots
        fig3, ax3 = plt.subplots()
        ss = {'Group C:''\n''0 tutorials': grpC_avg,
    'Group B:''\n''1-4 tutorials': grpB_avg, 
    'Group A:''\n''5 tutorials': grpA_avg}
        A,B,C = [0,grpC_avg],[1, grpB_avg],[2, grpA_avg]
        ABx_values = [A[0], B[0]]
        ABy_values = [A[1], B[1]]
        BCx_values = [B[0], C[0]]
        BCy_values = [B[1], C[1]]
        ax3.plot(ABx_values, ABy_values, 'k-', linewidth=1.0)
        ax3.plot(BCx_values, BCy_values, 'k-', linewidth=1.0)
        ax3.scatter(ABx_values, ABy_values,  marker = 's', s = 100.0, c = 'k')
        ax3.scatter(BCx_values, BCy_values, marker = 's', s = 100.0, c = 'k')
        ax3.bar(ss.keys(), ss.values(), color = ['darkblue', 'firebrick', 'grey'], alpha =1.0, width = 0.4)
        plt.text(x=1.9,y=grpA_avg+3,s = '%.2f'%grpA_avg+'%', fontsize=10)
        plt.text(x=0.9,y=grpB_avg+3,s = '%.2f'%grpB_avg+'%', fontsize=10)
        plt.text(x=-0.1,y=grpC_avg+3,s = '%.2f'%grpC_avg+'%', fontsize=10)
        ax3.grid(color='k', linestyle='--', linewidth=.1)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Final Marks (%)')
        ax3.set_title('Student Final Marks Per Independent Group.')
        plt.savefig('anova_'+fac+'_'+term+'.png')

        pdf.multi_cell(0, 5, txt = str(s14), align = 'L')
        pdf.cell(0, 5, txt = '', align = 'C', ln=13)
        pdf.ln(0.25)
        pdf.set_font('Arial','B',10.0)
        pdf.cell(0, 5, txt = '3. Understanding tutorial attendance in relation to academic performance.', ln =14, align = 'L')
        pdf.cell(0, 5, txt = '', ln =15, align = 'C')
        pdf.ln(0.25)
        pdf.set_font('Arial','',10.0)
        sa = ('The following section presents the findings from the analysis of tutorial attendance in relation to academic performance \
for each module in the faculty. Data between 2019 - 2022 obtained from attendance lists and academic performance is analysed to determine \
whether students who attend tutorials perform better academically than the students who do not attend tutorials,  and whether attending at least 5  tutorials makes a difference to academic performance.  Furthermore, the relationship between \
academic performance and tutorial attendance is investigated. Finally, the relationship between tutorial attendance and \
academic performance is investigated within the top performing academic group and the lowest performing academic \
group.').format()
        pdf.multi_cell(0, 5, txt = str(sa), align = 'L')
        pdf.cell(0, 5, txt = '', ln =16, align = 'C')
        pdf.multi_cell(0, 5, txt = 'For each faculty a summary of results is presented first to allow for an overall understanding of tutorial attendance in relation to academic performance. Thereafter, a section entitled Evidence is presented which details all of the results from the statistical analysis.', align = 'L')
        pdf.cell(0, 5, txt = '', ln =17, align = 'C')
        pdf.image('kindpng_2007834.jpeg', x = 10, y = 255, w = 187, h = 5, type = 'PNG')
        pdf.cell(0, 5, txt = '', ln =17, align = 'C')
        pdf.cell(0, 5, txt = '', ln =17, align = 'C')
        #pdf.cell(0, 5, txt = '', ln =17, align = 'C')
        #pdf.cell(0, 5, txt = '', ln =17, align = 'C')
        pdf.cell(0, 5, txt = '1', ln =17, align = 'C')

        # running the t-test analysis
        X, Y = GroupA['FINAL.MARK'], GroupB['FINAL.MARK']

        t_analysis = Ttest.T_test(X,Y)

        try:
            t, p_test = t_analysis[0], t_analysis[1]
            #break
            #if np.isnan(t) == True:
            #t = 0.2
            #p_test = 0.15
        except (TypeError, ValueError):
            print('x')
            #continue

        #ANOVA
        anova = OneWay.anova(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'])
        l = [len(GroupA['FINAL.MARK']), len(GroupB['FINAL.MARK'])]
        if l[0] > 0 and l[1] > 0:
            d = Ttest.cohend(GroupA['FINAL.MARK'], GroupB['FINAL.MARK'])
            print('Cohens d: %.3f' % d)
        elif l[0] == 0 or l[1] == 0:
            print('One of Group A and/or B is missing data.')

        #The determination coefficient between student attendance and final mark...
        att, perf = np.array([GroupA['freq']]) , np.array([GroupA['FINAL.MARK']])
        x, y = att.reshape(-1,1), perf.reshape(-1,1)
        res = OneWay.Regression(x,y)
        R2 = res
        R2_perc = R2/100

        #corr_dat = {'FINAL.MARK': GroupA['FINAL.MARK'], 'freq': GroupA['freq'], 'GR_12_ADSCORE':pd.to_numeric(GroupA['GR_12_ADSCORE'])}

        corr_dat = {'FINAL.MARK': GroupA['FINAL.MARK'], 'freq': GroupA['freq']}

        GroupAtt = pd.DataFrame.from_dict(data = corr_dat)

        f, p, eta2= anova[0], anova[1], anova[2]
        if p < 0.05:
    
            Post_Hoc = Posthoc.Post_hoc(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'], GroupAtt, f, p)
            ppost, A, B  = Post_Hoc['pval'], Post_Hoc['A'], Post_Hoc['B']

            c = ppost.dropna()
            cavg = np.average(c)
            cc = Post_Hoc.loc[lambda Post_Hoc: ppost < 0.05]
        elif p >= 0.05:
            print(' The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')

        try:
            coR, _ = pearsonr(GroupA['freq'], GroupA['FINAL.MARK'])
            corr = coR.round(2)
            print('Pearsons correlation: %.3f' % corr)
            r = Correlate.Corr(GroupAtt)
            pval = float(r[0]['p-val'])
            print(pval)
        except (ValueError, AttributeError):
            print('x')
            #lnXcr = 'No'
            
        def tabfill(x, y, z, e, t):
            if x <= 0.05:
                if 0 < y <= 0.2:
                    ttXst = 'Yes'
                elif 0.2 < y <= 0.5:
                    ttXst = 'Yes'
                elif 0.5 < y < 0.8:
                    ttXst = 'Yes*'
                elif y >= 0.8:
                    ttXst = 'Yes*'                        
                elif y <= 0:
                    ttXst = 'No'
                if np.isnan(y) == True:
                    ttXst = 'No'
            elif x > 0.05:
                if 0 < y <= 0.2:
                    ttXst = 'No'
                elif 0.2 < y <= 0.5:
                    ttXst = 'No'
                elif 0.5 < y < 0.8:
                    ttXst = 'No*'
                elif y >= 0.8:
                    ttXst = 'No*'
                elif y <= 0:
                    ttXst = 'No'
                if np.isnan(y) == True:
                    ttXst = 'No'
            if np.isnan(x) == True:
                ttXst = 'No'
    
            if z <= 0.05:
                if 0 < e <= 0.2:
                    anXva = 'Yes'
                elif 0.2 < e <= 0.5:
                    anXva = 'Yes'
                elif 0.5 < e < 0.8:
                    anXva = 'Yes*'
                elif e >= 0.8:
                    anXva = 'Yes*'
                elif e <= 0:
                    anXva = 'No'
                if np.isnan(e) == True:
                    anXva = 'Yes'
            elif z > 0.05:
                if e <= 0.2:
                    anXva = 'No'
                elif 0.2 < e <= 0.5:
                    anXva = 'No'
                elif 0.5 < e < 0.8:
                    anXva = 'No*'
                elif e >= 0.8:
                    anXva = 'No*'
                elif e <= 0:
                    anXva = 'No'
                if np.isnan(e) == True:
                    anXva = 'No'
            
            if np.isnan(z) == True:
                anXva = 'No'
        
            if t <= 0.05:
                lnXcr = 'Yes'
            elif t > 0.05:
                lnXcr = 'No'
        
            if np.isnan(t) == True:
                lnXcr = 'No'
        
            return [ttXst, anXva, lnXcr]
        k = []
        ana = []
        for j, i in enumerate(un):
    
            dt = dataGrab.dataSelector1(fac, term, i, camp)

            GroupA, GroupB, grpA, grpB, grpC, s1, s2 = dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dt[6]
    
    
            # running the t-test analysis
            X, Y = GroupA['FINAL.MARK'], GroupB['FINAL.MARK']

            t_analysis = Ttest.T_test(X,Y)
            try:
                t1, p_test1 = t_analysis[0], t_analysis[1]
            except (TypeError, ValueError):
                p_test1 = np.nan
                continue
            l = [len(grpA['FINAL.MARK']), len(grpB['FINAL.MARK']), len(grpC['FINAL.MARK'])]
            Av = [np.average(grpA['FINAL.MARK']), np.average(grpB['FINAL.MARK']), np.average(grpC['FINAL.MARK'])]
    
            try:        
                d1 = Ttest.cohend(GroupA['FINAL.MARK'], GroupB['FINAL.MARK'])
                print('Cohens d: %.3f' % d1)
            except(ZeroDivisionError):
                d1 = np.nan
                continue
            anova1 = OneWay.anova(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'])   # if Av[0] and Av[1] and Av[2] > 1:
            try:                
                f1, p1, eta2_2 = anova1[0], anova1[1], anova1[2]
            except F_onewayBadInputSizesWarning:
                f1, p1, eta_2 = np.nan, np.nan, np.nan
                continue

    
            att, perf = np.array([GroupA['freq']]) , np.array([GroupA['FINAL.MARK']])
            x, y = att.reshape(-1,1), perf.reshape(-1,1)
            try:
                res1 = OneWay.Regression(x,y)
                R22 = res1
                R2_perc2 = R22/100
            except ValueError:
                R22 = np.nan
                R2_perc2 = np.nan
                continue
            corr_dat = {'FINAL.MARK': GroupA['FINAL.MARK'], 'freq': GroupA['freq']}

            GroupAtt = pd.DataFrame.from_dict(data = corr_dat)

    
            if p1 < 0.05:
    
                Post_Hoc = Posthoc.Post_hoc(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'], GroupAtt, f1, p1)
                ppost, A, B  = Post_Hoc['pval'], Post_Hoc['A'], Post_Hoc['B']

                c = ppost.dropna()
                cavg = np.average(c)
                cc = Post_Hoc.loc[lambda Post_Hoc: ppost < 0.05]
            elif p1 >= 0.05:
                #cc = 0
                print(' The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')    
    
            try:
                coR, _ = pearsonr(GroupA['freq'], GroupA['FINAL.MARK'])
                corr1 = coR.round(2)
                print('Pearsons correlation: %.3f' % corr)
                r = Correlate.Corr(GroupAtt)
                pval1 = float(r[0]['p-val'])
                print(pval1)
            except (ValueError, AttributeError):
                pval1 = np.nan
                #print(pval1)
                continue
        
            dec = tabfill(p_test1, d1, p1, eta2_2, pval1)
            stat_val = [i, t1, p_test1, d1, f1, p1, eta2_2, R22, R2_perc2, corr1, pval1]
            k.append(dec)
            ana.append(stat_val)
        pdf.add_page()
        pdf.set_font("Arial", 'B', size = 9)
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = 'Table 1. A summary of primary findings from an analysis of a sample of '+fac+' modules. ', ln =22, align = 'L')
        pdf.ln(0.25)
        #pdf.set_font("Arial", size = 10)
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.set_font("Arial", 'B', size = 10)
        line_height = pdf.font_size * 2.5
        epw = pdf.w - 2*pdf.l_margin
        th = pdf.font_size
        col_width = epw / 4
        size = len(ana)
        dec = tabfill(p_test, d, p, eta2, pval)
        ttXst, anXva, lnXcr = dec[0], dec[1], dec[2]
        tab = {'Modules':[1, 1, 1, 1],'Do students who attend tutorials perfom significantly better than students who do not?' : [1, 1, 1, 1],
     'Do students who attend at least 5 tutorials perfom signifiicantly better than students who attend between 1-4 AND students that do not attend?' : [1, 1, 1, 1],
     'Is there a significant positive correlation between tutorial attendance and perfomance?' : [1, 1, 1, 1]}
        tabular = pd.DataFrame.from_dict(tab)
        col_list = list(tabular.columns)
        pdf.multi_cell(20, 40, str(col_list[0]), 1, 0, 'C')
        pdf.set_xy(30, 25)
        pdf.multi_cell(50, 10, str(col_list[1]), 1, 0, 'C')
        pdf.set_xy(80, 25)
        pdf.multi_cell(70, 10, str(col_list[2]), 1, 0, 'C')
        pdf.set_xy(150, 25)
        pdf.multi_cell(50, 10, str(col_list[3]), 1, 0, 'C')
        pdf.cell(20, 15, 'Faculty', border = 1, ln= 7, align = 'C')
        pdf.ln(0.25) 
        pdf.set_font("Arial", '', size = 10)
        pdf.set_text_color(255, 0, 0)
        pdf.set_xy(30, 65)
        pdf.cell(50, 15, txt = str(ttXst), border = 1, ln=7, align = 'C')
        pdf.set_xy(80, 65)
        pdf.cell(70, 15, txt = str(anXva), border = 1, ln=7, align = 'C')
        pdf.set_xy(150, 65)
        pdf.cell(50, 15, txt = str(lnXcr), border = 1, ln=7, align = 'C')
        pdf.set_text_color(0, 0, 0)
        if size == 1:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')   
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
        elif size == 2:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')

            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
        elif size == 3:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')    

            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
        elif size == 4:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')    

            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
        elif size == 5:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')

            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C')
        elif size == 6:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
    
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
        elif size == 7:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[6][0], border = 1, ln= 9, align = 'C')  
    
    
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 170)
            pdf.cell(50, 15, txt = str(k[6][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 170)
            pdf.cell(70, 15, txt = str(k[6][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 170)
            pdf.cell(50, 15, txt = str(k[6][2]), border = 1, ln=7, align = 'C')
        elif size == 8:
    
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[6][0], border = 1, ln= 9, align = 'C')  
            pdf.cell(20, 15, ana[7][0], border = 1, ln= 9, align = 'C')
    
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 170)
            pdf.cell(50, 15, txt = str(k[6][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 170)
            pdf.cell(70, 15, txt = str(k[6][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 170)
            pdf.cell(50, 15, txt = str(k[6][2]), border = 1, ln=7, align = 'C')  

            pdf.set_xy(30, 185)
            pdf.cell(50, 15, txt = str(k[7][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 185)
            pdf.cell(70, 15, txt = str(k[7][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 185)
            pdf.cell(50, 15, txt = str(k[7][2]), border = 1, ln=7, align = 'C')
        elif size == 9:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[6][0], border = 1, ln= 9, align = 'C')  
            pdf.cell(20, 15, ana[7][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[8][0], border = 1, ln= 9, align = 'C')
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 170)
            pdf.cell(50, 15, txt = str(k[6][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 170)
            pdf.cell(70, 15, txt = str(k[6][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 170)
            pdf.cell(50, 15, txt = str(k[6][2]), border = 1, ln=7, align = 'C')  

            pdf.set_xy(30, 185)
            pdf.cell(50, 15, txt = str(k[7][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 185)
            pdf.cell(70, 15, txt = str(k[7][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 185)
            pdf.cell(50, 15, txt = str(k[7][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 200)
            pdf.cell(50, 15, txt = str(k[8][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 200)
            pdf.cell(70, 15, txt = str(k[8][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 200)
            pdf.cell(50, 15, txt = str(k[8][2]), border = 1, ln=7, align = 'C')
        elif size == 10:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[6][0], border = 1, ln= 9, align = 'C')  
            pdf.cell(20, 15, ana[7][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[8][0], border = 1, ln= 9, align = 'C') 
            pdf.cell(20, 15, ana[9][0], border = 1, ln= 9, align = 'C')

            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 170)
            pdf.cell(50, 15, txt = str(k[6][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 170)
            pdf.cell(70, 15, txt = str(k[6][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 170)
            pdf.cell(50, 15, txt = str(k[6][2]), border = 1, ln=7, align = 'C')
            pdf.set_xy(30, 185)
            pdf.cell(50, 15, txt = str(k[7][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 185)
            pdf.cell(70, 15, txt = str(k[7][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 185)
            pdf.cell(50, 15, txt = str(k[7][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 200)
            pdf.cell(50, 15, txt = str(k[8][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 200)
            pdf.cell(70, 15, txt = str(k[8][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 200)
            pdf.cell(50, 15, txt = str(k[8][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 215)
            pdf.cell(50, 15, txt = str(k[9][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 215)
            pdf.cell(70, 15, txt = str(k[9][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 215)
            pdf.cell(50, 15, txt = str(k[9][2]), border = 1, ln=7, align = 'C')
        elif size >= 11:
            pdf.ln(0.25) 
            pdf.set_font("Arial", 'B', size = 10)
            pdf.cell(20, 15, ana[0][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[1][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[2][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[3][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[4][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[5][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[6][0], border = 1, ln= 9, align = 'C')  
            pdf.cell(20, 15, ana[7][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[8][0], border = 1, ln= 9, align = 'C') 
            pdf.cell(20, 15, ana[9][0], border = 1, ln= 9, align = 'C')
            pdf.cell(20, 15, ana[10][0], border = 1, ln= 9, align = 'C')
            pdf.ln(0.25) 
            pdf.set_font("Arial", '', size = 10)
            pdf.set_text_color(0, 0, 255)
            pdf.set_xy(30, 80)
            pdf.cell(50, 15, txt = str(k[0][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 80)
            pdf.cell(70, 15, txt = str(k[0][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 80)
            pdf.cell(50, 15, txt = str(k[0][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 95)
            pdf.cell(50, 15, txt = str(k[1][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 95)
            pdf.cell(70, 15, txt = str(k[1][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 95)
            pdf.cell(50, 15, txt = str(k[1][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 110)
            pdf.cell(50, 15, txt = str(k[2][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 110)
            pdf.cell(70, 15, txt = str(k[2][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 110)
            pdf.cell(50, 15, txt = str(k[2][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 125)
            pdf.cell(50, 15, txt = str(k[3][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 125)
            pdf.cell(70, 15, txt = str(k[3][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 125)
            pdf.cell(50, 15, txt = str(k[3][2]), border = 1, ln=7, align = 'C')
            pdf.set_xy(30, 140)
            pdf.cell(50, 15, txt = str(k[4][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 140)
            pdf.cell(70, 15, txt = str(k[4][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 140)
            pdf.cell(50, 15, txt = str(k[4][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 155)
            pdf.cell(50, 15, txt = str(k[5][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 155)
            pdf.cell(70, 15, txt = str(k[5][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 155)
            pdf.cell(50, 15, txt = str(k[5][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 170)
            pdf.cell(50, 15, txt = str(k[6][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 170)
            pdf.cell(70, 15, txt = str(k[6][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 170)
            pdf.cell(50, 15, txt = str(k[6][2]), border = 1, ln=7, align = 'C')  

            pdf.set_xy(30, 185)
            pdf.cell(50, 15, txt = str(k[7][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 185)
            pdf.cell(70, 15, txt = str(k[7][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 185)
            pdf.cell(50, 15, txt = str(k[7][2]), border = 1, ln=7, align = 'C')
    
            pdf.set_xy(30, 200)
            pdf.cell(50, 15, txt = str(k[8][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 200)
            pdf.cell(70, 15, txt = str(k[8][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 200)
            pdf.cell(50, 15, txt = str(k[8][2]), border = 1, ln=7, align = 'C') 
    
            pdf.set_xy(30, 215)
            pdf.cell(50, 15, txt = str(k[9][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 215)
            pdf.cell(70, 15, txt = str(k[9][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 215)
            pdf.cell(50, 15, txt = str(k[9][2]), border = 1, ln=7, align = 'C')
            pdf.set_xy(30, 230)
            pdf.cell(50, 15, txt = str(k[10][0]), border = 1, ln=7, align = 'C')
            pdf.set_xy(80, 230)
            pdf.cell(70, 15, txt = str(k[10][1]), border = 1, ln=7, align = 'C')
            pdf.set_xy(150, 230)
            pdf.cell(50, 15, txt = str(k[10][2]), border = 1, ln=7, align = 'C')
        
        pdf.set_text_color(0, 0, 0)    
        pdf.ln(0.25)
        pdf.cell(0, 5, txt = '', ln =22, align = 'C') 
        pdf.cell(0, 5, txt = 'Note:', ln =22, align = 'L')
        pdf.cell(0, 5, txt = '* Is used to indicate results that have moderate to large effect sizes.:', ln =22, align = 'L')
        pdf.cell(0, 5, txt = '^ Is used to indicate year modules.:', ln =22, align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.ln(0.25)
        #pdf.cell(0, 5, txt = '2', ln = 22, align = 'C')
        pdf.add_page()
        pdf.set_font('Arial','B',10.0)
        pdf.cell(0, 5, txt = 'Faculty of '+' '+str(fac), ln = 21, align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = 'Summary.', align = 'L', ln =23)
        pdf.cell(0, 5, txt = '', ln =24, align = 'C')

        s5 = 'anova_'+fac+'_'+term+'.png'
        s6 = 'ttest_'+fac+'_'+term+'.png'
        pdf.image(str(s6), x = 20, y = 45, w = 80, h = 60, type = 'PNG')
        pdf.image(str(s5), x = 105, y = 45, w = 80, h = 60, type = 'PNG')
        pdf.ln(0.25)
        pdf.set_font("Arial", size = 10)
        siz = len(un)
        if p_test <= 0.05:
            if 0 < d <= 0.2:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was small.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < d <= 0.5:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was medium.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < d < 0.8:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was intermediate.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d >= 0.8:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was large.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')                        
            elif d <= 0:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was very small.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        elif p_test > 0.05:
            if d <= 0.2:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did not perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was small.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < d <= 0.5:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did not perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was medium.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < d < 0.8:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did not perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was intermediate.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d >= 0.8:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did not perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was large.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d <= 0:
                s4 = ('Between 2019 - 2022, there were seven semesters, from the {} modules analysed, we find that students who attended tutorial sessions did perform significantly better in the faculty of '+fac+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was very small.').format(siz)
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')

        if np.isnan(p_test) == True:
            s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was not calculable.').format(siz)
            pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
            pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')    
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')    
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.ln(0.25)
        pdf.set_font("Arial", '', size = 10)
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = '**Evidence**', align = 'L')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', ln =20, align = 'C')
        pdf.cell(0, 5, txt = '', ln =19, align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.cell(-190, 5)
        pdf.multi_cell(0, 5, txt = str(s12 +'  '+ s13), align = 'L')
        pdf.cell(0, 5, txt = '', align = 'C')
        pdf.ln(0.25)
        if p_test <= 0.05:
            T, D =  round(t, 2), round(d, 2)
            if 0 < d <= 0.2:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < d <= 0.5:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a medium effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < d < 0.8:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a intermediate effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d >= 0.8:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associiated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a large effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d <= 0:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}.  The effect size was determined using the Cohen\'s d method [2] and a very small effect was found (d = {}). Since d < 0, it means students in Group B performed significantly better than students in Group A.').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        elif p_test > 0.05:
            T, D =  round(t, 2), round(d, 2)
            if 0 < d <= 0.2:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < d <= 0.5:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < d < 0.8:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a intermediate effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif d >= 0.8:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a large effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            
            elif d <= 0:
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the faculty of '+fac+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none. The effect size was determined using the Cohen\'s d method [2] and a very small effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        if p <= 0.05:
            F, ETA =  round(f, 2), round(eta2, 2)
            R2, R2_perc = round(R2, 2), round(R2_perc, 2)
            if 0 < eta2 <= 0.2:
                s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended betweeen 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < eta2 <= 0.5:
                s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a medium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < eta2 < 0.8:
                s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did nt attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to an intermedium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')

            elif eta2 >= 0.8:
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a large effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        elif p > 0.05:
            F, ETA =  round(f, 2), round(eta2, 2)
            R2, R2_perc = round(R2, 2), round(R2_perc, 2)
            if 0 < eta2 <= 0.2:
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.2 < eta2 <= 0.5:
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a medium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif 0.5 < eta2 < 0.8:
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a intermedium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif eta2 >= 0.8:
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a large effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            
            if np.isnan(p) == True:
                F, ETA =  round(f, 2), round(eta2, 2)
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a very small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        if p < 0.05:
            clen = len(cc['A'])
            clen
            if 1 <= clen <= 2:
                s10 = ('Specifically, the Post hoc analysis using the Games-Howell test [5] indicates a significant difference between students who attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B).').format()
                pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif clen > 2:
                s10 = ('Specifically, the Post hoc analysis using the Games-Howell test [5] indicates a significant difference between students who attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A).').format() 
                pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif clen == 0:
                s10 = ('The Post hoc analysis using the Games-Howell test [5] does not indicate significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A).').format() 
                pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        elif p >= 0.05:
            pdf.multi_cell(0, 5, txt = 'The Null Hypothesis has been accepted i.e, there\'s no significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A). Therefore, the Games Howell Post-hoc analysis was [5] omitted.', align = 'L')
            pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            print(' The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')
        
        if np.isnan(p) == True:
            pdf.multi_cell(0, 5, txt = 'The Null Hypothesis has been accepted i.e, there\'s no significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A). Therefore, the Games Howell Post-hoc analysis [5] was omitted.', align = 'L')
            pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        if pval <= 0.05:
            R2, R2_perc = round(R2, 2), round(R2_perc, 2)
            if 0 < corr <= 0.3:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically significant,  weakly positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
            elif 0.3 < corr <= 0.6:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically significant,  medium positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
            elif 0.6 < corr <= 1:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically significant,  strong positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
            elif corr < 0:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically significant,  weakly negative linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
        elif pval > 0.05:
            R2, R2_perc = round(R2, 2), round(R2_perc, 2)
            s11 = ('There was a statistically insignificant linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
            s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
            pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
        
        if np.isnan(pval) == True:
            R2, R2_perc = round(R2, 2), round(R2_perc, 2)
            s11 = ('There was a statistically insignificant linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = nan, pval = nan).').format()
            s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
            pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
            
        for j, i in enumerate(un):
    
            dt = dataGrab.dataSelector1(fac, term, i, camp)
    
            GroupA, GroupB, grpA, grpB, grpC, s1, s2 = dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dt[6]
    
    
            # running the t-test analysis
            X, Y = GroupA['FINAL.MARK'], GroupB['FINAL.MARK']

            t_analysis = Ttest.T_test(X,Y)
            try:
                t, p_test = t_analysis[0], t_analysis[1]
            except (TypeError, ValueError):
                continue
            l = [len(GroupA['FINAL.MARK']), len(GroupB['FINAL.MARK'])]
            Av = [np.average(GroupA['FINAL.MARK']), np.average(GroupB['FINAL.MARK'])]
    
            if l[0] and l[1] > 1:
                if Av[0] and Av[1] > 0:
                    d = Ttest.cohend(GroupA['FINAL.MARK'], GroupB['FINAL.MARK'])
                    print('Cohens d: %.3f' % d)
                elif Av[0] or Av[1] == 0:
                    print('One of Group A and/or B is missing final marks.')
            elif l[0] or l[1] == 1:
                print('Only a single student attended in Group A and/or B.')
            elif l[0] or l[1] == 0:
                print('One of Group A and/or B is missing data.')
    
            anova = OneWay.anova(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'])

            att, perf = np.array([GroupA['freq']]) , np.array([GroupA['FINAL.MARK']])
            x, y = att.reshape(-1,1), perf.reshape(-1,1)
    
            try:
                res = OneWay.Regression(x,y)
                R22 = res1
                R2_perc = R22/100
            except ValueError:
                R22 = np.nan
                R22_perc2 = np.nan
                continue
        
            corr_dat = {'FINAL.MARK': GroupA['FINAL.MARK'], 'freq': GroupA['freq']}

            GroupAtt = pd.DataFrame.from_dict(data = corr_dat)

            f, p, eta2 = anova[0], anova[1], anova[2]
            if p < 0.05:
    
                Post_Hoc = Posthoc.Post_hoc(grpA['FINAL.MARK'], grpB['FINAL.MARK'], grpC['FINAL.MARK'], GroupAtt, f, p)
                ppost, A, B  = Post_Hoc['pval'], Post_Hoc['A'], Post_Hoc['B']

                c = ppost.dropna()
                cavg = np.average(c)
                cc = Post_Hoc.loc[lambda Post_Hoc: ppost < 0.05]
            elif p >= 0.05:
                print(' The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')    
    
            try:
                coR, _ = pearsonr(GroupA['freq'], GroupA['FINAL.MARK'])
                corr = coR.round(2)
                print('Pearsons correlation: %.3f' % corr)
                r = Correlate.Corr(GroupAtt)
                pval = float(r[0]['p-val'])
                print(pval)
            except (ValueError, AttributeError):
                print('x')
                continue
            fig5, ax5 = plt.subplots()
            np.random.seed(1234)
            GroupTtest_final_marks = {('Students who did not' '\n' 'attend any tutorial'): GroupB['FINAL.MARK'], 
                             ('Students who attended''\n''at least one tutorial'): GroupA['FINAL.MARK']}
            Aavg, Bavg = np.average(GroupA['FINAL.MARK']), np.average(GroupB['FINAL.MARK'])
            dF1 = pd.DataFrame.from_dict(data = GroupTtest_final_marks)
            a,b = [1,Bavg],[2, Aavg]
            x_values = [a[0], b[0]]
            y_values = [a[1], b[1]]
            ax5.plot(x_values, y_values, 'r-', linewidth=1.0)             
            dF1.boxplot(grid=True, showmeans=True)
            ax5.set_ylabel('Final Marks (%)')
            plt.text(x=2.1,y=Aavg,s = '%.2f'%Aavg+'%', fontsize=10)
            plt.text(x=1.1,y=Bavg,s = '%.2f'%Bavg+'%', fontsize=10)
            ax5.set_title('Student Final Marks Per Independent Group Selected From '+i+'.')
            #plt.savefig('ttest_'+fac+'_'+i+'_'+term+'.png')
            plt.close()
            fig6, ax6 = plt.subplots()
            sa = {('Students who did not''\n''attend any tutorial'): Bavg,
    ('Students who attended''\n''at least one tutorial'): Aavg}
            a,b = [0,Bavg],[1, Aavg]
            x_values = [a[0], b[0]]
            y_values = [a[1], b[1]]
            ax6.plot(x_values, y_values, 'k-', linewidth=1.0)
            ax6.scatter(x_values, y_values,  marker = 's', s = 100.0, c = 'k')
            ax6.bar(sa.keys(), sa.values(), color = ['darkblue', 'firebrick'], alpha =1.0, width = 0.35)
            ax6.set_ylim(0, 100)
            ax6.set_ylabel('Final Marks (%)')
            ax6.grid(color='k', linestyle='--', linewidth=.1)
            plt.text(x=1.0,y=Aavg+3,s = '%.2f'%Aavg+'%', fontsize=10)
            plt.text(x=0.0,y=Bavg+3,s = '%.2f'%Bavg+'%', fontsize=10)
            ax6.set_title('Student Final Marks Per Independent Group.')
            plt.savefig('ttest_'+fac+'_'+i+'_'+term+'.png')
            plt.close()
            fig7, ax7 = plt.subplots()

            ss = {'Group C:''\n''0 tutorials': grpC_avg,
    'Group B:''\n''1-4 tutorials': grpB_avg, 
    'Group A:''\n''5 tutorials': grpA_avg}
            A,B,C = [0,grpC_avg],[1, grpB_avg],[2, grpA_avg]
            ABx_values = [A[0], B[0]]
            ABy_values = [A[1], B[1]]
            BCx_values = [B[0], C[0]]
            BCy_values = [B[1], C[1]]
            ax7.plot(ABx_values, ABy_values, 'k-', linewidth=1.0)
            ax7.plot(BCx_values, BCy_values, 'k-', linewidth=1.0)
            ax7.scatter(ABx_values, ABy_values,  marker = 's', s = 100.0, c = 'k')
            ax7.scatter(BCx_values, BCy_values, marker = 's', s = 100.0, c = 'k')
            ax7.bar(ss.keys(), ss.values(), color = ['darkblue', 'firebrick', 'grey'], alpha =1.0, width = 0.4)
            plt.text(x=1.9,y=grpA_avg+3,s = '%.2f'%grpA_avg+'%', fontsize=10)
            plt.text(x=0.9,y=grpB_avg+3,s = '%.2f'%grpB_avg+'%', fontsize=10)
            plt.text(x=-0.1,y=grpC_avg+3,s = '%.2f'%grpC_avg+'%', fontsize=10)
            ax7.grid(color='k', linestyle='--', linewidth=.1)
            ax7.set_ylim(0, 100)
            ax7.set_ylabel('Final Marks (%)')
            ax7.set_title('Student Final Marks Per Independent Group.')
            plt.savefig('anova_'+fac+'_'+i+'_'+term+'.png')
            plt.close()
            pdf.add_page()
            pdf.set_font('Arial','B',10.0)
            pdf.cell(0, 5, txt = str(i), ln = 21, align = 'L')
            pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            pdf.cell(0, 5, txt = 'Summary.', align = 'L', ln =23)
            pdf.cell(0, 5, txt = '', ln =24, align = 'C')
            pdf.set_font("Arial", size = 10)

            if p_test <= 0.05:
                if 0 < d <= 0.2:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was small.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < d <= 0.5:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was medium.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < d < 0.8:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was intermediate.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif d >= 0.8:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was large.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')                        
                elif d <= 0:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was very small.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif p_test > 0.05:
                if 0 < d <= 0.2:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was small.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < d <= 0.5:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was medium.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < d < 0.8:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was intermediate.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif d >= 0.8:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was large.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif d <= 0:
                    s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was very small.').format()
                    pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            if np.isnan(p_test) == True:
                s4 = ('Between 2019 - 2022, there were seven semesters, we find that students who attended tutorial sessions did not perform significantly better in the module '+i+' than students who did \
not attend any tutorial sessions. The practical significance of the difference in the means was not calculable.').format()
                pdf.multi_cell(0, 5, txt = str(s4), align = 'L', fill = False)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
    
            s5 = 'anova_'+fac+'_'+i+'_'+term+'.png'
            s6 = 'ttest_'+fac+'_'+i+'_'+term+'.png'
            pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            pdf.image(str(s6), x = 20, y = 50, w = 80, h = 60, type = 'PNG')
            pdf.image(str(s5), x = 105, y = 50, w = 80, h = 60, type = 'PNG')
            pdf.ln(0.25)
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.ln(0.25)
            pdf.set_font("Arial", '', size = 10)
            pdf.cell(0, 5, txt = '', ln =19, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', ln =19, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', ln =19, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', ln =19, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '**Evidence**', align = 'L')
            pdf.cell(0, 5, txt = '', ln =19, align = 'C')
            pdf.cell(0, 5, txt = '', ln =20, align = 'C')
            pdf.cell(0, 5, txt = '', align = 'C')
            pdf.cell(0, 5, txt = '', align = 'C')
            pdf.cell(-190, 5)
            pdf.multi_cell(0, 5, txt = str(s1 +'  '+ s2), align = 'L')
            if p_test <= 0.05:
                T, D =  round(t, 2), round(d, 2)
                if 0 < d <= 0.2:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < d <= 0.5:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a medium effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < d < 0.8:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a intermediate effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif d >= 0.8:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associiated with p-value equal to {:.3g}. Since p-value is less than 0.05, we reject the null hypothesis i.e, students who attended at least one tutorial session performed significantly better than students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a large effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            
                elif d <= 0:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}.  The effect size was determined using the Cohen\'s d method [2] and a very small effect was found (d = {}). Since d < 0, it means students in Group B performed significantly better than students in Group A.').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif p_test > 0.05:
                T, D =  round(t, 2), round(d, 2)
                if 0 < d <= 0.2:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < d <= 0.5:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a small effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < d < 0.8:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a intermediate effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif d >= 0.8:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a large effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            
                elif d <= 0:
                    s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is greater than 0.05, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a large effect was found (d = {}).').format(T, p_test, D)
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            
            if np.isnan(p_test) == True:
                T, D =  round(t, 2), round(d, 2)
                s7 = ('An independent two-samples T-test analysis [1] was conducted for comparing the final marks of students who attended at least one tutorial session, and students who attended none for the module '+i+'. The test shows that the t-statistic is equal to {}, associated with p-value equal to {:.3g}. Since p-value is nan, we accept the null hypothesis i.e, students who attended at least one tutorial session performed similar to students who attended none.  The effect size was determined using the Cohen\'s d method [2] and a very small effect was found (d = {}).').format(T, p_test, D)
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                pdf.multi_cell(0, 5, txt = str(s7), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            if p <= 0.05:
                F, ETA =  round(f, 2), round(eta2, 2)
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                if eta2 <= 0.2:
                    s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended betweeen 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < eta2 <= 0.5:
                    s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a medium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < eta2 < 0.8:
                    s8 = ('A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did nt attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to an intermedium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif eta2 >= 0.8:
                    s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we reject the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a large effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif p > 0.05:
                F, ETA =  round(f, 2), round(eta2, 2)
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
        
                if eta2 <= 0.2:
                    s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.2 < eta2 <= 0.5:
                    s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a medium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif 0.5 < eta2 < 0.8:
                    s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a intermedium effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif eta2 >= 0.8:
                    s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a large effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                    pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    
            if np.isnan(p) == True:
                F, ETA =  round(f, 2), round(eta2, 2)
                s8 = (r'A one-way analysis of variance [3] was conducted to compare the average academic performances of Group  A,  B and  C, these are students who did not attend any tutorial sessions (Group C), students who attended between 1-4 of the tutorial sessions (Group  B) and,  students who attended at least 5 of the tutorial sessions (Group  A).  The analysis returns an F-statistic equal to {}, associated with p-value equal to {:.3g}. Using the 5% significance level we accept the null hypothesis of no difference in averages of final marks for students in the three groups. The effect size [4] determined points to a very small effect between mean final marks of students in the three groups (eta2 = {}).').format(F, p, ETA)
                pdf.multi_cell(0, 5, txt = str(s8), align = 'L')
                pdf.cell(0, 5, txt = '', ln =22, align = 'C')
    
            if p < 0.05:
                clen = len(cc['A'])
                clen
                if 1 <= clen <= 2:
                    s10 = ('Specifically, the Post hoc analysis using the Games-Howell test [5] indicates a significant difference between students who attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B).').format()
                    pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif clen > 2:
                    s10 = ('Specifically, the Post hoc analysis using the Games-Howell test [5] indicates a significant difference between students who attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A).').format() 
                    pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                elif clen == 0:
                    s10 = ('The Post hoc analysis using the Games-Howell test [5] does not indicate significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A).').format() 
                    pdf.multi_cell(0, 5, txt = str(s10), align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
            elif p >= 0.05:
                    pdf.multi_cell(0, 5, txt = 'The Null Hypothesis has been accepted i.e, there\'s no significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A). Therefore, the Games Howell Post-hoc analysis [5] was omitted.', align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
                    print(' The Null Hypothesis has been accepted and therefore no need for a post-hoc analysis...')
        
            if np.isnan(p) == True:
                    pdf.multi_cell(0, 5, txt = 'The Null Hypothesis has been accepted i.e, there\'s no significant difference between students attended no tutorial sessions (Group C) and students who attended between 1-4 tutorial sessions (Group B), as well as between students who attended no tutorial sessions (Group C) and students who attended at least 5 tutorial sessions (Group A). Therefore, the Games Howell Post-hoc analysis [5] was omitted.', align = 'L')
                    pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        
            if pval <= 0.05:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                if 0 < corr <= 0.3:
                    R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                    s11 = ('There was a statistically significant,  weakly positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                    s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                    pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
                elif 0.3 < corr <= 0.6:
                    R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                    s11 = ('There was a statistically significant,  medium positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                    s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                    pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
                elif 0.6 < corr <= 1:
                    R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                    s11 = ('There was a statistically significant,  strong positive linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                    s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                    pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
                elif corr < 0:
                    R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                    s11 = ('There was a statistically significant,  weakly negative linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                    s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                    pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
            elif pval > 0.05:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically insignificant linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = {}, pval = {:.3g}).').format(corr, pval)
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
        
            if np.isnan(pval) == True:
                R2, R2_perc = round(R2, 2), round(R2_perc, 2)
                s11 = ('There was a statistically insignificant linear relationship (at a 5% significance level) between attendance of the tutorial sessions and the final mark obtained by the students (r = nan, pval = nan).').format()
                s9 = ('Linear regression analysis [6] of student attendance and final marks returns a determination coefficient that is equal to {}, i.e. {}% of the variation in final marks of students can be explained by the variation in tutorial attendance.').format(R2_perc, R2)
                pdf.multi_cell(0, 5, txt = str(s9+' '+s11), align = 'L')
        pdf.ln(0.25)
        pdf.add_page()
        pdf.set_font('Times','B',10.0)
        pdf.cell(0, 5, txt = 'Bibliography', ln = 21, align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        tt = ('Cressie, N. A. C., & Whitford, H. J. (1986). How to use the two sample t-test. Biometrical Journal, 28(2), 131-148.')
        coh = ('Ialongo, C. (2016). Understanding the effect size and its measures. Biochemia medica, 26(2), 150-163.')
        tano = ('Rice, W. R., & Gaines, S. D. (1989). One-way analysis of variance with unequal variances. Proceedings of the National Academy of Sciences, 86(21), 8183-8184.')
        ano_coh = ('Cortina, J. M., & Nouri, H. (2000). Effect size for ANOVA designs (No. 129). Sage.')
        pos_hoc = ('Hilton, A., & Armstrong, R. A. (2006). Statnote 6: post-hoc ANOVA tests. Microbiologist, 2006, 34-36.')
        reg = ('Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). Introduction to linear regression analysis. John Wiley & Sons.')
        t_ref_n = ('[1]')
        coh_ref_n = ('[2]')
        ano_ref_n = ('[3]')
        ano_coh_ref_n = ('[4]')
        pos_hoc_n = ('[5]')
        reg_n = ('[6]')
        pdf.ln(0.25)
        pdf.set_font('Times','I',10.0)
        pdf.multi_cell(0, 5, txt = str(t_ref_n+' '+tt), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.multi_cell(0, 5, txt = str(coh_ref_n+' '+coh), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.multi_cell(0, 5, txt = str(ano_ref_n+' '+tano), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.multi_cell(0, 5, txt = str(ano_coh_ref_n+' '+ano_coh), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.multi_cell(0, 5, txt = str(pos_hoc_n+' '+pos_hoc), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.multi_cell(0, 5, txt = str(reg_n+' '+reg), align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_font('Times','B',10.0)
        pdf.cell(0, 5, txt = 'Acknowledgements', ln = 21, align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_font('Times','I',9.0)
        pdf.multi_cell(0, 5, txt = 'This research was conducted with perfomance data from the Directorate for Institutional Research and Academic Planning (DIRAP). We thank our colleagues from DIRAP who also provided insight and expertise that greatly assisted the research.', align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_font('Times','BI',10.0)
        pdf.cell(0, 5, txt = 'Contact Details', ln = 21, align = 'L')
        #s37 = 'eicon.jpeg'
        pdf.image('email.jpeg', x =5, y = 122, w = 40, h = 30, type = 'JPEG')
        pdf.image('globe.jpeg', x =120, y = 122, w = 40, h = 30, type = 'JPEG')
        pdf.set_font('Times','BI',9.0)
        pdf.set_text_color(0, 0, 255)
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '                                         \x95 email: mbonanits@ufs.ac.za', 
        ln =22, align = 'L', fill = False)
        pdf.cell(0, 5, txt = '                                         \x95 email: motsokobie@ufs.ac.za                                                                                                        \x95 www.ufs.ac.za/ctl', 
        ln =22, align = 'L')
        pdf.cell(0, 5, txt = '                                         \x95 email: maribeg@ufs.ac.za', 
        ln =22, align = 'L')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_font('Arial','B',10.0)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 5, txt = 'The faculty of '+' '+str(fac), ln = 21, align = 'C')
        
        if str(fac) == 'EDUCATION':
            pdf.image('nas.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('hum.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('law.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'HUMANITIES':
            pdf.image('nas.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('hum.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('law.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'LAW':
            pdf.image('nas.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('law.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('hum.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'NATURAL AND AGRICULTURAL SCIENCES':
            pdf.image('law.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('nas.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('hum.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'ECONOMIC AND MANAGEMENT SCIENCES':
            pdf.image('law.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('hum.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('nas.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'HEALTH SCIENCES':
            pdf.image('law.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('hum.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('nas.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        elif str(fac) == 'THEOLOGY':
            pdf.image('law.jpeg', x =10, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('edu.jpeg', x =35, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('health.jpeg', x =60, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('theo.jpeg', x =87, y = 163, w = 40, h = 40, type = 'JPEG')
            pdf.image('hum.jpeg', x =129, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('nas.jpeg', x =154, y = 166, w = 25, h = 30, type = 'JPEG')
            pdf.image('ems.jpeg', x =179, y = 166, w = 25, h = 30, type = 'JPEG')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_fill_color(0, 0, 200)
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.cell(0, 5, txt = '', ln =22, align = 'C', fill=True)
        pdf.cell(0, 5, txt = '', ln =22, align = 'C')
        pdf.set_font('Times','',8.0)
        pdf.cell(0, 5, txt = '205 Nelson Mandela Drive | Park West, Bloemfontein 9301 | South Africa', ln =22, align = 'L')
        pdf.cell(0, 5, txt = 'P.O. Box 339 | Bloemfontein 9301 | South Africa |', ln =22, align = 'L')
        #pdf.set_xy(70, 262)
        #pdf.cell(50, 15, txt = 'www.ufs.ac.za', border = 0, ln=7, align = 'C')
        pdf.image('kindpng_2007834.jpeg', x = 10, y = 256, w = 189, h =3, type = 'PNG')
        pdf.image('ufs.jpeg', x = 175, y = 261, w = 30, h =30, type = 'JPEG')
        pdf.output('A_STEP_IR_'+fac+'_'+term+'_'+camp+'_2019_2022.pdf')

        removing_files = glob.glob('/Users/tekanombonani/Desktop/Faculty_Reports/*.png')
        for i in removing_files:
            os.remove(i)
    
        progress_bar = st.progress(0)
        for perc_completed in range(100):
            time.sleep(0.05)
            progress_bar.progress(perc_completed+1)
            
        #st.balloons()
        dF = df.loc[lambda df: (df['FACULTY'] == fac) & (df['Campus'] == camp) & (df['Term'] == term)]
        grpA = dF.loc[lambda dF: dF['freq'] == 0]
        grpB = dF.loc[lambda dF: (dF['freq'] >= 1) & (dF['freq'] <= 4)]
        grpC = dF.loc[lambda dF: dF['freq'] >= 5]
        dF1 = dF.loc[lambda dF: dF['freq'] >= 1]
        with st.expander(':blue[Read More:]'):
            st.write('The faculty of', fac,', on the', camp, 'campus had a total,', dF1['Attendee'].nunique(), 'unique A-STEP students in attendance during', term,'. The A-STEP\
             students accumulated attendance frequency equal to', dF1['freq'].sum(),', and obtained an average mark of', round(dF1['FINAL.MARK'].mean(),2),'%\
             in their final exams.', grpA['Attendee'].nunique(), 'students did not attend any A-STEP tutorials, and obtained an average final mark of', round(grpA['FINAL.MARK'].mean(),2),'%.\
             Let this assembly of students be **Group C**. In total, ', grpB['Attendee'].nunique(), 'students (**Group B**) attended at least one and at most four (1-4) A-STEP tutorials, and\
             reached an average final mark of', round(grpB['FINAL.MARK'].mean(),2),'%. Finally, let the', grpC['Attendee'].nunique(), 'students that attended at least five tutorials\
             and obtained an average of ', round(grpC['FINAL.MARK'].mean(),2),'% in their finals be **Group A**.')
            st.pyplot(fig1)
            st.pyplot(fig2)

        with open('A_STEP_IR_'+fac+'_'+term+'_'+camp+'_2019_2022.pdf', "rb") as file:
            btn = st.download_button(
            label="Download PDF Report",
            data=file,
            file_name='A_STEP_IR_'+fac+'_'+term+'_'+camp+'_2019_2022.pdf',
            mime="file/pdf"
              )
        st.success('Report Ready for Download', icon="✅")
 
  
    else:
        st.write('Press ***Generate Report*** to run the analysis.')
    colm1, colm2 = st.columns([0.05, 0.95], gap='small')
    with colm1:
        st.write(' ')
    with colm2:
        st.markdown("![Alt Text](https://i.postimg.cc/cHkqhhPP/ufsbar.png)")


