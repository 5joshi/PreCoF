# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:51:33 2022

@author: SGoethals
"""


#%%    
import os 
from pmlb import fetch_data
from functions_precof import *
from statistics import mean
import pandas as pd
#%%
student=pd.read_csv('DATA/student/student-por.csv', sep=';')

X=student.drop(columns=[ 'G1','G2','G3'])
y = student['G3']>=mean(student['G3'])

good_outcome=True
bad_outcome=False
sensitive_attribute='sex'
sensitive_value='F'
cat=['school', 'address', 'famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid', 'activities','nursery','higher','internet', 'romantic']
filename='results/student_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)

#%%t
adult = fetch_data('adult')
X=adult.drop(columns=[ 'target','fnlwgt','native-country'])
y = adult.loc[:, 'target']

good_outcome=0
bad_outcome=1
sensitive_attribute='sex'
sensitive_value=0
cat=['workclass','education','marital-status','occupation','relationship','race']
filename='results/adult_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)


#%% compas


compas=pd.read_csv('DATA/cox-violent-parsed.csv')
X=compas[['sex','age','age_cat', 'race', 'juv_fel_count','decile_score', 'juv_misd_count','juv_other_count', 'priors_count', 'c_charge_desc']]
y=compas['is_recid']
y=y.replace(to_replace=-1,value=0)
good_outcome=0
bad_outcome=1
X['African_American']=(X['race']=='African-American')
X=X.drop('race',axis=1)
sensitive_attribute='African_American'
sensitive_value=True
cat=['age_cat', 'sex','c_charge_desc']



filename='results/compas_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)


#%%


#Catalonia dataset
dfmain = pd.read_csv("https://github.com/nkundiushuti/savry/blob/master/dat/reincidenciaJusticiaMenors.csv?raw=true",low_memory=False)
dfmain = dfmain.rename(index=str, columns={"V2_estranger": "foreigner", "V1_sexe": "sex","V60_SAVRY_total_score": "full_score", \
                                  "V115_reincidencia_2015":'recid', "V4_nacionalitat_agrupat":"national_group", \
                                  "V5_edat_fet_agrupat":"age_group","V6_provincia":"province","V9_edat_final_programa": \
                                  "age_final", "V11_antecedents":"prior_crime", "V12_nombre_ante_agrupat": \
                                   "prior_crimerec","V13_nombre_fets_agrupat": "prior_crimes", "V15_fet_agrupat": \
                                  "crime_maincat", "V16_fet_violencia": "crime_violence", "V17_fet_tipus":"crime_type", \
                                  'V68_@4_fracas_intervencions_anteriors':'past_supervision_failure','V69_@5_intents_autolesio_suicidi_anteriors':'history_self_harm',\
                                      'V70_@6_exposicio_violencia_llar':'violent_home','V71_@7_historia_maltracte_infantil':'childhood_mistreatment', 'V72_@8_delinquencia_pares':'parental_criminality',\
                                          'V74_@10_baix_rendiment_escola':'poor_school','V75_@11_delinquencia_grup_iguals':'delinquent_peer_group','V76_@12_rebuig_grup_iguals':'rejected_by_peer_group',\
                                              'V79_@15_manca_suport_personal_social':'lack_of_social_support','V80_@16_entorn_marginal':'community_disorganization'})
dfmain['label_value'] = dfmain.recid == 'Sí'
dfmain.national_group=dfmain.national_group.fillna('Spanish')
dfaequi=dfmain[['id', 'recid','label_value','full_score','foreigner','sex','national_group','age_group','province','age_final', \
            'prior_crime','prior_crimerec','prior_crimes','crime_maincat','crime_violence', 'crime_type','past_supervision_failure','history_self_harm', 'violent_home',\
                'childhood_mistreatment','parental_criminality','poor_school', 'delinquent_peer_group','rejected_by_peer_group','lack_of_social_support','community_disorganization']]
#dfaequi=dfmain
dfaequi = dfaequi[np.isfinite(dfaequi['full_score'])]
dfaequi = dfaequi.loc[dfaequi['full_score']!=99]
dfaequi.age_group=dfaequi.age_group.fillna('16 i 17 anys') #replaced here missings by higher age group because final age is high
df = dfaequi
X=df.drop(['full_score', 'id', 'label_value','recid'],axis=1)
y=(df['label_value'])
good_outcome=False
bad_outcome=True
sensitive_attribute='foreigner'
sensitive_value='Estranger'
cat=['sex','national_group','age_group','province','prior_crime','prior_crimerec','prior_crimes','crime_maincat','crime_violence', 'crime_type','past_supervision_failure','history_self_harm', 'violent_home','childhood_mistreatment','parental_criminality','poor_school', 'delinquent_peer_group','rejected_by_peer_group','lack_of_social_support','community_disorganization']

filename='results/catalonia_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)



#%% German
credit = fetch_data('german')
X=credit.drop(columns=[ 'target'])
y = credit.loc[:, 'target']

good_outcome=1
bad_outcome=0
sensitive_attribute='Foreign'
sensitive_value=1
cat=['Status','Credit-history','Purpose','Savings-account', 'Employment','Installment-rate','Personal-status','Residence-time','Property','Housing','Job','Telephone']

filename='results/german_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)

#%% law school
law=pd.read_csv('DATA/fairness_dataset-main/fairness_dataset-main/experiments/data/law_school_clean.csv', sep=',')
X=law.drop(columns=[ 'pass_bar'])
y = law.loc[:, 'pass_bar']

good_outcome=1
bad_outcome=0
sensitive_attribute='race'
sensitive_value='Non-White'
cat=['fulltime', 'fam_inc', 'male', 'tier']


filename='results/law_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)


#%%
crime2=pd.read_csv('DATA/fairness_dataset-main/fairness_dataset-main/experiments/data/communities_crime.csv', sep=',')
X=crime2.drop(columns=['class','communityname','state', 'ViolentCrimesPerPop',
'racepctblack', 'racePctWhite', 'racePctAsian',
'racePctHisp', 'fold','AsianPerCap','HispPerCap','whitePerCap','blackPerCap','indianPerCap', 'HispPerCap'])
y = crime2['class']

good_outcome=False
bad_outcome=True
sensitive_attribute='Black'
sensitive_value=1
#cat=['state']
cat=None

filename='results/crime2_images.pdf'  
file=PdfPages(filename)
precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file)
