# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:44:34 2022

@author: SGoethals
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from collections import Counter
from statistics import mean
# nice.explainers import NICE
from nice import NICE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from statistics import median,mode
from matplotlib.backends.backend_pdf import PdfPages
#%%


   
def model( X, y, sensitive_attribute, cat=None, to_drop=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    if to_drop is not None:
        X_train=X_train.drop(columns=to_drop)
        X_testadapted=X_test.drop(columns=to_drop)
    if to_drop is None:
        X_testadapted=X_test
    feature_names=X_train.columns.tolist()
    print('Feature Names:{}'.format(feature_names))
    if cat==None:
        cat = X_train.select_dtypes(include=['category','object']).columns.tolist()
        cat_feat=[X_train.columns.get_loc(col) for col in cat]
        num =X_train._get_numeric_data().columns.tolist()
        num_feat=[X_train.columns.get_loc(col) for col in num]
    if cat!=None:
        feat=[X_train.columns.get_loc(col) for col in feature_names]
        cat_feat=[X_train.columns.get_loc(col) for col in cat]
        setje=(set(feat)-set(cat_feat))
        num_feat=list(setje)
        num=list(X_train.iloc[:,num_feat])
    numeric_transformer = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
            ])
    cate_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ])
    clf = Pipeline([
    ('PP',ColumnTransformer([
            ('num',numeric_transformer,num_feat),
            ('cate',cate_transformer,cat_feat)])),
    ('RF',RandomForestClassifier(random_state=0))])
    pipe_params = {
    "RF__n_estimators": [ 10,50,100, 500, 1000,5000],
    "RF__max_leaf_nodes":[10,100,500,None],
    }
    grid=GridSearchCV(clf, pipe_params, cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)
    clf=grid.best_estimator_
    accuracyglobal=accuracy_score(y_test,grid.predict(X_testadapted))
    print(grid.best_params_)
    print('The accuracy of the model  is {}'.format(accuracyglobal))
    if cat != (None or []):
        onehot_columns = clf.named_steps['PP'].named_transformers_['cate'].named_steps['onehot'].get_feature_names(input_features=cat)
        feat_after_pipeline=np.array(num+list(onehot_columns))
    if cat == (None or []):
        feat_after_pipeline=np.array(num)
    return clf, cat_feat, num_feat, feature_names,feat_after_pipeline,accuracyglobal

def calculate_fairness_metrics(X,y,clf,sensitive_attribute,good_outcome,bad_outcome,to_drop=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    TP,FP,TN,FN,PPV,PR,TPR,FPR,accuracy,balanced_accuracy={},{},{},{},{},{},{},{},{},{}
    if to_drop is not None:
        X_testadapted = X_test.drop(columns=to_drop)
    if to_drop is None:
        X_testadapted=X_test
    attribute_values=X_test[sensitive_attribute].unique()
    for attribute in attribute_values:
        TP[attribute], FP[attribute], FN[attribute],TN[attribute]=0,0,0,0
        for a in range(0,len(y_test)):
            if X_test.iloc[a][sensitive_attribute]==attribute:
                if clf.predict(X_testadapted.iloc[a:a+1,:])==good_outcome:
                    if y_test.iloc[a]==good_outcome:
                        TP[attribute]+=1
                    elif y_test.iloc[a]==bad_outcome:
                        FP[attribute]+=1
                if clf.predict(X_testadapted.iloc[a:a+1,:])==bad_outcome:
                    if y_test.iloc[a]==bad_outcome:
                        TN[attribute]+=1
                    elif y_test.iloc[a]==good_outcome:
                        FN[attribute]+=1
        PPV[attribute]=TP[attribute]/(TP[attribute]+FP[attribute]) #predictive parity
        PR[attribute]=(TP[attribute]+FP[attribute])/(TP[attribute]+FP[attribute]+TN[attribute]+FN[attribute]) #demographic parity
        TPR[attribute]=TP[attribute]/(TP[attribute]+FN[attribute])#equalized odds (and equal opportunity)
        FPR[attribute]=FP[attribute]/(FP[attribute]+TN[attribute])#equalized odds
        balanced_accuracy[attribute]=1/2*(TP[attribute])/(TP[attribute]+FN[attribute]) +1/2*(TN[attribute])/(TN[attribute]+FP[attribute])
        accuracy[attribute]=(TP[attribute]+TN[attribute])/(TP[attribute]+FP[attribute]+TN[attribute]+FN[attribute]) 
        print('The PPV of group:{} is {}'.format(attribute,PPV[attribute]))
        print('The PR of group:{} is {}'.format(attribute,PR[attribute]))
        print('The TPR of group:{} is {}'.format(attribute,TPR[attribute]))
        print('The FPR of group:{} is {}'.format(attribute,FPR[attribute]))
        print('The accuracy of group:{} is {}'.format(attribute,accuracy[attribute]))
        print('The balanced_accuracy of group:{} is {}'.format(attribute,balanced_accuracy[attribute]))
        print('The TP of group:{} is {}'.format(attribute,TP[attribute]))
        print('The FP of group:{} is {}'.format(attribute,FP[attribute]))
        print('The TN of group:{} is {}'.format(attribute,TN[attribute]))
        print('The FN of group:{} is {}'.format(attribute,FN[attribute]))
    return PPV,PR,TPR,FPR,TP,FP,TN,FN,accuracy,balanced_accuracy

def calculate_default_2(X,y,num_feat,cat_feat,feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    default_values={}
    threshold=len(X_train)/100
    for feature in feature_names:
        feature_names=list(feature_names)
        f=feature_names.index(feature)
        default_values[feature]=[]
        if f in num_feat:
            default_values[feature]=np.percentile(X_train[feature],[10,20,3,40,50,60,70,80,90,100])
            default_values[feature]=list(set( default_values[feature]))
        if f in cat_feat:
            valuessorted=X_train[feature].value_counts()
            for i in range(0,len(valuessorted)):
                value=valuessorted.index[i]
                if valuessorted[value]>=threshold:
                    default_values[feature].append(value)
                else:
                    break
            if len(default_values[feature])>10:
                default_values[feature]=valuessorted.index[0:10].tolist()
            if len(default_values[feature])==0:
                default_values[feature]=valuessorted.index[0:10].tolist()
    return default_values



def my_counterfactual(num_feat,cat_feat,X,y,sensitive_attribute,feature_names,clf,bad_outcome,good_outcome,default_values):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    X_testadapted = X_test.drop(columns=sensitive_attribute)
    dct,dct_with_value,factual = {},{},{}
    attribute_values=X_test[sensitive_attribute].unique()
    feat_list=[]
    feature_names=list(feature_names)
    for attribute in attribute_values:
        dct[attribute]=[]
        dct_with_value[attribute]=[]
        factual[attribute]=[]
        for a in range(0,len(y_test)):
            to_explain=X_testadapted.iloc[a:a+1,:]
            if clf.predict(to_explain)[0]==bad_outcome and X_test.iloc[a][sensitive_attribute]==attribute:
                for feature in feature_names:
                    f=feature_names.index(feature)
                    tel=0
                    CF=copy.deepcopy(to_explain)
                    for i in range(0,len(default_values[feature])):
                        CF[feature]=default_values[feature][i]
                        if clf.predict(CF)[0]==good_outcome:
                            tel+=1
                            if f in cat_feat:
                                dct_with_value[attribute].append(feature +':'+ str(default_values[feature][i]))
                                factual[attribute].append(feature +':'+ str(to_explain[feature].item()))
                            #if f in num_feat:
                                #if CF[feature].values>to_explain[feature].values:
                                    #dct_with_value[attribute].append(feature + ': increase')
                                    #factual[attribute].append(feature + ': increase')
                                #elif CF[feature].values<to_explain[feature].values:
                                    #dct_with_value[attribute].append(feature + ': decrease')
                                    #factual[attribute].append(feature + ': decrease')
                    if tel>0:
                        dct[attribute].append(feature)
    return dct,dct_with_value,factual,attribute_values
                
def count_list(dct,TN,FN,TP,FP, sensitive_attribute, X):
     attribute_values=X[sensitive_attribute].unique()
     dct_sorted = {}
     dct_count= {}
     for attribute in attribute_values:
         noemer=TN[attribute]+FN[attribute]
         print('noemer is rejected')
         #noemer=TN[attribute]+FN[attribute]+TP[attribute]+FP[attribute]
         #print('noemer is amount of people in the test set')
         a=Counter(dct[attribute])
         aa = {k: v / noemer for k, v in a.items()}
         dct_count[attribute]= aa
         dct_sorted[attribute]= sorted(aa.items(), key=lambda x: x[1], reverse=True)
     return dct_count,dct_sorted

            
def visualize_difference_2(dct_difference, feature_set, attribute_values, sensitive_attribute,file,method,  cf_algo=None):
    proxies={}
    for attribute in attribute_values:
        print(attribute)
        Z=pd.DataFrame({'Name': list(feature_set), 'Difference': dct_difference[attribute]}).sort_values('Difference', ascending=False)
        fig=plt.figure(figsize=(15, 5))
        plt.rcParams['font.size']='30'
        plt.barh(y='Name', width='Difference', data=Z[0:5],color='b')
        plt.gca().invert_yaxis()
        if method=='normal':
            plt.xlabel("$PreCoF$",fontsize=35)
        if method=='cf':
            plt.xlabel("$PreCoF_{c}$",fontsize=35)
        if method=='factual':
            plt.xlabel("$PreCoF_{f}$",fontsize=35)
        if cf_algo=='NICE':
            plt.suptitle("NICE",fontsize=20)
        plt.xticks(rotation=90)
        proxies[attribute]=Z.iloc[0,0]
        #plt.savefig(file, format = 'pdf', bbox_inches='tight')
        file.savefig(fig, bbox_inches='tight')
    return proxies
        
def calculate_difference_2 (dct_count,attribute_values):
    dct_difference={}
    feature_set=set()
    for attribute in attribute_values:
        feature_set.update(dct_count[attribute].keys())
    for attribute in attribute_values:
        dct_difference[attribute]=[]
        #for feature in feature_names:
        for feature in feature_set:
            temp_list=[]
            try:
                dct_count[attribute][feature]
            except KeyError:
                dct_count[attribute][feature] = 0
            for attribute2 in attribute_values:
                try:
                    dct_count[attribute2][feature]
                except KeyError:
                    dct_count[attribute2][feature]=0
                if attribute!=attribute2:
                    temp_list.append(dct_count[attribute2][feature])
            dct_difference[attribute].append(dct_count[attribute][feature]-mean(temp_list))
    return dct_difference, feature_set

def feature_importances(clf,feat_after_pipeline,sensitive_attribute):
    feat_values=clf['RF'].feature_importances_
    #new_feat=clf['PP'].named_transformers_['cat']['onehot'].get_feature_names()
    new_feat=feat_after_pipeline
    # Zip coefficients and names together and make a DataFrame
    zipped = zip(new_feat,feat_values)
    dffeat = pd.DataFrame(zipped, columns=["feature", "value"])
    # Sort the features by the absolute value of their coefficient
    dffeat["abs_value"] = dffeat["value"].apply(lambda x: abs(x))
    dffeat["colors"] = dffeat["value"].apply(lambda x: "green" if x > 0 else "red")
    dffeat = dffeat.sort_values("abs_value", ascending=False)
    import seaborn as sns
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sns.barplot(x="feature",
                y="value",
                data=dffeat.head(10),
                palette='mako')
               #palette=dffeat.head(20)["colors"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
    ax.set_title("Top 10 Features", fontsize=25)
    plt.suptitle('sensitive attribute: {}'.format(sensitive_attribute),fontsize=20)
    ax.set_ylabel("Feature importance", fontsize=22)
    ax.set_xlabel("Feature Name", fontsize=22)

def calculate_explicit_bias(X,y,sensitive_attribute,clf,bad_outcome,good_outcome):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    attribute_values=X_test[sensitive_attribute].unique()
    count={}
    factual={}
    explanation={}
    for attribute in attribute_values:
        factual[attribute] = 0
        explanation[attribute] = 0
    for attribute in attribute_values:
        for a in range(0,len(y_test)):
            to_explain=X_test.iloc[a:a+1,:]
            if clf.predict(to_explain)[0]==bad_outcome and X_test.iloc[a][sensitive_attribute]==attribute:
                CF=copy.deepcopy(to_explain)
                for attribute2 in attribute_values:
                    if attribute2!=attribute:
                        CF[sensitive_attribute]=attribute2
                        if clf.predict(CF)[0]==good_outcome:
                            factual[attribute]+=1
                            explanation[attribute2]+=1
    print('Explicit bias with our method. Factual: {}, Explanation {}'.format(factual,explanation))
    return factual,explanation


def visualize_explicit_bias(sensitive_attribute, factual, explanation,file,cf_algo=None):
    fig = plt.figure()
    plt.bar(factual.keys(), factual.values(), color='r', label = 'counterfactual values')
    plt.title('Explicit bias: ' + str(
        sensitive_attribute))
    if cf_algo=='NICE':
        plt.suptitle("NICE",fontsize=15)
    #plt.savefig(file, bbox_inches='tight')
    file.savefig(fig)
    fig = plt.figure()
    plt.bar(explanation.keys(), explanation.values(), color='g', label='factual values')
    plt.title('Explicit bias ' + str(
        sensitive_attribute))
    if cf_algo=='NICE':
        plt.suptitle("NICE",fontsize=15)
   # plt.savefig(file, format = 'pdf', bbox_inches='tight')
    file.savefig(fig, bbox_inches='tight')

def visualize_fairness_metrics(accuracyglobal1,accuracyglobal2,accuracyglobal3, PR1,PR2,PR3, PPV1, PPV2, PPV3,TPR1,TPR2,TPR3,FPR1,FPR2,FPR3,accuracy1, accuracy2, accuracy3):
    a1=max(PR1, key=PR1.get)
    a2=min(PR1, key=PR1.get)
    print('The unprotected group is {} and the protected group is {}'.format(a1,a2))
    print('The accuracy of the model trained on the dataset with the sensitive attribuut is :{}'.format(accuracyglobal1))
    print('The accuracy of the model trained on the dataset without the sensitive attribuut is :{}'.format(accuracyglobal2))
    print('The accuracy of the model trained on the dataset without the sensitive attribuut and the proxy is is :{}'.format(accuracyglobal3))
    print('The difference in positive rate for both sensitive groups: what is the difference in probability between the protected and unprotected group of being classified as positive?')
    print('The demographic disparity of the model trained on the dataset with the sensitive attribuut is :{}. The predictive rates are: {},{}'.format(PR1[a1]-PR1[a2],PR1[a1],PR1[a2]))
    print('The demographic disparity of the model trained on the dataset without the sensitive attribuut is :{}. The predictive rates are: {},{}'.format(PR2[a1]-PR2[a2],PR2[a1],PR2[a2]))
    print('The demographic disparity of the model trained on the dataset without the sensitive attribuut and the proxy is :{}. The predictive rates are: {},{}'.format(PR3[a1]-PR3[a2],PR3[a1],PR3[a2]))
    print('Predictive parity: if the protected and unprotected group have equal PPV: the probability of a subject with positive predicive value to truly belong to the positive class')
    print(' The fraction of correct positive predictions should be the same for both groups')
    print('The difference in predictive parity of the model trained on the dataset with the sensitive attribuut is :{}. The positive predicted values (PPV) are: {},{}'.format(PPV1[a1]-PPV1[a2],PPV1[a1], PPV1[a2]))
    print('The difference in predictive parity of the model trained on the dataset without the sensitive attribuut is :{}. The positive predicted values (PPV) are: {},{}'.format(PPV2[a1]-PPV2[a2],PPV2[a1], PPV2[a2]))
    print('The difference in predictive parity of the model trained on the dataset without the sensitive attribuut and the proxy is :{}. The positive predicted values (PPV) are: {},{}'.format(PPV3[a1]-PPV3[a2],PPV3[a1], PPV3[a2]))
    print('Equal opportunity:Similar results for people with a good outcome of both groups')
    print('The difference in equal opportunity in the model with the sensitive attribute is:{}'.format(TPR1[a1]-TPR1[a2]))
    print('The difference in equal opportunity in the model without the sensitive attribute is:{}'.format(TPR2[a1]-TPR2[a2]))
    print('The difference in equal opportunity in the model without the sensitive attribute and the proxy is:{}'.format(TPR3[a1]-TPR3[a2]))
    print('The accuracy of the model in each group trained on the dataset with the sensitive attribuut is :{},{}'.format(accuracy1[a1],accuracy1[a2]))
    print('The accuracy of the model in each group trained on the dataset without the sensitive attribuut is :{},{}'.format(accuracy2[a1],accuracy2[a2]))
    print('The accuracy of the model in each group trained on the dataset without the sensitive attribuut and the proxy is is :{},{}'.format(accuracy3[a1], accuracy3[a2]))
    
#%%    
def precof(X,y,sensitive_attribute,cat,good_outcome,bad_outcome,sensitive_value, file):
    print('Modelling with sensitive attribute')
    if cat is None:
        catws= [sensitive_attribute]
    if cat is not None:
        catws=cat+[sensitive_attribute]
    clf, cat_feat, num_feat, feature_names,feat_after_pipeline,accuracyglobal1 = model(X, y, sensitive_attribute,cat=catws,to_drop=None)
    feature_importances(clf,feat_after_pipeline,sensitive_attribute)
    PPV1,PR1,TPR1,FPR1,TP1,FP1,TN1,FN1,accuracy1,balanced_accuracy1=calculate_fairness_metrics(X, y, clf, sensitive_attribute, good_outcome, bad_outcome,to_drop=None)
    print('Calculate explicit bias')
    factual,explanation=calculate_explicit_bias(X,y,sensitive_attribute,clf,bad_outcome,good_outcome)
    visualize_explicit_bias(sensitive_attribute,  factual, explanation, file)
    print('Modelling without sensitive attribute')
    clf, cat_feat, num_feat, feature_names,feat_after_pipeline,accuracyglobal2 = model(X, y, sensitive_attribute,cat,to_drop=[sensitive_attribute])
    feature_importances(clf,feat_after_pipeline,sensitive_attribute)
    PPV2,PR2,TPR2,FPR2,TP2,FP2,TN2,FN2,accuracy2,balanced_accuracy2=calculate_fairness_metrics(X, y, clf, sensitive_attribute, good_outcome, bad_outcome,to_drop=sensitive_attribute)
    print('Generating counterfactuals')
    default_values=calculate_default_2(X,y,num_feat,cat_feat,feature_names)
    dct,dct_with_value,dct_factual,attribute_values=my_counterfactual(num_feat,cat_feat,X,y,sensitive_attribute,feature_names,clf,bad_outcome,good_outcome,default_values)
    print('Calculating and visualizing without value')
    method='Counterfactual attributes'
    dct_count,dct_sorted=count_list(dct,TN2,FN2,TP2,FP2,sensitive_attribute, X)
    dct_difference,feature_set=calculate_difference_2 (dct_count,attribute_values)      
    proxies=visualize_difference_2(dct_difference, feature_set, attribute_values, sensitive_attribute,file,method='normal')       
    print('Calculating and visualizing with value')
    if cat is not None:
        method='Counterfactual attributes with specific values'
        dct_count,dct_sorted=count_list(dct_with_value,TN2,FN2,TP2,FP2, sensitive_attribute, X)
        dct_difference,feature_set=calculate_difference_2 (dct_count,attribute_values)      
        proxyvalue=visualize_difference_2(dct_difference, feature_set, attribute_values, sensitive_attribute,file, method='cf')                 
        print('Calculate factual values')
        method='Factual values'
        dct_count,dct_sorted=count_list(dct_factual,TN2,FN2,TP2,FP2,sensitive_attribute, X)
        dct_difference,feature_set=calculate_difference_2 (dct_count,attribute_values)      
        proxyfactual=visualize_difference_2(dct_difference, feature_set, attribute_values, sensitive_attribute, file, method='factual')                 
    print('Modelling after dropping the feature coming out of the explanations')
    print('Modelling without sensitive attribute and without feature from explanation')
    #proxy = input("Enter your value: ")
    proxy=proxies[sensitive_value]
    print(proxy)
    if cat==None:
        catwithout=None
    elif proxy in cat:
        catwithout=cat.copy()
        catwithout.remove(proxy)
    else:
        catwithout=cat.copy()
    clf, cat_feat, num_feat, feature_names,feat_after_pipeline,accuracyglobal3 = model(X, y, sensitive_attribute,catwithout,to_drop=[sensitive_attribute,proxy])
    feature_importances(clf,feat_after_pipeline,sensitive_attribute)
    PPV3,PR3,TPR3,FPR3,TP3,FP3,TN3,FN3,accuracy3,balanced_accuracy3=calculate_fairness_metrics(X, y, clf, sensitive_attribute, good_outcome, bad_outcome,to_drop=[sensitive_attribute,proxy])
    visualize_fairness_metrics(accuracyglobal1,accuracyglobal2,accuracyglobal3, PR1,PR2,PR3, PPV1, PPV2, PPV3,TPR1,TPR2,TPR3,FPR1,FPR2,FPR3,accuracy1, accuracy2, accuracy3)

