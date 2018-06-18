from __future__ import print_function
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import compare
pd.options.display.max_rows = None
pd.options.display.max_columns = None
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import loader

def do_stats(df_X,y):
    """ For every column in df_X, compares truthful and bluffing groups using
        compare. A df with means, t-test, MWW, and Cohen's d is returned.
    """
    stats = []
    for feature in df_X.columns:
        A = df_X.ix[y,feature].dropna()
        B = df_X.ix[~y,feature].dropna()
        comp = compare.Compare(A,B,label_=feature,
                               x_label_='Truthful',y_label_='Lying')
        stats_f = comp.calc_stats()
        if(feature == 'smile'):
            comp.plot_all()
        stats.append(stats_f)
    
    df_stats = pd.DataFrame(stats,index=df_X.columns, columns=['mean T','mean B','t-test','MWW','Cohens d'])
    df_stats.sort_values(by=['MWW'],ascending=True,inplace=True)
    #display(df_stats)        
    return df_stats
    
#------------------------------------------------------------

def ml(df):
    """ Runs either logistic regression or KNN through sklearn using both 
        dev and test set. """
    
    N_FOLDS = 10
    ACC_THRESH = 0.01 # dev set accuracy must be x% better to use new param
    MODEL = 'LOGISTIC' 
    #MODEL = 'KNN' 
    
    FEATURES = ['S2_ AU12_r_var','SD_ AU23_c_avg', \
                'S1_ AU01_c_avg','S2_ AU06_r_var',\
                'S2_ AU25_c_avg', 'SD_abs_d_pose_Rx_var']
    #FEATURES = ['S2_ AU12_c_avg','S2_ AU06_c_avg']


    if MODEL == 'LOGISTIC':
        c_l = [0.01,0.03, 0.1, 0.3, 1,3,10,30,100,300,1000,3000,10000]
    elif MODEL == 'KNN':       
        c_l = [50,40,35,30,25,20,18,15]
    regularizer = 'l1'
    
    # only get features we want that are not nan
    df_good = df.loc[:,['y'] + FEATURES]
    df_good = df_good.dropna()
        
    X_nd = df_good.drop('y',axis=1).values
    X_nd = scale(X_nd) # magnitude has useful info?
    y_n = df_good['y'].values.astype(bool)
    skf = StratifiedKFold(shuffle = True, n_splits=N_FOLDS)

    acc_test_a = np.zeros(N_FOLDS)
    acc_train_a = np.zeros(N_FOLDS)
    for i, (train, test) in enumerate(skf.split(X_nd,y_n)):
        train_n = len(train)
        dev = train[:int(train_n/4)]  # empirically found that dev 1/4 is good
        sub_train = train[int(train_n/4):] # this is temporary train set
        best_acc = 0
        best_c = None
        # in this loop we find best hyper parameter for this split
        for c in c_l:
            if MODEL == 'LOGISTIC': 
                clf = linear_model.LogisticRegression(penalty=regularizer,C=c)
            elif MODEL == 'KNN':           
                clf = KNeighborsClassifier(n_neighbors=c, metric='euclidean',weights='uniform')

            clf.fit(X_nd[sub_train], y_n[sub_train])
            y_pred = clf.predict(X_nd[dev])
            acc = metrics.accuracy_score(y_pred,y_n[dev])
            if(acc > best_acc + ACC_THRESH):
                best_acc = acc
                best_c = c

        # retrain with all train data and best_c
        print('fold:',i,' best c:',best_c, ' dev:%.2f' % best_acc, ' dev_ones:%.2f' % (y_n[dev].sum()/len(dev)),end='')
        if MODEL == 'LOGISTIC': 
            clf = linear_model.LogisticRegression(penalty=regularizer,C=best_c)
        elif MODEL == 'KNN':           
            clf = KNeighborsClassifier(n_neighbors=best_c, metric='euclidean',weights='uniform')
        clf.fit(X_nd[train],y_n[train])
        y_pred = clf.predict(X_nd)
        acc_test_a[i] = metrics.accuracy_score(y_pred[test],y_n[test])
        acc_train_a[i] = metrics.accuracy_score(y_pred[train],y_n[train])
        print(' test:%.2f' % acc_test_a[i], ' train:%.2f' % acc_train_a[i])
    print('Avg test acc:%.3f' % acc_test_a.mean(),'Avg train acc:%.3f' % acc_train_a.mean())    

#----------------------------------------------------------------------

def plot(f0,f1,df_X,y):
    X_nd = df_X.loc[:,['S2_smile','SD_ AU23_c_avg']].values
    X_nd = df_X.loc[:,[f0,f1]].values
    nan_rows_i = np.isnan(X_nd).any(1)
    X_nd = X_nd[~nan_rows_i,:]
    y_n = y[~nan_rows_i]
    
    plt.scatter(X_nd[:,0],X_nd[:,1],c=['g' if x else 'r' for x in y_n],alpha=0.3)
    plt.xlabel(f0)
    plt.ylabel(f1)
    plt.show()

 
#=============================================================================
if __name__ == '__main__':

    avg_file = 'example/oface.120.avg.v2.csv'
    #avg_file = 'example/avg_response_time.csv'
    #avg_file = 'example/affdex2.120.v2.avg.csv'
    
    df = loader.load(avg_file)
    df_W = df[(df['filetype'] == 'W-B') | (df['filetype'] =='W-T')].copy()
    df_W['y'] = df_W['filetype'].apply(lambda s: 'T' in s)
    ml(df_W)

print('end')
