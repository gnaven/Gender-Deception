import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

pd.options.display.max_rows = None
pd.options.display.max_columns = None
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import metrics

#------------------------------------------------------------

def ml(dfin):
    """ Runs either logistic regression or KNN through sklearn using both 
        dev and test set. The hyperparameter is set using a dev set. Post
        hyperparameter training uses train and dev set for training.
        
        :df     -pandas df - must have a 'y' column that is bool, all other
                 columns will be cast to float and used as features.        
        :model  -'LOGISTIC' or 'KNN'
        """

    df = pd.DataFrame(dfin, dtype=float)
    assert('y' in df.columns)
    df = df.dropna()
    
    if df.shape[0] < 20:
        N_FOLDS = 2
    else:
        N_FOLDS = 10
    ACC_THRESH = 0.01 # dev set accuracy must be x% better to use new param   

    models = ['KNN','LOGISTIC','TREE']
    for model in models:
        print('\nMODEL: ', model)

        if model == 'LOGISTIC':
            c_l = [0.01,0.03, 0.1, 0.3, 1,3,10,30,100,300,1000,3000,10000]
        elif model == 'KNN':       
            c_l = [50,40,35,30,25,20,18,15]
        else:
            c_l = [3,4,5,6,7,8]
        regularizer = 'l1'
            
        X_nd = df.drop('y',axis=1).values
        #X_nd = scale(X_nd) # magnitude has useful info?
        y_n = df['y'].values.astype(bool)
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
                if model == 'LOGISTIC': 
                    clf = linear_model.LogisticRegression(penalty=regularizer,C=c)
                elif model == 'KNN':           
                    clf = KNeighborsClassifier(n_neighbors=c, metric='euclidean',weights='uniform')
                else:
                    clf = tree.DecisionTreeClassifier(max_leaf_nodes=c)
                clf.fit(X_nd[sub_train], y_n[sub_train])
                y_pred = clf.predict(X_nd[dev])
                acc = metrics.accuracy_score(y_pred,y_n[dev])
                if(acc > best_acc + ACC_THRESH):
                    best_acc = acc
                    best_c = c
    
            # retrain with all train data and best_c
            print('fold:',i,' best c:',best_c, ' dev:%.2f' % best_acc, ' dev_ones:%.2f' % (y_n[dev].sum()/len(dev)),end='')
            if model == 'LOGISTIC': 
                clf = linear_model.LogisticRegression(penalty=regularizer,C=best_c)
            elif model == 'KNN':           
                clf = KNeighborsClassifier(n_neighbors=best_c, metric='euclidean',weights='uniform')
            else:
                clf = tree.DecisionTreeClassifier(max_leaf_nodes=best_c)
            clf.fit(X_nd[train],y_n[train])
            y_pred = clf.predict(X_nd)
            acc_test_a[i] = metrics.accuracy_score(y_pred[test],y_n[test])
            acc_train_a[i] = metrics.accuracy_score(y_pred[train],y_n[train])
            print(' test:%.2f' % acc_test_a[i], ' train:%.2f' % acc_train_a[i])
        print('Avg test acc:%.3f' % acc_test_a.mean(),'Avg train acc:%.3f' % acc_train_a.mean())    

#----------------------------------------------------------------------

def plot(f0,f1,df):
    df = df[[f0,f1,'y']]
    df = df.dropna()
    
    plt.scatter(df[f0],df[f1],c=['g' if x else 'r' for x in df['y']],alpha=0.3)
    plt.xlabel(f0)
    plt.ylabel(f1)
    plt.show()

#=============================================================================
if __name__ == '__main__':

    datafile = 'cluster_dist_s2_w.csv'    
    df = pd.read_csv(datafile, skipinitialspace=True) 
    
    cluster_name2i = {'duchenne':0,'neutral':1,'strong duchenne':2,'6 only':3,'polite':4}
    cluster_names = list(cluster_name2i.keys())
    print('df.shape:',df.shape)
    for c in df.columns:
        print(c,end=',')
    usecols = ['y'] + cluster_names
    #usecols = ['y'] + ['strong duchenne','6 only']
    ml(df[usecols])
    #plot('strong duchenne','6 only',df)
print('end')
