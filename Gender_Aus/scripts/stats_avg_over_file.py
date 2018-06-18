#!/usr/bin/env python3
"""----------------------------------------------------
after specifying a splitting field and a compare field,
splits data into HI and LO by the splitting field, 
then compares the compare field between T and F separately for HI and then for LO.
provides t-test, MW U test, and Cohen's d

Does a brute force search of the above for all possible pairings of a splitting 
field with a compare field. 

"""

import compare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.max_rows = None
pd.options.display.max_columns = None

def do_stats(df_in, splitting_fields, compare_fields):
    """ For every splitting_field, partitions data into high and low groups
    by dividing at the value closest to the median that is not equal to either
    max or min. Then performs statistical comparisons for each compare_field. 
    A sorted df with means, t-test, MWW, and Cohen's d is returned.
    """
    #stats = []
    #for feature in df.columns.drop('y'):
    
    stats = []
    for sfield in splitting_fields:
        for cfield in compare_fields:
            print('PROCESSING: ' + sfield + ' ' + cfield)
            if sfield == cfield:
                continue
            if (sfield not in df_in.columns) or (cfield not in df_in.columns):
                continue
            df = df_in[['truth_val','root',sfield,cfield]].dropna() 
            amax = df[sfield].max()
            amin = df[sfield].min()
            amedian = df[sfield].median()
            if amin == amax:
                print('WARNING: no variation in splitting field ' + sfield)
            if amedian == amin:
                amedian += 0.0001
            elif amedian == amax:
                amedian -= 0.0001
            
            hi_quant = df[sfield].quantile(.75)
            lo_quant = df[sfield].quantile(.25)
                
            #HI = df[['root','truth_val',cfield]][df[sfield] > df[sfield].quantile(.75)].groupby('root').mean()
            #LO = df[['root','truth_val',cfield]][df[sfield] <= df[sfield].quantile(.25)].groupby('root').mean()
            HI = df[['root','truth_val',cfield]][df[sfield] > amedian].groupby('root').mean()
            LO = df[['root','truth_val',cfield]][df[sfield] <= amedian].groupby('root').mean()
            comp = compare.Compare(HI[cfield][HI['truth_val']==1],
                                   HI[cfield][HI['truth_val']==0],
                                   label_=cfield,
                                   x_label_=sfield+'_HIQ_T',y_label_=sfield+'_HIQ_B')
            stats_f = comp.calc_stats()
            #comp.plot_all()
            stats.append([sfield + '_HI',cfield] + list(stats_f))

            comp = compare.Compare(LO[cfield][LO['truth_val']==1],
                                   LO[cfield][LO['truth_val']==0],
                                   label_=cfield,
                                   x_label_=sfield+'_LOQ_T',y_label_=sfield+'_LOQ_B')
            stats_f = comp.calc_stats()
            #comp.plot_all()
            stats.append([sfield + '_LO',cfield] + list(stats_f) + [hi_quant, lo_quant])


    print(stats)
    #df_stats = pd.DataFrame(stats,columns=['sfield','cfield','mean T','mean B','t-test','MWW','Cohens d'])
    df_stats = pd.DataFrame(stats,columns=['sfield','cfield','mean T','mean B','t-test','MWW','Cohens d', 'HI quant', 'LO quant'])
    df_stats.sort_values(by=['MWW'],ascending=True,inplace=True)
    display(df_stats)        
    return df_stats

def do_boxplots(df_in, splitting_fields, compare_fields):
    """ For every splitting_field, partitions data into high and low groups
    by dividing at the value closest to the median that is not equal to either
    max or min. Then performs statistical comparisons for each compare_field. 
    A sorted df with means, t-test, MWW, and Cohen's d is returned.
    """
    #stats = []
    #for feature in df.columns.drop('y'):
    
    stats = []
    for sfield in splitting_fields:
        for cfield in compare_fields:
            print('PROCESSING: ' + sfield + ' ' + cfield)
            if sfield == cfield:
                continue
            if (sfield not in df_in.columns) or (cfield not in df_in.columns):
                continue
            df = df_in[['root',sfield,cfield]].dropna()
            amax = df[sfield].max()
            amin = df[sfield].min()
            amedian = df[sfield].median()
            if amin == amax:
                print('WARNING: no variation in splitting field ' + sfield)
            if amedian == amin:
                amedian += 0.0001
            elif amedian == amax:
                amedian -= 0.0001
                
            HI = df[['truth_val','root',cfield]][df[sfield] > amedian].groupby('root').mean()
            LO = df[['root',cfield]][df[sfield] <= amedian].groupby('root').mean()
            
            comp = compare.Compare(HI,LO,label_=cfield,
                                   x_label_=sfield+'_HI',y_label_=sfield+'_LO')
            stats_f = comp.calc_stats()
            #plt.boxplot((HI,LO))
            title = cfield + ' partitioned by ' + sfield + '\nMWW $\mu$ = ' 
            title += '' # TODO
            HI_label = sfield + '_HI\n' 
            HI_label += cfield + ' $\mu$ = ' + '%.2f' % stats_f[0] 
            LO_label = sfield + '_LO\n' 
            LO_label += cfield + ' $\mu$ = ' + '%.2f' % stats_f[1] 
            plt.xticks([1,2],[HI_label, LO_label])
            #f = comp.plot_all()
            #plt.savefig(compare.label + '_plot.png',dpi=70)
            #f.savefig(comp.label + '_plot.png',dpi=70)
            plt.show()            


if __name__ == '__main__':
    print('PROGRAM starting')
    #splitting_fields = FIELDS_CODES + FEATURES + FEATURES_ADV + FEATURES_DISC +\
    #    ['discDmind3','Aggressive_care_scoren']
    #compare_fields=splitting_fields
    
    df = pd.read_csv('../data/liwc_au.csv', skipinitialspace=True)
    AUS = [x for x in df.columns if '_r' in x]
    LIWC = ['Function words','Total pronouns','Personal pronouns','I','We',
            'You','SheHe','They','Impersonal pronouns','Articles',
            'Common Verbs','Auxiliary verbs','Past tense','Present tense',
            'Future tense','Adverbs','Prepositions','Conjunctions',
            'Negations','Quantifiers','Numbers','Social','Family','Friends','Humans','Affective',
            'Positive emotion','Negative Emotion','Anxiety','Anger','Sadness',
            'Cognitive','Insight','Causation','Discrepancy','Tentativeness',
            'Certainty','Inhibition','Inclusion','Exclusion','Perceptual','Seeing',
            'Hearing','Feeling','Biological','Body','Health','Sexual','Ingestion',
            'Relativity','Motion','Space','Time','Work','Achievement','Leisure',
            'Home','Money','Religion','death@Death','Assent','Non-fluencies',
            'Fillers', 'duration', 'total words',]
    
    df_statsa = do_stats(df, 
                         splitting_fields=AUS,
                        compare_fields=LIWC)

    df_statsb = do_stats(df,
                         splitting_fields=LIWC,
                         compare_fields=AUS)    
        
    df_stats = pd.concat([df_statsa, df_statsb])        
    df_stats.to_csv('../data/liwc_au_file_quantile_stats.csv')
    
    
    #do_boxplots(df3,
    #         splitting_fields=FEATURES_ADV,
    #         compare_fields=KEY_DETERMINERS)             
                
   
    print('PROGRAM complete')