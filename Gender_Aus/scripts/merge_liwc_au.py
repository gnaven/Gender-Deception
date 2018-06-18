#datafile = 'all_frames.pkl.xz' # FG/UBICOMP data N=151
#datafile = '../data/all_frames_wclust.pkl.xz' # with AU6_AU12 clusters
#datafile = '../data/all_frames.pkl.xz' # with AU6_AU12 clusters
datafile = '../data/all_frames_clust.km.q.pkl.xz'
liwc_datafile = '../data/liwc_data.csv'
CONFIDENCE_TOL = 0.90 # only use data with conf > this

#-----------------
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.3f}'.format
from IPython.display import display
import matplotlib.pyplot as plt
import compare

print('...loading AU data')
if 'pkl' in datafile:
    df = pd.read_pickle(datafile)
else:
    df = pd.read_csv(datafile, skipinitialspace=True) 
    
df = df[df['confidence'] >= CONFIDENCE_TOL]
df = df[(df['filetype'] == 'W-T') | (df['filetype'] == 'W-B') ]
n_tot = df.shape[0]

print('AU COLUMNS: ', df.columns, '\n')
print('n:',df.shape[0])

AUS = [x for x in df.columns if '_r' in x]


print('...loading LIWC data')
if 'pkl' in liwc_datafile:
    df_liwc = pd.read_pickle(liwc_datafile)
else:
    df_liwc = pd.read_csv(liwc_datafile, skipinitialspace=True) 

df_liwc['duration'] = df_liwc['ans_end_time'] - df_liwc['ans_start_time']
df_liwc['total words'] = df_liwc['Word_in_dict'] + df_liwc['Word_not_in_dict']

# note, removed sweaar
LIWC = ['Function words','Total pronouns','Personal pronouns','I','We','You',
        'SheHe','They','Impersonal pronouns','Articles','Common Verbs',
        'Auxiliary verbs','Past tense','Present tense','Future tense',
        'Adverbs','Prepositions','Conjunctions','Negations','Quantifiers',
        'Numbers','Social','Family','Friends','Humans','Affective',
        'Positive emotion','Negative Emotion','Anxiety','Anger','Sadness',
        'Cognitive','Insight','Causation','Discrepancy','Tentativeness',
        'Certainty','Inhibition','Inclusion','Exclusion','Perceptual','Seeing',
        'Hearing','Feeling','Biological','Body','Health','Sexual','Ingestion',
        'Relativity','Motion','Space','Time','Work','Achievement','Leisure',
        'Home','Money','Religion','death@Death','Assent','Non-fluencies','Fillers']
LIWC2 = LIWC + ['duration','total words']

# to normalize by total words
#df[LIWC] = df[LIWC].values/(df['total words'].values[:,np.newaxis])
print('LIWC COLUMNS: ', df_liwc.columns, '\n')

print('liwc n:',df_liwc.shape[0])
print('liwc # files = ', df_liwc['root'].nunique())

g = df.groupby(['Filename','question'])
df_au_avg = g[AUS].mean().reset_index()
df_au_avg['root'] = df_au_avg['Filename'].apply(lambda x: '-'.join(x.split('-')[:6]))

# merge liwc columns with au avg
df_merge = pd.merge(df_au_avg, df_liwc, how='inner', left_on=['root','question'], 
                    right_on=['root','question_num'])
df_merge.to_csv('../data/liwc_au.csv')