#!/usr/bin/env python3
"""
------------------------------------------------------------------------
calculates cluster distributions over files, puts in csv
      input: 
        data/all_frames_wclust.pkl.xz
      output: 
        data/cluster_dist_s2_w.csv
        data/cluster_dist_s2_w_i.csv
------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

datafile = 'data/all_frames_wclust.pkl.xz' # with AU6_AU12 clusters
#datafile = 'test_wclust.csv'    
CONFIDENCE_TOL = 0.90 # only use data with conf > this

print('...loading data')
if 'pkl' in datafile:
    df = pd.read_pickle(datafile)
else:
    df = pd.read_csv(datafile, skipinitialspace=True) 
df = df[df['confidence'] >= CONFIDENCE_TOL]
df = df[df['segment'] == 'S2']

cluster_name2i = {'duchenne':0,'neutral':1,'strong duchenne':2,'6 only':3,'polite':4}
cluster_names = list(cluster_name2i.keys())
for cluster_name in cluster_names:
    df[cluster_name] = (df['AU06_AU12_cluster'] == cluster_name2i[cluster_name]).astype(int)
df['y'] = df['filetype'].apply(lambda s: 'T' in s)
usecols = ['Filename','y'] + cluster_names
df_W = df[(df['filetype'] == 'W-B') | (df['filetype'] =='W-T')][usecols].copy()
g = df_W.groupby(['Filename'])
df_dist = g.mean()
df_dist.to_csv('data/cluster_dist_s2_w.csv')

usecols = ['Filename','filetype','y'] + cluster_names
df = df[usecols]
g = df.groupby(['Filename','filetype'])
df_dist = g.mean()
df_dist.to_csv('data/cluster_dist_s2_w_i.csv')

print('COMPLETE')