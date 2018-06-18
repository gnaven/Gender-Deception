#!/usr/bin/env python3
"""
------------------------------------------------------------------------
Using a k-means cluster file, adds a cluster column to the OpenFace data 
df and writes as a new pkl file. Low confidence dataframes get no cluster.

input:
  'data/face_clusters_AU06_r_AU12_r_5.csv'
  'data/all_frames.pkl.xz'
  
output:
  openface_data_wclustercol.pkl.xz
------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

CONFIDENCE_TOL = 0.9
#clusterfile = 'data/face_clusters_AU06_r_AU12_r_5.csv'
clusterfile = '../data/bmm_clusters_5_iternum1_sorted.csv'

clusterfile = '../data/bmm_clusters_9_iternum2.csv'
datafile = '../data/all_frames.pkl.xz'
#outfile = 'data/all_frames_clust.pkl.xz'
outfile = '../data/all_frames_clust.bmm9.pkl.xz'
newcolname = 'AU06_AU12_cluster'

#clusterfile = 'output/face_clusters_duchenne_neutral_strongduchenne_6only_polite_4.csv'
#datafile = '../data/cluster_dist_s2_w.csv'
#outfile = 'output/cluster_dist4wi.csv'
#newcolname = 'clusterdist4'

#clusterfile = 'face_clusters_AU06_r_AU12_r_5.csv'
#datafile = 'test.pkl.xz'
#outfile = 'test_wclust.csv'

if '.csv' in datafile:
    df = pd.read_csv(datafile, skipinitialspace=True)
else:
    df = pd.read_pickle(datafile)

if 'face_clusters' in clusterfile:
    # this is a kmeans cluster
    print('...calculating predictions from kmeans clusters')
    clusters_kd = pd.read_csv(clusterfile, skipinitialspace=True)
    k,d = clusters_kd.shape
    feats = list(clusters_kd.columns)

    dists = []
    for ki in range(k):
        dist = ((clusters_kd.iloc[ki,:] - df[feats])**2).sum(axis=1)
        dists.append(dist)
    dmat = np.vstack(dists)
    df[newcolname] = np.argmin(dmat, axis = 0)
    #z = df.apply(get_cluster, axis=1)
elif 'bmm' in clusterfile:
    # this is a BetaMixture cluster
    import beta
    print('...calculating predictions from BetaMixture clusters')
    bmm = beta.BetaMixture()
    bmm.read_bm(clusterfile)
    feats = ['AU06_r','AU12_r']
    X_nd = beta.Beta.rescale_data(df[feats].values.reshape((df.shape[0],2))/5)
    df[newcolname] = bmm.predict(X_nd)

# we will determine which CONF to use downstream
#if 'confidence' in df.columns:
#    lowconf_b = df['confidence'] < CONFIDENCE_TOL
#    df.ix[lowconf_b, newcolname] = np.nan

print('...writing outfile')
if 'confidence' in df.columns:
    usecols = ['Filename', 'filetype', 'confidence', 'segment', 'timestamp', newcolname]
else:
    usecols = ['Filename', 'y', newcolname]
    
if '.csv' in outfile:
    df[usecols].to_csv(outfile)
else:
    df[usecols].to_pickle(outfile)
