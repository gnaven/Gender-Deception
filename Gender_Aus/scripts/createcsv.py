#!/usr/bin/env python3
"""
------------------------------------------------------------------------
Concatenates a set of openface .txt files into a single df file. 
Only landmark columns are ignored. New columns are created for: Filename, 
filetype (XXX, I-T, I-B, W-T, W-B), and segment (xx, S1, S2, S3).

example usage: 

  ./createcsv.py -i 'OpenFace/*.csv' -l 'list.csv' -o 'all_frames.pkl.xz'
  ./createcsv.py -i 'example/*.txt' -l 'example/list.csv' -o 'all_frames.csv'

------------------------------------------------------------------------
"""
from __future__ import print_function

import pandas as pd
import numpy as np
import csv
import glob
import os 
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

S3_DURATION = 180

#-------------------------------------------------------------------------------
def read_time_list(fname):
    """ reads in the S1 and S2 times from csv file fname 
        returns time_interval_dict with root filename and list of file contents
    """
    
    with open(fname, 'rt') as f:
        time_interval_dict = {}
        try:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                fname = row[0]
                root = fname.split('.')[0]                
                time_interval_dict[root] = [float(row[1]),float(row[2])]
        finally:
            f.close()

    assert(header[0]=='Filename' or header[0]=='root')
    assert(header[1]=='S1' or header[1]=='s1')
    assert(header[2]=='S2' or header[2]=='s2')
    print(time_interval_dict)
    
    return time_interval_dict


#-------------------------------------------------------------------------------
def do_all(args):
    time_interval_dict = read_time_list(args.l)
    files = glob.glob(args.i)
    
    frame = pd.DataFrame()
    
    
    features = ['timestamp', 'confidence', 'success', 'pose_Tx', 'pose_Ty', \
                'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', \
                'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', \
                'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', \
                'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', \
                'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
                'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', \
                'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    #df = pd.concat((pd.read_csv(f, usecols=features) for f in files))
    df_list =[]
    
    
    for file in files:
        df = pd.read_csv(file, usecols=features, skipinitialspace=True)
        df.columns = [w.strip() for w in list(df.columns)]
        df.insert(0,'Filename',os.path.basename(file))
        df.insert(1,'filetype','XXX')
        df.insert(2,'segment','XX')
        
        df.loc[df.Filename.str.contains('-I-T-'),'filetype'] = 'I-T' 
        df.loc[df.Filename.str.contains('-I-B-'),'filetype'] = 'I-B' 
        df.loc[df.Filename.str.contains('-W-T-'),'filetype'] = 'W-T' 
        df.loc[df.Filename.str.contains('-W-B-'),'filetype'] = 'W-B' 
        
        basename = os.path.basename(file)
        root = '-'.join(basename.split('-')[0:6])
        if root in time_interval_dict:
            tS1,tS2 = time_interval_dict[root]
            tS3 = df['timestamp'].max() - S3_DURATION
            df.loc[((df['timestamp'] >= tS1) & (df['timestamp'] < tS2)),'segment'] = 'S1' 
            df.loc[((df['timestamp'] >= tS2) & (df['timestamp'] < tS3)),'segment'] = 'S2' 
            df.loc[(df['timestamp'] >= tS3),'segment'] = 'S3'
            df_list.append(df)
        else:
            print('no S1 S2 data exists for ' + file + '; skipping')


    
    df = pd.concat(df_list)
    
    if '.csv' in args.o:
        df.to_csv(args.o,index=False)
    else:
        df.to_pickle(args.o)                   
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input files', type=str, default='data/*.csv')
    parser.add_argument('-l', help='S1 S2 time list csv file', type=str, 
                        default='data/list.csv')
    parser.add_argument('-o', help='output file', type=str, default='data/all_frames.pkl.xz')
    args = parser.parse_args()
    
    print('args: ',args.i, args.l, args.o)

    if not os.path.isdir(os.path.dirname(args.i)):
        logging.error(args.i + ' directory does not exist')
        exit()
    if not os.path.exists(args.l):
        logging.error(args.l + ' file does not exist.\n')
        exit()
    
    do_all(args)
    print('COMPLETE')
