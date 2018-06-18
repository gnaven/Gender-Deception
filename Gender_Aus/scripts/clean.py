#!/usr/bin/env python3
"""
------------------------------------------------------------------------
this program will remove all OpenFace files for which there is no
corresponding transcript file
------------------------------------------------------------------------
"""
import glob
import os

tfiles = glob.glob('data/transcripts/*.csv')
ofiles = glob.glob('data/OpenFace/*.csv')

tbasenames = [os.path.basename(tfile) for tfile in tfiles]
for ofile in ofiles:
    found = False
    for tbasename in tbasenames:
        if os.path.splitext(tbasename)[0] in ofile:
            found = True
            break
    if not found:
        print('no transcipt for ',ofile)
        os.remove(ofile)
