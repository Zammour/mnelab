#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:09:38 2020

@author: zammour
"""


from __future__ import print_function
import warnings
from collections import defaultdict
import struct
import xml.etree.ElementTree as ETree




def read_raw_smr(fname, *args , **kwargs):
    """Read Spike2 file
    
    Parameters
    ----------
    fname : str
        Name of the smr file
        
    Returns
    -------
    raw : mne.io.Raw
        smr file data
    """
    
    
    import numpy as np
    import neo.io
    import mne
    
    # Read file
    
    reader = neo.io.Spike2IO(filename=fname)
    
    # Extract data from file
    
    segment = reader.read_segment(lazy=False, cascade=True,)
    
    # Info data
    
    n_channels = len(segment.analogsignals)
    sfreq = segment.analogsignals[0].sampling_rate
    channel_types = ['eeg']*len(segment.analogsignals) + ['stim']*len(segment.events)
    long = min(len(segment.analogsignals[f]) for f in range(len(segment.analogsignals)))

    # Extract signals & create channels_names
    
    data = np.empty((long,0))
    channel_names = []
    for i in range(len(segment.analogsignals)):
        channel_names.append(segment.analogsignals[i].name)
        data_app = np.array(segment.analogsignals[i])*10**-6
        if len(data_app) != long : data_app = data_app[:long]
        data = np.append(np.array(data), data_app, axis = 1)
        
    for k in range(len(segment.events)):
        channel_names.append('STIM_{}'.format(k))
            
    # Create STIM channels from extracted events
    
    events = np.empty((long,0))
    for j in range(len(segment.events)):
        event = np.array(segment.events[j])
        arr = np.zeros((long,1))
        arr[(event*sfreq).astype(int)] = 2**j
        events = np.append(events, arr, axis = 1)
                
    info = mne.create_info(channel_names, sfreq, channel_types)
                
    data = np.append(data , events, axis=1).T
                
    raw = mne.io.RawArray(data, info)
    
    return raw
