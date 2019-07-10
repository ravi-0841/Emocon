#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:12:27 2019

@author: ravi
"""
import numpy as np


def generate_context(features, axis=0, context=1):
    backward = features.copy()
    forward = features.copy()
    if axis==0:
        for c in range(context):
            backward = np.roll(backward, 1, axis=1)
            forward = np.roll(forward, -1, axis=1)
            backward[:,0] = 0
            forward[:,-1] = 0
            features = np.concatenate((backward, features, forward), axis=axis)
            
    else:
        for c in range(context):
            backward = np.roll(backward, 1, axis=0)
            forward = np.roll(forward, -1, axis=0)
            backward[0,:] = 0
            forward[-1,:] = 0
            features = np.concatenate((backward, features, forward), axis=axis)

    return features

def smooth(x,window_len=7,window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y

def smooth_contour(data, window=3):
    for i in range(data.shape[0]):
        x = smooth(data[i], window)
        data[i] = x[window-1:-1*(window-1)]
    return data