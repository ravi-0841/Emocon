#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:35:24 2019

@author: ravi
"""

import pyworld as pw
import scipy.io.wavfile as scwav
import numpy as np
from helper import generate_context, smooth_contour
from variables import global_dict
import scipy.signal as scisig
import librosa

def get_features(file_path):
    fs,src                      = scwav.read(file_path)
    src                         = np.asarray(src, np.float64)
    src                         = 1.0*((src - np.min(src)) \
                                       / (np.max(src) - np.min(src))) - 0.0
                                       
    filterbanks                 = librosa.filters.mel(fs, 1024, n_mels=global_dict['n_mels'], \
                                                fmin=0.0, fmax=None, htk=True, norm=1)
    
    f0_src, straight_src, ap    = pw.wav2world(src, fs, \
                                            frame_period=int(1000*global_dict['window_len']))
    
    (r,c)                       = np.where(straight_src<=0)
    for (i,j) in zip(r.tolist(), c.tolist()):
        straight_src[i,j]       = global_dict['underflow']
    
    f0_src                      = scisig.medfilt(f0_src, kernel_size=3)
    ec_src                      = np.sqrt(np.sum(np.square(straight_src),1))
    
    f0_context_src              = generate_context(f0_src.reshape(-1,1), \
                                                       axis=1, context=global_dict['context'])
    ec_context_src              = generate_context(ec_src.reshape(-1,1), \
                                                       axis=1, context=global_dict['context'])
    
    f0_context_src              = smooth_contour(f0_context_src, global_dict['smooth_param'])
    ec_context_src              = smooth_contour(ec_context_src, global_dict['smooth_param'])
    
    """
    Features extraction
    """
    
    straight_src                = straight_src.T
    straight_cep_src            = np.matmul(filterbanks, straight_src)
    
    mean_f0_src                 = np.mean(f0_src[np.where(f0_src>1.0)[0]])
    std_f0_src                  = np.std(f0_src[np.where(f0_src>1.0)[0]])
    nmz_f0_src                  = (f0_src - mean_f0_src) / std_f0_src
    straight_cep_f0_feat_src    = []
    straight_cep_ec_feat_src    = []
    for frames in range(straight_cep_src.shape[1]):
        straight_cep_f0_src     = list(straight_cep_src[:,frames])
        straight_cep_ec_src     = list(straight_cep_src[:,frames])
        
        straight_cep_f0_src.append(nmz_f0_src[frames])
        straight_cep_f0_src.extend(f0_context_src[frames].tolist())
        
        straight_cep_ec_src.append(nmz_f0_src[frames])
        straight_cep_ec_src.extend(ec_context_src[frames].tolist())
        
        straight_cep_f0_feat_src.append(straight_cep_f0_src)
        straight_cep_ec_feat_src.append(straight_cep_ec_src)
    
    straight_cep_f0_feat_src    = np.asarray(straight_cep_f0_feat_src, np.float32)
    straight_cep_ec_feat_src    = np.asarray(straight_cep_ec_feat_src, np.float32)
    
    f0_nz_idx                   = np.where(f0_src>1.0)[0]
    return_vars                 = {}
    return_vars['straight']     = straight_src
    return_vars['aperiod']      = ap
    return_vars['f0']           = f0_src
    return_vars['ec']           = ec_src
    return_vars['f0_feats']     = straight_cep_f0_feat_src
    return_vars['ec_feats']     = straight_cep_ec_feat_src
    return_vars['nz_idx']       = f0_nz_idx
    return return_vars