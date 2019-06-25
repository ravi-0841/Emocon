#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:49:33 2019

@author: ravi
"""

import pyworld as pw
import numpy as np
from helper import smooth
from variables import global_dict

def generate_waveform(predict_dict, features_dict):
    pred_f0                     = predict_dict['f0_predictions']
    pred_ec                     = predict_dict['ec_predictions']
    f0_src                      = features_dict['f0']
    ec_src                      = features_dict['ec']
    nz_idx                      = features_dict['nz_idx']
    straight_src                = features_dict['straight']
    ap                          = features_dict['aperiod']
    smooth_param                = global_dict['smooth_param']
    
    trans_f0                    = smooth(pred_f0.reshape(-1,), \
                                         window_len=smooth_param)
    mean_smooth_f0              = np.zeros((f0_src.shape[0],1))
    
    mean_smooth_f0[nz_idx]      = trans_f0[smooth_param-1:-1*(smooth_param-1)].reshape(-1,1)
    
    trans_ec                    = smooth(pred_ec.reshape(-1,), \
                                         window_len=smooth_param)
    new_ec                      = trans_ec[smooth_param-1:-1*(smooth_param-1)].reshape(-1,1)
    
    z_idx                       = np.where(new_ec<=0)[0]
    ec_src                      = ec_src.reshape(-1,1)
    new_ec[z_idx]               = ec_src[z_idx]

    ratio_ec                    = np.divide(new_ec.T, ec_src.T)
    straight_src                = np.multiply(straight_src, \
                                    np.matlib.repmat(ratio_ec, 513, 1))
    straight_src                = straight_src.T.copy(order='C')
    
    recon_sig                   = pw.synthesize(mean_smooth_f0.reshape((-1,)), \
                                                straight_src, ap, global_dict['fs'], \
                                                frame_period=int(1000*global_dict['window_len']))
    recon_sig                   = 2 * ((recon_sig - np.min(recon_sig)) \
                                       / (np.max(recon_sig) - np.min(recon_sig))) - 1
    return {'data':recon_sig, 'fs':global_dict['fs']}
