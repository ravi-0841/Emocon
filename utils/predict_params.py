#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:40:22 2019

@author: ravi
"""

from predict_pitch import predict_pitch
from predict_energy import predict_energy
from variables import global_dict

def predict_params(features_dict, emo):
    predicted_f0         = predict_pitch(features_dict['f0_feats'], \
                                    global_dict['context'], emo)
    predicted_ec         = predict_energy(features_dict['ec_feats'], \
                                     global_dict['context'], emo)

    return {'f0_predictions':predicted_f0, 'ec_predictions':predicted_ec}