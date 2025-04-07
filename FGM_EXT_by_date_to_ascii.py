# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 22:32:43 2025

@author: Test
"""


import shutil

import sys

import numpy as np

from datetime import date, datetime, time

#craft = str(sys.argv[1])
#year = str(sys.argv[2])
#month = str(sys.argv[3])
#day = str(sys.argv[4])

craft = 'C1'
year = '2001'
month = '03'
day = '22'


import os, fnmatch


def find_BS_file(date, path):

    pattern_B = craft + '_' + date + '_B.BS'

    pattern_K = craft + '_' + date + '_K.BS'

    pattern_A = craft + '_' + date + '_A.BS'

    for root, dirs, files in os.walk(path):

        for name in files:

            if fnmatch.fnmatch(name, pattern_B):
                return(os.path.join(root, name))

            elif fnmatch.fnmatch(name, pattern_K):
                return(os.path.join(root, name))

            elif fnmatch.fnmatch(name, pattern_A):
                return(os.path.join(root, name))

           
dumpdate = year[2:4] + month + day
print(dumpdate)

BS_filepath =  '/cluster/data/raw/' + year + '/' + month + '/'
print(BS_filepath)

BS_file_location = find_BS_file(dumpdate, BS_filepath)

print(BS_file_location)

def s16(val):
    
    value = int(val)
    
    return -(value & 0x8000) | (value & 0x7fff)

def packet_decoding_even(ext_bytes):
    
    packig = ext_bytes[68:7180]
    
    x_vals = []
    
    y_vals = []
    
    z_vals = []
    
    range_vals = []
    
    reset_vals = []
    
    reset_vals_hex = []
    
    
    for i in np.arange(0,len(packig), 16):
        
        byte_num = i/2
        
        if byte_num < 3552:
            
            x = s16(int(packig[i:i+4],16))
            
            y = s16(int(packig[i+4:i+8],16))
            
            z = s16(int(packig[i+8:i+12],16))
            
            range_val = s16(int(packig[i+12],16))
            
            reset_val = s16(int(packig[i+13:i+16]  + '0',16))
            
            reset_val_hex = packig[i+13:i+16]
            
        else:

            x= s16(int(packig[i:i+4],16))
            
            y = s16(int(packig[i+4:i+8],16))

            z = 'af'
            
            range_val = 'af'
            
            reset_val = 'af'
            
            reset_val_hex = 'af'
            
        
        x_vals.append(x)
        
        y_vals.append(y)
        
        z_vals.append(z)
        
        range_vals.append(range_val)
        
        reset_vals.append(reset_val)
        
        reset_vals_hex.append(reset_val_hex)
        
    
    df_p = pd.DataFrame(zip(reset_vals_hex, reset_vals, range_vals, x_vals, y_vals, z_vals))
    
    df_p.columns = ['reset_hex', 'reset', 'resolution', 'x', 'y', 'z']    
         
    return df_p


def packet_decoding_odd(ext_bytes):
    
    packig = ext_bytes[76:7180]
    
    partial_vec_end = ext_bytes[68:76]
    
    x_vals = ['bef']
    
    y_vals = ['bef']
    
    z_vals = [s16(int(partial_vec_end[0:4],16))]
    
    range_vals = [s16(int(partial_vec_end[4],16))]
    
    reset_vals = [s16(int(partial_vec_end[5:8] + '0',16))]
    
    reset_vals_hex = [partial_vec_end[5:8]]
    
    for i in np.arange(0, len(packig), 16):
                
        x = s16(int(packig[i:i+4],16))
            
        y = s16(int(packig[i+4:i+8],16))
            
        z = s16(int(packig[i+8:i+12],16))
            
        range_val = s16(int(packig[i+12],16))
            
        reset_val = s16(int(packig[i+13:i+16] + '0',16))
            
        reset_val_hex = packig[i+13:i+16]

        x_vals.append(x)
        
        y_vals.append(y)
        
        z_vals.append(z)
        
        range_vals.append(range_val)
        
        reset_vals.append(reset_val)
        
        reset_vals_hex.append(reset_val_hex)
    
    df_p = pd.DataFrame(zip(reset_vals_hex, reset_vals, range_vals, x_vals, y_vals, z_vals))
    
    df_p.columns = ['reset_hex', 'reset', 'resolution', 'x', 'y', 'z']    
         
    return df_p

