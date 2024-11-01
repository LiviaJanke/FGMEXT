# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:35:50 2024

@author: Livia
"""
#
#
#
#
#
#

# Description

# This program returns cleaned, decoded and calibrated Extended Mode data for a given craft and index (entry number)
# Also returns a data file containing the lines needed to run ext2tvec for the generated dataset


#
#
#
#
#
# craft and index variables (to change)


craft = 'C1'

index = 400
#
#
#
#
#
#
#


# the rest of the program 

# importing packages

import sys

# add your library folder to path

sys.path.append('C:/FGM_Extended_Mode/Lib')

import numpy as np

from fgmfiletools import fgmsave

import matplotlib.pyplot as plt

import pandas as pd

import os, fnmatch

from datetime import datetime, timedelta

# defining functions

#s16 changes unsigned 16 bit hex numbers to signed (positive and negative)

def s16(val):
    
    value = int(val)
    
    return -(value & 0x8000) | (value & 0x7fff)


# decodes the given packet as if it were 'even'
# characters 0 to 67 of the payload are the CDDS header (34 bytes)
# Only want to decode the data itself, so discarding the header 
# variable 'packig' contains the 7112 characters (3556 bytes) i

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


plt.rcParams['lines.linewidth'] = 0
plt.rcParams['lines.marker'] = '.'
plt.rcParams['lines.markersize'] = 0.5

def quickplot(titletext,xlabeltext,ylabeltext):
    plt.subplots(5,1,sharex=True,height_ratios=[2,2,2,2,1])
    plt.subplot(5,1,1);plt.plot(t,x,label='x');plt.grid();plt.legend();plt.ylabel(ylabeltext)
    plt.subplot(5,1,2);plt.plot(t,y,label='y');plt.grid();plt.legend();plt.ylabel(ylabeltext)
    plt.subplot(5,1,3);plt.plot(t,z,label='z');plt.grid();plt.legend();plt.ylabel(ylabeltext)
    b = np.sqrt(x**2+y**2+z**2)
    plt.subplot(5,1,4);plt.plot(t,b,label='B');plt.grid();plt.legend();plt.ylabel(ylabeltext)
    plt.subplot(5,1,5);plt.plot(t,r,label='range');plt.grid();plt.legend()
    plt.xlabel(xlabeltext)
    plt.suptitle(titletext,y=0.94)
    plt.show()
    return

def make_t(ext_entry, t_spin, ext_exit, x):
    t = []
    for i in range(0,len(x)):
        t.append(ext_entry + timedelta(seconds=i*t_spin))
    return t



def find_cal_file(pentry, pexit, path):

    pattern_entry = '* __' + pentry + '*'
    pattern_exit = '*' + pexit + '*'
    pattern_month_exit = '*' + pexit[:-1] + '*'
    pattern_month_entry = '*' + pexit[:-1] + '*'
    pattern_month_whole = '*' + pexit[:-2] + '*'
    for root, dirs, files in os.walk(path):
        
        for name in files:
                
            if fnmatch.fnmatch(name, pattern_entry):
                return(os.path.join(root, name))
                      
            elif fnmatch.fnmatch(name, pattern_exit):
                return(os.path.join(root, name))
                
            elif fnmatch.fnmatch(name, pattern_month_exit):
                return(os.path.join(root, name))
            
            elif fnmatch.fnmatch(name, pattern_month_entry):
                return(os.path.join(root, name))       
            
            elif fnmatch.fnmatch(name, pattern_month_whole):
                return(os.path.join(root, name))  
            

            

def find_BS_file(date, craft, path):

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
            

def closest_higher_date(date_list, test_date):
    sorted_list = sorted(date_list)
    for date in sorted_list:
        if date >= test_date:
            return date

    return sorted_list[-1]

# Defining constant variables, lists etc...

validphid=(0x1F,0x47,0x6F,0x97,0x26,0x4E,0x76,0x9E,0x2D,0x55,0x7D,0xA5)
sciphid=(0x1F,0x47,0x6F,0x97,0x26,0x4E,0x76,0x9E)
fgmhkphid=(0x2D,0x55,0x7D,0xA5)



starts_stops_spins_df = pd.read_csv('C:/FGM_Extended_Mode/Lib/' + craft + '_SATT_start_stop_spins',names = ['Starts', 'Stops', 'Spins'])

ext_entries_df = pd.read_csv('C:/FGM_Extended_Mode/Lib/' + craft + '_Ext_Entries', header = None)

ext_entries = pd.to_datetime(ext_entries_df[0])

del ext_entries_df

ext_exits_df = pd.read_csv('C:/FGM_Extended_Mode/Lib/' + craft + '_Ext_Exits', header = None)

ext_exits = pd.to_datetime(ext_exits_df[0])

del ext_exits_df

MSA_dumps_df = pd.read_csv('C:/FGM_Extended_Mode/Lib/' + craft + '_MSA_Dump_times', header = None)

MSA_dumps = pd.to_datetime(MSA_dumps_df[0])

del MSA_dumps_df


ext_entry = ext_entries[index]

next_ext_entry = ext_entries[index + 1]

if index >=1:
    prev_ext_entry = ext_entries[index - 1]
    prev_ext_exit = closest_higher_date(ext_exits, prev_ext_entry)
    prev_MSA_dump =  closest_higher_date(MSA_dumps, prev_ext_exit)

ext_exit = closest_higher_date(ext_exits, ext_entry)

next_ext_exit = closest_higher_date(ext_exits, next_ext_entry)


MSA_dump = closest_higher_date(MSA_dumps, ext_exit)

next_MSA_dump =  closest_higher_date(MSA_dumps, next_ext_exit)

duration = ext_exit - ext_entry

expected_packet_num = duration/timedelta(minutes=20, seconds=23)


if ext_exit > next_ext_entry:
        
     raise Exception("Unmatched Entry")

if MSA_dump > next_ext_exit:
        
    raise Exception("No Dump")
    # this covers the whole late and early half thing no?
    # If dump time is greater than next exit, then 'no dump' is raised
        
if duration <= timedelta(seconds = 0):
        
    raise Exception("Negatve/Zero Duration")
        


late_half = False
early_half = False

if MSA_dump.strftime('%Y%m%d') == next_MSA_dump.strftime('%Y%m%d'):
    early_half = True
    print('Dump Contains two EXT periods - this is the first')
        

if MSA_dump.strftime('%Y%m%d') == prev_MSA_dump.strftime('%Y%m%d'):
    late_half = True
    print('Dump Contains two EXT periods - this is the second')
    #raise Exception("Late Half - haven't covered this case in the code yet")
    
closest_start = np.min(abs(pd.to_datetime(starts_stops_spins_df['Starts']) - ext_entry))

closest_stop = np.min(abs(pd.to_datetime(starts_stops_spins_df['Stops']) - ext_exit))

if closest_start < closest_stop:
        
    SATT_index = list(abs(pd.to_datetime(starts_stops_spins_df['Starts']) - ext_entry)).index(closest_start)
        
if closest_start >= closest_stop:
        
    SATT_index = list(abs(pd.to_datetime(starts_stops_spins_df['Stops']) - ext_exit)).index(closest_stop)

t_spin = 60 / starts_stops_spins_df['Spins'].iloc[SATT_index]


duration = ext_exit - ext_entry

expected_packet_num = duration/timedelta(minutes=20, seconds=23)

print('EXT Entry Time')
print(ext_entry)
print('EXT Exit Time')
print(ext_exit)
print('EXT Duration')
print(duration)

dumpdate = MSA_dump.strftime('%Y%m%d')


datadate = ext_entry.strftime('%Y%m%d')

year = ext_entry.strftime('%y')

month = ext_entry.strftime('%m')

calparams_filepath = 'C:/FGM_Extended_Mode/calibration'

formatted_entry = ext_entry.strftime('%Y%m%d')

formatted_exit = ext_exit.strftime('%Y%m%d')

cal_filename = find_cal_file(formatted_entry, formatted_exit,  calparams_filepath)

cal_params = pd.read_csv(cal_filename, header = None, sep = ',|:', names = ['param', 'x', 'y', 'z'], on_bad_lines = 'skip', engine = 'python') 

x_offsets = cal_params[cal_params['param'].str.strip() == 'Offsets(nT)']['x'].astype(float).values.tolist()
x_gains = cal_params[cal_params['param'].str.strip() == 'Gains']['x'].astype(float).values.tolist()
y_gains = cal_params[cal_params['param'].str.strip() == 'Gains']['y'].astype(float).values.tolist()
z_gains = cal_params[cal_params['param'].str.strip() == 'Gains']['z'].astype(float).values.tolist()



while len(x_offsets) < 6:
    x_offsets.append(0.0)
        
while len(x_gains) < 6:
    x_gains.append(1.0)

while len(y_gains) < 6:
    y_gains.append(1.0)

while len(z_gains) < 6:
    z_gains.append(1.0)


yz_gains = []

for i in np.arange(0,6):

    yz_gain = (float(y_gains[i]) + float(z_gains[i])) / 2.0
        
    yz_gains.append(yz_gain)
        


calparams = {'x_offsets':  x_offsets,\
             'x_gains':    x_gains,\
             'yz_gains':   yz_gains}

    

class packet():

    counter=0

    # .cdds is the CDDS packet header bytes (15)
    # .size is the packet size from that CDDS header
    # .payload are the bytes of the payload packet, so everything that isn't the CDDS header
    # .reset is the packet reset count, from the appropriate part of the FGM header
    # .micros are the total microseconds from a combination of the days, milliseconds and microseconds
    # .scet is the time, in Python format, from the .micros
    # .pktcnt is a one-up count of each packet (ie order by presence in file)
    # .status is the status word
    # counter is a count of all the packets ever initialised
    

    def __init__(self,d):
        self.cdds=d[0:15]
        self.size=int.from_bytes(d[9:12],"big")
        self.payload=d[15:15+self.size]
        self.status = d[16]
        
        if self.cdds[8] in sciphid:
            self.reset=int.from_bytes(self.payload[12:14],"big")
        elif self.cdds[8] in fgmhkphid:
            self.reset=(int.from_bytes(self.payload[8:10],"big")+65537)%65536
        else:
            self.reset=-1
        self.micros= int.from_bytes(self.cdds[0:2],"big")*86400*1000000+int.from_bytes(self.cdds[2:6],"big")*1000+int.from_bytes(self.cdds[6:8],"big")
        self.scet=timedelta(microseconds=self.micros)+datetime(1958,1,1)
        
        self.pktcnt=packet.counter
        packet.counter+=1
        
        if self.status == 15 and self.size == 3596:
            self.odd_decode = packet_decoding_odd(self.payload.hex())
            self.even_decode = packet_decoding_even(self.payload.hex())
            ecount, eunique, etop, efreq =  self.even_decode['reset_hex'].describe()
            ocount, ounique, otop, ofreq = self.odd_decode['reset_hex'].describe()
            if eunique < 30:
                self.valid_decode = packet_decoding_even(self.payload.hex())
            elif ounique < 30:
                self.valid_decode = packet_decoding_odd(self.payload.hex())
            else:
                self.valid_decode = str('No_valid_decode')
    
    def __str__(self):
        return("{:7s}".format("#"+str(self.pktcnt))+" | "+" ".join('{:02X}'.format(n) for n in self.cdds)+" | "+" ".join('{:02X}'.format(n) for n in self.payload[0:30]))


BS_filename = find_BS_file(dumpdate[2:], craft, 'C:/FGM_Extended_Mode/BS_raw_files/')

file = open(BS_filename,"rb")

# this is the entire BS file retrieved on the dump date, including Burst Science data 
# D Burst Science packets have size 2232
# Normal Science and Data Dump (aka Extended Mode ?) both have size 3596


data=bytearray(file.read())
file.close()
datalen=len(data)    
        

packets=[]

offset=0


while True:
    packets.append(packet(data[offset:]))
    offset+=15+len(packets[-1].payload)
    #if packets[-1].payload[0]==0 and packets[-1].payload[1]==0x0E:
    #    packets=packets[:-1]
    if packets[-1].size != 3596 or packets[-1].status != 15:
        packets=packets[:-1]
    if offset>=datalen:
        break
        

del data


ext_resets = [i.reset for i in packets]

ext_nums = [i.pktcnt for i in packets]

valid_ext_packets = [i for i in packets if len(i.valid_decode) > 100]

if len(valid_ext_packets) == 0:
    raise Exception("No Valid Packets")

valid_ext_resets = [i.reset for i in valid_ext_packets]

valid_nums = [i.pktcnt for i in valid_ext_packets]

del packets





plt.plot(ext_nums, ext_resets, label = 'all')
plt.plot(valid_nums, valid_ext_resets, label = 'valid', marker = 'x')
plt.legend()
plt.show()

reset_counter = 1

resets = []

packet_series = []

while True:
    resets.append(valid_ext_resets[reset_counter - 1])
    packet_series.append(valid_ext_packets[reset_counter - 1])
    reset_diff = np.abs(valid_ext_resets[reset_counter - 1] - valid_ext_resets[reset_counter])
    #print(reset_diff)
    reset_counter +=1
    if reset_diff > 1 or len(resets) == len(valid_ext_resets) - 1:
        break


series_decodes = [i.valid_decode for i in packet_series]


# the bit below this doesn't really work 

#offset_late = 0

#if late_half == True:
#    reset_counter +=1
#    offset_late = len(resets)
#    resets = []
    
#    while True:
#        resets.append(ext_resets[reset_counter - 1])
#        reset_diff = np.abs(ext_resets[reset_counter - 1] - ext_resets[reset_counter])
        #print(reset_diff)
#        reset_counter +=1
#        if reset_diff > 1:
#            break


    
#valid_decodes = []

#for i in np.arange(1,len(resets) -1):
#    reset_val = resets[i - 1]
#    if reset_val in valid_ext_resets:
        #print(reset_val)
        #print(valid_ext_resets[i - 1])
        #valid_decodes.append(valid_ext_packets[i - 1 + offset_late].valid_decode)
#    else:
#        print('missing packet')
#        print(reset_val)
#        print(valid_ext_resets[i - 1])




sequential_data = pd.concat(series_decodes)

sequential_data.reset_index(drop = True, inplace = True)


bef_indices = sequential_data.loc[sequential_data['x'] == 'bef'].index.tolist()

af_indices = sequential_data.loc[sequential_data['z'] == 'af'].index.tolist()

plt.plot(bef_indices, label = 'bef', marker = 'x')

plt.plot(af_indices, label = 'af')

plt.title('AF and BEF indices (missing ends, missing starts)')
plt.legend()
plt.show()


for i in af_indices:
        
    if i <  len(sequential_data['reset']) - 1 and sequential_data.loc[i+1, 'x'] == 'bef':
        
        sequential_data.loc[i,'reset'] = sequential_data.loc[i+1, 'reset']
            
        sequential_data.loc[i,'reset_hex'] = sequential_data.loc[i+1, 'reset_hex']
        
        sequential_data.loc[i,'resolution'] = sequential_data.loc[i+1, 'resolution']
        
        sequential_data.loc[i,'z'] = sequential_data.loc[i+1,'z']
    
    elif i <  len(sequential_data['reset']) - 2 and sequential_data.loc[i+2, 'x'] == 'bef':

        sequential_data.loc[i,'reset'] = sequential_data.loc[i+2, 'reset']
            
        sequential_data.loc[i,'reset_hex'] = sequential_data.loc[i+2, 'reset_hex']
        
        sequential_data.loc[i,'resolution'] = sequential_data.loc[i+2, 'resolution']
        
        sequential_data.loc[i,'z'] = sequential_data.loc[i+2,'z']        
    
    else:
            
        bef_indices.append(i)
            


sequential_data.drop(labels = bef_indices, axis = 0, inplace = True)

sequential_data['reset'] = sequential_data['reset'].astype(float)

sequential_data['resolution'] = sequential_data['resolution'].astype(int)

sequential_data['x'] = sequential_data['x'].astype(float)

sequential_data['y'] = sequential_data['y'].astype(float)

sequential_data['z'] = sequential_data['z'].astype(float)

sequential_data.reset_index(drop = True, inplace = True)


plt.plot(sequential_data['reset'])
plt.title('Reset Vector for decoded EXT data')
plt.show()

filepath = 'C:/FGM_Extended_Mode/BS_ext_decoded_files' +'/' + craft + '_' + dumpdate + '_clean_decode' + '.csv'

sequential_data.to_csv(filepath)


# timestamping and scaling decoded file


# change to array

r = np.array(sequential_data['resolution'])
x = np.array(sequential_data['x'])
y = np.array(sequential_data['y'])
z = np.array(sequential_data['z'])

# make an estimated time axis

t = make_t(ext_entry, t_spin, ext_exit, x)

name = craft + '_' + datadate

quickplot(name + 'Raw Timestamped','time [UTC]','count [#]')

filebase_cal = 'C:/FGM_Extended_Mode/' + craft + '_EXT_Calibrated'


# nominal scaling
# nominal change from engineering units to nanotesla
# using +/-64nT with 15 bits in range 2

x = x * (2*64/2**15) * 4**(r-2)
y = y * (2*64/2**15) * 4**(r-2) * (np.pi/4)
z = z * (2*64/2**15) * 4**(r-2) * (np.pi/4)

quickplot(name + ' Nominal Scaling','time [UTC]','count [#]')

# apply approximate cal using orbit cal see notes 30-Jan-24

for i in range(0,len(t)):
    Ox = calparams['x_offsets'][r[i]-2]
    Gx = calparams['x_gains'][r[i]-2]
    Gyz = calparams['yz_gains'][r[i]-2]
    x[i] = (x[i] - Ox) / Gx
    y[i] = y[i] / Gyz
    z[i] = z[i] / Gyz
    
quickplot(name+'_calibrated','time [UTC]','[nT]')


zSCS = np.copy(x)
xSCS = np.copy(-y)
ySCS = np.copy(-z)
x = xSCS
y = ySCS
z = zSCS

quickplot(name+'_nominal_scs','time [UTC]','[nT]')


degrees = 146.5
theta = 2*np.pi*degrees/360
xx,yy = np.copy(x),np.copy(y)
x = xx*np.cos(theta) - yy*np.sin(theta)
y = xx*np.sin(theta) + yy*np.cos(theta)

quickplot(name +'_rotated_scs','time [UTC]','[nT]')



# Eliminating anomalous data points (more than 3 standard deviations beyond the mean)

x_outlier_indices = np.where(np.abs(x - np.mean(x)) > (np.std(x) * 3))[0]
y_outlier_indices = np.where(np.abs(y - np.mean(y)) > (np.std(y) * 3))[0]
z_outlier_indices = np.where(np.abs(z - np.mean(z)) > (np.std(z) * 3))[0]

outlier_indices = np.hstack((x_outlier_indices, y_outlier_indices, z_outlier_indices))

ordered_outliers = np.sort(outlier_indices)

x_cleaned = np.delete(x, ordered_outliers)
y_cleaned = np.delete(y, ordered_outliers)
z_cleaned = np.delete(z, ordered_outliers)
t_cleaned = np.delete(t, ordered_outliers)
r_cleaned = np.delete(r, ordered_outliers)

x = x_cleaned
y = y_cleaned
z = z_cleaned
t = t_cleaned
r = r_cleaned

quickplot(name +'_cleaned','time [UTC]','[nT]')


start_time = t[0].strftime('%Y%m%d_%H%M%S')

start_time_iso = t[0].strftime('%Y-%m-%dT%H:%M:%SZ') 

print('Timebase Start:')

print(start_time)

print('Timebase Stop:')
        
end_time = t[-1].strftime('%Y%m%d_%H%M%S')    

# Isoformat: 2001-03-24T23:25:54.000Z

end_time_iso = t[-1].strftime('%Y-%m-%dT%H:%M:%SZ')  

print(end_time)

timebase_duration = t[-1] - t[0]

print('Timebase Duration:')

print(timebase_duration)

savename = filebase_cal +  '/' + craft + '_' + start_time + '_' + end_time + '_calibrated.txt'

fgmsave(savename,t,x,y,z)
    
metadata_savename =  filebase_cal + '/' + craft + '_' + start_time + '_' + end_time + '_info.txt'


# some more quality control and despiking and general evaluation stuff required



f = open(metadata_savename, "w")
f.write('export PATH=$PATH:/cluster/operations/software/dp/bin/:/cluster/operations/software/caa \n')
#f.write('export FGMPATH=/cluster/operations/calibration/tubs_mirror/' + str(datadate[:4]) + '/' + str(datadate[4:6]) + '\n')
# pointing FGMPATH at my own calibration folder containing all calibration files; not sure if this is okay to do 
# should check wtih Chris/Tim on this
f.write('export FGMPATH=/home/lme19/calibration \n')
f.write('export SATTPATH=. \n')
f.write('export ORBITPATH=. \n')
f.write('putsatt /cluster/data/raw/' + str(datadate[:4]) + '/' + str(datadate[4:6]) + '/' +str(craft) + '_' + str(datadate[2:]) + '_B.SATT \n')
f.write('putsatt /cluster/data/raw/' + str(dumpdate[:4]) + '/' + str(dumpdate[4:6]) + '/' +str(craft) + '_' + str(dumpdate[2:]) + '_B.SATT \n')
f.write('putstof /cluster/data/raw/' + str(datadate[:4]) + '/' + str(datadate[4:6]) + '/' +str(craft) + '_' + str(datadate[2:]) + '_B.STOF \n')
f.write('putstof /cluster/data/raw/' + str(dumpdate[:4]) + '/' + str(dumpdate[4:6]) + '/' +str(craft) + '_' + str(dumpdate[2:]) + '_B.STOF \n')


f.write('./ext2tvec -i ' + str(craft) + '_EXT_Calibrated/' + str(craft) +  '_' + str(start_time) + '_' + str(end_time) + str('_calibrated.txt') + ' | fgmhrt -s gse | fgmpos | caavec -t 3 -m 3 -O ' + str(craft) + '_CP_FGM_EXTM_' + str(craft) + '_' + start_time + '_' + end_time + '_V01.cef -H /cluster/operations/calibration/caa_header_files/header_form_V10.txt TIME_SPAN ' + str(start_time_iso) + '/' + str(end_time_iso) + ' version 01')

f.close()
        






