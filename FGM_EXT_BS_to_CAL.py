#%% 
# libraries
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:35:50 2024

@author: Livia
"""
# from fgmfilepaths import craft,date_entry,lib_path,cal_filepath,BS_filepath,filebase_cal
# CC: I moved this back here as Python holds it in the cache and it was not being updated
import numpy as np
from fgmfiletools import fgmsave,fgmopen
from fgmplottools import fgmplot
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
#plt.rcParams['lines.marker'] = '.'
#plt.rcParams['lines.markersize'] = 1
import pandas as pd
import os, fnmatch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# no longer needed:
# from functions import quicksave,quickopen
#%%
# Defining constant variables, lists etc...
global t, x, y, z, r
craft = 'C3'
date_entry = '20020403'
lib_path = './Lib/'
if os.environ.get('LOGNAME') == 'cmcarr':
    cal_filepath = '/Volumes/cluster/calibration/'
    BS_filepath = '/Volumes/cluster/bs/'
else:
    cal_filepath = 'C:/Users/Test/Documents/FGM_Extended_Mode/calibration'
    #lib_path = 'C:/Users/Test/Documents/FGM_Extended_Mode/Lib/'
    BS_filepath = 'C:/Users/Test/Documents/FGM_Extended_Mode/BS/'

# save location for output data
# filebase_cal = './' + craft + '_EXT_Calibrated/'
save_path = './' + date_entry + '/' 
print('Craft: ' + craft)
print('Entry Date: ' + date_entry)
#%%
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

def make_t(date_time_entry, t_spin, date_time_exit, x):
    t = []
    for i in range(1,x+1): # CC edit: changed to 1 to x+1 for compatibility with my code
        t.append(date_time_entry + timedelta(seconds=i*t_spin))
    return t

def find_cal_file(craft,entry,path):
    
    for root, dirs, files in os.walk(path):
        
        for name in files:

            if len(name) == 58: 

                ncraft = name[0:2]
                nstartyear = int(name[16:20])
                nstartmonth = int(name[20:22])
                nstartday = int(name[22:24])
                nstarthour = int(name[25:27])
                nendyear = int(name[32:36])
                nendmonth = int(name[36:38])
                nendday = int(name[38:40])
                nendhour = int(name[41:43])
                nstart = pd.Timestamp(nstartyear, nstartmonth, nstartday, nstarthour, 0, 0)
                nend = pd.Timestamp(nendyear, nendmonth, nendday, nendhour, 0, 0)
    
                # test entry time within calfile start/end validity interval
                if ncraft == craft and entry >= nstart and entry < nend:
                    return os.path.join(root, name)
            
        print('No matching calibration file found for craft:', craft, 'between', entry, 'and', exit)
        return None
        
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
        sciphid=(0x1F,0x47,0x6F,0x97,0x26,0x4E,0x76,0x9E) # moved here (CC)
        fgmhkphid=(0x2D,0x55,0x7D,0xA5)
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


#%%
# Find EXT entry/exit/dump times
# validphid=(0x1F,0x47,0x6F,0x97,0x26,0x4E,0x76,0x9E,0x2D,0x55,0x7D,0xA5) # not used? (CC)

def find_times():
    starts_stops_spins_df = pd.read_csv(lib_path + craft + '_SATT_start_stop_spins',names = ['Starts', 'Stops', 'Spins'])
    ext_entries_df = pd.read_csv(lib_path + craft + '_Ext_Entries', header = None)
    ext_entries = pd.to_datetime(ext_entries_df[0])
    date_time_exits_df = pd.read_csv(lib_path + craft + '_Ext_Exits', header = None)
    date_time_exits = pd.to_datetime(date_time_exits_df[0])
    MSA_dumps_df = pd.read_csv(lib_path + craft + '_MSA_Dump_times', header = None)
    MSA_dumps = pd.to_datetime(MSA_dumps_df[0])

    these_entries = []
    for i in ext_entries:
        if str(i.strftime('%Y%m%d')) == date_entry:
            these_entries.append(np.where(ext_entries == i)[0])
            
    date_time_entry_index = these_entries[0][0]
    date_time_entry = ext_entries[date_time_entry_index]
    next_date_time_entry = ext_entries[date_time_entry_index + 1]

    if date_time_entry_index >=1:
        prev_date_time_entry = ext_entries[date_time_entry_index - 1]
        prev_date_time_exit = closest_higher_date(date_time_exits, prev_date_time_entry)
        prev_MSA_dump =  closest_higher_date(MSA_dumps, prev_date_time_exit)

    date_time_exit = closest_higher_date(date_time_exits, date_time_entry)
    next_date_time_exit = closest_higher_date(date_time_exits, next_date_time_entry)

    MSA_dump = closest_higher_date(MSA_dumps, date_time_exit)
    next_MSA_dump =  closest_higher_date(MSA_dumps, next_date_time_exit)

    duration = date_time_exit - date_time_entry

    # expected_packet_num = duration/timedelta(minutes=20, seconds=23)

    if date_time_exit > next_date_time_entry:            
        raise Exception("Unmatched Entry")

    if MSA_dump > next_date_time_exit:
        raise Exception("No Dump")
        # this covers the whole late and early half thing no?
        # If dump time is greater than next exit, then 'no dump' is raised
            
    if duration <= timedelta(seconds = 0):
        raise Exception("Negatve/Zero Duration")
            
    # late_half = False
    # early_half = False

    if MSA_dump.strftime('%Y%m%d') == next_MSA_dump.strftime('%Y%m%d'):
        # early_half = True
        print('Dump Contains two EXT periods - this is the first')
            
    if MSA_dump.strftime('%Y%m%d') == prev_MSA_dump.strftime('%Y%m%d'):
        # late_half = True
        print('Dump Contains two EXT periods - this is the second')
        
    closest_start = np.min(abs(pd.to_datetime(starts_stops_spins_df['Starts']) - date_time_entry))
    closest_stop = np.min(abs(pd.to_datetime(starts_stops_spins_df['Stops']) - date_time_exit))

    if closest_start < closest_stop:
        SATT_index = list(abs(pd.to_datetime(starts_stops_spins_df['Starts']) - date_time_entry)).index(closest_start)
            
    if closest_start >= closest_stop:
        SATT_index = list(abs(pd.to_datetime(starts_stops_spins_df['Stops']) - date_time_exit)).index(closest_stop)

    t_spin = 60 / starts_stops_spins_df['Spins'].iloc[SATT_index]

    # duration = date_time_exit - date_time_entry

    # expected_packet_num = duration/timedelta(minutes=20, seconds=23)

    print('EXT Entry Time',date_time_entry)
    print('EXT Exit Time',date_time_exit)
    print('EXT Duration',duration)

    date_dump = MSA_dump.strftime('%Y%m%d')
    date_data = date_time_entry.strftime('%Y%m%d')

    # year = date_time_entry.strftime('%y')
    # month = date_time_entry.strftime('%m')
    # formatted_entry = date_time_entry.strftime('%Y%m%d')=
    # formatted_exit = date_time_exit.strftime('%Y%m%d')

    print('MSA Dump Time', MSA_dump)

    return date_time_entry, date_time_exit, date_dump, date_data, t_spin

date_time_entry, date_time_exit, date_dump, date_data, t_spin = find_times()

#%%
# Find calibration file
cal_filename = find_cal_file(craft, date_time_entry, cal_filepath)
print('Calibration File \n',cal_filename)
#%%
# extract cal
def extract_cal():
    cal_params = pd.read_csv(cal_filename, header = None, sep = ',|:', names = ['param', 'x', 'y', 'z'], on_bad_lines = 'skip', engine = 'python') 

    x_offsets = cal_params[cal_params['param'].str.strip() == 'Offsets (nT)']['x'].astype(float).values.tolist()
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

    cal_params = {'x_offsets':  x_offsets,\
                'x_gains':    x_gains,\
                'yz_gains':   yz_gains}

    return cal_params

cal_params = extract_cal()
        
#%%        
# Find BS file
BS_filename = find_BS_file(date_dump[2:], craft, BS_filepath)
print('Burst Science file \n',BS_filename)

#%%
# process BS file
def process_bs_file():
    file = open(BS_filename,"rb")
    # del BS_filename, BS_filepath, cal_filepath, cal_filename

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
        if packets[-1].size != 3596 or packets[-1].status != 15:
            packets=packets[:-1]
        if offset>=datalen:
            break
            
    ext_resets = [i.reset for i in packets]
    ext_nums = [i.pktcnt for i in packets]
    valid_ext_packets = [i for i in packets if len(i.valid_decode) > 100]

    if len(valid_ext_packets) == 0:
        raise Exception("No Valid Packets")

    valid_ext_resets = [i.reset for i in valid_ext_packets]
    valid_nums = [i.pktcnt for i in valid_ext_packets]
    scets = [i.scet for i in packets]
    valid_scets = [i.scet for i in valid_ext_packets]

    # del data, datalen, packets


    reset_counter = 1
    resets = []
    packet_series = []

    while True:
        resets.append(valid_ext_resets[reset_counter - 1])
        packet_series.append(valid_ext_packets[reset_counter - 1])
        reset_diff = np.abs(valid_ext_resets[reset_counter - 1] - valid_ext_resets[reset_counter])
        reset_counter +=1
        if reset_diff > 2: 
            break
        elif len(resets) == len(valid_ext_resets) - 1:
            break 

    series_decodes = [i.valid_decode for i in packet_series]

    # next_series_start_reset = valid_ext_resets[reset_counter]

    # Can use incrementing set of scets etc
    # and next series start reset 
    # to develop second series
    # find the indices missing from sequential data here and use them to remove the corresponsing t values

    sequential_data = pd.concat(series_decodes)
    sequential_data.reset_index(drop = True, inplace = True)


    bef_indices = sequential_data.loc[sequential_data['x'] == 'bef'].index.tolist()
    af_indices = sequential_data.loc[sequential_data['z'] == 'af'].index.tolist()

    plt.figure(figsize=(8, 2))
    plt.plot(ext_nums, ext_resets, label = 'all', markersize = 1)
    plt.plot(valid_nums, valid_ext_resets, label = 'valid', marker = 'x', markersize = 1)
    plt.title('Packet Resets'); plt.legend()
    plt.figure(figsize=(8, 2))
    plt.plot(scets, ext_nums,  label = 'all')
    plt.plot(valid_scets, valid_nums,  label = 'valid')
    plt.title('Packet SCETs'); plt.figure(figsize=(8, 2))
    plt.plot(bef_indices, label = 'bef', marker = 'x')
    plt.plot(af_indices, label = 'af')
    plt.title('AF and BEF indices (missing ends, missing starts)'); plt.legend()

    # return sequential_data, bef_indices, af_indices

# sequential_data, bef_indices, af_indices = process_bs_file()


# extract EXT
# def extract_ext():
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
                
    # decoded and clean dataframe in sequential data 
    sequential_data.drop(labels = bef_indices, axis = 0, inplace = True)
    # Make a basic timebase using spin duration and number of vectors
    timebase_length = int(len(sequential_data))
    # make timeline
    t_actual = make_t(date_time_entry, t_spin, date_time_exit, timebase_length)
    # CC test added times to DF
    sequential_data.insert(0,'timestamp',t_actual)
    # CC Test: the first row usuually not valid
    sequential_data.drop(sequential_data.index[0],inplace=True) 
    sequential_data.reset_index(drop = True, inplace = True)

    # change to range array
    ra = np.array(sequential_data['resolution'].astype(int))

    #remove the vectors at which a range change occurs
    range_change_index = [i+1 if (ra[i+1] - ra[i]) != 0 else i-i for i in np.arange(0,len(ra)-1)]
                    
    #finding the index of the first vector after a range change
    #rci is an acronym for range change indices

    rci  = list(filter(lambda num: num != 0, range_change_index))

    #and the indices of the next 3 
    rci_h1 = [i+1 for i in rci]
    rci_h2 = [i+1 for i in rci]
    rci_h3 = [i+1 for i in rci]
    rci_h4 = [i+1 for i in rci]
    rci_h5 = [i+1 for i in rci]
    rci_h6 = [i+1 for i in rci]
    rci_h7 = [i+1 for i in rci]

    range_change_indices = rci + rci_h1 + rci_h2 +rci_h3 + rci_h4 + rci_h5 + rci_h6 + rci_h7
    sequential_data.drop(labels = range_change_indices, axis = 0, inplace = True)

    #change to array
    # resets = sequential_data['reset'].astype(float)
    r = np.array(sequential_data['resolution'].astype(int))
    x = np.array(sequential_data['x'].astype(float))
    y = np.array(sequential_data['y'].astype(float))
    z = np.array(sequential_data['z'].astype(float))
    # change_indices = []
    # BUG: need to remove the timestamps also as part of the above process
    # t = t_actual[0:len(x)]
    # BUG: need to remove the timestamps also as part of the above process
    # CC test
    t = list(sequential_data['timestamp'])
    # CC test end
    return t,x,y,z,r

t,x,y,z,r = process_bs_file()

#%%
# plot the raw timestamped data
quickplot(craft + '_' + date_data + ' Raw Timestamped','time [UTC]','count [#]')

#%%
# nominal scaling
# nominal change from engineering units to nanotesla
# using +/-64nT with 15 bits in range 2
x = x * (2*64/2**15) * 4**(r-2)
y = y * (2*64/2**15) * 4**(r-2) * (np.pi/4)
z = z * (2*64/2**15) * 4**(r-2) * (np.pi/4)

quickplot(craft + '_' + date_data + ' Nominal Scaling','time [UTC]','[nT]')
#%%
# apply approximate cal using orbit cal
def apply_cal():
    for i in range(0,len(t)):
        Ox = cal_params['x_offsets'][r[i]-2]
        Gx = cal_params['x_gains'][r[i]-2]
        Gyz = cal_params['yz_gains'][r[i]-2]
        x[i] = (x[i] - Ox) / Gx
        y[i] = y[i] / Gyz
        z[i] = z[i] / Gyz
    return x,y,z

x,y,z = apply_cal()

quickplot(craft + '_' + date_data + ' Calibrated','time [UTC]','[nT]')

#%%
# timespan of the data
def dataset_timespan():
    # note that the start of the interval needs to be  rounded DOWN to the nearest second
    # and the end of interval needs to be rounded UP 
    start = t[0].strftime('%Y%m%d_%H%M%S')
    start_iso = t[0].strftime('%Y-%m-%dT%H:%M:%SZ') 
    temp_t = t[-1].replace(microsecond=0) + timedelta(seconds=1) 
    end = temp_t.strftime('%Y%m%d_%H%M%S')
    end_iso = temp_t.strftime('%Y-%m-%dT%H:%M:%SZ')
    print('Dataset first vector time:', t[0].isoformat())
    print('Dataset last vector time:', t[-1].isoformat())
    print('Dataset Duration:',t[-1] - t[0])
    return start, start_iso, end, end_iso
dataset_start, dataset_start_iso, dataset_end, dataset_end_iso = dataset_timespan() 


#%%
# save the calibrated data to a file
def save_data():
    

    print('Timebase Start:')
    print(dataset_start)
    print('Timebase Stop:')
    print(dataset_end)

    savename = save_path + craft + '_' + dataset_start + '_' + dataset_end + '_calibrated.txt'
    fgmsave(savename,t,x,y,z,r)
    print('Saved to: ' + savename)
    return

save_data()
    
# %%
# if necessary, edit the _calibrated.txt file, directly in the editor, to remove any residual spikes
# then reopen here and plot to check before continuing with the processing steps
def finalcheck():
    global t, x, y, z, r
    # open the file
    filename = craft + '_' + dataset_start + '_' + dataset_end + '_calibrated.txt'
    dataset = fgmopen(save_path + '/',filename)
    t = dataset['t']
    x, y, z = dataset['Bx'], dataset['By'], dataset['Bz']
    r = dataset['range']
    # B = np.sqrt(x**2 + y**2 + z**2)
    # plot the data
    quickplot(filename + ' Final Check', 'time [UTC]', '[nT]')
    
    return

finalcheck() 

#%%
# print a string to scp the file to alsvid
def scp2alsvid():
    scp_script = 'scp ' + save_path + craft + '_' + dataset_start + '_' + dataset_end + '_calibrated.txt alsvid.sp.ph.ic.ac.uk:/home/cmcarr/ext' + '\n'
    print(scp_script)
    return

scp2alsvid()

#%%
# save the metadata file

metadata_savename =  save_path + craft + '_' + dataset_start + '_' + dataset_end + '_info.txt'
print('Metadata file: ' + metadata_savename)

f = open(metadata_savename, "w")

# CC modifications follow
# paths not needed as they are in my environment
# f.write('export PATH=$PATH:/cluster/operations/software/dp/bin/:/cluster/operations/software/caa \n')
# FGMPATH also in my environment, and points to /cluster/operations/calibration/default - TUBS mirror is not used / not correct
# f.write('export FGMPATH=$PATH:/cluster/operations/calibration/tubs_mirror/' + str(date_data[:4]) + '/' + str(date_data[4:6]) + ' \n')
# f.write('export FGMPATH=/home/lme19/calibration \n')
# f.write('export SATTPATH=. \n')
# f.write('export ORBITPATH=. \n')
f.write('putsatt /cluster/data/raw/' + str(date_data[:4]) + '/' + str(date_data[4:6]) + '/' +str(craft) + '_' + str(date_data[2:]) + '_B.SATT \n')
f.write('putsatt /cluster/data/raw/' + str(date_dump[:4]) + '/' + str(date_dump[4:6]) + '/' +str(craft) + '_' + str(date_dump[2:]) + '_B.SATT \n')
# works mostly, but ideally, should be next calendar day?
f.write('putstof /cluster/data/raw/' + str(date_data[:4]) + '/' + str(date_data[4:6]) + '/' +str(craft) + '_' + str(date_data[2:]) + '_B.STOF \n')
f.write('putstof /cluster/data/raw/' + str(date_dump[:4]) + '/' + str(date_dump[4:6]) + '/' +str(craft) + '_' + str(date_dump[2:]) + '_B.STOF \n')

# f.write('./ext2tvec -i ' + str(craft) + '_EXT_Calibrated/' + str(craft) +  '_' + str(dataset_start) + '_' + str(dataset_end) + str('_calibrated.txt') + ' | fgmhrt -s gse | fgmpos | caavec -t 3 -m 3 -O ' + str(craft) + '_CP_FGM_EXTM_' + str(craft) + '_' + dataset_start + '_' + dataset_end + '_V01.cef -H /cluster/operations/calibration/caa_header_files/header_form_V10.txt TIME_SPAN ' + str(dataset_start_iso) + '/' + str(dataset_end_iso) + ' version 01')
# CC edited:
f.write('ext2tvec -i ' + str(craft) + '_' + str(dataset_start) + '_' + str(dataset_end) + str('_calibrated.txt') + ' | fgmhrt -s gse | fgmpos | caavec -t 3 -m 3 -O ' + str(craft) + '_CP_FGM_EXTM__' + dataset_start + '_' + dataset_end + '_V01.cef -H /cluster/operations/calibration/caa_header_files/header_form_V11.txt TIME_SPAN ' + str(dataset_start_iso) + '/' + str(dataset_end_iso) + ' version 01')

f.close()

#%%
# print a string to scp the file from alsvid to local folder
def scp2local():
    cefname = str(craft) + '_CP_FGM_EXTM__' + dataset_start + '_' + dataset_end + '_V01.cef'
    scp_script = 'scp alsvid.sp.ph.ic.ac.uk:/home/cmcarr/ext/' + save_path[2:] + cefname + ' ' + save_path 
    print(scp_script)
    return

scp2local()
# %%
# plot the CEF file
def checkplot():
    cefname = str(craft) + '_CP_FGM_EXTM__' + dataset_start + '_' + dataset_end + '_V01.cef'
    dataset = fgmopen(save_path,cefname)
    fgmplot(dataset)
    return

checkplot()
# %%
# optional - plot with other datasets for final validation
def valplot():
    cefname = str(craft) + '_CP_FGM_EXTM__' + dataset_start + '_' + dataset_end + '_V01.cef'
    dataset = fgmopen(save_path,cefname)
    spin = fgmopen(save_path, 'C3_CP_FGM_SPIN__20020403_175500_20020404_120500_V00.cef')
    fgmplot([dataset,spin])
    return

valplot()