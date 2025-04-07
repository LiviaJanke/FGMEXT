#!/usr/bin/env python
# coding: utf-8

# Load packages
# from datetime import timedelta
# from numpy import pi,sqrt,arccos,arctan2,array,linspace
# from matplotlib.pyplot import xlabel,ylabel,subplot,grid
# from matplotlib.pyplot import suptitle,subplots
# from matplotlib.pyplot import rcParams
# rcParams['text.usetex'] = False
# rcParams['mathtext.default'] = 'regular'
# from scipy.fft import fft,fftfreq
# from scipy.integrate import trapz
# from scipy.signal import spectrogram,welch
# from numpy import log10,shape,argmax,linspace,pi,arange
# from numpy import sqrt,argwhere
# from numpy.random import normal
# from matplotlib.pyplot import pcolormesh,colorbar,loglog,title
# from matplotlib.pyplot import semilogy

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# filetools
from sys import path
#orbitpath = '../Lib'
#if not path.count(orbitpath):
#    path.append(orbitpath)
    
path.append('C:\FGM_Extended_Mode\Lib')
#path.append('C:\FGM_Extended_Mode\CEF_Files')
#path.append('C:\FGM_Extended_Mode\Example_CEF')


from fgmfiletools import fgmopen
from fgmplottools import fgmplot
# from orbitinfo import get_orbit_times,filename_finder



from fgmplotparams import fgmplotParams
fgmplotParams['figsize'] = (10,12)

# Open a file processed with the DP software
# C1 = fgmopen('/Volumes/cmcarr/dp','C1_240511_NS.txt')
# C3 = fgmopen('/Volumes/cmcarr/dp','C3_240511_NS.txt')
# C4 = fgmopen('/Volumes/cmcarr/dp','C4_240511_NS.txt')

# Plot whole file
# fgmplot([C1,C3,C4])

# Plot interval
# interval_start = dataset['data_start'] + timedelta(minutes=10)
# interval_end = interval_start + timedelta(minutes=20)
# fgmplot(dataset, interval_start, interval_end)

# find a file
# NB see better methods in multispacecraft
# orbit_number = 3517
# orbit_start,orbit_end = get_orbit_times(orbit_number,fgmdatasetParams['orbitpath'])
# filenamelist = filename_finder(orbit_start,orbit_end,spacecraft=['C1'],product='5VPS',version=1,orbitpath=fgmdatasetParams['orbitpath'])
# print(filenamelist)
# filename = filenamelist[0]['filenames'][0]
# print(filename)

# open CEF files in this folder

#C1_ESA = fgmopen('C:\FGM_Extended_Mode\Example_CEF','C1_CP_FGM_SPIN__20010419_175500_20010422_020500_V00.cef')

# C:\FGM_Extended_Mode\Example_CEF\C1

#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_EXTM_C1_20010403_050102_20010403_082833_V01.cef')
#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_EXTM_C1_20010420_032054_20010421_010710_V01.cef')
#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_EXTM_C1_20010705_205108_20010706_184317_V01.cef')
#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_EXTM_C1_20020703_111757_20020703_213953_V01.cef')
#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_EXTM_C1_20070111_144005_20070111_232214_V01.cef')

#ESA_C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_SPIN__20010402_235500_20010404_140500_V00.cef')
#ESA_C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_SPIN__20010419_175500_20010422_020500_V00.cef')
#ESA_C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_SPIN__20010705_145500_20010707_050500_V00.cef')
#ESA_C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_SPIN__20020703_045500_20020704_190500_V00.cef')
#ESA_C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C1', 'C1_CP_FGM_SPIN__20070110_235500_20070112_140500_V00.cef')

#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_EXTM_C2_20010607_213457_20010608_023104_V013.cef')
#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_EXTM_C2_20011104_072516_20011105_003624_V01.cef')
#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_EXTM_C2_20020120_071954_20020121_003541_V01.cef')
#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_EXTM_C2_20020624_134454_20020625_123852_V01.cef')
#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_EXTM_C2_20061123_034054_20061123_132839_V01.cef')

#ESA_C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_SPIN__20010606_235500_20010608_140500_V00.cef')
#ESA_C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_SPIN__20011103_235500_20011105_140500_V00.cef')
#ESA_C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_SPIN__20020119_235500_20020121_140500_V00.cef')
#ESA_C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_SPIN__20020624_045500_20020625_190500_V00.cef')
#ESA_C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C2', 'C2_CP_FGM_SPIN__20061122_235500_20061123_220500_V00.cef')

#C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', 'C3_CP_FGM_EXTM_C3_20010613_140954_20010614_125509_V01.cef')
#C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', 'C3_CP_FGM_EXTM_C3_20020214_085227_20020214_211352_V01.cef')
#C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', 'C3_CP_FGM_EXTM_C3_20090211_105454_20090211_153144_V01.cef')

#ESA_C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', '')
#ESA_C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', 'C3_CP_FGM_SPIN__20020213_235500_20020215_140500_V00.cef')
#ESA_C3 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C3', 'C3_CP_FGM_SPIN__20090210_235500_20090212_140500_V00.cef')

C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_EXTM_C4_20010827_070954_20010827_223857_V01.cef')
#C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_EXTM_C4_20020126_035726_20020127_023848_V01.cef')
#C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_EXTM_C4_20020706_104354_20020706_203759_V01.cef')
#C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_EXTM_C4_20131015_085602_20131015_170437_V01.cef')
#C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_EXTM_C4_20180308_235128_20180309_134109_V01.cef')

ESA_C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_SPIN__20010826_235500_20010828_140500_V00.cef')
#ESA_C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_SPIN__20020125_235500_20020127_140500_V00.cef')
#ESA_C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_SPIN__20020705_235500_20020707_140500_V00.cef')
#ESA_C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', 'C4_CP_FGM_SPIN__20131014_235500_20131016_140500_V00.cef')
#ESA_C4 = fgmopen('C:\FGM_Extended_Mode\Example_CEF\C4', '')

#C :\FGM_Extended_Mode\Example_CEF\
# C1 = fgmopen('./','C1_CP_FGM_5VPS__20010102_055913_20010104_150531_V01.cef')
# C2 = fgmopen('./','C2_CP_FGM_5VPS__20010102_055913_20010104_150531_V01.cef')
# C3 = fgmopen('./','C3_CP_FGM_5VPS__20010102_055913_20010104_150531_V01.cef')
# C4 = fgmopen('./','C4_CP_FGM_5VPS__20010102_055913_20010104_150531_V01.cef')

# open CEF files in this folder
#C1 = fgmopen('C:\FGM_Extended_Mode\CEF_Files','C1_CP_FGM_EXTM_C1_20150312_185954_20150313_094110_V01_withoutm3.cef')
#C1 = fgmopen('C:\FGM_Extended_Mode\Example_CEF','C1_CP_FGM_EXTM_C1_20010420_032054_20010421_010710_V01.cef')
#C2 = fgmopen('C:\FGM_Extended_Mode\Example_CEF', 'C2_CP_FGM_EXTM_C2_20020120_071954_20020121_003541_V01.cef')
# C2 = fgmopen('./','C2_CP_FGM_5VPS__20010104_150531_20010107_001002_V01.cef')
# C3 = fgmopen('./','C3_CP_FGM_5VPS__20010104_150531_20010107_001002_V01.cef')
# C4 = fgmopen('./','C4_CP_FGM_5VPS__20010104_150531_20010107_001002_V01.cef')

#%% Plot all
fgmplotParams['rangemax']=7
# fgmplot([C1,C2,C3,C4])
#fgmplot([C1, ESA_C1], interval_end = datetime(2007,1,12,2,0,0))
#fgmplot([C2, ESA_C2])#, interval_end = datetime(2007,1,12,2,0,0))
#fgmplot([C3, ESA_C3], interval_start = datetime(2009,2,10,23,0,0), interval_end = datetime(2009,2,11,20,0,0))
fgmplot([C4, ESA_C4])#, interval_end = datetime(2013,10,15,23,0,0))
plt.savefig('C4_2001_08_27')

#fgmplot(C1_ESA)
#fgmplot(C1)
#plt.savefig('EXT')

#fgmplot(ESA)
#plt.savefig('ESA')


#fgmplotParams['rangemax']=7
# fgmplot([C1,C2,C3,C4])
#fgmplot([C1, C1_ESA], interval_end = datetime(2001,4,21,6,0,0))
#plt.savefig('ESA_and_EXT_start')

# plot range 2
#fgmplotParams['rangemax']=2
#fgmplot(C1)

# Custom interval
# Here, need to specify range is passed as position is not
# fgmplot(dataset,interval_start=datetime(2010,9,6),interval_end=datetime(2010,9,6,1,0,0))


#fgmplotParams['rangemax']=7
# fgmplot([C1,C2,C3,C4])
#fgmplot(C2)

#%%

