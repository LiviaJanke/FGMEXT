#%% 
# packages
from pandas import read_csv
# from matplotlib.pyplot import savefig,suptitle,xlabel,ylabel,plot,grid,legend,subplot,subplots
from datetime import datetime #,timedelta
from numpy import sqrt,array,zeros,size,pi,sin,cos,arctan
# FGM tools
from fgmfiletools import fgmopen,fgmsave
from fgmplottools import fgmplot
from fgmplotparams import fgmplotParams
fgmplotParams['magnitudescale'] = 'linear'
# from functions import openraw,quickplot,quicksave,quickopen,make_t,apply_calparams,nominal_scaling
import functions as fn
#%%
# STEP 0 - input parameters
# local file location, spacecraft and date
spacecraft,filedate = 'C1','020403'
# entry/exit timing
# C1
# 2002-04-03T21:19:54.000Z Entry 9.27 hours
#  Midnight crossed
# 2002-04-04T06:36:00.000Z Exit
# 2002-04-04T06:44:06.000Z MSA Dump
# C234 all same as above

# # find /cluster/data/raw/2002/  -name "C?_020403_B.SATT" | sort -n | xargs strings
# 01 R 2002-04-02T06:20:13Z 2002-04-17T06:20:13Z 103.19 -64.48 14.934581 333.948 760.4  -0.1   0.1  0.00  0.06 2002-04-02T15:24:22Z
# 02 R 2002-04-02T09:26:18Z 2002-04-17T09:26:18Z 103.21 -64.55 15.009333 333.885 768.6  -0.2   0.0  0.00  0.02 2002-04-02T15:27:54Z
# 03 R 2002-04-02T12:33:31Z 2002-04-17T12:33:31Z 102.97 -64.51 14.975621 333.966 759.3   0.2  -0.1 -0.07 -0.02 2002-04-02T15:29:53Z
# 04 R 2002-04-02T15:05:05Z 2002-04-17T15:05:05Z 103.03 -64.52 15.012543 333.900 769.1  -0.1  -0.1  0.02  0.09 2002-04-02T15:31:32Z

if spacecraft == 'C1':
    ext_entry = datetime.fromisoformat('2002-04-03T21:19:54.000')
    ext_exit = datetime.fromisoformat('2002-04-04T06:36:00')
    rpm = 14.934581
if spacecraft == 'C2':
    ext_entry = datetime.fromisoformat('2002-04-03T21:19:54.000')
    ext_exit = datetime.fromisoformat('2002-04-04T06:36:00')
    rpm = 15.009333
if spacecraft == 'C3':
    ext_entry = datetime.fromisoformat('2002-04-03T21:19:54.000')
    ext_exit = datetime.fromisoformat('2002-04-04T06:36:00')
    rpm = 14.975621
if spacecraft == 'C4':
    ext_entry = datetime.fromisoformat('2002-04-03T21:19:54.000')
    ext_exit = datetime.fromisoformat('2002-04-04T06:36:00')
    rpm = 15.012543

# Cal params 
# need only x offset, x,y,z gains, for the ranges used - execute on embarr

# find /cluster/caa/calibration/C[1234]_CC_FGM_CALF__20020401*.fgmcal -print -exec sed -n '59,62p' {} \; -exec sed -n '66,69p' {} \; -exec sed -n '73,76p' {} \; 
# /cluster/caa/calibration/C1_CC_FGM_CALF__20020401_172927_20020404_023251_V01.fgmcal
# Components  :          X,        Y,        Z
# Offsets (nT):     -2.876,   +4.865,   +0.673
# Gains       :   +0.95159, +0.95081, +0.96690
# Theta (deg) :     +0.770,  +90.162,  +90.357
# Components  :          X,        Y,        Z
# Offsets (nT):     -2.806,   +5.029,   +0.747
# Gains       :   +0.96854, +0.96903, +0.98451
# Theta (deg) :     +0.779,  +90.235,  +90.353
# Components  :          X,        Y,        Z
# Offsets (nT):    -37.449,  +18.836,   -1.870
# Gains       :   +0.97911, +0.98461, +0.99551
# Theta (deg) :     +0.771,  +90.206,  +90.360
# /cluster/caa/calibration/C2_CC_FGM_CALF__20020401_172927_20020404_023251_V01.fgmcal
# Components  :          X,        Y,        Z
# Offsets (nT):     +0.298,   -2.256,   -1.472
# Gains       :   +0.95879, +0.95281, +0.94717
# Theta (deg) :     +0.368,  +89.470,  +89.930
# Components  :          X,        Y,        Z
# Offsets (nT):     +0.349,   -2.310,   -1.456
# Gains       :   +0.97587, +0.96971, +0.96373
# Theta (deg) :     +0.360,  +89.422,  +89.925
# Components  :          X,        Y,        Z
# Offsets (nT):     +2.857,   -2.092,   -0.698
# Gains       :   +0.98655, +0.98125, +0.98562
# Theta (deg) :     +0.363,  +89.456,  +89.941
# /cluster/caa/calibration/C3_CC_FGM_CALF__20020401_172927_20020404_023251_V01.fgmcal
# Components  :          X,        Y,        Z
# Offsets (nT):     -2.006,   -5.060,   -2.484
# Gains       :   +0.95990, +0.96553, +0.94801
# Theta (deg) :     +0.832,  +89.529,  +89.809
# Components  :          X,        Y,        Z
# Offsets (nT):     -1.982,   -5.100,   -2.523
# Gains       :   +0.97587, +0.98040, +0.96391
# Theta (deg) :     +0.826,  +89.504,  +89.779
# Components  :          X,        Y,        Z
# Offsets (nT):     +4.178,   +1.282,   +2.404
# Gains       :   +0.99547, +0.99397, +0.98087
# Theta (deg) :     +0.829,  +89.531,  +89.781
# /cluster/caa/calibration/C4_CC_FGM_CALF__20020401_172927_20020404_023251_V01.fgmcal
# Components  :          X,        Y,        Z
# Offsets (nT):    -12.747,   -3.031,   +4.518
# Gains       :   +0.96016, +0.92837, +0.95295
# Theta (deg) :     +0.336,  +89.588,  +90.147
# Components  :          X,        Y,        Z
# Offsets (nT):    -12.931,   -3.062,   +4.610
# Gains       :   +0.97802, +0.94530, +0.96964
# Theta (deg) :     +0.333,  +89.542,  +90.112
# Components  :          X,        Y,        Z
# Offsets (nT):     -5.105,   +3.438,  +12.747
# Gains       :   +0.99546, +0.95831, +0.98500
# Theta (deg) :     +0.342,  +89.558,  +90.152

# C1 EXTM has R3,4
if spacecraft == 'C1': 
    calparams = {'x_offsets':  [0,-2.806,-37.449,0,0,0],\
                'x_gains':     [1,0.96854,0.97911,1,1,1],\
                'yz_gains':    [1,(+0.96903+0.98451)/2,(+0.98461+0.99551)/2,1,1,1]}
    
# C2 EXTM has R3,4,5 but no R5 cal so replicate R4 values with manual correction - see wiki
if spacecraft == 'C2': 
    calparams = {'x_offsets':  [0,+0.349,2.857,2.857,0],\
                'x_gains':     [1,0.97587,0.98655,0.98655,1,1],\
                'yz_gains':    [1,(+0.96971+0.96373)/2,(+0.98125+0.98562)/2,(500/492)*(+0.98125+0.98562)/2,1,1]}

# C3 EXTM has R3,4,5 but no R5 cal so replicate R4 values with manual correction - see wiki
if spacecraft == 'C3': 
    calparams = {'x_offsets':  [0,-1.982,4.178,4.178,0,0],\
                'x_gains':     [1,0.97587,0.99547,0.99547,1,1],\
                'yz_gains':    [1,(+0.98040+0.96391)/2,(+0.99397+0.98087)/2,(500/492)*(+0.99397+0.98087)/2,1,1]}
# if spacecraft == 'C4': 
#     calparams = {'x_offsets':  [-12.279,-12.452,-4.451,0,0,0],\
#                 'x_gains':     [+0.95886,0.97786,0.99546,1,1,1],\
#                 'yz_gains':    [(+0.92872+0.95367)/2,(+0.94466+0.96876)/2,(+0.95831+0.98496)/2,1,1,1]}


#%% 
# STEP 1 - open raw edited
# open raw edited extended mode data file
t,x,y,z,r = fn.openraw(filedate,spacecraft)
# first visual inspection
filename = filedate+'/'+spacecraft+'_'+filedate+'_ext_1_edited.txt'
fn.quickplot(t,x,y,z,r,filename,'sample #','count [#]')
# Now check plot for any major errors 

#%% 
# STEP 2 - timing
t_spin = 60/rpm
# build timeline
t = fn.make_t(t_spin,len(r),ext_entry,ext_exit)
del rpm,t_spin
# second visual inspection
filename = spacecraft+'_'+filedate+'_ext_2_timestamped.txt'
fn.quickplot(t,x,y,z,r,filename,'time [UTC]','count [#]')
# temporary save for purposes of despiking
filename = filedate + '/' + spacecraft +'_' + filedate + 'ext_2_timestamped.txt'
fn.quicksave(filename,t,x,y,z,r)
del filename

#%% 
# STEP 3 - de-spiking
# duplicate file with 3_despiked 
# manually edit to remove spikes
# re-open
filename = filedate+'/'+spacecraft+'_'+filedate+'_ext_3_despiked.txt'
t,x,y,z,r = fn.quickopen(filename)
# re-plot
fn.quickplot(t,x,y,z,r,filename,'time [UTC]','count [#]')
# check correctly despiked before proceeding
# NB only need to run this step in the case that changes were made above

#%% 
# STEP 4 - apply scaling
# make sure to only apply ONCE
# scaling using +/-64nT with 15 bits in range 2
x,y,z = fn.nominal_scaling(x,y,z,r)
filename = filedate+'/'+spacecraft+'_'+filedate+'_ext_4_scaled.txt'
fn.quickplot(t,x,y,z,r,filename,'time [UTC]','[nT]')
fgmsave(filename,t,x,y,z,r)

#%%
# apply calibration
x,y,z = fn.apply_calparams(x,y,z,r,calparams)
filename = filedate+'/'+spacecraft+'_'+filedate+'_ext_4_calibrated.txt'
fn.quickplot(t,x,y,z,r,filename,'time [UTC]','[nT]')
# save
fgmsave(filename,t,x,y,z,r)


#%% Validation - GSE/CEF - vs CSA data
# compare before/after NM from interactive plotting tool
plotlist = []
plotlist.append(fgmopen('020403/C1_CP_FGM_EXTM__20020403_212002_20020404_063604_V01.cef'))
# plotlist.append(fgmopen('020403/C2_CP_FGM_EXTM__20020403_212001_20020404_063558_V01.cef'))
# plotlist.append(fgmopen('020403/C3_CP_FGM_EXTM__20020403_212002_20020404_063537_V01.cef'))
# plotlist.append(fgmopen('020403/C1_CP_FGM_SPIN__20020403_175500_20020404_120500_V00.cef'))
# plotlist.append(fgmopen('020403/C2_CP_FGM_SPIN__20020403_175500_20020404_120500_V00.cef'))
# plotlist.append(fgmopen('020403/C3_CP_FGM_SPIN__20020403_175500_20020404_120500_V00.cef'))
fgmplot(plotlist)


# %%
