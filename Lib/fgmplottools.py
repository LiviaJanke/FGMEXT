#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:08:07 2023

@author: cmcarr
"""

#%% Load packages

from numpy import sqrt,ndarray
from datetime import datetime,timedelta
from matplotlib.pyplot import plot,xlabel,ylabel,subplot,grid,legend,ylim
from matplotlib.pyplot import suptitle,subplots,yticks,semilogy,scatter

from fgmplotparams import fgmplotParams

#%% Function definitions
# Plot

def fgmplot(datasets, interval_start=None, interval_end=None, titletext=None):
    earth_radius = 6371
    colours = {'C1' : 'k', 'C2' : 'c' , 'C3' : 'm' , 'C4' : 'b' , 'SPIN' : 'g', 'EXTM' :'r'}
    # list or single dataset dict
    if type(datasets) is dict:
        datasets = [datasets] # cast to a list
        
    if len(datasets) > 8:
        labelFlag = False
    else:
        labelFlag = True    
        
    if interval_start is None:
        # find earliest start time
       interval_start = datetime(2030,1,1,0,0,0)
       for dataset in datasets:
           if not dataset['data_start'] is None:
               if dataset['data_start'] < interval_start:
                   interval_start = dataset['data_start'] 
    if interval_end is None:
        # find latest end time
       interval_end = datetime(2000,1,1,0,0,0)
       for dataset in datasets:
           if not dataset['data_end'] is None:
               if dataset['data_end'] > interval_end:
                   interval_end = dataset['data_end'] 
    
    text = interval_start.isoformat(timespec='seconds') + ' - ' + interval_end.isoformat(timespec='seconds')
    if titletext is None:
        titletext = text
    else:
        titletext += ' ' + text        
    
    # find if any datasets contain position or range info           
    positionPanel, rangePanel = False, False # flags determine if panels are shown
    for dataset in datasets:
        if dataset['positionFlag'] and fgmplotParams['showposition']:
            positionPanel = True # a panel will be shown
        if dataset['rangeFlag'] and fgmplotParams['showrange']:
            rangePanel = True # panel will be shown
        
    num_panels = 4 # just Bx,By,Bz,B
    height_ratios = [2,2,2,2]
    if (positionPanel and not rangePanel) or (not positionPanel and rangePanel):# xor
        num_panels = 5
        height_ratios = [2,2,2,2,1]
    elif positionPanel and rangePanel:
        num_panels = 6
        height_ratios = [2,2,2,2,1,1]
        
    # match num_panels:
    #     case 4:
    #         height_ratios = [2,2,2,2]
    #     case 5:
    #         height_ratios = [2,2,2,2,1]
    #     case 6:
    #         height_ratios = [2,2,2,2,1,1]
    
    fig,ax = subplots(num_panels,1,figsize=fgmplotParams['figsize'],sharex=True,height_ratios=height_ratios)
    
    for dataset in datasets:
        # for each dataset to plot find the indices of times in the inyterval
        # dataDict = theDict[key]
        
        # handle empty files
        if type(dataset['t']) is ndarray:
            # print('DatasetID {:s}'.format(dataset['dataset_id']))
            numvals = len(dataset['t'])
            scid = dataset['dataset_id'][0:2]
            
            data_cat = dataset['dataset_id'][10:14]
            
            if labelFlag:
                if len(dataset['dataset_id']) > 25:
                    label = dataset['dataset_id'][0:24] + '...'
                else:
                    label = dataset['dataset_id']
            else:
                label = None
                
            #colour=colours[scid]
            
            colour = colours[data_cat]
            
            index = []
            for i in range(numvals):
                if (dataset['t'][i] > interval_start and dataset['t'][i] < interval_end):
                    if dataset['rangeFlag']:
                        if dataset['range'][i] >= fgmplotParams['rangemin'] \
                            and dataset['range'][i] <= fgmplotParams['rangemax']:
                            index.append(i)
                    else:
                        index.append(i)
            t = dataset['t'][index]
            Bx = dataset['Bx'][index]
            By = dataset['By'][index]
            Bz = dataset['Bz'][index]
            B = sqrt(Bx**2 + By**2 + Bz**2)
            if positionPanel and dataset['positionFlag']:
                Px = dataset['Px'][index]
                Py = dataset['Py'][index]
                Pz = dataset['Pz'][index]
                P = sqrt(Px**2 + Py**2 + Pz**2)/earth_radius
            if rangePanel and dataset['rangeFlag']:
                # if scid == 'C1': 
                #     rngoffset = 0
                # elif scid == 'C2':
                #     rngoffset = 0.1
                # elif scid == 'C3':
                #     rngoffset = 0.2
                # else:
                #     rngoffset = 0.3
                rng = dataset['range'][index] #+ rngoffset
            
            # print('Checking for gaps {:s}'.format(scid))
            gapindex = gap_detect(t)
        
            subplot(num_panels,1,1); plot_nogaps(t,Bx,gapindex,colour,fgmplotParams['linewidth'])
            subplot(num_panels,1,2); plot_nogaps(t,By,gapindex,colour,fgmplotParams['linewidth'])
            subplot(num_panels,1,3); plot_nogaps(t,Bz,gapindex,colour,fgmplotParams['linewidth'])
            subplot(num_panels,1,4); plot_nogaps(t,B,gapindex,colour,fgmplotParams['linewidth'],label=label,scale=fgmplotParams['magnitudescale'])
            if labelFlag:
                legend()
            if dataset['positionFlag'] and positionPanel:
                    subplot(num_panels,1,5); plot_nogaps(t,P,gapindex,colour,fgmplotParams['linewidth']) 
            if dataset['rangeFlag'] and rangePanel:
                    subplot(num_panels,1,num_panels); plot_nogaps(t,rng,gapindex,colour,fgmplotParams['linewidth']) 
        else:
            print('Empty dataset')
    subplot(num_panels,1,1);grid();ylabel("x [nT]")
    subplot(num_panels,1,2);grid();ylabel("y [nT]")
    subplot(num_panels,1,3);grid();ylabel("z [nT]")
    subplot(num_panels,1,4);grid();ylabel("|B| [nT]");
    # if type(dataset['t']) is ndarray: 
        
    if positionPanel:
        subplot(num_panels,1,5);grid();ylabel("R_E")
    if rangePanel:
        subplot(num_panels,1,num_panels);grid();ylabel("range")
        ylim([1,8]); yticks(ticks=[2,4,6])
    xlabel('time [UTC]')
    suptitle(titletext,y=0.92)
    return fig,ax

def gap_detect(t,dtmax=6):
    dtmax = timedelta(seconds=dtmax)
    num = len(t)
    myindex = []
    if num >=2:
        for i in range(1,num):
            if t[i]-t[i-1] > dtmax:
                # print('Gap between '+str(t[i-1])+' and '+str(t[i]))
                myindex.append(i-1)
    return myindex
        
def plot_nogaps(t,data,gapindex,colour,linewidth,label=None,scale='linear'):
    start = 0
    if len(gapindex) != 0:
        for i in range(len(gapindex)):
            end = gapindex[i]
            if scale == 'linear':
                plot(t[start:end],data[start:end],color=colour,linewidth=linewidth,label=None)
            else:
                semilogy(t[start:end],data[start:end],color=colour,linewidth=linewidth,label=None)
            start = end+1
    end = len(t)
    if scale == 'linear':
        plot(t[start:end],data[start:end],color=colour,linewidth=linewidth,label=label)
    else:
        semilogy(t[start:end],data[start:end],color=colour,linewidth=linewidth,label=label)
    return

#%% Plot FGM data
# Can be used with CEF data or output from DP software
# positional and range info is optional, as is time intervals 
# def plot_fgm(dataset_id,t,Bx,By,Bz,Px=None,Py=None,Pz=None,rng=None,\
#                         interval_start=None,interval_end=None,system='cartesian'):
#     earth_radius = 6371
#     cluster_colours = ['k','r','g','b']
#     cluster_spacecraft = ['1','2','3','4']
#     colour = cluster_colours[cluster_spacecraft.index(dataset_id[1])]
#     if interval_start == None:
#         interval_start = t[0]
#     if interval_end == None:
#         interval_end = t[len(t)-1]
#     index=[]
#     for j in range(len(t)):
#         if t[j] > interval_start and t[j] < interval_end:
#             index.append(j)
#     B = sqrt(Bx**2+By**2+Bz**2)
#     theta = ((pi/2) - arccos(Bz/B))*360/(2*pi)
#     phi = arctan2(By,Bx)*360/(2*pi)
#     if not (Px is None or Py is None or Pz is None):
#         altitude = sqrt(Px**2+Py**2+Pz**2)/earth_radius # in R_E
    
#     if system == 'cartesian':
#         rows,height_ratios = 6,[2,2,2,2,1,1]
#         if (Px is None or Py is None or Pz is None):
#             if rng is None:
#                 rows, height_ratios = 4,[2,2,2,2]
#             else:
#                 rows, height_ratios = 5,[2,2,2,2,1]  
#         else:
#             if rng is None:
#                 rows, height_ratios = 5,[2,2,2,2,1]
#         row = 1
#         fig,ax = subplots(rows,1,figsize=(8,10),sharex=True,height_ratios=height_ratios)
#         subplot(rows,1,row)
#         plot(t[index],Bx[index],color=colour,linewidth='0.5')
#         grid(); ylabel("Bx [nT]"); row += 1
#         subplot(rows,1,row)
#         plot(t[index],By[index],color=colour,linewidth='0.5')
#         grid(); ylabel("By [nT]"); row += 1
#         subplot(rows,1,row)
#         plot(t[index],Bz[index],color=colour,linewidth='0.5')
#         grid(); ylabel("Bz [nT]"); row += 1
#         subplot(rows,1,row)
#         plot(t[index],B[index],color=colour,linewidth='0.5')
#         ylabel("|B| [nT]"); grid(); row += 1
#         if not Px is None:
#             subplot(rows,1,row)
#             plot(t[index],altitude[index],color=colour,linewidth='0.5')
#             grid(); ylabel("Alt.[R_E]"); row +=1
#         if not rng is None:
#             subplot(rows,1,row)
#             plot(t[index],rng[index],color=colour,linewidth='2')
#             ylim([1,8]); yticks(ticks=[2,4,6])
#             grid(); ylabel("Range")
#     elif system == 'polar':
#         rows,height_ratios = 5,[2,2,2,1,1]
#         if (Px is None or Py is None or Pz is None):
#             if rng is None:
#                 rows, height_ratios = 3,[2,2,2]
#             else:
#                 rows, height_ratios = 4,[2,2,2,1]  
#         else:
#             if rng is None:
#                 rows, height_ratios = 4,[2,2,2,1]
#         row = 1
#         fig,ax = subplots(rows,1,figsize=(8,10),sharex=True,height_ratios=height_ratios)
#         subplot(rows,1,row)
#         plot(t[index],B[index],color=colour,linewidth='0.5')
#         grid(); ylabel('|B| [nT]'); row += 1
#         subplot(rows,1,row)
#         plot(t[index],theta[index],color=colour,linewidth='0.5')
#         grid(); ylabel("$\\theta$ [$\\degree$]"); row += 1
#         subplot(rows,1,row)
#         plot(t[index],phi[index],color=colour,linewidth='0.5')
#         grid(); ylabel("$\\phi$ [$\\degree$]"); row += 1
#         if not Px is None:
#             subplot(rows,1,row)
#             plot(t[index],altitude[index],color=colour,linewidth='0.5')
#             grid(); ylabel("${R_E}$"); row += 1
#         if not rng is None:
#             subplot(rows,1,rows)
#             plot(t[index],rng[index],color=colour,linewidth='2')
#             ylim([1,8]); yticks(ticks=[2,4,6])
#             grid(); ylabel("Range")
#     else:
#         return
#     xlabel('time [UTC]')
#     suptitle(dataset_id,y=0.92)
#     return fig,ax

    