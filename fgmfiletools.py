#!/usr/bin/env python
# coding: utf-8
# FGM file handling function

#%% Load packages
from datetime import datetime
from numpy import array
from os.path import isfile

#%% Function definitions

# open ascii file from dp software


# save in ascii format as per dp output
def fgmsave(filename,t,x,y,z):
    file = open(filename,'w')
    for i in range(0,len(t)):
        aline = t[i].isoformat(timespec='milliseconds')[0:23] + 'Z'
        aline += " {0: 10.4f} {1: 10.4f} {2: 10.4f}\n".format(x[i],y[i],z[i])
        file.write(aline)
    file.close()
    return

#ADD SAVING RANGE VALUES IN NEW COLUMN TO FGMSAVE - leading spaces in front

# open a CEF format FGM file
def fgmopen_cef(lines): # open FGM CEF file
    """Returns
    -------
    dataset dictionary with keys: dataset_id,data_start,data_end,t,\
        Bx,By,Bz,B,Px,Py,Pz,range,mode,positionFlag,rangeFlag"""
    # find some metadata
    dataset_id = None
    data_start = None
    data_end = None
    t = None
    Bx, By, Bz = None, None, None
    Px, Py, Pz = None, None, None
    rng, mode = None, None
    positionFlag, rangeFlag = None, None
    dataFlag = False
    
    for i in range(0,len(lines)):    
        
        if lines[i].find('START_META') != -1 and lines[i].find('LOGICAL_FILE_ID') != -1:
            dataset_id = lines[i+1][lines[i+1].index('\"')+1:len(lines[i+1])-2]
        if lines[i].find('DATA_UNTIL') != -1:
            if len(lines) > i+1:
                datastart = i+1
                dataFlag = True
   
    # print(datastart)
    # print(len(lines))
    
    if dataFlag: 
        t = [] # ISO time
        a = [] # half average interval [s] (discarded)
        Bx = [] # components [nT]
        By = []
        Bz = []
        B = [] # magnitude [nT] (discarded)
        Px = [] # position [km]
        Py = []
        Pz = []
        rng = [] # range
        mode = [] # mode [sps]
        
        for i in range(datastart,len(lines)):
            aline = lines[i]
            alist = aline.split(",")
            if len(alist) == 11:
                t.append(datetime.fromisoformat(alist[0]).replace(tzinfo=None))
                a.append(float(alist[1]))
                Bx.append(float(alist[2]))
                By.append(float(alist[3]))
                Bz.append(float(alist[4]))
                B.append(float(alist[5]))
                Px.append(float(alist[6]))
                Py.append(float(alist[7]))
                Pz.append(float(alist[8]))
                rng.append(int(alist[9]))
                mode.append(int(alist[10][0:2]))
            
        print("Dataset_ID %s" % dataset_id)
        records = len(t)
        data_start = t[0]
        data_end = t[len(t)-1]
        print("{} records start {} end {}".format(records,data_start.isoformat(timespec='seconds'),data_end.isoformat(timespec='seconds')))
        # print("Start time %s" % data_start.isoformat(timespec='seconds'))
        # print("End time   %s" % data_end.isoformat(timespec='seconds'))
        t = array(t)
        a = array(a)
        Bx = array(Bx)
        By = array(By)
        Bz = array(Bz)
        B = array(B)
        Px = array(Px)
        Py = array(Py)
        Pz = array(Pz)
        rng = array(rng)
        mode = array(mode)
        positionFlag = True
        rangeFlag = True

    DatasetDict = {'dataset_id': dataset_id, 'data_start': data_start,\
               'data_end':data_end, 't':t, \
                   'Bx':Bx, 'By':By, 'Bz':Bz, 'Px':Px, 'Py':Py, 'Pz':Pz,\
                       'range': rng, 'mode': mode, 'positionFlag':positionFlag,\
                           'rangeFlag':rangeFlag}
    return DatasetDict


def fgmopen_dp(lines): # open a dp output file
    t = []
    x = []
    y = []
    z = []
    # set the content flags; consider only two simple cases
    #  either with or without position data
    rangeFlag = False # not possibile with fgmvec or igmvec
    num_columns = len(lines[0].split())
    if num_columns == 4:
        positionFlag = False
    elif num_columns == 7:
        positionFlag = True
    else:
        print('Unexpected number of columns')
        return
    # set empty position lists
    if positionFlag:
        px = [] # optional position
        py = []
        pz = []
    # decode lines    
    for i in range(0,len(lines)):
        aline = lines[i]
        alist = aline.split()
        if alist[0][17:20] == '60.':
           print('Discarding bad time at line {:d}'.format(i+1)) # for now simply ignore bad times
        else:
            timestring = alist[0][0:len(alist[0])-1]
            t.append(datetime.fromisoformat(timestring).replace(tzinfo=None))
            x.append(float(alist[1]))
            y.append(float(alist[2]))
            z.append(float(alist[3]))
            if positionFlag:
                px.append(float(alist[4]))
                py.append(float(alist[5]))
                pz.append(float(alist[6]))
    # print results 
    records = len(t)
    print("Found %i records" % records)
    data_start = t[0]
    data_end = t[len(t)-1]
    print("Start time %s" % data_start.isoformat(timespec='seconds'))
    print("End time   %s" % data_end.isoformat(timespec='seconds'))
    
    t = array(t)
    x = array(x)
    y = array(y)
    z = array(z)
    
    if positionFlag:
        print("File contains position data")
        px = array(px)
        py = array(py)
        pz = array(pz)
    else:
        px = None
        py = None
        pz = None
        
    DatasetDict = {'dataset_id': None, 'data_start': data_start,\
                'data_end': data_end, 't': t, \
                    'Bx': x, 'By': y, 'Bz':z, 'Px': px, 'Py': py, 'Pz':pz,\
                        'range': None, 'mode': None, 'positionFlag':positionFlag,\
                            'rangeFlag':rangeFlag}
    return DatasetDict

    
def fgmopen(path,filename):
    # will return either a dict or None if the file can't be decoded
    # note that the dict may be empty if the file is a valid CEF but containing no data
    fullpath = path + '/' + filename
    # print("Opening {:s}".format(fullpath))
    extension = filename[len(filename)-3:len(filename)]
    lines = [] 
    if isfile(fullpath):
        with open(fullpath) as f:
            for row in f:
                lines.append(row)    
        if (extension == 'cef' or extension == 'CEF')\
            and (lines[0][0] == '!' or lines[0][0] == 'F'):
            DatasetDict = fgmopen_cef(lines)
        elif (extension == 'txt' or extension == 'asc') and lines[0][0:2] == '20':
            DatasetDict = fgmopen_dp(lines)
            dataset_id = filename[0:len(filename)-4]
            DatasetDict['dataset_id'] = dataset_id
        else:
            print('Unknown file format')
            return None
        return DatasetDict
    else:
        print('File does not exist!')
        return None
