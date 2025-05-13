# packages
from matplotlib.pyplot import suptitle,xlabel,ylabel,plot,grid,legend,subplot,subplots
from datetime import datetime,timedelta
from numpy import sqrt,array,zeros,size,pi
from pandas import read_csv

# function definitions
def openraw(filedate,spacecraft):
    # open raw edited extended mode data file
    filename = filedate+'/'+spacecraft+'_'+filedate+'_ext_1_edited.txt'
    data = read_csv(filename,header=None)
    # change to array
    r = array(data[2][:])
    x = array(data[3][:])
    y = array(data[4][:])
    z = array(data[5][:])
    # make a default time-axis
    t = range(0,len(r)) 
    return t,x,y,z,r

def quickplot(t,x,y,z,r,titletext,xlabeltext,ylabeltext):
    subplots(5,1,sharex=True,height_ratios=[2,2,2,2,1])
    subplot(5,1,1);plot(t,x,label='x');grid();legend();ylabel(ylabeltext)
    subplot(5,1,2);plot(t,y,label='y');grid();legend();ylabel(ylabeltext)
    subplot(5,1,3);plot(t,z,label='z');grid();legend();ylabel(ylabeltext)
    b = sqrt(x**2+y**2+z**2)
    subplot(5,1,4);plot(t,b,label='B');grid();legend();ylabel(ylabeltext)
    subplot(5,1,5);plot(t,r,label='range');grid();legend()
    xlabel(xlabeltext)
    suptitle(titletext,y=0.94)
    # savefig(filepath+'/'+titletext+'.png',dpi=150)
    return

def quicksave(filename,t,x,y,z,r):
    file = open(filename,'w')
    for i in range(0,len(t)):
        # aline = t[i].isoformat(timespec='milliseconds')[0:23] + 'Z'
        aline = t[i].isoformat(timespec='milliseconds')
        aline += ", {0: 5d}, {1: 5d}, {2: 5d}, {3: 1d}\n".format(x[i],y[i],z[i],r[i])
        file.write(aline)
    file.close()
    return

def quickopen(filename):
    lines = [] 
    with open(filename) as f:
        for row in f:
            lines.append(row)    
        
    t = []
    x = []
    y = []
    z = []
    r = []
    for i in range(0,len(lines)):
        aline = lines[i]
        alist = aline.split(',')
        timestring = alist[0]#[0:len(alist[0])-1]
        t.append(datetime.fromisoformat(timestring))
        x.append(int(alist[1]))
        y.append(int(alist[2]))
        z.append(int(alist[3]))
        r.append(int(alist[4]))

    t = array(t)
    x = array(x)
    y = array(y)
    z = array(z)
    r = array(r)
    return t,x,y,z,r

def make_t(t_spin,length,ext_entry,ext_exit):
    t = []
    for i in range(1,length+1):
        t.append(ext_entry + timedelta(seconds=i*t_spin))
    print('Last vector time {}'.format(t[len(t)-1]))
    print('Ext Exit time {}'.format(ext_exit))
    print('Difference {}'.format(ext_exit - t[len(t)-1]))
    return t

def nominal_scaling(x,y,z,r):
    xx = x * (2*64/2**15) * 4**(r-2)
    yy = y * (2*64/2**15) * 4**(r-2) * (pi/4)
    zz = z * (2*64/2**15) * 4**(r-2) * (pi/4)
    return xx,yy,zz

def apply_calparams(x,y,z,r,calmatrix):
    xx,yy,zz = zeros(size(r)),zeros(size(r)),zeros(size(r))
    for i in range(0,len(r)):
        Ox = calmatrix['x_offsets'][r[i]-2]
        Gx = calmatrix['x_gains'][r[i]-2]
        Gyz = calmatrix['yz_gains'][r[i]-2]
        xx[i] = (x[i] - Ox) / Gx
        yy[i] = y[i] / Gyz
        zz[i] = z[i] / Gyz
    return xx,yy,zz