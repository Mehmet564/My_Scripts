# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:08:32 2021

@author: phmeay
"""

import numpy as np
from scipy import signal
import  pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz
import mne
from scipy import ndimage
import mplcursors
import scipy.signal
%matplotlib 
# reading edf file 
file = "20210224104858_test1.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
info = data.info
channels = data.ch_names


# reading csv file with pandas this file edited from matlab due to some 
# problems. at some point delimiters are not same, and float data decimate by comma instead dot.
df = pd.read_csv('myData.csv', delimiter=' ')

# columns decimated by comma 
df2 = df.loc[:,'ax_g_':'T___']

# decimations converted to dot
df2 = df2.apply(lambda x: x.str.replace(',','.'))

#converting from pandas to numpy 
dataframe = df2.to_numpy().astype('float64') 

# C3,C1, Cz,C2,C4, and Oz data extracted from raw_data
EEG_signals = raw_data[1:8,:]
# plt.plot(EEG_signals[1])
# time data extracted from data frame
time = df.loc[:,'Time_s_']

# time to unix 
from datetime import datetime

date = '24.02.2021'
unixfile= []
for i in range(len(time)):
    time_stamp = str(date)+' ' + time[i]
    dt_obj = datetime.strptime(time_stamp,
                               '%d.%m.%Y %H.%M.%S.%f')
    millisec = dt_obj.timestamp() * 1000
    unixfile.append(millisec)
unixfile = np.array(unixfile)
n = len(dataframe) % 4
wx = dataframe[n:,3]


# defining zero crossing point where data change the sign from + to - 
def crossings_zero_pos2neg(data):
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0]

values =[-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
# removing value above and below 50 that make noise
k = np.where( (wx >-50) & (wx <50), 0, wx) 
k = np.where(k==0, 0.001, k)
c = crossings_zero_pos2neg(k)

EEG_signal = [EEG_signals[0,:],EEG_signals[1,:],EEG_signals[2,:],EEG_signals[3,:],EEG_signals[4,:],EEG_signals[5,:]]
EEG_signal_name =['C3','C1','Cz','C2','C4','Oz']
for x,j in zip(EEG_signal, EEG_signal_name):
    b, a = scipy.signal.butter(2, 0.035)
    Signal = scipy.signal.filtfilt(b, a, x)*1000000
    
    
    
    exracted_data =[]
    
    movement_time_in_edf=[]
    for i in range(len(c)):
        starting_point = c[i]
        corresponding_time_stamp = unixfile[starting_point]
        
        firstEEGtimestamp = 1614156547000.0
        
        movement_time_in_edf1 = int(abs(firstEEGtimestamp-corresponding_time_stamp)/2)
        if movement_time_in_edf1 > len(EEG_signals[0,:]) :
            pass
        
        else:
            movement_time_in_edf.append(movement_time_in_edf1)
                
    movement_time_in_edf = np.array(movement_time_in_edf)
    
    for k in range(len(movement_time_in_edf)):
        extracted_dataa = Signal[int(movement_time_in_edf[k]-1500):int(movement_time_in_edf[k]+500)]
        exracted_data.append(extracted_dataa)
        

    plt.show()       
    exracted_data = np.array(exracted_data)
    

    average = np.mean(exracted_data, axis =0)
    fig, ax = plt.subplots()
    plt.plot(average)
    plt.axvline(x=1500, c ='r')
    ax.set_title('Averaging of all segments of the {} with filter'.format(j))
    
    positions = (np.arange(0,2001,250))
    labels = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
    plt.xticks(positions, labels)
    plt.xlabel('Second (s)',loc='center')
    plt.ylabel('Amplitude (uV)',loc='center')
    plt.savefig('Averaging of all segments of the {} with filter'.format(j))
   
    mplcursors.cursor()
    plt.show()
    
    
    
    fig, ax = plt.subplots()
    plt.plot(Signal)
    ax.set_title('{} signal with filter'.format(j))
    
    mplcursors.cursor()
    # plt.show()
    for xc in movement_time_in_edf:
        plt.axvline(x=xc, c ='r')
    plt.savefig('{} signal with filter'.format(j))
    plt.show()





    fig, ax = plt.subplots()
    for i in exracted_data:
        
        plt.plot(i)
        plt.axvline(x=1500, c ='r')
        ax.set_title('All segments of the {} '.format(j))
               
        positions = (np.arange(0,2001,250))
        labels = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
        plt.xticks(positions, labels)
        plt.xlabel('Second (s)',loc='center')
        plt.ylabel('Amplitude (uV)',loc='center')
        plt.savefig('All segments of the {} '.format(j)) 
        # ax.invert_yaxis()
        mplcursors.cursor()
    plt.show()
    









